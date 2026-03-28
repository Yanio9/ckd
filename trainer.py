import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from dataset.utils import split_vfl
from models.MIMIC_models import SimplexAggregator
from tqdm import tqdm
from utils import *
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
import os
from torch.utils.data import DataLoader, Subset
from itertools import combinations
import copy
from random import sample
from attack.attack import attack_LFBA, get_anchor_LFBA, get_near_index
import time
# from Marvell.marvell_torch import MarvellConfig, MarvellSumKLPerturber
# from Marvell.marvell_torch_multiclass_ovr import MarvellConfig, MarvellMultiClassOVRPerturber
from Marvell.marvell_torch_multiclass_structured import MarvellConfig, MarvellMultiClassStructuredPerturber
from Marvell.marvell_torch import MarvellSumKLPerturber, BinaryMarvellConfig


def auc_from_logits(y_true: torch.Tensor, logits: torch.Tensor) -> float:
    """
    y_true: [N] int
    logits: [N,C]
    binary: AUC on prob[:,1]
    multiclass: macro OVR AUC
    """
    y = y_true.detach().cpu().numpy().astype(int)
    prob = F.softmax(logits.detach(), dim=1).cpu().numpy()
    if prob.shape[1] == 2:
        return float(roc_auc_score(y, prob[:, 1]))
    return float(roc_auc_score(y, prob, multi_class="ovr", average="macro"))

@torch.no_grad()
def clc_encode_binary_from_logits(active_logits: torch.Tensor,
                                 y: torch.Tensor,
                                 eps: float = 1e-6):
    """
    active_logits: [B,2]  (active party output logits f_theta)
    y:            [B]     (0/1)
    returns:
      w:     [B]          (normalized sample weights, sum=1 over batch)
      r_vec: [B,2]        (pseudo-residual vector in logit space)
      p1:    [B]          (p(y=1|x0))
    """
    # p1 = σ(fθ(i)) for class 1
    p = F.softmax(active_logits, dim=1)[:, 1]                      # [B]
    p = p.clamp(min=eps, max=1.0 - eps)

    # w_raw = p(1-p)
    w_raw = p * (1.0 - p)                                          # [B]
    w = w_raw / (w_raw.sum().clamp_min(eps))                       # batch-normalized weights

    # residual r = (y - p)/(p(1-p))
    y_f = y.float()
    r = (y_f - p) / (w_raw.clamp_min(eps))                         # [B]

    #
    r_clip = 10.0
    r = torch.clamp(r, min=-r_clip, max=r_clip)

    # map scalar residual to 2-logit residual
    r_vec = torch.stack([-0.5 * r, 0.5 * r], dim=1)                            # [B,2]
    # w = w_raw
    return w, r_vec, p


def weighted_mse(pred: torch.Tensor, target: torch.Tensor, w: torch.Tensor):
    # pred/target: [B,2], w: [B]
    per = (pred - target).pow(2).sum(dim=1)   # [B]
    return (w * per).sum() / (w.sum().clamp_min(1e-12))


def weighted_kl(teacher_prob: torch.Tensor, student_logprob: torch.Tensor, w: torch.Tensor):
    # teacher_prob: [B,2] softmax
    # student_logprob: [B,2] log_softmax
    # return scalar
    kl_per = F.kl_div(student_logprob, teacher_prob, reduction='none').sum(dim=1)  # [B]
    return (w * kl_per).sum() / (w.sum().clamp_min(1e-12))


# 多分类OVR版本的 CLC 编码构造函数
def build_ovr_clc_targets(
    active_logits: torch.Tensor,
    labels: torch.Tensor,
    eps: float = 1e-6,
    r_clip: float = 10.0,
):
    """
    基于主动方本地 logits 构造多分类 OvR-CLC 目标

    Args:
        active_logits: [B, C]
        labels:        [B]，类别索引
        eps:           数值稳定项
        r_clip:        对 pseudo-residual 做裁剪，防止爆炸

    Returns:
        W_cls:         [B, C]，按类别的 CLC 权重
        R_cls:         [B, C]，按类别的 OvR pseudo-residual
        W_sample:      [B]，给 p2a / p2p 用的样本级权重（标量）
        probs:         [B, C]，active local softmax 概率
    """
    assert active_logits.dim() == 2
    B, C = active_logits.shape

    probs = F.softmax(active_logits, dim=1)                    # [B, C]
    y_onehot = F.one_hot(labels, num_classes=C).float()        # [B, C]

    # OvR “二阶项”
    curvature = probs * (1.0 - probs)                          # [B, C]
    curvature = curvature.clamp_min(eps)

    # OvR pseudo-residual
    R_cls = (y_onehot - probs) / curvature                     # [B, C]
    R_cls = torch.clamp(R_cls, min=-r_clip, max=r_clip)

    # 类别维权重：每个类别内部归一化，最贴近二分类 Eq.(6) 的扩展思路
    W_cls = curvature / (curvature.sum(dim=0, keepdim=True) + eps)   # [B, C]

    # 样本级权重：给 p2a / p2p 用一个标量版本
    # 这里取各类别 curvature 的均值，再归一到均值约为 1，避免损失尺度乱飘
    W_sample = curvature.mean(dim=1)                           # [B]
    W_sample = W_sample / (W_sample.mean().detach() + eps)     # [B]

    return W_cls.detach(), R_cls.detach(), W_sample.detach(), probs.detach()


def clc_ovr_loss(
    h_pas: torch.Tensor,
    R_cls: torch.Tensor,
    W_cls: torch.Tensor,
):
    """
    多分类 OvR-CLC 损失

    Args:
        h_pas: [B, C]
        R_cls:[B, C]
        W_cls:[B, C]

    Returns:
        scalar loss
    """
    per_sample = (W_cls * (h_pas - R_cls).pow(2)).sum(dim=1)   # [B]
    return per_sample.mean()


def weighted_kl_div(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    T: float = 4.0,
    sample_weight: torch.Tensor = None,
):
    """
    带样本权重的 KL 蒸馏损失
    KL( teacher || student )

    Args:
        student_logits: [B, C]
        teacher_logits: [B, C]
        T: temperature
        sample_weight: [B] or None
    """
    log_p_student = F.log_softmax(student_logits / T, dim=1)   # [B, C]
    p_teacher = F.softmax(teacher_logits / T, dim=1)           # [B, C]

    kl = F.kl_div(log_p_student, p_teacher, reduction="none").sum(dim=1)  # [B]
    kl = kl * (T * T)

    if sample_weight is not None:
        return (kl * sample_weight).sum() / sample_weight.sum().clamp_min(1e-12)
    
    return kl.mean()

# CKD多分类严格扩展 
@torch.no_grad()
def build_strict_clc_targets(
    active_logits: torch.Tensor,
    labels: torch.Tensor,
    eps: float = 1e-6,
    r_clip: float = 10.0,
    second_order: str = "pinv",      # "pinv" | "diag"
    weight_mode: str = "uncertainty" # "uncertainty" | "entropy" | "uniform"
):
    """
    严格版 CLC 目标构造：
    - 二分类：调用原 CKD exact CLC
    - 多分类：softmax 残差 + Hessian 二阶缩放 + 中心化

    Args:
        active_logits: [B,C] 或 [B] / [B,1]
        labels:        [B]
        eps:           数值稳定项
        r_clip:        残差逐维裁剪
        second_order:
            - "pinv": 使用 softmax Hessian 的 Moore-Penrose 逆
            - "diag": 使用对角近似 p*(1-p)
        weight_mode:
            - "uncertainty": w_i ∝ trace(H_i)=sum_c p_c(1-p_c)
            - "entropy":     w_i ∝ entropy(p_i)
            - "uniform":     w_i = 1

    Returns:
        W_sample: [B]    样本权重（相对权重，均值归一到约 1）
        R_clc:    [B,C]  CLC 残差目标
        probs:    [B,C]  active local softmax 概率
        info:     dict   便于日志打印
    """
    # -------------------------
    # binary: keep exact CKD
    # -------------------------
    if active_logits.dim() == 1 or (active_logits.dim() == 2 and active_logits.size(1) <= 2):
        if active_logits.dim() == 1:
            active_logits = active_logits.unsqueeze(1)

        if active_logits.size(1) == 1:
            zeros = torch.zeros_like(active_logits)
            active_logits = torch.cat([zeros, active_logits], dim=1)

        W_sample, R_clc, p1 = clc_encode_binary_from_logits(
            active_logits=active_logits,
            y=labels,
            eps=eps
        )

        # 为了和你现有日志尺度一致，归一到 mean≈1
        W_sample = W_sample / W_sample.mean().clamp_min(eps)

        probs = F.softmax(active_logits, dim=1)
        info = {
            "mode": "binary_exact",
            "second_order": "exact",
            "weight_mode": "p(1-p)",
            "W_mean": float(W_sample.mean().item()),
            "W_max": float(W_sample.max().item()),
            "R_mean": float(R_clc.mean().item()),
            "R_std": float(R_clc.std().item()),
            "R_abs_max": float(R_clc.abs().max().item()),
        }
        return W_sample.detach(), R_clc.detach(), probs.detach(), info

    # -------------------------
    # multiclass strict CLC
    # -------------------------
    assert active_logits.dim() == 2
    B, C = active_logits.shape

    probs = F.softmax(active_logits, dim=1)                          # [B,C]
    y_onehot = F.one_hot(labels.long(), num_classes=C).float()      # [B,C]

    # 一阶残差：主动方尚未解释掉的部分
    g = y_onehot - probs                                             # [B,C]

    # 样本权重
    if weight_mode == "uncertainty":
        # trace(H) = 1 - ||p||^2 = sum_c p_c(1-p_c)
        W_sample = (probs * (1.0 - probs)).sum(dim=1)               # [B]
    elif weight_mode == "entropy":
        ent = -(probs.clamp_min(eps) * probs.clamp_min(eps).log()).sum(dim=1)
        W_sample = ent / np.log(C)
    elif weight_mode == "uniform":
        W_sample = torch.ones(B, device=active_logits.device, dtype=active_logits.dtype)
    else:
        raise ValueError(f"Unknown weight_mode={weight_mode}")

    # 归一到 mean≈1，与你现有 weighted_mse / weighted_kl_div 更协调
    W_sample = W_sample / W_sample.mean().clamp_min(eps)

    # 二阶缩放
    if second_order == "diag":
        h_diag = (probs * (1.0 - probs)).clamp_min(eps)             # [B,C]
        R_clc = g / h_diag

    elif second_order == "pinv":
        # H = diag(p) - p p^T
        H = torch.diag_embed(probs) - torch.bmm(
            probs.unsqueeze(2), probs.unsqueeze(1)
        )                                                           # [B,C,C]

        # pinv 用 double 更稳，再转回原 dtype
        H_pinv = torch.linalg.pinv(H.double(), hermitian=True).to(active_logits.dtype)
        R_clc = torch.bmm(H_pinv, g.unsqueeze(-1)).squeeze(-1)      # [B,C]
    else:
        raise ValueError(f"Unknown second_order={second_order}")

    # softmax 子空间中心化：sum_c r_c = 0
    R_clc = R_clc - R_clc.mean(dim=1, keepdim=True)

    # 裁剪，防止少数极端样本爆炸
    R_clc = torch.clamp(R_clc, min=-r_clip, max=r_clip)

    info = {
        "mode": "multiclass_strict_clc",
        "second_order": second_order,
        "weight_mode": weight_mode,
        "num_classes": int(C),
        "W_mean": float(W_sample.mean().item()),
        "W_max": float(W_sample.max().item()),
        "R_mean": float(R_clc.mean().item()),
        "R_std": float(R_clc.std().item()),
        "R_abs_max": float(R_clc.abs().max().item()),
    }

    return W_sample.detach(), R_clc.detach(), probs.detach(), info


@torch.no_grad()
def build_margin_target_logits(
    active_logits: torch.Tensor,
    labels: torch.Tensor,
    teacher_margin: float = 5.0,
):
    """
    构造 logit-space teacher target:
    active_logits: [B, C]
    labels:        [B]
    return:
        z_target:   [B, C]
    """
    assert active_logits.dim() == 2
    B, C = active_logits.shape

    y_onehot = F.one_hot(labels.long(), num_classes=C).float()

    if C == 1:
        # 理论上你现在代码基本不会走到这里，因为全流程都按 [B,C] 多类 logits 写的
        z_target = (2.0 * labels.float().view(-1, 1) - 1.0) * teacher_margin
    else:
        neg_val = -teacher_margin / max(C - 1, 1)
        z_target = y_onehot * teacher_margin + (1.0 - y_onehot) * neg_val

    return z_target


# 多分类教师残差蒸馏版 CLC 目标构造函数
@torch.no_grad()
def build_teacher_residual_targets(
    active_logits: torch.Tensor,
    labels: torch.Tensor,
    mode: str = "logit_margin",
    teacher_margin: float = 5.0,
    weight_mode: str = "uncertainty",
    teacher_logits: torch.Tensor = None,
    eps: float = 1e-6,
):
    """
    教师残差蒸馏版 CLC 目标构造

    Args:
        active_logits: [B, C]
        labels:        [B]
        mode:
            - "logit_margin":   R = z_target_margin - z_active
            - "prob_residual":  R = onehot(y) - softmax(z_active)
            - "logit_teacher":  R = z_teacher - z_active
        weight_mode:
            - "uniform"
            - "uncertainty"   : 1 - max prob
            - "entropy"
            - "residual"      : ||R||^2
        teacher_logits:
            仅在 mode="logit_teacher" 时使用
    Returns:
        W_sample: [B]
        R_teacher:[B, C]
        probs:    [B, C]
    """
    assert active_logits.dim() == 2
    B, C = active_logits.shape

    probs = F.softmax(active_logits, dim=1)  # [B, C]

    # -------- residual target --------
    if mode == "logit_margin":
        z_target = build_margin_target_logits(
            active_logits=active_logits,
            labels=labels,
            teacher_margin=teacher_margin,
        )
        R_teacher = z_target - active_logits

    elif mode == "prob_residual":
        y_onehot = F.one_hot(labels.long(), num_classes=C).float()
        R_teacher = y_onehot - probs

    elif mode == "logit_teacher":
        if teacher_logits is None:
            raise ValueError("teacher_logits is required when mode='logit_teacher'")
        assert teacher_logits.shape == active_logits.shape
        R_teacher = teacher_logits - active_logits

    else:
        raise ValueError(f"Unknown teacher residual mode: {mode}")

    # -------- sample weights --------
    if weight_mode == "uniform":
        W_sample = torch.ones(B, device=active_logits.device)

    elif weight_mode == "uncertainty":
        # 主动方越不确定，权重越大
        W_sample = 1.0 - probs.max(dim=1).values

    elif weight_mode == "entropy":
        entropy = -(probs.clamp_min(eps) * probs.clamp_min(eps).log()).sum(dim=1)
        W_sample = entropy / np.log(C)

    elif weight_mode == "residual":
        W_sample = R_teacher.pow(2).mean(dim=1)

    else:
        raise ValueError(f"Unknown teacher residual weight mode: {weight_mode}")

    # 归一化到均值约为 1，避免损失尺度乱飘
    W_sample = W_sample / (W_sample.mean().clamp_min(eps))

    return W_sample.detach(), R_teacher.detach(), probs.detach()


class Trainer:
    def __init__(self, args, model_list, optimizer_list, criterion, train_loader, val_loader,
                test_loader, test_asr_loader, device, simplex, logger, trigger_dimensions=None):
        self.model_list = model_list
        self.optimizer_list = optimizer_list
        self.criterion = criterion
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.test_asr_loader = test_asr_loader
        self.args = args
        self.device = device
        self.kld = nn.KLDivLoss(reduction="none")
        self.simplex = simplex.to(self.device)
        self.logger = logger
        self.trigger_dimensions = trigger_dimensions

        # Marvell 防御：初始化 perturber（只初始化一次）
        self.marvell = None
        if getattr(self.args, "defense", None) == "marvell":
            cfg = MarvellConfig(
                p_frac="pos_frac",                 # 与 TF 对齐：按 batch 正类比例
                # dynamic=False,                     # 先用非 dynamic，复现实验更稳
                dynamic=getattr(self.args, "marvell_dynamic", False),
                error_prob_lower_bound=self.args.marvell_error_prob_lower_bound
                    if hasattr(self.args, "marvell_error_prob_lower_bound") else None,
                sumKL_threshold=self.args.marvell_sumKL_threshold
                    if hasattr(self.args, "marvell_sumKL_threshold") else None,
                init_scale=self.args.marvell_init_scale
                    if hasattr(self.args, "marvell_init_scale") else 1.0,
                uv_choice=getattr(self.args, "marvell_uv_choice", "uv"),
                dynamic_scale_mul=getattr(self.args, "marvell_dynamic_scale_mul", 1.5),

                # 新增
                # multiclass_mode=getattr(self.args, "marvell_multiclass_mode", "ovr"),
                # multiclass_budget=getattr(self.args, "marvell_multiclass_budget", "uniform"),
                # min_class_count=getattr(self.args, "marvell_min_class_count", 2),
                # eps=getattr(self.args, "marvell_eps", 1e-12),

                # 严格多分类结构版
                multiclass_mode="structured",
                min_class_count=getattr(self.args, "marvell_min_class_count", 2),
                eps=getattr(self.args, "marvell_eps", 1e-12),
                svd_eps=getattr(self.args, "marvell_svd_eps", 1e-8),
                max_iter=getattr(self.args, "marvell_max_iter", 500),
                lr=getattr(self.args, "marvell_lr", 0.1),
                tol=getattr(self.args, "marvell_tol", 1e-10),
                use_perp_budget=getattr(self.args, "marvell_use_perp_budget", False),
                budget_base=getattr(self.args, "marvell_budget_base", "trace_sb"),
                structured_weight=getattr(self.args, "marvell_structured_weight", "papb"),
                class_var_mode=getattr(self.args, "marvell_class_var_mode", "mean_sq_norm"),
            )
            # self.marvell = MarvellSumKLPerturber(cfg)
            # self.marvell = MarvellMultiClassOVRPerturber(cfg)
            self.marvell = MarvellMultiClassStructuredPerturber(cfg)


            # marvell 原论文二分类
            # ===== binary Marvell: 原论文二分类版本 =====
            binary_cfg = BinaryMarvellConfig(
                p_frac="pos_frac",
                dynamic=getattr(self.args, "marvell_dynamic", False),
                init_scale=getattr(self.args, "marvell_init_scale", 1.0),
                uv_choice=getattr(self.args, "marvell_uv_choice", "uv"),
                dynamic_scale_mul=getattr(self.args, "marvell_dynamic_scale_mul", 1.5),
                error_prob_lower_bound=getattr(self.args, "marvell_error_prob_lower_bound", None),
                sumKL_threshold=getattr(self.args, "marvell_sumKL_threshold", None),
            )
            self.marvell_binary = MarvellSumKLPerturber(binary_cfg)


        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            )

    def adjust_learning_rate(self, epoch):
        lr = self.args.lr * (0.1) ** (epoch // 5)
        for opt in self.optimizer_list:
            for param_group in opt.param_groups:
                param_group['lr'] = lr

    def train(self):
        start_time_train = time.time()
        self.logger.info(
            "Start training: epochs=%s, client_num=%s, kd=%s, kd_T=%s, device=%s",
            self.args.epochs,
            self.args.client_num,
            self.args.kd,
            self.args.kd_T,
            self.device,
        )
        best_acc = 0.0
        train_losses = []
        val_losses = []
        acc = []
        trian_step_losses = []
        best_epoch = 0
        best_score = 0.0
        no_change = 0
        
        # LFBA
        best_trade_off = 0
        asr_for_best_epoch = 0
        target_for_best_epoch = 0
        total_time_GPC = 0
        total_time_HS = 0
        self.select_his = torch.zeros(self.train_loader.dataset.data.shape[0])
        epoch_loss_list = []

        # simplex layer freezing epoch (optional, can help stabilize training in early epochs, not required by CKD core)
        freeze_simplex_after = getattr(self.args, "freeze_simplex_after", 3)
        simplex_frozen = False

        # Create directory for saving best models
        save_dir = getattr(self.args, "results_dir", "./checkpoints")
        os.makedirs(save_dir, exist_ok=True) # 不存在就创建，存在就直接用
        best_path = os.path.join(save_dir, f"best_model.pt")
        best_metric = getattr(self.args, "best_metric", "auc")  # "auc" or "acc"

        
        # 初始化模型，移动到设备
        for model in self.model_list:
            model.to(self.device)
        self.simplex.to(self.device)

        
        best_full_auc = -1.0
        best_full_epoch = -1
        best_robust_score = -1.0
        best_robust_epoch = -1

        history = {
            "epoch": [],
            "train_loss": [],
            "val_auc": [],
            "L_util_auc": [],
            "full_auc": [],
            "robust_score": [],
            "pmc_mean_auc": [],   # 如果你没有 PMC 评估函数，可以先记 None
        }

        best_full_state = None
        best_robust_state = None
        
        # active party cold start epochs (local warmup)
        cold_epochs = getattr(self.args, "cold_start_epochs", 0)
        if cold_epochs > 0:
            self.cold_start_active_only(cold_epochs)
        
        # save cold start model
        torch.save(self.model_list[0].state_dict(), os.path.join(self.args.results_dir, "active_coldstart.pt"))

        # 改良后无需使用list存储
        # global_soft_predict = []
        # local_soft_predict = []
        # train loop
        early_stopped = False
        stop_epoch = None
        for epoch in range(self.args.epochs):
            self.logger.info("Epoch %s/%s started", epoch + 1, self.args.epochs)
            # 模型训练阶段
            for model in self.model_list:
                model.train()
            # self.simplex.train()
            if not simplex_frozen:
                self.simplex.train()

            # 单形层冻结（可选）
            if (not simplex_frozen) and (epoch + 1 >= freeze_simplex_after):
                self.freeze_simplex()
                simplex_frozen = True
            
            running_train_loss = 0.0
            batch_loss_list = []
            total = 0
            correct = 0
            marvell_logged_this_epoch = False  # Marvell：控制每个 epoch 只打印一次

            # LFBA
            if epoch >= 1 and self.args.attack == 'LFBA':
                self.train_features, self.train_labels, self.train_indexes = self.grad_vec_epoch, self.target_epoch, self.indexes_epoch
                self.train_features, self.train_labels, self.train_indexes = self.train_features.cpu(), self.train_labels.cpu(), self.train_indexes.cpu()
                self.num_poisons = int(self.args.poison_rate * len(self.train_loader.dataset.data))
                self.num_select = int(self.num_poisons * self.args.select_rate)

                if epoch == 1:
                    start_time = time.time()
                    self.anchor_idx_t = torch.nonzero(self.train_indexes == self.args.anchor_idx).squeeze()
                    self.indexes = get_near_index(self.train_features[self.anchor_idx_t], self.train_features,
                                                    self.num_poisons) # 
                    end_time = time.time()
                    print("The poison set construction time: {}".format((end_time - start_time)))
                    total_time_GPC += (end_time - start_time)
                    self.poison_indexes = self.train_indexes[self.indexes] # 取出毒化集的全局索引并存储
                    self.consistent_rate = float(
                        (self.train_labels[self.indexes] == int(self.train_labels[self.anchor_idx_t])).sum() / len(
                            self.indexes))  # 直接取出的train_labels[self.anchor_idx_t]是一个tensor，如果不强制转int，后续计算会因为广播问题报错
                    # 个人验证
                    anchor_label = int(self.train_labels[self.anchor_idx_t])
                    self.logger.info(
                        "[LFBA] repr=%s epoch=%d anchor_global_idx=%d anchor_local_idx=%d "
                        "anchor_label=%d poison_set_size=%d poison_consistent_rate=%.6f",
                        self.args.attack_repr,
                        epoch + 1,
                        int(self.args.anchor_idx),
                        int(self.anchor_idx_t),
                        anchor_label,
                        len(self.indexes),
                        self.consistent_rate,
                    )

                # For replace poisoning
                self.indexes = np.isin(self.train_indexes.numpy(), torch.tensor(self.poison_indexes).numpy())
                temp = np.array(range(len(self.train_indexes))) 
                self.indexes = temp[self.indexes] # 得到毒化集D_p的局部索引，即train_indexes数组的下标(索引)，可通过train_indexes找到毒化集的全局下标
                self.l2_norm_features = torch.norm(self.train_features[self.indexes], p=2, dim=1)
                start_time = time.time()
                # 得到D_s的局部索引，l2_norm_features数组中的下标(索引)
                self.poison_features, self.select_indexes = self.l2_norm_features.topk(self.num_select, dim=0,
                                                                                        largest=True,
                                                                                        sorted=True)
                
                # 个人验证
                anchor_label = int(self.train_labels[self.anchor_idx_t])

                selected_local_idx_in_train = self.indexes[self.select_indexes]
                selected_labels = self.train_labels[selected_local_idx_in_train]

                selected_target_ratio = float(
                    (selected_labels == anchor_label).sum().item() / max(1, len(selected_labels))
                )

                self.logger.info(
                    "[LFBA] repr=%s epoch=%d selected_num=%d selected_target_ratio=%.6f "
                    "selected_norm_mean=%.6f selected_norm_max=%.6f",
                    self.args.attack_repr,
                    epoch + 1,
                    len(self.select_indexes),
                    selected_target_ratio,
                    float(self.poison_features.float().mean().item()),
                    float(self.poison_features.float().max().item()),
                )

                end_time = time.time()
                print("The hard-sample selection time: {}".format((end_time - start_time)))
                total_time_HS += (end_time - start_time)
                num_of_replace = int(len(self.poison_indexes) * self.args.select_rate)
                replace_all_list = list(set(self.train_indexes.numpy()).difference(set(torch.tensor(self.poison_indexes).numpy())))
                replace_indexes_others = sample(replace_all_list, num_of_replace) # 替换样本的全局索引(随机抽取方式选出)
                random_indexes_target = sample(list(self.poison_indexes), num_of_replace) # 待替换样本的全局索引(随机抽取方式选出)
                selected_indexes_target = self.train_indexes[self.indexes[self.select_indexes]] # 待替换样本的全局索引(模长topk方式选出)

                # poison_all = True + random_select = True 时，表示在毒化集D_p中随机选取子集进行注入，不替换样本特征 RGPC
                # poison_all = False + random_select = False 时，表示在毒化集D_p中选取模长topk子集进行注入，替换样本特征(替换的样本是随机选取) LFBA
                # poison_all = False + random_select = True 时，表示在毒化集D_p中随机选取子集进行注入，替换样本特征(替换的样本是随机选取) RS-GPC
                # poison_all = True + random_select = False 时，表示对整个毒化集D_p进行注入，不替换样本特征 DGPC
                if self.args.poison_all:
                    if self.args.random_select:
                        self.poison_indexes_t = sample(list(self.poison_indexes), self.num_select)
                        self.indexes = np.isin(self.train_indexes.numpy(), torch.tensor(self.poison_indexes_t).numpy())
                    self.poisoning_labels = np.array(self.train_labels)[self.indexes]
                    self.anchor_label = int(self.train_labels[self.train_indexes == self.args.anchor_idx])
                    self.args.target_label = self.anchor_label
                    self.logger.info('Target label:{}'.format(self.anchor_label))
                    self.clean_data_p = copy.deepcopy(self.train_loader.dataset.data_p)
                    if self.args.random_select:
                        self.train_loader.dataset.data = attack_LFBA(self.args, self.logger, [],
                                                                    [], self.train_indexes,
                                                                    self.poison_indexes_t,
                                                                    self.clean_data_p, 
                                                                    self.train_loader.dataset.targets,
                                                                    self.trigger_dimensions,
                                                                    self.args.poison_rate, 'train')
                    else:
                        self.train_loader.dataset.data = attack_LFBA(self.args, self.logger, [],
                                                                    [], self.train_indexes,
                                                                    self.poison_indexes,
                                                                    self.clean_data_p,
                                                                    self.train_loader.dataset.targets,
                                                                    self.trigger_dimensions,
                                                                    self.args.poison_rate, 'train')
                else:
                    if self.args.random_select:
                        replace_indexes_target = random_indexes_target
                    else:
                        replace_indexes_target = selected_indexes_target
                    self.poisoning_labels = np.array(self.train_labels)[self.indexes]
                    self.anchor_label = int(self.train_labels[self.train_indexes == self.args.anchor_idx])
                    self.clean_data_p = copy.deepcopy(self.train_loader.dataset.data_p)
                    
                    # 个人验证 cifa-10
                    # print(self.clean_data_p.shape) # shape: (len(train_data), 32, 32, 3), HWC

                    self.train_loader.dataset.data = attack_LFBA(self.args, self.logger, replace_indexes_others,
                                                                replace_indexes_target, self.train_indexes,
                                                                self.poison_indexes,
                                                                self.clean_data_p,
                                                                self.train_loader.dataset.targets,
                                                                self.trigger_dimensions,
                                                                self.args.poison_rate, 'train')
                    self.args.target_label = self.anchor_label
                    self.logger.info('Target label:{}'.format(self.anchor_label))

            elif self.args.attack == 'rsa' or self.args.attack == 'lra' or self.args.attack is None:
                pass

            self.logger.info("=> Start Training for Injecting Backdoor...")

            self.grad_vec_epoch = []
            self.indexes_epoch = []
            self.target_epoch = []

            # 没有使用LFBA攻击之前的loader循环
            # for step, (inputs, labels) in enumerate(tqdm(self.train_loader, total=len(self.train_loader), desc=f"Epoch {epoch+1}/{self.args.epochs} - Training")):
            # LFBA攻击之后的loader循环
            for step, (inputs, x_p, labels, index) in enumerate(tqdm(self.train_loader, total=len(self.train_loader), desc=f"Epoch {epoch+1}/{self.args.epochs} - Training")):
   
                # alpha_ckd = 1.5     # merge α
                # beta_kd   = 3.0     # KD β
                alpha_ckd = self.args.alpha
                beta_kd = self.args.beta
                t = self.args.kd_T
                eps = 1e-4
                
                inputs = inputs.to(self.device).float()
                labels = labels.to(self.device).long()
                index = index.to(self.device).long()
                # split inputs for active and passive parties
                inputs_list = split_vfl(inputs, self.args)

                # if step == 0 and epoch >= 1 and self.args.kd:
                #     print(f"CE={local_loss.item():.4f} KD={kd_loss.item():.4f} alpha*KD={(alpha*kd_loss).item():.4f}")

                # 前向传播
                local_outputs_logits_list = [] # 存储每个模型的logits输出(embedding输出)
                local_outputs_classifier_list = [] # 存储每个模型的预测输出
                global_input_list = [] # 存储每个模型输入单形层的特征（即logits输出）
                # for model in self.model_list[1:]:
                #     outputs = model(inputs)
                #     local_outputs_list.append(outputs)
                for i in range(self.args.client_num):
                    local_classifier_output, local_outputs_logits = self.model_list[i + 1](inputs_list[i + 1])
                    local_outputs_logits_list.append(local_outputs_logits)
                    local_outputs_classifier_list.append(local_classifier_output)

                # LFBA: get the global model inputs, recording the gradients
                for i in range(self.args.client_num):
                    global_input_t = local_outputs_classifier_list[i].detach().clone()
                    global_input_t.requires_grad_(True)
                    global_input_list.append(global_input_t)
                    local_outputs_classifier_list[i].requires_grad_(True)
                    local_outputs_classifier_list[i].retain_grad()
                    inputs_list[i + 1].requires_grad_(True)
                    inputs_list[i + 1].retain_grad()
                    local_outputs_logits_list[i].requires_grad_(True)
                    local_outputs_logits_list[i].retain_grad()

                
                # # 清空上一轮的软标签
                # global_soft_predict = []
                # local_soft_predict = []

                # 全局模型前向传播
                # global_inputs = torch.tensor(local_outputs_logits_list)
                # global_inputs = local_outputs_logits_list
                # # global_inputs = torch.tensor([item.cpu().detach().numpy() for item in local_outputs_logits_list]).to(self.device)
                # # print(global_inputs.shape)
                # global_outputs = self.model_list[0](global_inputs)

                # # 全局模型前向传播 利用单形层聚合后输入全局模型
                # simplex_outputs, simplex_weights = self.simplex(local_outputs_logits_list) # [B, hidden_dim], [K], 这里的hidden_dim是单形层输出的维度
                # global_outputs = self.model_list[0](simplex_outputs)

                # 全局模型前向传播 利用单形层聚合后输入全局模型，且单形层输出维度与每个参与方logits维度相同（即不使用额外的线性层调整维度）
                # 设置掩码，模拟参与方缺失（训练时不使用，测试时使用） available_mask: [B,K]
                B = inputs.size(0)
                K = self.args.client_num
                avail_mask = torch.ones(B, K, dtype=torch.bool, device=self.device)
                # 组装 party_logits 输入单形层，维度为 [B, K, C]，其中C是每个参与方logits的维度
                party_logits = torch.stack(local_outputs_classifier_list, dim = 1) # [B, K, C], 这里的C是类别数量
                # simplex 聚合被动方：输出就是聚合后的分类 logits，维度为 [B, C]，权重维度为 [B, K]
                # simplex_outputs, simplex_weights, simplex_outputs_embedding = self.simplex(party_logits, avail_mask) # [B, C], [B, K], 这里的C是类别数量
                simplex_outputs, simplex_weights, simplex_outputs_embedding, simplex_aux, simplex_outputs_pre = self.simplex(party_logits, avail_mask)
                # 主动方
                active_outputs_classifier, active_outputs_logits = self.model_list[0](inputs_list[0]) # [B, C]
                # global_outputs = self.model_list[0](simplex_outputs)
                # global_outputs = active_output + self.args.alpha * simplex_outputs # [B, C]
                global_outputs = active_outputs_classifier + alpha_ckd * simplex_outputs

                # clc encode (w, r)   论文中的二分类CLC编码构造函数，基于主动方的分类输出logits构造CLC权重和残差
                # w, r_vec, _p1 = clc_encode_binary_from_logits(active_outputs_classifier.detach(), labels, eps=eps)  # detach per paper logic

                # 多分类版本的 clc encode (W_cls, R_cls, W_sample)  基于主动方的分类输出logits构造多分类 OvR-CLC 权重和残差
                # ===== 3) 用 active local logits 构造 OvR-CLC 目标 =====
                # with torch.no_grad():
                #     W_cls, R_cls, W_sample, active_probs = build_ovr_clc_targets(
                #         active_logits=active_outputs_classifier.detach(),
                #         labels=labels,
                #         eps=getattr(self.args, "ovr_eps", 1e-6),
                #         r_clip=getattr(self.args, "ovr_r_clip", 10.0),
                #     )

                # 多分类教师残差蒸馏版 CLC 目标构造，基于主动方的分类输出logits构造教师残差R_teacher和权重W_sample，mode和weight_mode可选
                # with torch.no_grad():
                #     W_sample, R_teacher, active_probs = build_teacher_residual_targets(
                #         active_logits=active_outputs_classifier.detach(),
                #         labels=labels,
                #         mode=getattr(self.args, "teacher_residual_space", "logit_margin"),
                #         teacher_margin=getattr(self.args, "teacher_margin", 5.0),
                #         weight_mode=getattr(self.args, "clc_weight_mode", "uncertainty"),
                #         teacher_logits=None,   # 现在先不用额外 teacher，后面想扩展再传
                #         eps=getattr(self.args, "teacher_residual_eps", 1e-6),
                #     )

                # CKD 严格多分类 CLC目标构造
                with torch.no_grad():
                    W_sample, R_clc, active_probs, clc_info = build_strict_clc_targets(
                        active_logits=active_outputs_classifier.detach(),
                        labels=labels,
                        eps=getattr(self.args, "clc_eps", 1e-6),
                        r_clip=getattr(self.args, "clc_r_clip", 10.0),
                        second_order=getattr(self.args, "clc_second_order", "pinv"),
                        weight_mode=getattr(self.args, "clc_weight_mode", "uncertainty"),
                    )


                if step == 0:
                    self.logger.info(
                        "Epoch %s first batch prepared: batch_size=%s, simplex_output_shape=%s, simplex_weights=%s",
                        epoch + 1,
                        B,
                        tuple(simplex_outputs.shape),
                        simplex_weights[0].detach().cpu().numpy()
                    )
                    
                    # CKD 多分类版本2
                    # self.logger.info(
                    #     "Epoch %s residual target stats: mode=%s margin=%.4f weight_mode=%s "
                    #     "R_teacher_mean=%.6f R_teacher_std=%.6f W_sample_mean=%.6f W_sample_max=%.6f",
                    #     epoch + 1,
                    #     getattr(self.args, "teacher_residual_space", "logit_margin"),
                    #     float(getattr(self.args, "teacher_margin", 5.0)),
                    #     getattr(self.args, "clc_weight_mode", "uncertainty"),
                    #     float(R_teacher.mean().item()),
                    #     float(R_teacher.std().item()),
                    #     float(W_sample.mean().item()),
                    #     float(W_sample.max().item()),
                    # )
                    # print("simplex w:", simplex_weights[0].detach().cpu().numpy())
                    # print("simplex weights:", simplex_weights.detach().cpu().numpy())

                    # CKD 严格多分类版本
                    self.logger.info(
                        "Epoch %s CLC stats: mode=%s second_order=%s weight_mode=%s "
                        "R_mean=%.6f R_std=%.6f R_abs_max=%.6f W_mean=%.6f W_max=%.6f",
                        epoch + 1,
                        clc_info["mode"],
                        clc_info["second_order"],
                        clc_info["weight_mode"],
                        clc_info["R_mean"],
                        clc_info["R_std"],
                        clc_info["R_abs_max"],
                        clc_info["W_mean"],
                        clc_info["W_max"],
                    )
                
                # 计算损失(采用clc encode) 教师模型采用simplex聚合后的输出logits作为软标签，蒸馏采用两条路线p2p和p2a
                # ===== losses =====

                # active local CE  
                L_loc = self.criterion(active_outputs_classifier, labels)

                # passive CLC (THIS is the key)  二分类CLC损失，教师模型采用simplex聚合后的输出logits作为软标签，残差r_vec作为目标，权重w作为样本权重
                # L_clc = weighted_mse(simplex_outputs, r_vec, w)

                # passive OvR-CLC  多分类损失，教师模型采用 simplex 聚合后的输出 logits 作为软标签，残差 R_cls 作为目标，类别权重 W_cls 作为权重
                # L_clc = clc_ovr_loss(simplex_outputs_pre, R_cls, W_cls)
                # L_clc = clc_ovr_loss(simplex_outputs, R_cls, W_cls)

                # 教师残差蒸馏版 CLC 损失，教师模型采用 simplex 聚合后的输出 logits 作为软标签，残差 R_teacher 作为目标，样本权重 W_sample 作为权重
                # L_clc = weighted_mse(simplex_outputs, R_teacher, W_sample)

                # CKD 严格多分类CLC损失
                L_clc = weighted_mse(simplex_outputs, R_clc, W_sample)

                # entropy of simplex weights (encourage exploration, optional)
                L_entropy = -simplex_aux["mean_entropy"]

                if epoch >= self.args.kd_start_epoch and self.args.kd:
                    if self.args.p2p:
                        # ---- p2p: teacher = simplex_outputs, students = each passive output ----
                        # teacher_p = F.softmax(simplex_outputs.detach() / t, dim=1)
                        # teacher_p = F.softmax(simplex_outputs_pre.detach() / t, dim=1) # 如果单形层输出前没有额外的线性层调整维度，那么 simple_outputs_pre 就是输入全局模型的聚合后 logits，可以直接用来做 p2p 蒸馏目标，更符合 CKD 论文中“教师模型为单形层聚合后输出”的描述
                        teacher_p = F.softmax(simplex_outputs.detach() / t, dim=1)
                        L_p2p = 0.0
                        for i in range(self.args.client_num):
                            student_logp = F.log_softmax(local_outputs_classifier_list[i] / t, dim=1)
                            L_p2p = L_p2p + (t * t) * weighted_kl(teacher_p, student_logp, W_sample)
                        L_p2p = L_p2p / self.args.client_num
                    else:
                        L_p2p = torch.tensor(0.0, device=self.device)


                    if self.args.p2a:
                        # ---- p2a: teacher = f_ckd, student = active ----
                        simplex_outputs_detached = simplex_outputs.detach()
                        f_ckd = active_outputs_classifier + alpha_ckd * simplex_outputs_detached
                        teacher_p2a = F.softmax(f_ckd.detach() / t, dim=1)
                        student_logp2a = F.log_softmax(active_outputs_classifier / t, dim=1)
                        L_p2a = (t * t) * weighted_kl(teacher_p2a, student_logp2a, W_sample)
                    else:
                        L_p2a = torch.tensor(0.0, device=self.device)

                    # totals (paper style)
                    L_passive = L_clc + beta_kd * L_p2p + self.args.simplex_entropy * L_entropy
                    L_active  = L_loc + beta_kd * L_p2a

                    if step % 50 == 0:
                        self.logger.info(
                            "Epoch %s Step %s KD losses: L_loc=%.6f L_clc=%.6f L_p2p=%.6f L_p2a=%.6f, entropy=%.6f",
                            epoch + 1,
                            step,
                            L_loc.item(),
                            L_clc.item(),
                            L_p2p.item(),
                            L_p2a.item(),
                            simplex_aux["mean_entropy"].item(),
                        )

                    # backward order (passive then active) – 原本流程保持不变
                else:
                    # no KD: still do CKD-style split training
                    L_passive = L_clc + self.args.simplex_entropy * L_entropy
                    L_active  = L_loc
                    if step % 50 == 0:
                        self.logger.info(
                            "Epoch %s Step %s losses: L_loc=%.6f L_clc=%.6f, entropy=%.6f",
                            epoch + 1,
                            step,
                            L_loc.item(),
                            L_clc.item(),
                            L_entropy.item()
                        )

                # 梯度清零
                for optimizer in self.optimizer_list:
                    optimizer.zero_grad()

                
                # ===== Marvell：在 backward 前对 cut-layer 梯度做扰动（防御 LFBA）=====

                # Marvell调用初版，可能会因为数据分布问题执行崩溃（比如 batch 内单类导致 Marvell 计算均值/方差时崩溃），所以加了很多保护措施，且默认不开启 Marvell 的动态调整（dynamic=False）。如果后续版本稳定了，可以考虑简化代码并开启 dynamic。
                # hook_handles = []
                # if self.marvell is not None:
                #     # 只对攻击方客户端做（更省开销，不符合实际）
                #     # defend_clients = [self.args.attack_client_num]
                #     # 对所有客户端都防御
                #     defend_clients = list(range(self.args.client_num))

                #     for i in defend_clients:
                #         def _make_hook(y_batch):
                #             def _hook(grad):
                #                 # 打印扰动前的梯度统计量
                #                 self.logger.info("[before Marvell] grad shape: {}, grad mean: {:.4g}, grad std: {:.4g}".format(
                #                     grad.shape, grad.mean().item(), grad.std().item()
                #                 ))
                #                 # print("before marvell grad:", grad)
                #                 # 将标签转为 int64 类型，符合 Marvell 要求
                #                 y_int = y_batch.view(-1).to(dtype=torch.int64)
                #                 # 避免 batch 单类导致 Marvell 均值/方差在空集上崩溃
                #                 if (y_int == 1).sum().item() == 0 or (y_int == 0).sum().item() == 0:
                #                     return grad
                #                 g_out, _info = self.marvell(grad, y_int)
                #                 # 扰动后梯度
                #                 self.logger.info("[after Marvell] grad shape: {}, grad mean: {:.4g}, grad std: {:.4g}".format(
                #                     g_out.shape, g_out.mean().item(), g_out.std().item()
                #                 ))
                #                 # print("after marvell grad:", g_out)
                #                 # ===== Marvell：每个 epoch 只在第一个 batch 打一次关键统计量 =====
                #                 nonlocal marvell_logged_this_epoch
                #                 if (not marvell_logged_this_epoch):
                #                     marvell_logged_this_epoch = True
                #                     self.logger.info(
                #                         "[Marvell] ep={} step={} scale={:.4g} sumKL={:.4g} "
                #                         "lam10={:.4g} lam20={:.4g} lam11={:.4g} lam21={:.4g} "
                #                         "p={:.3f} gdiff2={:.4g} implied_e_lb={:.4g}".format(
                #                             epoch + 1, step,
                #                             _info["scale"], _info["sumKL"],
                #                             _info["lam10"], _info["lam20"], _info["lam11"], _info["lam21"],
                #                             _info["p"], _info["g_diff_norm2"],
                #                             _info["error_prob_lower_bound_implied"]
                #                         )
                #                     )
                #                 return g_out
                #             return _hook

                #         h = local_outputs_classifier_list[i].register_hook(_make_hook(labels))
                #         hook_handles.append(h)

                # # ===== Marvell：在 backward 前对 cut-layer 梯度做扰动（防御 LFBA） 改良稳定版=====
                # hook_handles = []
                # if self.marvell is not None:
                #     # 实际部署里不知道谁是攻击方，因此对所有被动方都做防御
                #     defend_clients = list(range(self.args.client_num))

                #     for i in defend_clients:
                #         def _make_hook(y_batch, client_id):
                #             def _hook(grad):
                #                 nonlocal marvell_logged_this_epoch

                #                 y_int = y_batch.view(-1).to(dtype=torch.int64)
                #                 grad_f = grad.float()

                #                 # -------- 基础统计 --------
                #                 grad_mean = float(grad_f.mean().item())
                #                 grad_std = float(grad_f.std().item())
                #                 grad_min = float(grad_f.min().item())
                #                 grad_max = float(grad_f.max().item())
                #                 has_nan = bool(torch.isnan(grad_f).any().item())
                #                 has_inf = bool(torch.isinf(grad_f).any().item())

                #                 pos_cnt = int((y_int == 1).sum().item())
                #                 neg_cnt = int((y_int == 0).sum().item())
                #                 uniq = torch.unique(y_int).detach().cpu().tolist()

                #                 self.logger.info(
                #                     "[Marvell-check] ep=%d step=%d client=%d shape=%s "
                #                     "labels=%s pos=%d neg=%d "
                #                     "grad_mean=%.6e grad_std=%.6e grad_min=%.6e grad_max=%.6e "
                #                     "has_nan=%s has_inf=%s",
                #                     epoch + 1,
                #                     step,
                #                     client_id,
                #                     tuple(grad.shape),
                #                     str(uniq),
                #                     pos_cnt,
                #                     neg_cnt,
                #                     grad_mean,
                #                     grad_std,
                #                     grad_min,
                #                     grad_max,
                #                     has_nan,
                #                     has_inf,
                #                 )

                #                 # -------- 1) 标签必须是二分类 0/1 --------
                #                 if not set(uniq).issubset({0, 1}):
                #                     self.logger.warning(
                #                         "[Marvell-skip] ep=%d step=%d client=%d non-binary labels=%s",
                #                         epoch + 1, step, client_id, str(uniq)
                #                     )
                #                     return grad

                #                 # -------- 2) batch 太小直接跳过 --------
                #                 if y_int.numel() < 8:
                #                     self.logger.warning(
                #                         "[Marvell-skip] ep=%d step=%d client=%d batch too small: B=%d",
                #                         epoch + 1, step, client_id, y_int.numel()
                #                     )
                #                     return grad

                #                 # -------- 3) 单类 batch 跳过 --------
                #                 if pos_cnt == 0 or neg_cnt == 0:
                #                     self.logger.warning(
                #                         "[Marvell-skip] ep=%d step=%d client=%d single-class batch: pos=%d neg=%d",
                #                         epoch + 1, step, client_id, pos_cnt, neg_cnt
                #                     )
                #                     return grad

                #                 # -------- 4) 某类样本过少跳过 --------
                #                 if pos_cnt < 2 or neg_cnt < 2:
                #                     self.logger.warning(
                #                         "[Marvell-skip] ep=%d step=%d client=%d too few samples per class: pos=%d neg=%d",
                #                         epoch + 1, step, client_id, pos_cnt, neg_cnt
                #                     )
                #                     return grad

                #                 # -------- 5) grad 含 nan/inf 跳过 --------
                #                 if has_nan or has_inf:
                #                     self.logger.warning(
                #                         "[Marvell-skip] ep=%d step=%d client=%d grad has nan/inf",
                #                         epoch + 1, step, client_id
                #                     )
                #                     return grad

                #                 # -------- 6) 整体方差过小跳过 --------
                #                 if grad_std < 1e-8:
                #                     self.logger.warning(
                #                         "[Marvell-skip] ep=%d step=%d client=%d grad std too small: %.6e",
                #                         epoch + 1, step, client_id, grad_std
                #                     )
                #                     return grad

                #                 # -------- 7) 两类梯度均值差过小跳过 --------
                #                 pos_grad = grad_f[y_int == 1]
                #                 neg_grad = grad_f[y_int == 0]

                #                 if pos_grad.numel() == 0 or neg_grad.numel() == 0:
                #                     self.logger.warning(
                #                         "[Marvell-skip] ep=%d step=%d client=%d empty class grad after mask",
                #                         epoch + 1, step, client_id
                #                     )
                #                     return grad

                #                 mean_gap = float((pos_grad.mean(dim=0) - neg_grad.mean(dim=0)).norm(p=2).item())
                #                 if mean_gap < 1e-8:
                #                     self.logger.warning(
                #                         "[Marvell-skip] ep=%d step=%d client=%d class mean gap too small: %.6e",
                #                         epoch + 1, step, client_id, mean_gap
                #                     )
                #                     return grad

                #                 # -------- 8) 调用 Marvell；某个 client 失败时只回退该 client --------
                #                 try:
                #                     g_out, _info = self.marvell(grad_f.detach(), y_int)
                #                 except RecursionError as e:
                #                     self.logger.warning(
                #                         "[Marvell-fallback] ep=%d step=%d client=%d RecursionError: %s",
                #                         epoch + 1, step, client_id, str(e)
                #                     )
                #                     return grad
                #                 except Exception as e:
                #                     self.logger.warning(
                #                         "[Marvell-fallback] ep=%d step=%d client=%d solver failed: %s",
                #                         epoch + 1, step, client_id, str(e)
                #                     )
                #                     return grad

                #                 # 每个 epoch 只打一条关键信息，避免日志过大
                #                 if not marvell_logged_this_epoch:
                #                     marvell_logged_this_epoch = True
                #                     self.logger.info(
                #                         "[Marvell] ep=%d step=%d client=%d scale=%.4g sumKL=%.4g "
                #                         "lam10=%.4g lam20=%.4g lam11=%.4g lam21=%.4g "
                #                         "p=%.3f gdiff2=%.4g implied_e_lb=%.4g",
                #                         epoch + 1,
                #                         step,
                #                         client_id,
                #                         float(_info["scale"]),
                #                         float(_info["sumKL"]),
                #                         float(_info["lam10"]),
                #                         float(_info["lam20"]),
                #                         float(_info["lam11"]),
                #                         float(_info["lam21"]),
                #                         float(_info["p"]),
                #                         float(_info["g_diff_norm2"]),
                #                         float(_info["error_prob_lower_bound_implied"]),
                #                     )

                #                 return g_out.to(dtype=grad.dtype)
                #             return _hook

                #         h = local_outputs_classifier_list[i].register_hook(_make_hook(labels, i))
                #         hook_handles.append(h)

                # # 小改动稳定版，上面是大改动稳定版，上面的更稳定，限制更多，过滤更强
                # hook_handles = []
                # if self.marvell is not None:
                #     defend_clients = list(range(self.args.client_num))

                #     for i in defend_clients:
                #         def _make_hook(y_batch, client_id):
                #             def _hook(grad):
                #                 y_int = y_batch.view(-1).to(dtype=torch.int64)

                #                 self.logger.info(
                #                     "[before Marvell] client=%d grad shape=%s mean=%.4g std=%.4g",
                #                     client_id, tuple(grad.shape), grad.mean().item(), grad.std().item()
                #                 )

                #                 # 只保留最必要保护，尽量不改原始逻辑
                #                 if (y_int == 1).sum().item() == 0 or (y_int == 0).sum().item() == 0:
                #                     self.logger.warning(
                #                         "[Marvell-skip] client=%d single-class batch", client_id
                #                     )
                #                     return grad

                #                 if torch.isnan(grad).any() or torch.isinf(grad).any():
                #                     self.logger.warning(
                #                         "[Marvell-skip] client=%d grad has nan/inf", client_id
                #                     )
                #                     return grad

                #                 try:
                #                     g_out, _info = self.marvell(grad, y_int)
                #                 except RecursionError as e:
                #                     self.logger.warning(
                #                         "[Marvell-fallback] client=%d RecursionError: %s", client_id, str(e)
                #                     )
                #                     return grad
                #                 except Exception as e:
                #                     self.logger.warning(
                #                         "[Marvell-fallback] client=%d solver failed: %s", client_id, str(e)
                #                     )
                #                     return grad

                #                 nonlocal marvell_logged_this_epoch
                #                 if not marvell_logged_this_epoch:
                #                     marvell_logged_this_epoch = True
                #                     self.logger.info(
                #                         "[Marvell] ep=%d step=%d client=%d scale=%.4g sumKL=%.4g "
                #                         "lam10=%.4g lam20=%.4g lam11=%.4g lam21=%.4g "
                #                         "p=%.3f gdiff2=%.4g implied_e_lb=%.4g",
                #                         epoch + 1, step, client_id,
                #                         _info["scale"], _info["sumKL"],
                #                         _info["lam10"], _info["lam20"], _info["lam11"], _info["lam21"],
                #                         _info["p"], _info["g_diff_norm2"],
                #                         _info["error_prob_lower_bound_implied"]
                #                     )

                #                 self.logger.info(
                #                     "[after Marvell] client=%d grad shape=%s mean=%.4g std=%.4g",
                #                     client_id, tuple(g_out.shape), g_out.mean().item(), g_out.std().item()
                #                 )
                #                 return g_out
                #             return _hook

                #         h = local_outputs_classifier_list[i].register_hook(_make_hook(labels, i))
                #         hook_handles.append(h)

                # 小改动稳定版 + 多分类ovr版本
                # hook_handles = []
                # if self.marvell is not None:
                #     defend_clients = list(range(self.args.client_num))

                #     for i in defend_clients:
                #         def _make_hook(y_batch, client_id):
                #             def _hook(grad):
                #                 y_int = y_batch.view(-1).to(dtype=torch.int64)
                #                 grad_f = grad.float()

                #                 uniq, cnt = torch.unique(y_int, sorted=True, return_counts=True)
                #                 uniq_list = uniq.detach().cpu().tolist()
                #                 cnt_list = cnt.detach().cpu().tolist()

                #                 self.logger.info(
                #                     "[before Marvell] client=%d grad shape=%s mean=%.4g std=%.4g labels=%s counts=%s",
                #                     client_id, tuple(grad.shape), grad.mean().item(), grad.std().item(),
                #                     str(uniq_list), str(cnt_list)
                #                 )

                #                 if uniq.numel() < 2:
                #                     self.logger.warning(
                #                         "[Marvell-skip] client=%d single-class batch labels=%s counts=%s",
                #                         client_id, str(uniq_list), str(cnt_list)
                #                     )
                #                     return grad

                #                 if torch.isnan(grad_f).any() or torch.isinf(grad_f).any():
                #                     self.logger.warning(
                #                         "[Marvell-skip] client=%d grad has nan/inf", client_id
                #                     )
                #                     return grad

                #                 try:
                #                     g_out, _info = self.marvell(grad_f.detach(), y_int)
                #                 except RecursionError as e:
                #                     self.logger.warning(
                #                         "[Marvell-fallback] client=%d RecursionError: %s", client_id, str(e)
                #                     )
                #                     return grad
                #                 except Exception as e:
                #                     self.logger.warning(
                #                         "[Marvell-fallback] client=%d solver failed: %s", client_id, str(e)
                #                     )
                #                     return grad

                #                 if _info.get("skipped", False):
                #                     self.logger.warning(
                #                         "[Marvell-skip] client=%d reason=%s",
                #                         client_id, str(_info.get("skip_reason", "unknown"))
                #                     )
                #                     return grad

                #                 nonlocal marvell_logged_this_epoch
                #                 if not marvell_logged_this_epoch:
                #                     marvell_logged_this_epoch = True
                #                     if _info.get("mode") == "multiclass_ovr":
                #                         cls_logs = []
                #                         for x in _info["per_class"]:
                #                             if x.get("skipped", False):
                #                                 cls_logs.append(f"c={x['ovr_class']}:skip")
                #                             else:
                #                                 cls_logs.append(
                #                                     f"c={x['ovr_class']}:w={x['ovr_weight']:.3f},"
                #                                     f"sumKL={x['sumKL']:.4g},scale={x['scale']:.4g},"
                #                                     f"gdiff2={x['g_diff_norm2']:.4g}"
                #                                 )
                #                         self.logger.info(
                #                             "[Marvell-OVR] ep=%d step=%d client=%d labels=%s counts=%s "
                #                             "active_ovr=%d mean_sumKL=%s max_sumKL=%s detail=%s",
                #                             epoch + 1, step, client_id,
                #                             str(_info["labels"]), str(_info["counts"]),
                #                             int(_info["num_active_ovr"]),
                #                             str(_info["mean_sumKL"]),
                #                             str(_info["max_sumKL"]),
                #                             " | ".join(cls_logs)
                #                         )
                #                     else:
                #                         self.logger.info(
                #                             "[Marvell] ep=%d step=%d client=%d scale=%.4g sumKL=%.4g "
                #                             "lam10=%.4g lam20=%.4g lam11=%.4g lam21=%.4g "
                #                             "p=%.3f gdiff2=%.4g implied_e_lb=%.4g",
                #                             epoch + 1, step, client_id,
                #                             float(_info["scale"]), float(_info["sumKL"]),
                #                             float(_info["lam10"]), float(_info["lam20"]),
                #                             float(_info["lam11"]), float(_info["lam21"]),
                #                             float(_info["p"]), float(_info["g_diff_norm2"]),
                #                             float(_info["error_prob_lower_bound_implied"]),
                #                         )

                #                 self.logger.info(
                #                     "[after Marvell] client=%d grad shape=%s mean=%.4g std=%.4g",
                #                     client_id, tuple(g_out.shape), g_out.mean().item(), g_out.std().item()
                #                 )
                #                 return g_out.to(dtype=grad.dtype)
                #             return _hook

                #         h = local_outputs_classifier_list[i].register_hook(_make_hook(labels, i))
                #         hook_handles.append(h)

                # marvell 多分类版 structured版本
                # 多分类版本 Marvell 结构化版本
                hook_handles = []
                if self.marvell is not None:
                    defend_clients = list(range(self.args.client_num))

                    for i in defend_clients:
                        def _make_hook(y_batch, client_id):
                            def _hook(grad):
                                y_int = y_batch.view(-1).to(dtype=torch.int64)
                                grad_f = grad.float()

                                uniq, cnt = torch.unique(y_int, sorted=True, return_counts=True)
                                uniq_list = uniq.detach().cpu().tolist()
                                cnt_list = cnt.detach().cpu().tolist()

                                self.logger.info(
                                    "[before Marvell] client=%d grad shape=%s mean=%.4g std=%.4g labels=%s counts=%s",
                                    client_id, tuple(grad.shape), grad.mean().item(), grad.std().item(),
                                    str(uniq_list), str(cnt_list)
                                )

                                if uniq.numel() < 2:
                                    self.logger.warning(
                                        "[Marvell-skip] client=%d single-class batch labels=%s counts=%s",
                                        client_id, str(uniq_list), str(cnt_list)
                                    )
                                    return grad
                                if torch.isnan(grad_f).any() or torch.isinf(grad_f).any():
                                    self.logger.warning(
                                        "[Marvell-skip] client=%d grad has nan/inf", client_id
                                    )
                                    return grad
                                if uniq.numel() == 2:
                                    try:
                                        uniq_sorted, _ = torch.sort(uniq)
                                        y01 = (y_int == uniq_sorted[1]).to(torch.int64) # 将较大标签映射为 1，较小标签映射为 0，符合 Marvell 二分类要求
                                        g_out, _info = self.marvell_binary(grad_f.detach(), y01)
                                        solver_mode = "binary_original"
                                    except RecursionError as e:
                                        self.logger.warning(
                                            "[Marvell-fallback] client=%d RecursionError: %s", client_id, str(e)
                                        )
                                        return grad
                                    except Exception as e:
                                        self.logger.warning(
                                            "[Marvell-fallback] client=%d solver failed: %s", client_id, str(e)
                                        )
                                        return grad
                                elif uniq.numel() > 2:
                                    try:
                                        g_out, _info = self.marvell(grad_f.detach(), y_int)
                                        solver_mode = "multiclass_structured"
                                    except RecursionError as e:
                                        self.logger.warning(
                                            "[Marvell-fallback] client=%d RecursionError: %s", client_id, str(e)
                                        )
                                        return grad
                                    except Exception as e:
                                        self.logger.warning(
                                            "[Marvell-fallback] client=%d solver failed: %s", client_id, str(e)
                                        )
                                        return grad

                                if _info.get("skipped", False):
                                    self.logger.warning(
                                        "[Marvell-skip] client=%d reason=%s",
                                        client_id, str(_info.get("skip_reason", "unknown"))
                                    )
                                    return grad

                                nonlocal marvell_logged_this_epoch
                                if not marvell_logged_this_epoch:
                                    marvell_logged_this_epoch = True
                                    if solver_mode == "binary_original":
                                        self.logger.info(
                                            "[Marvell] ep={} step={} scale={:.4g} sumKL={:.4g} "
                                            "lam10={:.4g} lam20={:.4g} lam11={:.4g} lam21={:.4g} "
                                            "p={:.3f} gdiff2={:.4g} implied_e_lb={:.4g}".format(
                                                epoch + 1, step,
                                                _info["scale"], _info["sumKL"],
                                                _info["lam10"], _info["lam20"], _info["lam11"], _info["lam21"],
                                                _info["p"], _info["g_diff_norm2"],
                                                _info["error_prob_lower_bound_implied"]
                                            )
                                        )
                                    elif solver_mode == "multiclass_structured":
                                        if _info.get("mode") == "multiclass_structured":
                                            alpha_arr = np.asarray(_info.get("alpha", []), dtype=np.float64)
                                            beta_arr = np.asarray(_info.get("beta", []), dtype=np.float64)
                                            svals_arr = np.asarray(_info.get("singular_values", []), dtype=np.float64)

                                            alpha_summary = "empty"
                                            if alpha_arr.size > 0:
                                                alpha_summary = (
                                                    f"mean={float(alpha_arr.mean()):.4g},"
                                                    f"max={float(alpha_arr.max()):.4g},"
                                                    f"min={float(alpha_arr.min()):.4g}"
                                                )

                                            beta_summary = "empty"
                                            if beta_arr.size > 0:
                                                beta_summary = (
                                                    f"mean={float(beta_arr.mean()):.4g},"
                                                    f"max={float(beta_arr.max()):.4g},"
                                                    f"min={float(beta_arr.min()):.4g}"
                                                )

                                            svals_summary = "empty"
                                            if svals_arr.size > 0:
                                                topn = min(5, svals_arr.size)
                                                svals_summary = str([float(x) for x in svals_arr[:topn]])

                                            self.logger.info(
                                                "[Marvell-Structured] ep=%d step=%d client=%d labels=%s counts=%s "
                                                "K=%d d=%d r=%d perp_dim=%d scale=%.4g P=%.4g trace_used=%.4g sumKL=%.4g "
                                                "Sb_trace=%.4g converged=%s iters=%d use_perp=%s alpha=%s beta=%s svals_top=%s",
                                                epoch + 1, step, client_id,
                                                str(_info["labels"]), str(_info["counts"]),
                                                int(_info["num_classes"]),
                                                int(_info["d"]),
                                                int(_info["r"]),
                                                int(_info["perp_dim"]),
                                                float(_info["scale"]),
                                                float(_info["P"]),
                                                float(_info["trace_used"]),
                                                float(_info["sumKL"]),
                                                float(_info["between_scatter_trace"]),
                                                str(_info["converged"]),
                                                int(_info["num_iter"]),
                                                str(_info["use_perp_budget"]),
                                                alpha_summary,
                                                beta_summary,
                                                svals_summary,
                                            )
                                        else:
                                            self.logger.info(
                                                "[Marvell] ep=%d step=%d client=%d scale=%.4g sumKL=%.4g",
                                                epoch + 1, step, client_id,
                                                float(_info.get("scale", 0.0)),
                                                float(_info.get("sumKL", 0.0)),
                                            )

                                self.logger.info(
                                    "[after Marvell] client=%d grad shape=%s mean=%.4g std=%.4g",
                                    client_id, tuple(g_out.shape), g_out.mean().item(), g_out.std().item()
                                )
                                return g_out.to(dtype=grad.dtype)
                            return _hook

                        h = local_outputs_classifier_list[i].register_hook(_make_hook(labels, i))
                        hook_handles.append(h)
                        
                # # backward (不采用clc encode)
                # if self.args.kd and epoch >= 1:
                #     # 先更新被动方
                #     L_passive.backward(retain_graph=True)
                #     # for i in range(1, self.args.client_num + 1):
                #     #     self.optimizer_list[i].step()
                    
                #     # 再更新主动方
                #     L_active.backward()
                #     # self.optimizer_list[0].step()
                # else:
                #     loss.backward()
                #     # for optimizer in self.optimizer_list:
                #     #     optimizer.step()

                # backward 之后、step 之前
                # for pi in range(self.args.client_num):
                #     m = self.model_list[pi + 1]
                #     # 随便取一个参数看梯度是否存在且非 0
                #     p = next(m.parameters())
                #     print(f"client{pi} grad_norm =", None if p.grad is None else p.grad.norm().item())

                # step(不采用clc encode) 注意如果使用了 KD，且分开更新主动方和被动方，必须先 backward 被动方的 loss，再 step 被动方的 optimizer，然后再 backward 主动方的 loss，最后 step 主动方的 optimizer，否则会出现梯度覆盖的问题（因为主动方和被动方的 loss 都依赖于 global_outputs，如果先 step 了主动方，那么 global_outputs 的计算图就被清空了，导致被动方无法正确计算梯度）
                # if self.args.kd and epoch >= 1:
                #     # 先更新被动方
                #     for i in range(1, self.args.client_num + 1):
                #         self.optimizer_list[i].step()
                    
                #     # 再更新主动方
                #     self.optimizer_list[0].step()

                #     # 更新simplex层
                #     self.optimizer_list[-1].step()
                # else:
                #     for optimizer in self.optimizer_list:
                #         optimizer.step()

                # 采用clc encode的backward和step
                # backward
                L_passive.backward(retain_graph=True)

                # # debug simplex层梯度
                # grad_before = None
                # if self.simplex.raw.grad is not None:
                #     grad_before = self.simplex.raw.grad.clone()

                L_active.backward()

                # hook 用完移除
                for h in hook_handles:
                    h.remove()

                #扰动后的梯度
                # self.logger.info("[Marvell] After perturbation:")
                # print("after marvell grad:", local_output_list[self.args.attack_client_num].grad)

                if self.args.attack == 'LFBA':
                    """
                    这行的原文逻辑在这里可能有问题，可能会越界，
                    因为 local_outputs_classifier_list 的长度是参与方数量，
                    而 self.args.attack_client_num 是指定的攻击目标参与方的索引，
                    如果 self.args.attack_client_num >= self.args.client_num 就会越界
                    暂时先不动，报错了再动 --> 已报越界错，改为 self.args.attack_client_num - 1，攻击目标参与方的索引从1开始（因为0是主动方），这样就不会越界了
                    """ 
                    if self.args.attack_repr == 'classifier_output':
                        self.grad_vec_epoch.append(local_outputs_classifier_list[self.args.attack_client_num - 1].grad.to(self.device))
                    elif self.args.attack_repr == 'embedding_output':
                        self.grad_vec_epoch.append(local_outputs_logits_list[self.args.attack_client_num - 1].grad.to(self.device))
                    self.indexes_epoch.append(index)
                    self.target_epoch.append(labels)

                # # debug simplex层梯度
                # if grad_before is not None:
                #     print((self.simplex.raw.grad - grad_before).abs().max())

                # step
                # passive models
                for i in range(1, self.args.client_num + 1):
                    self.optimizer_list[i].step()
                
                self.optimizer_list[0].step() # 主动方 model

                # self.optimizer_list[-1].step() # simplex layer
                # 如果单形层冻结了，就不更新单形层参数(测试用)
                if not simplex_frozen:
                    self.optimizer_list[-1].step()

                loss = self.criterion(global_outputs, labels)
                running_train_loss += loss.item()
                trian_step_losses.append(loss.item())

                # LFBA: 记录每个 batch 的 loss 以便后续分析
                batch_loss_list.append(loss.item())

                # calculate the training accuracy
                _, predicted = global_outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # train_acc
                train_acc = correct / total
                current_loss = sum(batch_loss_list) / len(batch_loss_list)

                if step % self.args.print_steps == 0:
                    self.logger.info(
                        'Epoch: {}, {}/{}: train loss: {:.4f}, train main task accuracy: {:.4f}'.format(epoch + 1,
                                                                                                        step + 1,
                                                                                                        len(self.train_loader),
                                                                                                        current_loss,
                                                                                                        train_acc))

                if step % 100 == 0:
                    self.logger.info(
                        "Epoch %s Step %s train_loss=%.6f",
                        epoch + 1,
                        step,
                        loss.item(),
                    )

                # 改良后无需使用list存储，故不用清空
                # if self.args.kd:
                #     # 清空已经使用过的记录的软标签，为下一轮做准备
                #     # global_soft_predict = []
                #     # local_soft_predict = []

            avg_train_loss = running_train_loss / len(self.train_loader)
            train_losses.append(avg_train_loss)

            if self.args.attack == 'LFBA':
                self.grad_vec_epoch = torch.cat(self.grad_vec_epoch) # 默认dim = 0
                self.indexes_epoch = torch.cat(self.indexes_epoch)
                self.target_epoch = torch.cat(self.target_epoch)
            
            epoch_loss = sum(batch_loss_list) / len(batch_loss_list)
            epoch_loss_list.append(epoch_loss)
            self.adjust_learning_rate(epoch + 1)
            # 模型验证阶段以及ASR测试
            test_acc, test_poison_accuracy, test_target, test_asr, result = self.test(epoch, val_losses, acc)
            test_trade_off = (test_acc + test_asr) / 2

            # # 模型验证阶段(写在 test 函数里了)
            # self.logger.info("Epoch %s validation started", epoch + 1)
            # for model in self.model_list:
            #     model.eval()
            # self.simplex.eval()
            
            # running_val_loss = 0.0
            # correct = 0
            # total = 0
            # with torch.no_grad():
            #     all_val_logits = []
            #     all_val_labels = []
            #     for inputs, labels in tqdm(self.val_loader, total=len(self.val_loader), desc=f"Epoch {epoch+1}/{self.args.epochs} - Validation"):
            #         inputs = inputs.to(self.device)
            #         labels = labels.to(self.device)
            #         # split inputs for active and passive parties
            #         inputs_list = split_vfl(inputs, self.args)

            #         # 前向传播
            #         local_outputs_logits_list = [] # 存储每个模型的logits输出
            #         local_outputs_classifier_list = []
            #         # for model in self.model_list[1:]:
            #         #     outputs = model(inputs)
            #         #     local_outputs_list.append(outputs)
            #         for i in range(self.args.client_num):
            #             local_classifier_output, local_outputs_logits = self.model_list[i + 1](inputs_list[i + 1])
            #             local_outputs_logits_list.append(local_outputs_logits)
            #             local_outputs_classifier_list.append(local_classifier_output)
                    
            #         # 全局模型前向传播
            #         # global_inputs = local_outputs_logits_list
            #         # global_outputs = self.model_list[0](global_inputs)

            #         # 全局模型前向传播 利用单形层聚合后输入全局模型
            #         # simplex_outputs, simplex_weights = self.simplex(local_outputs_logits_list) # [B, hidden_dim], [K], 这里的hidden_dim是单形层输出的维度
            #         # simplex_outputs, simplex_weights = self.simplex(local_outputs_logits_list, ) # [B, C], [B, K], 这里的C是类别数量
                    
            #         # 全局模型前向传播 利用单形层聚合后输入全局模型，且单形层输出维度与每个参与方logits维度相同（即不使用额外的线性层调整维度）
            #         # 设置掩码，模拟参与方缺失（训练时不使用，测试时使用） available_mask: [B,K]
            #         B = inputs.size(0)
            #         K = self.args.client_num
            #         avail_mask = torch.ones(B, K, dtype=torch.bool, device=self.device)
            #         # 组装 party_logits 输入单形层，维度为 [B, K, C]，其中C是每个参与方logits的维度
            #         party_logits = torch.stack(local_outputs_classifier_list, dim = 1) # [B, K, C], 这里的C是类别数量
            #         # simplex 聚合被动方：输出就是聚合后的分类 logits，维度为 [B, C]，权重维度为 [B, K]
            #         # simplex_outputs, simplex_weights, simplex_outputs_embedding = self.simplex(party_logits, avail_mask) # [B, C], [B, K], 这里的C是类别数量
            #         simplex_outputs, simplex_weights, simplex_outputs_embedding, simplex_aux = self.simplex(party_logits, avail_mask)
            #         # 主动方
            #         active_outputs_classifier, active_outputs_logits = self.model_list[0](inputs_list[0]) # [B, C]
            #         global_outputs = active_outputs_classifier + self.args.alpha * simplex_outputs

            #         all_val_logits.append(global_outputs.detach().cpu())
            #         all_val_labels.append(labels.detach().cpu())

            #         # 计算损失
            #         loss = self.criterion(global_outputs, labels)
            #         running_val_loss += loss.item()

            #         # 计算准确率
            #         _, predicted = torch.max(global_outputs, 1)
            #         total += labels.size(0)
            #         correct += (predicted == labels).sum().item()

            # avg_val_loss = running_val_loss / len(self.val_loader)
            # val_losses.append(avg_val_loss)
            # val_acc = correct / total
            # acc.append(val_acc)

            self.logger.info(
                "Epoch %s finished: train_loss=%.6f val_loss=%.6f val_acc=%.6f",
                epoch + 1,
                avg_train_loss,
                result["avg_val_loss"],
                result["val_acc"]
            )
            if self.args.dataset == "avazu":  # avazu 是二分类，评 AUC
                # val_auc = auc_from_logits(torch.cat(result["all_val_labels"], dim=0),
                #             torch.cat(result["all_val_logits"], dim=0))
                val_auc = auc_from_logits(result["all_val_labels"], result["all_val_logits"])
                self.logger.info("Epoch %s val_auc=%.6f", epoch + 1, val_auc)

                # 测试
                # ===== strict FL-AUC diagnostic =====
                L_util_auc, summary, detail = self.eval_fl_auc_subset_stats(
                    self.val_loader,
                    alpha_ckd=self.args.alpha
                )

                K = self.args.client_num
                full_auc = summary[K]["mean"]
                robust_score = float(np.mean([summary[m]["mean"] for m in range(1, K + 1)]))

                # 如果想顺手看 PMC，就打开；不想每轮都跑可以先设 None
                pmc_mean_auc = None
                # pmc_mean_auc, _ = self.eval_privacy_pmc_auc(
                #     self.make_aux_loader_from_train(self.train_loader, aux_k=getattr(self.args, "pmc_aux_k", 100),
                #                                     seed=getattr(self.args, "pmc_aux_seed", getattr(self.args, "seed", 0))),
                #     self.val_loader,
                #     use_prob=getattr(self.args, "pmc_use_prob", False)
                # )

                history["epoch"].append(epoch + 1)
                history["train_loss"].append(float(avg_train_loss))
                history["val_auc"].append(float(val_auc))
                history["L_util_auc"].append(float(L_util_auc))
                history["full_auc"].append(float(full_auc))
                history["robust_score"].append(float(robust_score))
                history["pmc_mean_auc"].append(None if pmc_mean_auc is None else float(pmc_mean_auc))

                mean_curve = {m: round(summary[m]["mean"], 6) for m in summary}
                max_curve  = {m: round(summary[m]["max"], 6) for m in summary}

                self.logger.info(
                    "[Epoch %d Diagnostic] val_auc=%.6f | L_util=%.6f | full_auc=%.6f | robust_score=%.6f",
                    epoch + 1, val_auc, L_util_auc, full_auc, robust_score
                )
                self.logger.info("[Epoch %d Diagnostic] FL_AUC_mean_curve=%s", epoch + 1, mean_curve)
                self.logger.info("[Epoch %d Diagnostic] FL_AUC_max_curve=%s", epoch + 1, max_curve)

            # score = val_auc if best_metric == "auc" else val_acc
            if best_metric == "auc":
                score = val_auc
            elif best_metric == "acc":
                score = result["val_acc"]
            elif best_metric == "trade_off":
                score = test_trade_off
            
            # 测试
            # ===== best full-party checkpoint =====
            if self.args.dataset == "avazu" and full_auc > best_full_auc:
                best_full_auc = full_auc
                best_full_epoch = epoch + 1
                best_full_state = {
                    "epoch": epoch + 1,
                    "active_state_dict": copy.deepcopy(self.model_list[0].state_dict()),
                    "passive_state_dicts": [copy.deepcopy(m.state_dict()) for m in self.model_list[1:]],
                    "simplex_state_dict": copy.deepcopy(self.simplex.state_dict()),
                    "val_auc": float(val_auc),
                    "L_util_auc": float(L_util_auc),
                    "full_auc": float(full_auc),
                    "robust_score": float(robust_score),
                    "args": vars(self.args) if hasattr(self.args, "__dict__") else str(self.args),
                }
                torch.save(best_full_state, os.path.join(save_dir, "best_full_checkpoint.pt"))
                self.logger.info("[Best-Full] saved at epoch %d, full_auc=%.6f", epoch + 1, full_auc)

            # ===== best robustness checkpoint =====
            if self.args.dataset == "avazu" and robust_score > best_robust_score:  # 目前只在 avazu 上测试了 robustness 相关的评测，其他数据集先限定不保存
                best_robust_score = robust_score
                best_robust_epoch = epoch + 1
                best_robust_state = {
                    "epoch": epoch + 1,
                    "active_state_dict": copy.deepcopy(self.model_list[0].state_dict()),
                    "passive_state_dicts": [copy.deepcopy(m.state_dict()) for m in self.model_list[1:]],
                    "simplex_state_dict": copy.deepcopy(self.simplex.state_dict()),
                    "val_auc": float(val_auc),
                    "L_util_auc": float(L_util_auc),
                    "full_auc": float(full_auc),
                    "robust_score": float(robust_score),
                    "args": vars(self.args) if hasattr(self.args, "__dict__") else str(self.args),
                }
                torch.save(best_robust_state, os.path.join(save_dir, "best_robust_checkpoint.pt"))
                self.logger.info("[Best-Robust] saved at epoch %d, robust_score=%.6f", epoch + 1, robust_score)

            # 保存最后一次的模型 ckpt（非最佳 ckpt），每轮覆盖同一个文件
            if self.args.dataset == "avazu": # 目前只在 avazu 上测试了 robustness 相关的评测，其他数据集先限定不保存最后一次的 ckpt
                last_ckpt_path = os.path.join(save_dir, "last_model.pt")
                ckpt = {
                    "epoch": epoch + 1,
                    "best_metric": best_metric,
                    "best_score": best_score,
                    "val_loss": float(result["avg_val_loss"]),
                    "val_acc": float(result["val_acc"]),
                    "val_auc": float(val_auc),

                    "active_state_dict": self.model_list[0].state_dict(),
                    "passive_state_dicts": [m.state_dict() for m in self.model_list[1:]],
                    "simplex_state_dict": self.simplex.state_dict(),
                    "args": vars(self.args) if hasattr(self.args, "__dict__") else str(self.args),
                }
                torch.save(ckpt, last_ckpt_path)
                self.logger.info("[LAST] saved: %s (epoch=%d, %s=%.6f)",
                                last_ckpt_path, epoch + 1, best_metric, score)
                print(f"[LAST] saved: {last_ckpt_path} | epoch={epoch + 1} | {best_metric}={score:.6f}")

            if score > best_score:
                # best accuracy
                best_score = float(score)
                best_epoch = epoch + 1
                no_change = 0
                best_acc = test_acc
                best_trade_off = test_trade_off
                poison_acc_for_best_epoch = test_poison_accuracy
                asr_for_best_epoch = test_asr
                target_for_best_epoch = test_target

                # if self.args.dataset == "avazu":
                if best_metric == "auc": # 如果评测指标是 auc，就保存对应的 ckpt
                    ckpt = {
                        "epoch": best_epoch,
                        "best_metric": best_metric,
                        "best_score": best_score,
                        "val_loss": float(result["avg_val_loss"]),
                        "val_acc": float(result["val_acc"]),
                        "val_auc": float(val_auc),

                        "active_state_dict": self.model_list[0].state_dict(),
                        "passive_state_dicts": [m.state_dict() for m in self.model_list[1:]],
                        "simplex_state_dict": self.simplex.state_dict(),
                        "optimizer_states": [opt.state_dict() for opt in self.optimizer_list],
                        "args": vars(self.args) if hasattr(self.args, "__dict__") else str(self.args),
                    }
                # elif self.args.dataset == "phishing" and best_metric == "acc": # phishing 是多分类，评准确率
                elif best_metric == "acc": # 评准确率
                    ckpt = {
                        "epoch": best_epoch,
                        "best_metric": best_metric,
                        "best_score": best_score,
                        "val_loss": float(result["avg_val_loss"]),
                        "val_acc": float(result["val_acc"]),

                        "active_state_dict": self.model_list[0].state_dict(),
                        "passive_state_dicts": [m.state_dict() for m in self.model_list[1:]],
                        "simplex_state_dict": self.simplex.state_dict(),
                        "optimizer_states": [opt.state_dict() for opt in self.optimizer_list],
                        "args": vars(self.args) if hasattr(self.args, "__dict__") else str(self.args),
                    }
                elif best_metric == "trade_off": # 可能的特殊情况：如果评测指标是 trade_off（综合考虑准确率和攻击成功率的指标），就保存对应的 ckpt
                    ckpt = {
                        "epoch": best_epoch,
                        "best_metric": best_metric,
                        "best_acc": best_acc,
                        "best_score": best_score,
                        "val_loss": float(result["avg_val_loss"]),
                        "val_acc": float(result["val_acc"]),
                        "test_trade_off": float(test_trade_off),
                        "test_target": target_for_best_epoch,
                        "asr": asr_for_best_epoch,
                        'poison_acc': poison_acc_for_best_epoch,

                        "active_state_dict": self.model_list[0].state_dict(),
                        "passive_state_dicts": [m.state_dict() for m in self.model_list[1:]],
                        "simplex_state_dict": self.simplex.state_dict(),
                        "optimizer_states": [opt.state_dict() for opt in self.optimizer_list],
                        "args": vars(self.args) if hasattr(self.args, "__dict__") else str(self.args),
                    }
                torch.save(ckpt, best_path)
                self.logger.info("[BEST] saved: %s (epoch=%d, %s=%.6f)",
                                best_path, best_epoch, best_metric, best_score)
                print(f"[BEST] saved: {best_path} | epoch={best_epoch} | {best_metric}={best_score:.6f}")
            else:
                no_change += 1
                # self.logger.info("No improvement for %d epochs", no_change)
                self.logger.info("No improvement for %d epochs (best_epoch=%d best_%s=%.6f current=%.6f)",
                         no_change, best_epoch, best_metric, best_score, score)
                self.logger.info(
                '=> End Epoch: {}, early stop epochs: {}, best epoch: {}, best trade off accuracy: {:.4f}, main task accuracy: {:.4f}, test target accuracy: {:.4f}, test asr: {:.4f}'.format(
                    epoch + 1,
                    no_change,
                    best_epoch, best_trade_off, best_acc, target_for_best_epoch, asr_for_best_epoch))
                if no_change >= getattr(self.args, "early_stopping", 5):
                    # # 保存最后一次的模型 ckpt（非最佳 ckpt）
                    # last_ckpt_path = os.path.join(save_dir, f"last_model_epoch{epoch+1}.pt")
                    # ckpt = {
                    #     "epoch": epoch + 1,
                    #     "best_metric": best_metric,
                    #     "best_score": best_score,
                    #     "val_loss": float(avg_val_loss),
                    #     "val_acc": float(val_acc),
                    #     "val_auc": float(val_auc),

                    #     "active_state_dict": self.model_list[0].state_dict(),
                    #     "passive_state_dicts": [m.state_dict() for m in self.model_list[1:]],
                    #     "simplex_state_dict": self.simplex.state_dict(),
                    #     "args": vars(self.args) if hasattr(self.args, "__dict__") else str(self.args),
                    # }
                    # torch.save(ckpt, last_ckpt_path)
                    # self.logger.info("[LAST] saved: %s (epoch=%d, %s=%.6f)",
                    #                 last_ckpt_path, epoch + 1, best_metric, score)
                    # print(f"[LAST] saved: {last_ckpt_path} | epoch={epoch + 1} | {best_metric}={score:.6f}")
                    # self.logger.info("Early stopping triggered after %d epochs with no improvement", no_change)
                    self.logger.info("Early stopping triggered: no improvement for %d epochs (best_epoch=%d best_%s=%.6f)",
                    no_change, best_epoch, best_metric, best_score)
                    early_stopped = True
                    stop_epoch = epoch + 1
                    break
            print(f"Epoch [{epoch+1}/{self.args.epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {result['avg_val_loss']:.4f}, Val Acc: {result['val_acc']:.4f}")
            self.logger.info("===== Epoch Finished =====")
            self.logger.info("Best val checkpoint: epoch=%d, %s=%.6f", best_epoch, best_metric, best_score)
            self.logger.info("Best full checkpoint: epoch=%d, full_auc=%.6f", best_full_epoch, best_full_auc)
            self.logger.info("Best robust checkpoint: epoch=%d, robust_score=%.6f", best_robust_epoch, best_robust_score)
            self.logger.info("History=%s", history)

            print("===== Epoch Finished =====")
            print(f"Best val checkpoint: epoch={best_epoch}, {best_metric}={best_score:.6f}")
            print(f"Best full checkpoint: epoch={best_full_epoch}, full_auc={best_full_auc:.6f}")
            print(f"Best robust checkpoint: epoch={best_robust_epoch}, robust_score={best_robust_score:.6f}")


        # LFBA
        end_time_train = time.time()
        print("The total training time: {}".format((end_time_train - start_time_train)))
        print("The average training time of each epoch: {}".format(((end_time_train - start_time_train)) / (epoch + 1)))
        print("The poison set construction time: {}".format(total_time_GPC))
        print("The average hard-sample selection time: {}".format(total_time_HS / (epoch + 1)))
        print("The total hard-sample selection time: {}".format(total_time_HS))   


        # ===== training finished (normal or early stop) =====
        self.logger.info("Best checkpoint: epoch=%d %s=%.6f path=%s",
                        best_epoch, best_metric, best_score, best_path)
        print(f"Best checkpoint: epoch={best_epoch} {best_metric}={best_score:.6f} path={best_path}")


        if self.args.dataset == "avazu": # avazu 才做后续的各种评估(PMC/Utility)，其他数据集先不评
            # ===== load best and run final evaluation =====
            try:
                self.logger.info("Loading best checkpoint for final evaluation: %s", best_path)
                print(f"Loading best checkpoint for final evaluation: {best_path}")
                self._load_best_ckpt(best_path)
            except Exception as e:
                # 如果 best_path 不存在（极端情况），就用当前权重评估
                self.logger.warning("Failed to load best ckpt for final eval, fallback to current weights. err=%s", str(e))
                print("[WARN] Failed to load best ckpt, fallback to current weights:", e)

            # 无论是否 early stop，都执行最终评估
            self._final_ckd_eval()

            # 额外进行一次最后一次模型的评测（如果 early stopped），看看最后一次模型和最佳模型的差距
            self.logger.info("Loading last model checkpoint for evaluation: %s", last_ckpt_path)
            print(f"Loading last model checkpoint for evaluation: {last_ckpt_path}")
            self._load_best_ckpt(last_ckpt_path)
            self._final_ckd_eval()

            # 额外进行最佳鲁棒性模型的评测，看看最佳鲁棒性模型和最佳性能模型的差距
            self.logger.info("Loading best robust checkpoint for evaluation: %s", os.path.join(save_dir, "best_robust_checkpoint.pt"))
            print(f"Loading best robust checkpoint for evaluation: {os.path.join(save_dir, 'best_robust_checkpoint.pt')}")
            self._load_best_ckpt(os.path.join(save_dir, "best_robust_checkpoint.pt"))
            self._final_ckd_eval()
        
        # 训练结束评估
        self._load_best_ckpt(best_path)
        self._final_ckd_eval()

        self.logger.info("Training completed (early_stopped=%s, stop_epoch=%s)", str(early_stopped), str(stop_epoch))
        print(f"Training completed (early_stopped={early_stopped}, stop_epoch={stop_epoch})")
            
            
            # ===== CKD paper-style utility metrics =====
            # if epoch == self.args.epochs - 1:  # 最后一个 epoch 评测
            #     L_util_auc, fl_auc_curve, full_auc = self.eval_utility_auc_curve(
            #         loader=self.val_loader,
            #         alpha_ckd=1.5,
            #         party_dropout="random",  # 或 "first_m"
            #         seed=self.args.seed
            #     )
            #     self.logger.info(
            #         "[CKD-Utility] L_Util(AUC)=%.6f | Full_FL_AUC(m=K)=%.6f | FL_AUC_curve=%s",
            #         L_util_auc, full_auc, fl_auc_curve
            #     )
            #     print("[CKD-Utility]", "L_Util(AUC)=", L_util_auc, "Full_FL_AUC=", full_auc, "Curve=", fl_auc_curve)

                # ===== Optional: privacy leakage (PMC) =====
                # 需要你提前准备 self.pmc_train_loader / self.pmc_test_loader
                # if hasattr(self, "pmc_train_loader") and hasattr(self, "pmc_test_loader"):
                #     priv_auc, per_party = self.eval_privacy_pmc_auc(self.pmc_train_loader, self.pmc_test_loader, use_prob=False)
                #     self.logger.info("[CKD-Priv] Priv(mean AUC)=%.6f | per_party=%s", priv_auc, per_party)
                #     print("[CKD-Priv]", "Priv(mean AUC)=", priv_auc, "per_party=", per_party)
        # self.logger.info("Best checkpoint: epoch=%d %s=%.6f path=%s",
        #          best_epoch, best_metric, best_score, best_path)
        # print(f"Best checkpoint: epoch={best_ep   och} {best_metric}={best_score:.6f} path={best_path}")
        self.logger.info("Training completed")

    # LFBA 的测试函数，评测主任务准确率、攻击成功率（ASR）和目标标签的准确率
    def test(self, epoch, val_losses, acc):
        self.logger.info("Epoch %s validation started", epoch + 1)
        # self.logger.info("=> Test ASR...")
        model_list = self.model_list
        model_list = [model.eval() for model in model_list]
        self.simplex.eval()

        # test main task accuracy
        running_val_loss = 0.0
        batch_loss_list = []
        total = 0
        correct = 0
        total_target = 0
        correct_target = 0

        all_val_logits = []
        all_val_labels = []
        for step, (inputs, x_p, labels, index) in enumerate(tqdm(self.val_loader, total=len(self.val_loader), desc=f"Epoch {epoch+1}/{self.args.epochs} - Validation")):
            x = inputs.to(self.device).float()
            y = labels.to(self.device).long()
            # split data for vfl
            inputs_list = split_vfl(x, self.args)
            local_outputs_logits_list = []
            local_outputs_classifier_list = []
            global_input_list = []
            # get the local model outputs
            for i in range(self.args.client_num):
                local_classifier_output, local_outputs_logits = model_list[i + 1](inputs_list[i + 1])
                local_outputs_logits_list.append(local_outputs_logits)
                local_outputs_classifier_list.append(local_classifier_output)
            # get the global model inputs, recording the gradients
            for i in range(self.args.client_num):
                global_input_t = local_outputs_classifier_list[i].detach().clone()
                global_input_t.requires_grad_(True)
                global_input_list.append(global_input_t)

            # 全局模型前向传播 利用单形层聚合后输入全局模型，且单形层输出维度与每个参与方logits维度相同（即不使用额外的线性层调整维度）
            # 设置掩码，模拟参与方缺失（训练时不使用，测试时使用） available_mask: [B,K]
            B = inputs.size(0)
            K = self.args.client_num
            avail_mask = torch.ones(B, K, dtype=torch.bool, device=self.device)
            # 组装 party_logits 输入单形层，维度为 [B, K, C]，其中C是每个参与方logits的维度
            party_logits = torch.stack(local_outputs_classifier_list, dim = 1) # [B, K, C], 这里的C是类别数量
            # simplex 聚合被动方：输出就是聚合后的分类 logits，维度为 [B, C]，权重维度为 [B, K]
            # simplex_outputs, simplex_weights, simplex_outputs_embedding = self.simplex(party_logits, avail_mask) # [B, C], [B, K], 这里的C是类别数量
            simplex_outputs, simplex_weights, simplex_outputs_embedding, simplex_aux, simplex_outputs_pre = self.simplex(party_logits, avail_mask)
            # 主动方
            active_outputs_classifier, active_outputs_logits = self.model_list[0](inputs_list[0]) # [B, C]
            global_outputs = active_outputs_classifier + self.args.alpha * simplex_outputs

            all_val_logits.append(global_outputs.detach().cpu())
            all_val_labels.append(labels.detach().cpu())

            # global model backward
            loss = self.criterion(global_outputs, y)
            batch_loss_list.append(loss.item())
            running_val_loss += loss.item()

            # calculate the testing accuracy
            _, predicted = global_outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            total_target += (y == self.args.target_label).float().sum()
            correct_target += predicted.eq(y)[y == self.args.target_label].float().sum().item()

        avg_val_loss = running_val_loss / len(self.val_loader)
        val_losses.append(avg_val_loss)
        val_acc = correct / total
        acc.append(val_acc)

        result = {
            "avg_val_loss": float(avg_val_loss),
            "val_acc": float(val_acc),
            "all_val_logits": torch.cat(all_val_logits, dim=0),
            "all_val_labels": torch.cat(all_val_labels, dim=0),
        }
        
        # test poison accuracy and asr
        self.logger.info("=> Test ASR...")
        total_poison = 0
        correct_poison = 0
        total_asr = 0
        correct_asr = 0
        for step, (inputs, x_p, labels, index) in enumerate(tqdm(self.test_asr_loader, total=len(self.test_asr_loader), desc=f"Epoch {epoch+1}/{self.args.epochs} - ASR Testing")):
            x = inputs.to(self.device).float()
            y = labels.to(self.device).long()
            y_attack_target = torch.ones(size=y.shape).to(self.device).long()
            y_attack_target *= self.args.target_label
            # split data for vfl
            inputs_list = split_vfl(x, self.args)
            local_outputs_logits_list = []
            local_outputs_classifier_list = []
            global_input_list = []
            # get the local model outputs
            for i in range(self.args.client_num):
                local_classifier_output, local_outputs_logits = model_list[i + 1](inputs_list[i + 1])
                local_outputs_logits_list.append(local_outputs_logits)
                local_outputs_classifier_list.append(local_classifier_output)
            # get the global model inputs, recording the gradients
            for i in range(self.args.client_num):
                global_input_t = local_outputs_classifier_list[i].detach().clone()
                global_input_t.requires_grad_(True)
                global_input_list.append(global_input_t)

            # 全局模型前向传播 利用单形层聚合后输入全局模型，且单形层输出维度与每个参与方logits维度相同（即不使用额外的线性层调整维度）
            # 设置掩码，模拟参与方缺失（训练时不使用，测试时使用） available_mask: [B,K]
            B = inputs.size(0)
            K = self.args.client_num
            avail_mask = torch.ones(B, K, dtype=torch.bool, device=self.device)
            # 组装 party_logits 输入单形层，维度为 [B, K, C]，其中C是每个参与方logits的维度
            party_logits = torch.stack(local_outputs_classifier_list, dim = 1) # [B, K, C], 这里的C是类别数量
            # simplex 聚合被动方：输出就是聚合后的分类 logits，维度为 [B, C]，权重维度为 [B, K]
            # simplex_outputs, simplex_weights, simplex_outputs_embedding = self.simplex(party_logits, avail_mask) # [B, C], [B, K], 这里的C是类别数量
            simplex_outputs, simplex_weights, simplex_outputs_embedding, simplex_aux, simplex_outputs_pre = self.simplex(party_logits, avail_mask)
            # 主动方
            active_outputs_classifier, active_outputs_logits = self.model_list[0](inputs_list[0]) # [B, C]
            global_outputs = active_outputs_classifier + self.args.alpha * simplex_outputs


            # calculate the poison accuracy
            _, predicted = global_outputs.max(1)
            total_poison += y.size(0)
            correct_poison += predicted.eq(y).sum().item()
            # calculate the asr
            total_asr += (y != self.args.target_label).float().sum()
            correct_asr += (predicted[y != self.args.target_label] == self.args.target_label).float().sum()

        # main task accuracy, poison_acc and asr
        test_acc = correct / total
        test_poison_accuracy = correct_poison / total_poison
        test_asr = correct_asr / total_asr
        test_target = correct_target / total_target
        epoch_loss = sum(batch_loss_list) / len(batch_loss_list)
        test_trade_off = (test_acc + test_asr) / 2
        # main task accuracy on target set
        self.logger.info(
            '=> Test Epoch: {}, main task samples: {}, attack samples: {}, test loss: {:.4f}, test trade off: {:.4f}, test main task '
            'accuracy: {:.4f}, test target accuracy: {:.4f}, test asr: {:.4f}'.format(epoch + 1,
                                                                                      len(self.val_loader.dataset),
                                                                                      len(self.test_asr_loader.dataset),
                                                                                      epoch_loss,
                                                                                      test_trade_off, test_acc,
                                                                                      test_target, test_asr))

        return test_acc, test_poison_accuracy, test_target, test_asr, result

    @staticmethod
    def make_aux_loader_from_train(train_loader, aux_k=100, seed=0):
        ds = train_loader.dataset
        n = len(ds)
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=min(aux_k, n), replace=False)
        aux_set = Subset(ds, idx.tolist())
        aux_loader = DataLoader(
            aux_set,
            batch_size=min(aux_k, 128),
            shuffle=True,
            num_workers=getattr(train_loader, "num_workers", 0),
            pin_memory=getattr(train_loader, "pin_memory", False),
        )
        return aux_loader

    def _final_ckd_eval(self):
        # ===== Utility: prefer TEST loader =====
        util_loader = self.test_loader if self.test_loader is not None else self.val_loader
        if self.test_loader is None:
            self.logger.warning("[CKD-Utility] test_loader is None, fallback to val_loader")

        # 随机求解可能子集的FL-AUC
        # L_util_auc, fl_auc_curve, full_auc = self.eval_utility_auc_curve(
        #     loader=util_loader,
        #     alpha_ckd=self.args.alpha,
        #     party_dropout="random",
        #     seed=getattr(self.args, "seed", 0)
        # )

        # 求解所有可能子集组合的平均 FL-AUC ，得到更稳定的评测结果
        L_util_auc, fl_auc_curve, full_auc = self.eval_utility_auc_curve_strict(
            loader=util_loader,
            alpha_ckd=self.args.alpha
        )

        self.logger.info(
            "[CKD-Utility] L_Util(AUC)=%.6f | Full_FL_AUC(m=K)=%.6f | FL_AUC_curve=%s",
            L_util_auc, full_auc, fl_auc_curve
        )
        print("[CKD-Utility]", "L_Util(AUC)=", L_util_auc, "Full_FL_AUC=", full_auc, "Curve=", fl_auc_curve)

        # ===== PMC: train on small aux from TRAIN, test on TEST (or VAL fallback) =====
        aux_k = getattr(self.args, "pmc_aux_k", 100)        # 50-100
        aux_seed = getattr(self.args, "pmc_aux_seed", getattr(self.args, "seed", 0))
        pmc_use_prob = getattr(self.args, "pmc_use_prob", False)

        # 构造 pmc_train_loader（辅助数据）
        pmc_train_loader = self.make_aux_loader_from_train(self.train_loader, aux_k=aux_k, seed=aux_seed)

        # pmc_test_loader：优先 test，否则 val
        pmc_test_loader = self.test_loader if self.test_loader is not None else self.val_loader
        if self.test_loader is None:
            self.logger.warning("[CKD-Priv] test_loader is None, PMC test fallback to val_loader")

        priv_auc, per_party = self.eval_privacy_pmc_auc(
            pmc_train_loader,
            pmc_test_loader,
            use_prob=pmc_use_prob
        )
        self.logger.info("[CKD-Priv] PMC(mean AUC)=%.6f | per_party=%s", priv_auc, per_party)
        print("[CKD-Priv]", "PMC(mean AUC)=", priv_auc, "per_party=", per_party)

        priv_acc, per_party = self.eval_privacy_pmc_acc(
            pmc_train_loader,
            pmc_test_loader,
            use_prob=pmc_use_prob
        )
        self.logger.info("[CKD-Priv-ACC] PMC(mean ACC)=%.6f | per_party=%s", priv_acc, per_party)
        print("[CKD-Priv-ACC]", "PMC(mean ACC)=", priv_acc, "per_party=", per_party)

        L_util_auc, summary, detail = self.eval_fl_auc_subset_stats(
            self.val_loader,
            alpha_ckd=self.args.alpha
        )

        self.logger.info(f"L_Util(AUC)={L_util_auc:.6f}")

        mean_curve = {m: summary[m]['mean'] for m in summary}
        max_curve  = {m: summary[m]['max'] for m in summary}
        min_curve  = {m: summary[m]['min'] for m in summary}

        self.logger.info(f"FL_AUC_mean_curve={mean_curve}")
        self.logger.info(f"FL_AUC_max_curve={max_curve}")
        self.logger.info(f"FL_AUC_min_curve={min_curve}")

        for m in summary:
            self.logger.info(
                f"[m={m}] mean={summary[m]['mean']:.6f}, "
                f"max={summary[m]['max']:.6f}, "
                f"min={summary[m]['min']:.6f}, "
                f"median={summary[m]['median']:.6f}, "
                f"std={summary[m]['std']:.6f}, "
                f"num_subsets={summary[m]['num_subsets']}"
            )

    # simplex层的冻结，用于测试
    def freeze_simplex(self):
        for p in self.simplex.parameters():
            p.requires_grad_(False)
            p.grad = None  # 直接清空梯度，防止误用
        self.simplex.eval()
        self.logger.info("[Simplex] frozen")
        print("[Simplex] frozen")

    # simplex层的解冻，用于测试
    def unfreeze_simplex(self):
        for p in self.simplex.parameters():
            p.requires_grad_(True)
        self.simplex.train()
        self.logger.info("[Simplex] unfrozen")
        print("[Simplex] unfrozen")


    def _load_best_ckpt(self, best_path):
        ckpt = torch.load(best_path, map_location=self.device)

        self.model_list[0].load_state_dict(ckpt["active_state_dict"])
        for m, sd in zip(self.model_list[1:], ckpt["passive_state_dicts"]):
            m.load_state_dict(sd)
        self.simplex.load_state_dict(ckpt["simplex_state_dict"])

        return ckpt        
    
    @torch.no_grad()
    def eval_utility_auc_curve(self, loader, alpha_ckd=1.5, party_dropout="random", seed=0):
        """
        复用你现有推理逻辑，只改 avail_mask 以模拟缺失被动方。
        返回：
        L_util_auc: active-only AUC
        curve: {m: auc}  m=0..K
        full_auc: m=K 的 auc（等价于你现在 val 的 global_outputs AUC）
        """
        rng = np.random.default_rng(seed)
        device = self.device
        K = self.args.client_num

        # -------- L-Util (active only) --------
        y_all, a_logits_all = [], []
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            inputs_list = split_vfl(inputs, self.args)

            active_outputs_classifier, _ = self.model_list[0](inputs_list[0])  # [B,C]
            y_all.append(labels)
            a_logits_all.append(active_outputs_classifier)

        y_all = torch.cat(y_all, dim=0)
        a_logits_all = torch.cat(a_logits_all, dim=0)
        L_util_auc = auc_from_logits(y_all, a_logits_all)

        # -------- FL-AUC curve over m aligned parties --------
        curve = {}
        full_auc = None

        for m_aligned in range(K + 1):
            y_all, g_logits_all = [], []

            for inputs, labels in loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                inputs_list = split_vfl(inputs, self.args)
                B = inputs.size(0)

                # passive forward
                local_outputs_classifier_list = []
                for i in range(K):
                    local_classifier_output, _ = self.model_list[i + 1](inputs_list[i + 1])
                    local_outputs_classifier_list.append(local_classifier_output)

                # build avail_mask: [B,K]
                if m_aligned == 0:
                    avail_mask = torch.zeros(B, K, dtype=torch.bool, device=device)
                elif m_aligned == K:
                    avail_mask = torch.ones(B, K, dtype=torch.bool, device=device)
                else:
                    avail_mask = torch.zeros(B, K, dtype=torch.bool, device=device)
                    if party_dropout == "first_m":
                        avail_mask[:, :m_aligned] = True
                    else:
                        idx = rng.choice(K, size=m_aligned, replace=False)
                        avail_mask[:, idx] = True

                party_logits = torch.stack(local_outputs_classifier_list, dim=1)  # [B,K,C]
                # simplex_outputs, simplex_weights, simplex_outputs_embedding = self.simplex(party_logits, avail_mask)  # [B,C]
                simplex_outputs, simplex_weights, simplex_outputs_embedding, simplex_aux, simplex_outputs_pre = self.simplex(party_logits, avail_mask)  # [B,C]
                
                # active
                active_outputs_classifier, _ = self.model_list[0](inputs_list[0])  # [B,C]

                # global (与你 val 完全一致)
                if m_aligned == 0:
                    global_outputs = active_outputs_classifier
                else:
                    global_outputs = active_outputs_classifier + alpha_ckd * simplex_outputs

                y_all.append(labels)
                g_logits_all.append(global_outputs)

            y_all = torch.cat(y_all, dim=0)
            g_logits_all = torch.cat(g_logits_all, dim=0)
            auc_m = auc_from_logits(y_all, g_logits_all)

            curve[m_aligned] = auc_m
            if m_aligned == K:
                full_auc = auc_m

        return L_util_auc, curve, full_auc
    

    @torch.no_grad()
    def eval_utility_auc_curve_strict(self, loader, alpha_ckd=1.5):
        """
        严格枚举组合版 FL-AUC 评估

        返回：
        L_util_auc: active-only AUC
        curve: {m: auc}，其中 auc 是“所有 |S|=m 的 passive 子集”的 AUC 平均
        full_auc: m=K 时的 AUC
        detail: {m: {"subsets": [...], "subset_auc": [...]} }，方便你调试/分析
        """
        device = self.device
        K = self.args.client_num

        # =========================================================
        # 1) L-Util：active-only AUC
        # =========================================================
        y_all, a_logits_all = [], []

        for inputs, x_p, labels, index in loader:
            inputs = inputs.to(device).float()
            labels = labels.to(device).long()
            inputs_list = split_vfl(inputs, self.args)

            active_outputs_classifier, _ = self.model_list[0](inputs_list[0])  # [B,C]

            y_all.append(labels.detach())
            a_logits_all.append(active_outputs_classifier.detach())

        y_all = torch.cat(y_all, dim=0)
        a_logits_all = torch.cat(a_logits_all, dim=0)
        L_util_auc = auc_from_logits(y_all, a_logits_all)

        # =========================================================
        # 2) FL-AUC curve：严格枚举每个 m 的所有组合
        # =========================================================
        curve = {}
        detail = {}
        full_auc = None

        passive_ids = list(range(K))  # 注意：这里是被动方编号 0..K-1，对应 model_list[i+1]

        for m_aligned in range(K + 1):
            subset_auc_list = []
            subset_list = list(combinations(passive_ids, m_aligned))

            # m=0 时 combinations 会返回 [()]
            # m=K 时 combinations 会返回 [(0,1,...,K-1)]

            for subset in subset_list:
                y_sub, g_logits_sub = [], []

                for inputs, x_p, labels, index in loader:
                    inputs = inputs.to(device).float()
                    labels = labels.to(device).long()
                    inputs_list = split_vfl(inputs, self.args)
                    B = inputs.size(0)

                    # ---------- passive forward ----------
                    local_outputs_classifier_list = []
                    for i in range(K):
                        local_classifier_output, _ = self.model_list[i + 1](inputs_list[i + 1])
                        local_outputs_classifier_list.append(local_classifier_output)

                    party_logits = torch.stack(local_outputs_classifier_list, dim=1)  # [B,K,C]

                    # ---------- build fixed avail_mask for this subset ----------
                    avail_mask = torch.zeros(B, K, dtype=torch.bool, device=device)
                    if m_aligned > 0:
                        avail_mask[:, list(subset)] = True

                    # ---------- simplex aggregation ----------
                    simplex_outputs, simplex_weights, simplex_outputs_embedding, simplex_aux, simplex_outputs_pre = self.simplex(
                        party_logits, avail_mask
                    )  # [B,C]

                    # ---------- active ----------
                    active_outputs_classifier, _ = self.model_list[0](inputs_list[0])  # [B,C]

                    # ---------- global output ----------
                    if m_aligned == 0:
                        global_outputs = active_outputs_classifier
                    else:
                        global_outputs = active_outputs_classifier + alpha_ckd * simplex_outputs

                    y_sub.append(labels.detach())
                    g_logits_sub.append(global_outputs.detach())

                y_sub = torch.cat(y_sub, dim=0)
                g_logits_sub = torch.cat(g_logits_sub, dim=0)

                auc_subset = auc_from_logits(y_sub, g_logits_sub)
                subset_auc_list.append(float(auc_subset))

            auc_m = float(np.mean(subset_auc_list))
            curve[m_aligned] = auc_m
            detail[m_aligned] = {
                "subsets": subset_list,
                "subset_auc": subset_auc_list,
            }

            if m_aligned == K:
                full_auc = auc_m

        # return L_util_auc, curve, full_auc, detail
        return L_util_auc, curve, full_auc
    

    # 子集枚举版的 FL-AUC 评估，输出更丰富的统计信息，便于分析不同子集的表现差异
    @torch.no_grad()
    def eval_fl_auc_subset_stats(self, loader, alpha_ckd=1.5):
        """
        对每个 m=0..K，严格枚举所有大小为 m 的 passive 子集，
        分别计算 AUC，然后输出多种汇总口径。

        Returns:
            L_util_auc: float
            summary: dict
                {
                m: {
                    "mean": ...,
                    "max": ...,
                    "min": ...,
                    "median": ...,
                    "std": ...,
                    "num_subsets": ...
                    }
                }
            detail: dict
                {
                m: {
                    "subsets": [...],
                    "subset_auc": [...]
                    }
                }
        """
        device = self.device
        K = self.args.client_num

        # =====================================================
        # 1) L-Util: active-only AUC
        # =====================================================
        y_all, a_logits_all = [], []
        for inputs, x_p, labels, index in loader:
            inputs = inputs.to(device).float()
            labels = labels.to(device).long()
            inputs_list = split_vfl(inputs, self.args)

            active_outputs_classifier, _ = self.model_list[0](inputs_list[0])
            y_all.append(labels.detach())
            a_logits_all.append(active_outputs_classifier.detach())

        y_all = torch.cat(y_all, dim=0)
        a_logits_all = torch.cat(a_logits_all, dim=0)
        L_util_auc = float(auc_from_logits(y_all, a_logits_all))

        # =====================================================
        # 2) FL-AUC subset statistics
        # =====================================================
        passive_ids = list(range(K))
        summary = {}
        detail = {}

        for m_aligned in range(K + 1):
            subset_list = list(combinations(passive_ids, m_aligned))
            subset_auc_list = []

            for subset in subset_list:
                y_sub, g_logits_sub = [], []

                for inputs, x_p, labels, index in loader:
                    inputs = inputs.to(device).float()
                    labels = labels.to(device).long()
                    inputs_list = split_vfl(inputs, self.args)
                    B = inputs.size(0)

                    # ---------- passive forward ----------
                    local_outputs_classifier_list = []
                    for i in range(K):
                        local_classifier_output, _ = self.model_list[i + 1](inputs_list[i + 1])
                        local_outputs_classifier_list.append(local_classifier_output)

                    party_logits = torch.stack(local_outputs_classifier_list, dim=1)  # [B,K,C]

                    # ---------- fixed avail_mask for this subset ----------
                    avail_mask = torch.zeros(B, K, dtype=torch.bool, device=device)
                    if m_aligned > 0:
                        avail_mask[:, list(subset)] = True

                    # ---------- simplex ----------
                    simplex_outputs, simplex_weights, simplex_outputs_embedding, simplex_aux, simplex_outputs_pre = self.simplex(
                        party_logits, avail_mask
                    )

                    # ---------- active ----------
                    active_outputs_classifier, _ = self.model_list[0](inputs_list[0])

                    # ---------- global ----------
                    if m_aligned == 0:
                        global_outputs = active_outputs_classifier
                    else:
                        global_outputs = active_outputs_classifier + alpha_ckd * simplex_outputs

                    y_sub.append(labels.detach())
                    g_logits_sub.append(global_outputs.detach())

                y_sub = torch.cat(y_sub, dim=0)
                g_logits_sub = torch.cat(g_logits_sub, dim=0)

                auc_subset = float(auc_from_logits(y_sub, g_logits_sub))
                subset_auc_list.append(auc_subset)

            arr = np.array(subset_auc_list, dtype=np.float64)

            summary[m_aligned] = {
                "mean": float(arr.mean()),
                "max": float(arr.max()),
                "min": float(arr.min()),
                "median": float(np.median(arr)),
                "std": float(arr.std(ddof=0)),
                "num_subsets": len(subset_list),
            }
            detail[m_aligned] = {
                "subsets": subset_list,
                "subset_auc": subset_auc_list,
            }

        return L_util_auc, summary, detail


    @torch.no_grad()
    def _collect_passive_logits(self, loader, party_id: int):
        """
        party_id: 1..K 对应 self.model_list[party_id]
        返回 Z:[N,C], Y:[N]
        """
        device = self.device
        Zs, Ys = [], []
        for inputs, x_p, labels, index in loader:
            inputs = inputs.to(device).float()
            labels = labels.to(device).long()
            inputs_list = split_vfl(inputs, self.args)

            local_classifier_output, _ = self.model_list[party_id](inputs_list[party_id])  # [B,C]
            Zs.append(local_classifier_output.detach().cpu())
            Ys.append(labels.detach().cpu())
        return torch.cat(Zs, dim=0).numpy(), torch.cat(Ys, dim=0).numpy().astype(int)

    def eval_privacy_pmc_acc(self, pmc_train_loader, pmc_test_loader, use_prob=False):
        """
        每个被动方 k 用其输出 z_k 训练 label predictor，输出泄露 ACC。
        返回:
            mean_acc: float
            per_party: List[float]
        """
        K = self.args.client_num
        per_party = []

        for k in range(1, K + 1):
            Ztr, Ytr = self._collect_passive_logits(pmc_train_loader, k)
            Zte, Yte = self._collect_passive_logits(pmc_test_loader, k)

            if use_prob:
                Ztr = F.softmax(torch.tensor(Ztr), dim=1).cpu().numpy()
                Zte = F.softmax(torch.tensor(Zte), dim=1).cpu().numpy()

            clf = LogisticRegression(
                max_iter=2000,
                n_jobs=-1,
            )
            clf.fit(Ztr, Ytr)

            pred = clf.predict(Zte)
            acc_k = accuracy_score(Yte, pred)
            per_party.append(float(acc_k))

        return float(np.mean(per_party)), per_party

    def eval_privacy_pmc_auc(self, pmc_train_loader, pmc_test_loader, use_prob=False):
        """
        每个被动方 k 用其输出 z_k 训练 label predictor，输出泄露 AUC。
        """
        K = self.args.client_num
        per_party = []

        for k in range(1, K + 1):
            Ztr, Ytr = self._collect_passive_logits(pmc_train_loader, k)
            Zte, Yte = self._collect_passive_logits(pmc_test_loader, k)

            if use_prob:
                Ztr = F.softmax(torch.tensor(Ztr), dim=1).numpy()
                Zte = F.softmax(torch.tensor(Zte), dim=1).numpy()

            clf = LogisticRegression(max_iter=2000, n_jobs=-1)
            clf.fit(Ztr, Ytr)
            proba = clf.predict_proba(Zte)

            if proba.shape[1] == 2:
                auc_k = roc_auc_score(Yte, proba[:, 1])
            else:
                auc_k = roc_auc_score(Yte, proba, multi_class="ovr", average="macro")
            per_party.append(float(auc_k))

        return float(np.mean(per_party)), per_party
    
    @torch.no_grad()
    def _eval_active_only_auc(self, loader):
        ys, logits = [], []
        self.model_list[0].eval()
        for inputs, x_p, labels, index in loader:
            inputs = inputs.to(self.device).float()
            labels = labels.to(self.device).long()

            inputs_list = split_vfl(inputs, self.args)
            x0 = inputs_list[0]

            active_logits, _ = self.model_list[0](x0)  # [B,C]
            ys.append(labels.detach().cpu())
            logits.append(active_logits.detach().cpu())

        return auc_from_logits(torch.cat(ys, 0), torch.cat(logits, 0))


    def cold_start_active_only(self, cold_epochs: int):
        """
        Stage-0: active-only cold start.
        数据仍然来自 train_loader，只是每个 batch 动态 split，然后只训练 active。
        """
        if cold_epochs <= 0:
            return

        self.logger.info("[ColdStart] active-only for %d epochs", cold_epochs)

        # 1) 冻结 passive + simplex（避免误更新）
        for m in self.model_list[1:]:
            for p in m.parameters():
                p.requires_grad_(False)
            m.eval()
        for p in self.simplex.parameters():
            p.requires_grad_(False)
        self.simplex.eval()

        # 2) 训练 active
        active = self.model_list[0]
        for p in active.parameters():
            p.requires_grad_(True)

        opt = self.optimizer_list[0]  # 复用 active optimizer

        for ep in range(cold_epochs):
            active.train()
            running = 0.0
            for step, (inputs, x_p, labels, index) in enumerate(self.train_loader):
                inputs = inputs.to(self.device).float()
                labels = labels.to(self.device).long()

                inputs_list = split_vfl(inputs, self.args)
                x0 = inputs_list[0]

                opt.zero_grad(set_to_none=True)
                active_logits, _ = active(x0)

                # debugger
                # 调试信息：放在 loss 前
                print("active_logits.shape =", active_logits.shape)
                print("active_logits.dtype =", active_logits.dtype)
                print("labels.shape =", labels.shape)
                print("labels.dtype =", labels.dtype)

                labels_flat = labels.view(-1)
                print("labels min/max =", labels_flat.min().item(), labels_flat.max().item())
                print("labels unique =", torch.unique(labels_flat).detach().cpu().tolist()[:20])

                assert active_logits.dim() == 2, f"active_logits.dim={active_logits.dim()}, shape={active_logits.shape}"
                num_classes = active_logits.size(1)

                assert labels_flat.dim() == 1, f"labels must be 1D after view(-1), got {labels.shape}"
                assert labels_flat.dtype == torch.long, f"labels dtype must be long, got {labels_flat.dtype}"
                assert labels_flat.min().item() >= 0, f"labels contain negative values: min={labels_flat.min().item()}"
                assert labels_flat.max().item() < num_classes, (
                    f"labels out of range: max={labels_flat.max().item()}, num_classes={num_classes}"
                )

                loss = self.criterion(active_logits, labels)
                loss.backward()
                opt.step()
                running += loss.item()

            avg_loss = running / max(1, len(self.train_loader))
            val_auc = self._eval_active_only_auc(self.val_loader)
            val_acc = self._eval_active_only_acc(self.val_loader)
            self.logger.info("[ColdStart] ep %d/%d loss=%.6f active_val_auc=%.6f",
                            ep + 1, cold_epochs, avg_loss, val_auc)
            self.logger.info("[ColdStart] ep %d/%d active_val_acc=%.6f", ep + 1, cold_epochs, val_acc)
            print(f"[ColdStart] ep {ep+1}/{cold_epochs} active_val_acc={val_acc:.4f}")
            print(f"[ColdStart] ep {ep+1}/{cold_epochs} loss={avg_loss:.4f} active_val_auc={val_auc:.4f}")

        # 3) 解冻 passive + simplex，进入联合训练
        for m in self.model_list[1:]:
            for p in m.parameters():
                p.requires_grad_(True)
        for p in self.simplex.parameters():
            p.requires_grad_(True)

        self.logger.info("[ColdStart] finished")

    
    def _eval_active_only_acc(self, loader):
        """仅使用 active 模型计算验证集准确率"""
        active = self.model_list[0]
        active.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, x_p, labels, index in loader:
                inputs = inputs.to(self.device).float()
                labels = labels.to(self.device).long()
                inputs_list = split_vfl(inputs, self.args)
                x0 = inputs_list[0]
                logits, _ = active(x0)
                pred = logits.argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        acc = correct / total if total > 0 else 0.0
        return acc
