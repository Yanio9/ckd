import torch
import torch.nn as nn
import torch.optim as optim


class PartyModel(nn.Module):
    def __init__(self, in_features: int, hidden: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict(self, emb: torch.Tensor) -> torch.Tensor:
        return self.head(emb)


class ServerModel(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


def vertical_federated_training(
    x_a: torch.Tensor,
    x_b: torch.Tensor,
    y: torch.Tensor,
    hidden: int = 16,
    epochs: int = 5,
    lr: float = 1e-2,
    temperature: float = 2.0,
    alpha: float = 0.5,
) -> None:
    """Simple vertical FL with two parties and a central server.

    Party A owns features x_a, Party B owns features x_b, and labels y live on the server.
    The parties only send intermediate embeddings to the server; gradients are returned.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_a = x_a.to(device)
    x_b = x_b.to(device)
    y = y.to(device)

    num_classes = len(torch.unique(y))
    party_a = PartyModel(x_a.shape[1], hidden, num_classes).to(device)
    party_b = PartyModel(x_b.shape[1], hidden, num_classes).to(device)
    server = ServerModel(hidden * 2, num_classes=num_classes).to(device)

    opt_a = optim.SGD(party_a.parameters(), lr=lr)
    opt_b = optim.SGD(party_b.parameters(), lr=lr)
    opt_s = optim.SGD(server.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    kd_criterion = nn.KLDivLoss(reduction="batchmean")

    for epoch in range(epochs):
        opt_a.zero_grad()
        opt_b.zero_grad()
        opt_s.zero_grad()

        # Parties compute local embeddings.
        # 中文注释：各参与方在本地计算中间表示（不共享原始特征）。
        emb_a = party_a(x_a)
        emb_b = party_b(x_b)

        # Send embeddings to server for prediction (detached to simulate VFL).
        # 中文注释：将中间表示发送到服务器，使用 detach 模拟纵向联邦的安全传输。
        emb_a_det = emb_a.detach().requires_grad_(True)
        emb_b_det = emb_b.detach().requires_grad_(True)
        emb = torch.cat([emb_a_det, emb_b_det], dim=1)
        logits = server(emb)
        ce_loss = criterion(logits, y)

        # Knowledge distillation: server (teacher) -> parties (students).
        # 中文注释：服务器作为教师模型，将软标签蒸馏给参与方学生模型。
        with torch.no_grad():
            teacher_prob = torch.softmax(logits / temperature, dim=1)
        logit_a = party_a.predict(emb_a)
        logit_b = party_b.predict(emb_b)
        kd_loss_a = kd_criterion(
            torch.log_softmax(logit_a / temperature, dim=1),
            teacher_prob,
        )
        kd_loss_b = kd_criterion(
            torch.log_softmax(logit_b / temperature, dim=1),
            teacher_prob,
        )
        kd_loss = (kd_loss_a + kd_loss_b) * (temperature**2)

        # Backprop on server to get gradients for party embeddings.
        # 中文注释：服务器端反向传播得到对各参与方中间表示的梯度。
        ce_loss.backward()
        grad_a = emb_a_det.grad
        grad_b = emb_b_det.grad

        # Parties apply KD loss + server gradients locally.
        # 中文注释：参与方在本地结合蒸馏损失和服务器梯度更新参数。
        (alpha * kd_loss).backward(retain_graph=True)
        emb_a.backward(grad_a)
        emb_b.backward(grad_b)

        loss = ce_loss + alpha * kd_loss

        opt_a.step()
        opt_b.step()
        opt_s.step()

        pred = torch.argmax(logits, dim=1)
        acc = (pred == y).float().mean().item()
        print(f"epoch={epoch + 1} loss={loss.item():.4f} acc={acc:.3f}")


def main() -> None:
    torch.manual_seed(42)
    num_samples = 256
    x_a = torch.randn(num_samples, 8)
    x_b = torch.randn(num_samples, 6)

    # Create a synthetic label based on both feature partitions.
    logits = x_a[:, 0] + 0.5 * x_b[:, 1] - 0.3 * x_a[:, 2]
    y = (logits > 0).long()

    vertical_federated_training(x_a, x_b, y)


if __name__ == "__main__":
    main()
