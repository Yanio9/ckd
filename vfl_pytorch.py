import torch
import torch.nn as nn
import torch.optim as optim


class PartyModel(nn.Module):
    def __init__(self, in_features: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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
) -> None:
    """Simple vertical FL with two parties and a central server.

    Party A owns features x_a, Party B owns features x_b, and labels y live on the server.
    The parties only send intermediate embeddings to the server; gradients are returned.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_a = x_a.to(device)
    x_b = x_b.to(device)
    y = y.to(device)

    party_a = PartyModel(x_a.shape[1], hidden).to(device)
    party_b = PartyModel(x_b.shape[1], hidden).to(device)
    server = ServerModel(hidden * 2, num_classes=len(torch.unique(y))).to(device)

    opt_a = optim.SGD(party_a.parameters(), lr=lr)
    opt_b = optim.SGD(party_b.parameters(), lr=lr)
    opt_s = optim.SGD(server.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        opt_a.zero_grad()
        opt_b.zero_grad()
        opt_s.zero_grad()

        # Parties compute local embeddings.
        emb_a = party_a(x_a)
        emb_b = party_b(x_b)

        # Send embeddings to server for prediction.
        emb = torch.cat([emb_a, emb_b], dim=1)
        logits = server(emb)
        loss = criterion(logits, y)

        # Backprop on server.
        loss.backward()

        # Server sends gradients back to parties.
        grad_a, grad_b = emb_a.grad, emb_b.grad
        if grad_a is None:
            grad_a = torch.autograd.grad(loss, emb_a, retain_graph=True)[0]
        if grad_b is None:
            grad_b = torch.autograd.grad(loss, emb_b, retain_graph=True)[0]

        # Parties apply gradients locally.
        emb_a.backward(grad_a)
        emb_b.backward(grad_b)

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
