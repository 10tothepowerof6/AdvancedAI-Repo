"""Mạng policy cho Caro 11×11 và cách nạp trọng số đã huấn luyện."""

from pathlib import Path

import torch
from torch import nn


class CaroPolicy(nn.Module):
    """Nhận hai kênh quân mình/đối thủ, trả một logit cho mỗi ô."""

    VERSION = "caro-policy-v1"

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1), nn.Flatten(start_dim=1),
        )

    def forward(self, boards: torch.Tensor) -> torch.Tensor:
        return self.layers(boards)


def mask_occupied(logits: torch.Tensor, boards: torch.Tensor) -> torch.Tensor:
    """Loại các ô đã có quân khỏi phân phối và thứ hạng của policy."""
    occupied = boards.ne(0).any(dim=1).flatten(start_dim=1)
    return logits.masked_fill(occupied, -torch.inf)


def save_policy(path: Path, model: CaroPolicy, *, epoch: int, dataset_sha256: str) -> None:
    """Lưu trọng số và thông tin cần để truy vết mô hình."""
    torch.save({"model_version": model.VERSION, "state_dict": model.state_dict(),
                "epoch": epoch, "dataset_sha256": dataset_sha256}, path)


def load_policy(path: Path | str, device: torch.device | str = "cpu") -> CaroPolicy:
    """Nạp trọng số vào mô hình ở chế độ đánh giá."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint["model_version"] != CaroPolicy.VERSION:
        raise ValueError("Phiên bản kiến trúc trong checkpoint không tương thích")
    model = CaroPolicy().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()
