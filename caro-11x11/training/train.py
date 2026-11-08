"""Huấn luyện CNN Policy có giám sát và chọn checkpoint bằng validation loss."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from training.dataset import PolicyDataset
from training.generate_data import DEFAULT_OUTPUT as DEFAULT_DATA
from training.model import CaroPolicy, load_policy, mask_occupied, save_policy

DEFAULT_OUTPUT = Path("models/caro-policy-v1")


def _sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _read_split(data: Path, manifest: dict, split: str) -> PolicyDataset:
    filename = f"{split}.npz"
    if _sha256(data / filename) != manifest["sha256"][filename]:
        raise ValueError(f"SHA-256 của tập {split} không khớp manifest")
    dataset = PolicyDataset(data, split, augment=split == "train")
    if len(dataset) == 0:
        raise ValueError(f"Tập {split} không có mẫu")
    return dataset


def _run_epoch(model: CaroPolicy, loader: DataLoader, device: torch.device,
               optimizer: torch.optim.Optimizer | None = None) -> dict[str, float]:
    is_training = optimizer is not None
    model.train(is_training)
    totals = torch.zeros(3, device=device)
    samples = 0
    with torch.set_grad_enabled(is_training):
        for boards, labels in loader:
            boards, labels = boards.to(device), labels.to(device)
            logits = mask_occupied(model(boards), boards)
            loss = nn.functional.cross_entropy(logits, labels)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            # Cộng theo số mẫu để lô cuối ngắn hơn vẫn có trọng số đúng.
            batch_size = labels.size(0)
            top = logits.detach().topk(3, dim=1).indices.eq(labels[:, None])
            totals[0] += loss.detach() * batch_size
            totals[1] += top[:, 0].sum()
            totals[2] += top.any(dim=1).sum()
            samples += batch_size

    loss, top1, top3 = (totals / samples).tolist()
    if not math.isfinite(loss):
        raise ValueError("Loss không hữu hạn. Kiểm tra trọng số và tính hợp lệ của nhãn")
    return {"loss": loss, "top1": top1, "top3": top3}


def train(data: Path, output: Path, *, epochs: int, batch_size: int, lr: float,
          patience: int, seed: int, device_name: str) -> dict:
    """Ghi báo cáo và checkpoint tốt nhất vào một thư mục mới."""
    if output.exists():
        raise FileExistsError(f"Thư mục kết quả đã tồn tại: {output}")
    if min(epochs, batch_size, patience) < 1 or seed < 0 or not math.isfinite(lr) or lr <= 0:
        raise ValueError("Epoch, batch size, patience và learning rate phải dương; seed không âm")
    cuda_available = torch.cuda.is_available()
    if device_name == "cuda" and not cuda_available:
        raise ValueError("Đã yêu cầu CUDA nhưng PyTorch không phát hiện GPU khả dụng")
    device = torch.device("cuda" if device_name == "auto" and cuda_available else
                          "cpu" if device_name == "auto" else device_name)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    manifest_path = data / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_sha = _sha256(manifest_path)
    train_loader = DataLoader(_read_split(data, manifest, "train"), batch_size=batch_size,
                              shuffle=True, generator=torch.Generator().manual_seed(seed))
    validation_loader = DataLoader(_read_split(data, manifest, "val"), batch_size=batch_size)
    model = CaroPolicy().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    best_epoch = 0
    best_loss = math.inf

    print(f"Huấn luyện trên {device}; {len(train_loader.dataset)} mẫu train sau tăng cường.", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".caro-train-") as temporary:
        staging = Path(temporary) / "run"
        staging.mkdir()
        checkpoint = staging / "policy.pt"
        start = perf_counter()
        for epoch in range(1, epochs + 1):
            train_metrics = _run_epoch(model, train_loader, device, optimizer)
            validation = _run_epoch(model, validation_loader, device)
            history.append({"epoch": epoch, "train": train_metrics, "validation": validation})
            print(f"Epoch {epoch:02d} | loss train {train_metrics['loss']:.4f} | "
                  f"loss val {validation['loss']:.4f} | top-1 val {validation['top1']:.2%}", flush=True)
            if validation["loss"] < best_loss:
                best_epoch, best_loss = epoch, validation["loss"]
                save_policy(checkpoint, model, epoch=epoch, dataset_sha256=manifest_sha)
            if epoch - best_epoch >= patience:
                print(f"Dừng sớm sau {patience} epoch không cải thiện validation loss.", flush=True)
                break

        training_seconds = perf_counter() - start
        # Chỉ đọc test sau khi đã cố định checkpoint theo validation.
        best_model = load_policy(checkpoint, device)
        test_loader = DataLoader(_read_split(data, manifest, "test"), batch_size=batch_size)
        test_metrics = _run_epoch(best_model, test_loader, device)
        report = {
            "model_version": model.VERSION,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "data": {"teacher": manifest["teacher"], "manifest_sha256": manifest_sha,
                     "samples": {split: stats["samples"] for split, stats in manifest["splits"].items()},
                     "augmented_train_samples": len(train_loader.dataset)},
            "config": {"seed": seed, "optimizer": "Adam", "learning_rate": lr, "batch_size": batch_size,
                       "max_epochs": epochs, "patience": patience, "selection_metric": "validation_loss"},
            "runtime": {"device": str(device), "cpu": platform.processor() or platform.machine(),
                        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                        "python": platform.python_version(), "numpy": np.__version__,
                        "torch": str(torch.__version__), "cuda": torch.version.cuda,
                        "training_seconds": training_seconds},
            "history": history, "best_epoch": best_epoch, "test": test_metrics,
            "checkpoint": {"file": checkpoint.name, "bytes": checkpoint.stat().st_size,
                           "sha256": _sha256(checkpoint)},
        }
        (staging / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        staging.rename(output)

    print(f"Checkpoint epoch {best_epoch} | test loss {test_metrics['loss']:.4f} | "
          f"top-1 {test_metrics['top1']:.2%} | top-3 {test_metrics['top3']:.2%}", flush=True)
    print(f"Đã lưu kết quả vào {output}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Huấn luyện mạng CNN Policy Caro từ dữ liệu tự đấu.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Thư mục bộ dữ liệu đã sinh")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Thư mục kết quả chưa tồn tại")
    parser.add_argument("--epochs", type=int, default=20, help="Số epoch tối đa")
    parser.add_argument("--batch-size", type=int, default=64, help="Số mẫu trong mỗi lô")
    parser.add_argument("--lr", type=float, default=0.001, help="Tốc độ học")
    parser.add_argument("--patience", type=int, default=5, help="Số epoch chờ cải thiện validation loss")
    parser.add_argument("--seed", type=int, default=42, help="Seed để tái lập")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="Thiết bị huấn luyện")
    args = parser.parse_args()
    try:
        train(args.data, args.output, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
              patience=args.patience, seed=args.seed, device_name=args.device)
    except (OSError, ValueError) as error:
        parser.exit(2, f"Không thể huấn luyện: {error}\n")


if __name__ == "__main__":
    main()
