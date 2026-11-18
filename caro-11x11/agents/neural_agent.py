"""Tác tử dùng CNN Policy đã huấn luyện để chọn nước Caro."""

from pathlib import Path
from pickle import UnpicklingError

import numpy as np
import torch

from agents.base_agent import BaseAgent
from config.search_budget import SearchBudget
from training.dataset import normalize_board
from training.model import load_policy, mask_occupied

DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[1] / "models/caro-policy-v1/policy.pt"


class NeuralAgent(BaseAgent):
    """Nạp mô hình một lần và suy luận trên CPU với một luồng tính toán."""

    def __init__(self, checkpoint: Path | str = DEFAULT_CHECKPOINT, *,
                 budget: SearchBudget | None = None):
        super().__init__("CNN Policy", budget=budget)
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy checkpoint: {path}")
        torch.set_num_threads(1)
        try:
            self.model = load_policy(path, "cpu")
        except (OSError, RuntimeError, ValueError, KeyError, UnpicklingError) as error:
            raise ValueError(f"Không nạp được checkpoint {path}: {error}") from error

    def get_action(self, board: np.ndarray, player: int) -> tuple[int, int]:
        normalized = normalize_board(board, player)
        channels = np.stack((normalized == 1, normalized == -1)).astype(np.float32)
        inputs = torch.from_numpy(channels).unsqueeze(0)
        with torch.inference_mode():
            logits = self.model(inputs)
            if not torch.isfinite(logits).all():
                raise ValueError("Mô hình trả về logit không hữu hạn")
            action = int(mask_occupied(logits, inputs).argmax(dim=1).item())
        return divmod(action, board.shape[0])
