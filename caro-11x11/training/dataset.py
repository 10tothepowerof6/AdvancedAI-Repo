"""Biểu diễn và đọc dữ liệu Caro cho bài toán dự đoán nước đi."""

from numbers import Integral
from pathlib import Path

import numpy as np

from config.game_config import BOARD_SIZE, PLAYER_O, PLAYER_X

SPLITS = ("train", "val", "test")
ACTION_COUNT = BOARD_SIZE ** 2


def _symmetries() -> np.ndarray:
    cells = np.arange(ACTION_COUNT).reshape(BOARD_SIZE, BOARD_SIZE)
    rotations = [np.rot90(cells, turn) for turn in range(4)]
    return np.array([view.ravel() for view in rotations + [np.fliplr(view) for view in rotations]])


_PERMUTATIONS = _symmetries()
_INVERSES = np.argsort(_PERMUTATIONS, axis=1)


def normalize_board(board: np.ndarray, player: int) -> np.ndarray:
    """Đưa quân mình về 1 và quân đối thủ về -1, giữ ô trống bằng 0."""
    if (not isinstance(board, np.ndarray) or board.shape != (BOARD_SIZE, BOARD_SIZE)
            or not np.isin(board, (-1, 0, 1)).all()):
        raise ValueError("Bàn cờ phải có kích thước 11×11 và chỉ chứa -1, 0, 1")
    if isinstance(player, bool) or not isinstance(player, Integral) or player not in (PLAYER_X, PLAYER_O):
        raise ValueError("Người chơi phải là X hoặc O")
    return board.astype(np.int8) * int(player)


def transform_sample(board: np.ndarray, action: int, symmetry: int) -> tuple[np.ndarray, int]:
    """Áp dụng cùng phép xoay/lật cho bàn cờ và chỉ số nước đi."""
    if not 0 <= symmetry < len(_PERMUTATIONS) or not 0 <= action < ACTION_COUNT:
        raise ValueError("Mã đối xứng hoặc chỉ số nước đi không hợp lệ")
    transformed = board.ravel()[_PERMUTATIONS[symmetry]].reshape(board.shape)
    return transformed, int(_INVERSES[symmetry, action])


def canonical_key(board: np.ndarray) -> bytes:
    """Tạo khóa chung cho các bàn cờ tương đương qua xoay hoặc lật."""
    return min(view.tobytes() for view in board.ravel()[_PERMUTATIONS])


class PolicyDataset:
    """Trả về cặp (hai kênh float32, chỉ số nước đi) cho từng mẫu."""

    def __init__(self, directory: Path | str, split: str, *, augment: bool = False):
        if split not in SPLITS:
            raise ValueError(f"Tập dữ liệu phải là một trong {SPLITS}")
        if augment and split != "train":
            raise ValueError("Chỉ tăng cường dữ liệu trên tập train")
        with np.load(Path(directory) / f"{split}.npz", allow_pickle=False) as arrays:
            self.boards = arrays["boards"]
            self.actions = arrays["actions"]
            self.game_ids = arrays["game_ids"]
            self.plies = arrays["plies"]
        self.augment = augment

    def __len__(self) -> int:
        return len(self.actions) * (len(_PERMUTATIONS) if self.augment else 1)

    def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
        sample, symmetry = divmod(index, len(_PERMUTATIONS)) if self.augment else (index, 0)
        board, action = self.boards[sample], int(self.actions[sample])
        if symmetry:
            board, action = transform_sample(board, action, symmetry)
        channels = np.stack((board == 1, board == -1)).astype(np.float32)
        return channels, action
