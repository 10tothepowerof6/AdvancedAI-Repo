"""Tác tử chiến thuật dùng để tự đấu và sinh nhãn cho mạng policy."""

from functools import lru_cache

import numpy as np

from agents.base_agent import BaseAgent
from config.game_config import EMPTY, WIN_LENGTH
from core.game_engine import Action, CaroGame
from core.game_utils import candidate_actions


@lru_cache
def _winning_lines(size: int) -> np.ndarray:
    lines = []
    for row in range(size):
        for col in range(size):
            for dr, dc in CaroGame.DIRECTIONS:
                end_row = row + (WIN_LENGTH - 1) * dr
                end_col = col + (WIN_LENGTH - 1) * dc
                if 0 <= end_row < size and 0 <= end_col < size:
                    lines.append([(row + i * dr) * size + col + i * dc
                                  for i in range(WIN_LENGTH)])
    return np.asarray(lines, dtype=np.intp).reshape(-1, WIN_LENGTH)


class TacticalAgent(BaseAgent):
    """Ưu tiên thắng, chặn, tạo đe dọa rồi kiểm soát cụm quân gần nhất."""

    VERSION = "tactical-v1"
    RADIUS = 2

    def __init__(self, seed: int):
        super().__init__("Tác tử sinh dữ liệu chiến thuật")
        self.rng = np.random.default_rng(seed)

    def get_action(self, board: np.ndarray, player: int) -> Action:
        size = board.shape[0]
        flat = board.ravel()
        lines = _winning_lines(size)
        values = flat[lines]
        own = np.count_nonzero(values == player, axis=1)
        other = np.count_nonzero(values == -player, axis=1)
        empty = values == EMPTY

        # Một ô trống trên đoạn có sẵn bốn quân cho biết nước thắng chính xác.
        winning = (own == WIN_LENGTH - 1) & (other == 0)
        if np.any(winning):
            choices = np.unique(lines[winning[:, None] & empty])
        else:
            losing = (other == WIN_LENGTH - 1) & (own == 0)
            threats = np.unique(lines[losing[:, None] & empty])
            choices = threats if len(threats) else np.array(
                [row * size + col for row, col in candidate_actions(board, self.RADIUS)], dtype=np.intp
            )

            # Đếm các ô thắng khác nhau sẽ xuất hiện sau khi lấp một trong hai ô trống.
            almost = (own == WIN_LENGTH - 2) & (other == 0)
            next_wins: dict[int, set[int]] = {}
            for first, second in lines[almost][empty[almost]].reshape(-1, 2):
                next_wins.setdefault(int(first), set()).add(int(second))
                next_wins.setdefault(int(second), set()).add(int(first))
            forks = np.array([len(next_wins.get(int(cell), ())) for cell in choices])
            choices = choices[forks == forks.max()]

            # Các đoạn chưa bị chặn và có nhiều quân được ưu tiên hơn vị trí rời rạc.
            weights = np.where(other == 0, 4 ** own - 1, 0)
            weights += np.where(own == 0, 3 ** other - 1, 0)
            scores = np.bincount(lines.ravel(), np.repeat(weights, WIN_LENGTH), minlength=flat.size)
            choices = choices[scores[choices] == scores[choices].max()]

        return divmod(int(self.rng.choice(choices)), size)
