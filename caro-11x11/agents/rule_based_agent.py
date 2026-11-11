"""Tác tử Rule-based cho Caro 11×11 — Giai đoạn 1: luật thắng ngay và chặn thua.

Thứ tự luật (dừng ở tầng đầu tiên có nước):
1. ``thang_ngay`` — bên đang đi có nước tạo chuỗi từ ``win_length`` quân trở lên.
2. ``chan_thua`` — đối thủ có nước thắng ngay ở lượt sau; đi vào chính ô đó.
3. ``du_phong`` — chưa có đe dọa trực tiếp; chọn ô ứng viên theo điểm vị trí đơn giản.

Tầng 1 và 2 quét **toàn bộ ô trống** để không bỏ sót đe dọa nằm ngoài bán kính ứng viên.
Tầng 3 chỉ là baseline giữ cho tác tử luôn trả nước hợp lệ; nhận diện pattern (ba mở,
bốn mở, chuỗi có khoảng trống...) thuộc Giai đoạn 2.

Mọi hàm chỉ đọc ``board``: nước thử được kiểm tra bằng cách đếm quân hai phía của ô trống,
không đặt quân tạm lên mảng.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from agents.base_agent import BaseAgent
from config.game_config import EMPTY, WIN_LENGTH
from config.search_budget import SearchBudget
from core.game_engine import Action
from core.game_utils import candidate_actions

# Ngang, dọc, chéo chính, chéo phụ — cùng thứ tự với CaroGame.DIRECTIONS.
DIRECTIONS: tuple[tuple[int, int], ...] = ((0, 1), (1, 0), (1, 1), (1, -1))
DEFAULT_RADIUS = 2

RULE_WIN = "thang_ngay"
RULE_BLOCK = "chan_thua"
RULE_FALLBACK = "du_phong"
RULES: tuple[str, ...] = (RULE_WIN, RULE_BLOCK, RULE_FALLBACK)


def _check_positive_int(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} phải là số nguyên dương")
    return int(value)


def line_length(board: np.ndarray, row: int, col: int, player: int, direction: tuple[int, int]) -> int:
    """Độ dài chuỗi của ``player`` đi qua (row, col) theo ``direction`` nếu ô này là quân ``player``.

    Ô (row, col) luôn được tính là một quân, kể cả khi đang trống (nước thử);
    hai phía chỉ đếm quân ``player`` liền kề, dừng ở ô trống, quân đối thủ hoặc mép bàn.
    """
    size = board.shape[0]
    dr, dc = direction
    length = 1
    for sign in (-1, 1):
        r, c = row + sign * dr, col + sign * dc
        while 0 <= r < size and 0 <= c < size and board[r, c] == player:
            length += 1
            r, c = r + sign * dr, c + sign * dc
    return length


def is_winning_move(board: np.ndarray, action: Action, player: int, win_length: int = WIN_LENGTH) -> bool:
    """Đúng khi ô ``action`` trống và ``player`` đặt vào đó tạo chuỗi ≥ ``win_length`` ở một trong 4 hướng."""
    row, col = action
    if board[row, col] != EMPTY:
        return False
    return any(line_length(board, row, col, player, d) >= win_length for d in DIRECTIONS)


def winning_moves(board: np.ndarray, player: int, win_length: int = WIN_LENGTH) -> list[Action]:
    """Mọi ô trống giúp ``player`` thắng ngay, quét toàn bàn, sắp theo (hàng, cột)."""
    return [(int(r), int(c)) for r, c in np.argwhere(board == EMPTY)
            if is_winning_move(board, (int(r), int(c)), player, win_length)]


def has_winner(board: np.ndarray, win_length: int = WIN_LENGTH) -> bool:
    """Đúng khi trên bàn đã có chuỗi ≥ ``win_length`` quân của một bên (ván đã kết thúc)."""
    for r, c in np.argwhere(board != EMPTY):
        player = int(board[r, c])
        if any(line_length(board, int(r), int(c), player, d) >= win_length for d in DIRECTIONS):
            return True
    return False


def position_score(board: np.ndarray, action: Action, player: int) -> int:
    """Điểm vị trí của tầng dự phòng: 2 × quân mình + 1 × quân đối thủ trong 8 ô kề.

    Ưu tiên nối quân mình hơn áp sát quân đối thủ. Đây là điểm baseline, không nhận diện pattern.
    """
    size = board.shape[0]
    row, col = action
    block = board[max(0, row - 1):min(size, row + 2), max(0, col - 1):min(size, col + 2)]
    return 2 * int(np.count_nonzero(block == player)) + int(np.count_nonzero(block == -player))


def rank_key(board: np.ndarray, action: Action, player: int) -> tuple[int, int, int, int, int]:
    """Khóa xếp hạng tất định, nhỏ hơn là tốt hơn; dùng chung để phá hòa ở cả ba tầng.

    Thứ tự: điểm vị trí cao hơn → gần tâm hơn (Chebyshev, rồi Manhattan) → hàng nhỏ hơn → cột nhỏ hơn.
    """
    row, col = action
    center = board.shape[0] // 2
    dr, dc = abs(row - center), abs(col - center)
    return (-position_score(board, action, player), max(dr, dc), dr + dc, row, col)


@dataclass(frozen=True)
class Decision:
    """Truy vết một quyết định; bất biến để bên ngoài không sửa được dữ liệu nội bộ.

    ``acceptable`` là tập nước chấp nhận được của tầng đã kích hoạt (mọi nước thắng ngay,
    hoặc mọi ô chặn, hoặc mọi ứng viên cùng điểm cao nhất ở tầng dự phòng).
    """

    rule: str
    action: Action
    acceptable: tuple[Action, ...]
    reason: str


class RuleBasedAgent(BaseAgent):
    """Rule-based Giai đoạn 1: thắng ngay → chặn thua → dự phòng vị trí; hoàn toàn tất định.

    ``stats["nodes_visited"]`` đếm số ô được đánh giá trong lượt: số ô trống ở tầng thắng,
    cộng số ô trống ở tầng chặn (nếu tới), cộng số ứng viên ở tầng dự phòng (nếu tới).
    ``simulations`` và ``pruned_nodes`` luôn 0. Tác tử trả về ngay, không dùng hết ngân sách thời gian.
    """

    def __init__(self, name: str = "Rule-based", *, budget: SearchBudget | None = None,
                 radius: int = DEFAULT_RADIUS, win_length: int = WIN_LENGTH):
        super().__init__(name, budget=budget)
        self.radius = _check_positive_int(radius, "Bán kính ứng viên")
        self.win_length = _check_positive_int(win_length, "Độ dài thắng")
        self.last_decision: Decision | None = None

    def explain_last_action(self) -> Decision | None:
        """Quyết định của lượt gần nhất (None nếu chưa đi nước nào)."""
        return self.last_decision

    def decide(self, board: np.ndarray, player: int) -> Decision:
        """Áp dụng ba tầng luật và trả về quyết định kèm lý do; không sửa ``board``.

        Gọi trực tiếp được (ví dụ trong notebook); ``stats["nodes_visited"]`` được ghi đè
        bằng số ô đã đánh giá của lần gọi này, không cộng dồn giữa các lần gọi.
        """
        if (not isinstance(board, np.ndarray) or board.ndim != 2 or board.shape[0] == 0
                or board.shape[0] != board.shape[1] or not np.isin(board, (-1, 0, 1)).all()):
            raise ValueError("Bàn cờ phải là mảng vuông không rỗng, chỉ chứa -1, 0, 1")
        if isinstance(player, (bool, np.bool_)) or not isinstance(player, Integral) or player not in (-1, 1):
            raise ValueError("Người chơi phải có giá trị 1 hoặc -1")
        player = int(player)
        if self.win_length > board.shape[0]:
            raise ValueError("Độ dài thắng không được lớn hơn kích thước bàn cờ")
        n_empty = int(np.count_nonzero(board == EMPTY))
        if n_empty == 0:
            raise ValueError("Không còn nước đi hợp lệ")
        if has_winner(board, self.win_length):
            raise ValueError("Ván đấu đã kết thúc, không chọn nước đi")

        self.stats["nodes_visited"] = n_empty
        wins = winning_moves(board, player, self.win_length)
        if wins:
            return self._pick(board, player, RULE_WIN, wins,
                              f"có {len(wins)} nước thắng ngay")

        self.stats["nodes_visited"] += n_empty
        threats = winning_moves(board, -player, self.win_length)
        if threats:
            note = " (đe dọa kép: chỉ chặn được một ô)" if len(threats) > 1 else ""
            return self._pick(board, player, RULE_BLOCK, threats,
                              f"đối thủ có {len(threats)} ô thắng ngay{note}")

        candidates = candidate_actions(board, self.radius)
        self.stats["nodes_visited"] += len(candidates)
        best = max(position_score(board, a, player) for a in candidates)
        top = [a for a in candidates if position_score(board, a, player) == best]
        return self._pick(board, player, RULE_FALLBACK, top,
                          f"chưa có đe dọa; {len(candidates)} ứng viên bán kính {self.radius}, "
                          f"điểm vị trí cao nhất {best}")

    @staticmethod
    def _pick(board: np.ndarray, player: int, rule: str, actions: list[Action], reason: str) -> Decision:
        action = min(actions, key=lambda a: rank_key(board, a, player))
        return Decision(rule, action, tuple(actions), f"{reason}; chọn {action}")

    def get_action(self, board: np.ndarray, player: int) -> Action:
        self.last_decision = self.decide(board, player)
        return self.last_decision.action
