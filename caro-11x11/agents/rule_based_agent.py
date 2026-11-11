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

Chế độ ``time``: nước dự phòng hợp lệ được chuẩn bị trước khi quét, mốc dừng hợp tác được kiểm tra
trước mỗi ô; hết giờ thì trả quyết định tốt nhất đã có (xem ``RuleBasedAgent``).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real
from time import perf_counter

import numpy as np

from agents.base_agent import BaseAgent
from config.game_config import EMPTY, WIN_LENGTH
from config.search_budget import SearchBudget
from core.game_engine import Action
from core.game_utils import candidate_actions

# Ngang, dọc, chéo chính, chéo phụ — cùng thứ tự với CaroGame.DIRECTIONS.
DIRECTIONS: tuple[tuple[int, int], ...] = ((0, 1), (1, 0), (1, 1), (1, -1))
DEFAULT_RADIUS = 2
# Biên an toàn chế độ time: dừng ở 1,9 giây trên hạn 2,0 giây (theo docs).
DEFAULT_TIME_MARGIN_S = 0.1

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
    hoặc mọi ô chặn, hoặc mọi ứng viên cùng điểm cao nhất ở tầng dự phòng). Khi ``timed_out``
    là True, tập này chỉ gồm những ô đã kịp xét trước mốc dừng.
    """

    rule: str
    action: Action
    acceptable: tuple[Action, ...]
    reason: str
    timed_out: bool = False


def _center_key(action: Action, size: int) -> tuple[int, int, int, int]:
    """Khóa gần tâm (Chebyshev, Manhattan, hàng, cột); chỉ dùng phép tính số học, không đọc bàn cờ."""
    row, col = action
    center = size // 2
    dr, dc = abs(row - center), abs(col - center)
    return (max(dr, dc), dr + dc, row, col)


class RuleBasedAgent(BaseAgent):
    """Rule-based Giai đoạn 1: thắng ngay → chặn thua → dự phòng vị trí; hoàn toàn tất định.

    ``stats["nodes_visited"]`` đếm số ô **thực sự đã xét** trong lượt: ô trống ở tầng thắng,
    ô trống ở tầng chặn (nếu tới), ứng viên được chấm điểm ở tầng dự phòng (nếu tới).
    ``simulations`` và ``pruned_nodes`` luôn 0.

    Ngân sách:
      ``fixed``: không có giới hạn thời gian, luôn xét hết các tầng cần thiết.
      ``time``: mốc dừng hợp tác = deadline − min(time_margin_s, 0,5 × time_limit_s). Trước khi quét,
      tác tử chuẩn bị sẵn nước dự phòng hợp lệ (ứng viên gần tâm nhất); mốc dừng được kiểm tra trước
      mỗi ô ở cả ba tầng. Hết giờ thì trả quyết định tốt nhất đã có: nước thắng hoặc ô chặn đã tìm thấy,
      nếu không thì ứng viên điểm cao nhất đã chấm, cuối cùng là nước dự phòng chuẩn bị sẵn.
    """

    def __init__(self, name: str = "Rule-based", *, budget: SearchBudget | None = None,
                 radius: int = DEFAULT_RADIUS, win_length: int = WIN_LENGTH,
                 time_margin_s: float = DEFAULT_TIME_MARGIN_S):
        super().__init__(name, budget=budget)
        self.radius = _check_positive_int(radius, "Bán kính ứng viên")
        self.win_length = _check_positive_int(win_length, "Độ dài thắng")
        if (isinstance(time_margin_s, (bool, np.bool_)) or not isinstance(time_margin_s, Real)
                or not isfinite(time_margin_s) or time_margin_s < 0):
            raise ValueError("Biên an toàn thời gian phải là số hữu hạn không âm")
        self.time_margin_s = float(time_margin_s)
        self.last_decision: Decision | None = None

    def explain_last_action(self) -> Decision | None:
        """Quyết định của lượt gần nhất (None nếu chưa đi nước nào)."""
        return self.last_decision

    def _stop_at(self, deadline: float | None) -> float | None:
        """Mốc dừng hợp tác: None ở chế độ fixed; chế độ time bắt buộc có deadline."""
        if self.budget.mode != "time":
            return None
        if deadline is None:
            raise ValueError("Chế độ time cần deadline cho lượt hiện tại")
        return deadline - min(self.time_margin_s, 0.5 * self.budget.time_limit_s)

    def decide(self, board: np.ndarray, player: int) -> Decision:
        """Gọi trực tiếp (ví dụ trong notebook): áp dụng ba tầng luật, không sửa ``board``.

        Ở chế độ time, lần gọi này tự đặt deadline riêng = thời điểm gọi + time_limit_s,
        không dùng ``self.deadline`` (có thể là None hoặc mốc cũ của lượt trước).
        ``stats["nodes_visited"]`` được ghi đè theo lần gọi này, không cộng dồn.
        """
        deadline = perf_counter() + self.budget.time_limit_s if self.budget.mode == "time" else None
        return self._decide(board, player, deadline)

    def _decide(self, board: np.ndarray, player: int, deadline: float | None) -> Decision:
        """Ba tầng luật với deadline đã xác định; cập nhật ``stats["nodes_visited"]``."""
        if (not isinstance(board, np.ndarray) or board.ndim != 2 or board.shape[0] == 0
                or board.shape[0] != board.shape[1] or not np.isin(board, (-1, 0, 1)).all()):
            raise ValueError("Bàn cờ phải là mảng vuông không rỗng, chỉ chứa -1, 0, 1")
        if isinstance(player, (bool, np.bool_)) or not isinstance(player, Integral) or player not in (-1, 1):
            raise ValueError("Người chơi phải có giá trị 1 hoặc -1")
        player = int(player)
        if self.win_length > board.shape[0]:
            raise ValueError("Độ dài thắng không được lớn hơn kích thước bàn cờ")
        empty = [(int(r), int(c)) for r, c in np.argwhere(board == EMPTY)]
        if not empty:
            raise ValueError("Không còn nước đi hợp lệ")
        if has_winner(board, self.win_length):
            raise ValueError("Ván đấu đã kết thúc, không chọn nước đi")
        stop_at = self._stop_at(deadline)

        def out_of_time() -> bool:
            return stop_at is not None and perf_counter() >= stop_at

        # Nước dự phòng chuẩn bị trước khi quét: ứng viên gần tâm nhất, luôn hợp lệ.
        candidates = candidate_actions(board, self.radius) or empty
        fallback = min(candidates, key=lambda a: _center_key(a, board.shape[0]))
        examined = 0
        self.stats["nodes_visited"] = 0

        def scan(target: int) -> tuple[list[Action], bool]:
            # Quét toàn bộ ô trống tìm ô giúp `target` thắng ngay; dừng khi tới mốc.
            nonlocal examined
            found = []
            for action in empty:
                if out_of_time():
                    return found, True
                examined += 1
                if is_winning_move(board, action, target, self.win_length):
                    found.append(action)
            return found, False

        def finish(rule: str, actions: list[Action], reason: str, timed_out: bool) -> Decision:
            self.stats["nodes_visited"] = examined
            if timed_out:
                reason = f"hết thời gian sau {examined} ô đã xét; {reason}"
            return self._pick(board, player, rule, actions, reason, timed_out)

        wins, stopped = scan(player)
        if wins:
            return finish(RULE_WIN, wins, f"có {len(wins)} nước thắng ngay", stopped)
        if stopped:
            return finish(RULE_FALLBACK, [fallback], "dừng khi đang quét nước thắng, dùng nước dự phòng", True)

        threats, stopped = scan(-player)
        if threats:
            note = " (đe dọa kép: chỉ chặn được một ô)" if len(threats) > 1 else ""
            return finish(RULE_BLOCK, threats, f"đối thủ có {len(threats)} ô thắng ngay{note}", stopped)
        if stopped:
            return finish(RULE_FALLBACK, [fallback], "dừng khi đang quét đe dọa, dùng nước dự phòng", True)

        scores: dict[Action, int] = {}
        stopped = False
        for action in candidates:
            if out_of_time():
                stopped = True
                break
            examined += 1
            scores[action] = position_score(board, action, player)
        if not scores:
            return finish(RULE_FALLBACK, [fallback], "dừng trước khi chấm điểm, dùng nước dự phòng", True)
        best = max(scores.values())
        top = [a for a, score in scores.items() if score == best]
        return finish(RULE_FALLBACK, top,
                      f"chưa có đe dọa; chấm {len(scores)}/{len(candidates)} ứng viên bán kính {self.radius}, "
                      f"điểm vị trí cao nhất {best}", stopped)

    @staticmethod
    def _pick(board: np.ndarray, player: int, rule: str, actions: list[Action], reason: str,
              timed_out: bool) -> Decision:
        action = min(actions, key=lambda a: rank_key(board, a, player))
        return Decision(rule, action, tuple(actions), f"{reason}; chọn {action}", timed_out)

    def get_action(self, board: np.ndarray, player: int) -> Action:
        # choose_action của BaseAgent đã đặt self.deadline cho lượt này (None ở chế độ fixed).
        self.last_decision = self._decide(board, player, self.deadline)
        return self.last_decision.action
