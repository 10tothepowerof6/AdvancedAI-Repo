"""Minimax và Alpha-Beta cho cờ Caro 11×11.

Giai đoạn 1 — Trạng thái tìm kiếm và hàm lượng giá.
Giai đoạn 2 — Minimax giới hạn độ sâu, cắt tỉa Alpha-Beta, Iterative Deepening.
Giai đoạn 3 — Đối chứng cắt tỉa và khảo sát độ sâu.

Quy ước góc nhìn: mọi hàm lượng giá nhận tham số `perspective` (1 hoặc -1) và
trả điểm theo góc nhìn bên đó: dương là có lợi, âm là bất lợi, 0 là cân bằng.
Đổi góc nhìn chỉ đổi dấu: evaluate(s, X) == -evaluate(s, O).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import isfinite
from numbers import Integral
from time import perf_counter
from typing import Callable

import numpy as np

from agents.base_agent import BaseAgent
from config.game_config import BOARD_SIZE, EMPTY, WIN_LENGTH
from config.search_budget import SearchBudget
from core.game_engine import Action
from core.game_utils import candidate_actions

# Bốn hướng giống engine: ngang, dọc, chéo chính, chéo phụ.
DIRECTIONS: tuple[tuple[int, int], ...] = ((0, 1), (1, 0), (1, 1), (1, -1))

# Trọng số theo số quân trong một cửa sổ 5 ô chỉ có quân của một bên.
# Chỉ số = số quân (0..5). Mỗi quân thêm vào cửa sổ tăng điểm gấp 10 lần để
# một cửa sổ "gần thắng" hơn luôn nặng hơn vài cửa sổ yếu hơn cộng lại.
# Cửa sổ đủ 5 quân là trạng thái kết thúc, do terminal_score xử lý nên trọng số 0.
# Bằng default_weights(5, 11); luật thắng khác dùng default_weights(win_length, cạnh bàn).
WINDOW_WEIGHTS: tuple[int, ...] = (0, 1, 10, 100, 1_000, 0)

# Điểm thắng/thua tuyệt đối. Phải lớn hơn hẳn mọi điểm heuristic có thể đạt
# (xem heuristic_bound) để heuristic không bao giờ lấn át kết quả thật.
WIN_SCORE = 10_000_000


# Cửa sổ 5 ô — nền tảng cho cả phát hiện thắng và hàm lượng giá

@lru_cache(maxsize=None)
def _window_indices(size: int, win_length: int) -> np.ndarray:
    """Chỉ số phẳng của mọi đoạn win_length ô liên tiếp theo 4 hướng.

    Bàn 11×11, win_length 5: 77 ngang + 77 dọc + 49 chéo chính + 49 chéo phụ = 252.
    Kết quả được cache và chỉ đọc, dùng chung cho mọi lần lượng giá.
    """
    windows = []
    for dr, dc in DIRECTIONS:
        for r in range(size):
            for c in range(size):
                end_r, end_c = r + (win_length - 1) * dr, c + (win_length - 1) * dc
                if 0 <= end_r < size and 0 <= end_c < size:
                    windows.append([(r + k * dr) * size + (c + k * dc) for k in range(win_length)])
    result = np.array(windows, dtype=np.intp).reshape(-1, win_length)
    result.setflags(write=False)
    return result


def _validate_board(board: np.ndarray) -> None:
    if (not isinstance(board, np.ndarray) or board.ndim != 2
            or board.shape[0] == 0 or board.shape[0] != board.shape[1]
            or not np.isin(board, (-1, 0, 1)).all()):
        raise ValueError("Bàn cờ phải là mảng vuông không rỗng, chỉ chứa -1, 0, 1")


def _validate_player(player: int, name: str = "Người chơi") -> int:
    if isinstance(player, (bool, np.bool_)) or not isinstance(player, Integral) or player not in (-1, 1):
        raise ValueError(f"{name} phải có giá trị 1 hoặc -1")
    return int(player)


def _validate_win_length(win_length: int, board_size: int) -> int:
    if (isinstance(win_length, (bool, np.bool_)) or not isinstance(win_length, Integral)
            or not 1 <= win_length <= board_size):
        raise ValueError("win_length phải là số nguyên dương không lớn hơn cạnh bàn cờ")
    return int(win_length)


def window_counts(board: np.ndarray, win_length: int = WIN_LENGTH) -> dict[int, np.ndarray]:
    """Đếm số cửa sổ "còn sống" của mỗi bên theo số quân trong cửa sổ.

    Cửa sổ còn sống của X là cửa sổ không có quân O (X còn khả năng lấp đủ
    để thắng), và ngược lại. Trả về {1: mảng đếm của X, -1: mảng đếm của O},
    phần tử thứ k là số cửa sổ có đúng k quân. Cửa sổ trống được tính cho cả hai.
    """
    _validate_board(board)
    cells = board.ravel()[_window_indices(board.shape[0], win_length)]
    x_count = np.count_nonzero(cells == 1, axis=1)
    o_count = np.count_nonzero(cells == -1, axis=1)
    return {1: np.bincount(x_count[o_count == 0], minlength=win_length + 1),
            -1: np.bincount(o_count[x_count == 0], minlength=win_length + 1)}


def find_winner(board: np.ndarray, win_length: int = WIN_LENGTH) -> int | None:
    """Tìm bên có chuỗi >= win_length trên toàn bàn; None nếu chưa ai thắng.

    Chuỗi dài hơn 5 chứa ít nhất một cửa sổ 5 ô đồng màu nên vẫn được tính thắng,
    khớp luật freestyle của engine. Bàn có cả hai bên thắng là không hợp lệ.
    """
    counts = window_counts(board, win_length)
    x_wins, o_wins = counts[1][win_length] > 0, counts[-1][win_length] > 0
    if x_wins and o_wins:
        raise ValueError("Bàn cờ không hợp lệ: cả hai bên đều có chuỗi thắng")
    return 1 if x_wins else -1 if o_wins else None


# Trạng thái tìm kiếm

class SearchState:
    """Trạng thái tìm kiếm bất biến từ góc nhìn bên ngoài.

    apply() luôn trả về đối tượng mới với bản sao bàn cờ riêng, nên duyệt các
    nước con không bao giờ làm thay đổi trạng thái cha hay mảng board đầu vào.
    Mảng board ở chế độ chỉ đọc: ghi trực tiếp (state.board[r, c] = ...) bị
    numpy từ chối, vì như vậy sẽ làm result và move_count lệch khỏi bàn cờ.
    Quy ước theo engine: khi ván kết thúc, current_player giữ người vừa đi
    nước cuối; result theo góc nhìn X (None, 1, -1, 0).
    """

    __slots__ = ("board", "current_player", "last_action", "move_count", "win_length", "_result")

    def __init__(self, board: np.ndarray, current_player: int, *,
                 win_length: int = WIN_LENGTH):
        _validate_board(board)
        self.current_player = _validate_player(current_player)
        self.win_length = _validate_win_length(win_length, board.shape[0])
        self.board = board.astype(np.int8, copy=True)
        self.board.setflags(write=False)
        self.last_action: Action | None = None
        self.move_count = int(np.count_nonzero(self.board))
        winner = find_winner(self.board, self.win_length)
        if winner is not None:
            self._result: int | None = winner
        elif self.move_count == self.board.size:
            self._result = 0
        else:
            self._result = None

    @classmethod
    def initial(cls, board_size: int = BOARD_SIZE, win_length: int = WIN_LENGTH) -> "SearchState":
        """Bàn trống, X đi trước — giống CaroGame() mới khởi tạo."""
        return cls(np.zeros((board_size, board_size), dtype=np.int8), 1, win_length=win_length)

    @property
    def size(self) -> int:
        return self.board.shape[0]

    def is_terminal(self) -> bool:
        return self._result is not None

    def result(self) -> int | None:
        return self._result

    def candidates(self, radius: int = 2) -> list[Action]:
        """Tập nước đi rút gọn quanh các quân hiện có; rỗng khi ván đã kết thúc."""
        if self.is_terminal():
            return []
        return candidate_actions(self.board, radius)

    def apply(self, action: Action) -> "SearchState":
        """Trả về trạng thái con sau khi current_player đi action; cha giữ nguyên."""
        if self.is_terminal():
            raise ValueError("Không thể đi tiếp vì ván đấu đã kết thúc")
        if (not isinstance(action, tuple) or len(action) != 2
                or any(isinstance(x, (bool, np.bool_)) or not isinstance(x, Integral) for x in action)):
            raise ValueError("Nước đi phải là bộ hai số nguyên (hàng, cột)")
        row, col = map(int, action)
        if not (0 <= row < self.size and 0 <= col < self.size):
            raise ValueError("Nước đi nằm ngoài bàn cờ")
        if self.board[row, col] != EMPTY:
            raise ValueError("Ô cờ đã có quân")

        board = self.board.copy()  # bản sao của mảng chỉ đọc vẫn ghi được
        board[row, col] = self.current_player
        board.setflags(write=False)
        child = object.__new__(SearchState)
        child.board = board
        child.win_length = self.win_length
        child.last_action = (row, col)
        child.move_count = self.move_count + 1
        child.current_player = self.current_player
        if _is_winning_line(child.board, row, col, self.win_length):
            child._result = self.current_player
        elif child.move_count == child.board.size:
            child._result = 0
        else:
            child._result = None
            child.current_player = -self.current_player
        return child


def _is_winning_line(board: np.ndarray, row: int, col: int, win_length: int) -> bool:
    """Chỉ xét 4 đường đi qua nước vừa đi, giống CaroGame.check_winner()."""
    player = board[row, col]
    size = board.shape[0]
    for dr, dc in DIRECTIONS:
        count = 1
        for sign in (-1, 1):
            r, c = row + sign * dr, col + sign * dc
            while 0 <= r < size and 0 <= c < size and board[r, c] == player:
                count += 1
                r, c = r + sign * dr, c + sign * dc
        if count >= win_length:
            return True
    return False


# Hàm lượng giá
def terminal_score(result: int, perspective: int) -> int:
    """Điểm trạng thái kết thúc: thắng +WIN_SCORE, thua -WIN_SCORE, hòa 0."""
    if isinstance(result, (bool, np.bool_)) or result not in (-1, 0, 1):
        raise ValueError("Kết quả kết thúc phải là -1, 0 hoặc 1")
    return WIN_SCORE * int(result) * _validate_player(perspective, "Góc nhìn")


def _validate_board_size(board_size: int) -> int:
    if isinstance(board_size, (bool, np.bool_)) or not isinstance(board_size, Integral) or board_size < 1:
        raise ValueError("Cạnh bàn cờ phải là số nguyên dương")
    return int(board_size)


def default_weights(win_length: int = WIN_LENGTH, board_size: int = BOARD_SIZE) -> tuple[int, ...]:
    """Bộ trọng số mặc định cho luật thắng win_length quân trên bàn board_size.

    Cửa sổ có k quân (1 <= k < win_length) được base ** (k - 1); cửa sổ trống
    và cửa sổ đủ quân (đã là trạng thái kết thúc) được 0. base = 10 như
    WINDOW_WEIGHTS; chỉ khi chuỗi thắng quá dài khiến cận heuristic chạm
    WIN_SCORE thì mới giảm base (tối thiểu 2, vẫn tăng ngặt theo số quân).
    Luật 5 quân trên bàn 11×11 cho đúng WINDOW_WEIGHTS.
    """
    board_size = _validate_board_size(board_size)
    win_length = _validate_win_length(win_length, board_size)
    n_windows = len(_window_indices(board_size, win_length))
    for base in range(10, 1, -1):
        weights = (0, *(base ** k for k in range(win_length - 1)), 0)
        if n_windows * max(weights) < WIN_SCORE:
            return weights
    raise ValueError("Không có bộ trọng số mặc định giữ cận heuristic nhỏ hơn WIN_SCORE")


def _validate_weights(weights: tuple[int, ...], win_length: int, board_size: int) -> np.ndarray:
    """Kiểm tra trọng số và cận heuristic tương ứng phải nhỏ hơn WIN_SCORE.

    Nếu cận chạm WIN_SCORE, một thế đang chơi có thể được điểm cao hơn một ván
    đã thắng và Minimax sẽ bỏ qua nước thắng, nên cấu hình đó bị từ chối.
    """
    values = tuple(weights)
    if (len(values) != win_length + 1
            or any(isinstance(w, (bool, np.bool_)) or not isinstance(w, Integral) or w < 0 for w in values)):
        raise ValueError(f"Trọng số phải gồm {win_length + 1} số nguyên không âm")
    bound = len(_window_indices(board_size, win_length)) * int(max(values))
    if bound >= WIN_SCORE:
        raise ValueError(f"Cận heuristic {bound} phải nhỏ hơn WIN_SCORE = {WIN_SCORE} "
                         "để ván đã thắng luôn được điểm cao hơn mọi thế đang chơi")
    return np.array(values, dtype=np.int64)


def heuristic_bound(board_size: int = BOARD_SIZE, win_length: int = WIN_LENGTH,
                    weights: tuple[int, ...] | None = None) -> int:
    """Cận trên |heuristic_score|: mọi cửa sổ đều mang trọng số lớn nhất.

    weights=None dùng default_weights(win_length, board_size). Luôn nhỏ hơn
    WIN_SCORE vì cấu hình vượt cận bị _validate_weights từ chối.
    """
    board_size = _validate_board_size(board_size)
    win_length = _validate_win_length(win_length, board_size)
    if weights is None:
        weights = default_weights(win_length, board_size)
    weight_array = _validate_weights(weights, win_length, board_size)
    return len(_window_indices(board_size, win_length)) * int(weight_array.max())


def heuristic_score(board: np.ndarray, perspective: int, *,
                    win_length: int = WIN_LENGTH,
                    weights: tuple[int, ...] | None = None) -> int:
    """Điểm heuristic cho trạng thái chưa kết thúc, theo góc nhìn perspective.

    Tổng trọng số các cửa sổ còn sống của perspective trừ của đối thủ.
    Hàm đối xứng: không cộng thêm lợi thế cho bên đang đến lượt.
    weights=None dùng default_weights theo win_length và cạnh bàn cờ.
    """
    perspective = _validate_player(perspective, "Góc nhìn")
    _validate_board(board)
    size = board.shape[0]
    win_length = _validate_win_length(win_length, size)
    if weights is None:
        weights = default_weights(win_length, size)
    weight_array = _validate_weights(weights, win_length, size)
    return perspective * _raw_heuristic(board, weight_array, win_length)


def _raw_heuristic(board: np.ndarray, weight_array: np.ndarray, win_length: int) -> int:
    """Lõi heuristic theo góc nhìn X, không kiểm tra đầu vào (dùng trong vòng tìm kiếm).

    Cho cùng kết quả với tổng trọng số trên window_counts: cửa sổ không có quân O
    cộng W[số quân X], cửa sổ không có quân X trừ W[số quân O].
    """
    cells = board.ravel()[_window_indices(board.shape[0], win_length)]
    x_count = np.count_nonzero(cells == 1, axis=1)
    o_count = np.count_nonzero(cells == -1, axis=1)
    return int(weight_array[x_count[o_count == 0]].sum() - weight_array[o_count[x_count == 0]].sum())


def evaluate(state: SearchState, perspective: int, *,
             weights: tuple[int, ...] | None = None) -> int:
    """Lượng giá trạng thái: terminal dùng terminal_score, còn lại dùng heuristic.

    weights=None chọn bộ trọng số mặc định theo win_length và kích thước bàn
    của chính state, nên mọi trạng thái tạo được bằng SearchState đều lượng giá được.
    """
    if state.is_terminal():
        return terminal_score(state.result(), perspective)
    return heuristic_score(state.board, perspective, win_length=state.win_length, weights=weights)


# Thuật toán Minimax và cắt tỉa Alpha-Beta

# Giá trị vô cực dạng số nguyên, lớn hơn mọi điểm terminal.
_INF = WIN_SCORE * 10
DEFAULT_TIME_MARGIN_S = 0.1


class SearchTimeout(Exception):
    """Hết thời gian giữa chừng; kết quả của độ sâu đang dở bị bỏ."""


@dataclass(frozen=True)
class SearchResult:
    """Kết quả một lần tìm kiếm với độ sâu cố định.

    value theo góc nhìn bên đi ở nút gốc. nodes_visited đếm số nút đã duyệt
    (tính cả nút gốc), pruned_nodes đếm số lần cắt tỉa thực sự bỏ qua được
    ít nhất một nước con (không phải số nút con bị bỏ qua).
    """
    value: int
    action: Action | None
    depth: int
    nodes_visited: int
    pruned_nodes: int


def _validate_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} phải là số nguyên dương")
    return int(value)


class MinimaxSearcher:
    """Minimax giới hạn độ sâu; một ply là một nước đi của một bên.

    - Nút gốc là Max (bên đến lượt ở gốc); nút có người đến lượt khác gốc là Min.
    - Điểm luôn tính theo góc nhìn bên ở gốc. Nút terminal không mở rộng thêm;
      thắng được trừ đi số ply để ưu tiên thắng sớm, thua được cộng số ply để
      ưu tiên thua muộn. Nút lá không terminal dùng heuristic_score.
    - use_alpha_beta chỉ bật/tắt cắt tỉa; trạng thái, lượng giá, tập ứng viên
      và thứ tự xét nước giữ nguyên để hai chế độ so sánh được với nhau.
    - Phá hòa: khi nhiều nước đồng điểm, giữ nước gặp đầu tiên theo thứ tự xét.
    - move_ordering sắp xếp nước con theo heuristic của bên đi (tốt trước),
      chỉ áp dụng ở nút còn ít nhất 2 ply; sort ổn định nên vẫn tất định.
    - weights=None dùng default_weights theo luật thắng và cỡ bàn của trạng thái.
      Trọng số được kiểm tra theo luật của trạng thái khi tìm kiếm: cận heuristic
      phải nhỏ hơn WIN_SCORE - số ô của bàn, vì điểm thắng bị trừ tối đa ngần ấy ply.
    """

    def __init__(self, *, use_alpha_beta: bool = True, move_ordering: bool = True,
                 radius: int = 2, weights: tuple[int, ...] | None = None,
                 should_stop: Callable[[], bool] | None = None):
        if not isinstance(use_alpha_beta, bool) or not isinstance(move_ordering, bool):
            raise ValueError("use_alpha_beta và move_ordering phải là bool")
        self.use_alpha_beta = use_alpha_beta
        self.move_ordering = move_ordering
        self.radius = _validate_positive_int(radius, "Bán kính ứng viên")
        self.weights = None if weights is None else tuple(weights)
        self._weight_cache: dict[tuple[int, int], np.ndarray] = {}
        self.should_stop = should_stop
        self.nodes_visited = 0
        self.pruned_nodes = 0

    def weights_for(self, state: SearchState) -> np.ndarray:
        """Mảng trọng số đã kiểm tra cho luật thắng và cỡ bàn của state (có cache)."""
        key = (state.size, state.win_length)
        if key not in self._weight_cache:
            weights = self.weights if self.weights is not None else default_weights(state.win_length, state.size)
            array = _validate_weights(weights, state.win_length, state.size)
            bound = len(_window_indices(*key)) * int(array.max())
            if bound >= WIN_SCORE - state.board.size:
                raise ValueError(f"Cận heuristic {bound} phải nhỏ hơn WIN_SCORE - {state.board.size} "
                                 "để thắng ở ply sâu nhất vẫn cao điểm hơn mọi thế đang chơi")
            self._weight_cache[key] = array
        return self._weight_cache[key]

    def _evaluate(self, state: SearchState, perspective: int) -> int:
        """Như evaluate() nhưng bỏ kiểm tra đầu vào; trạng thái do chính SearchState sinh ra."""
        if state.is_terminal():
            return terminal_score(state.result(), perspective)
        return perspective * _raw_heuristic(state.board, self.weights_for(state), state.win_length)

    def ordered_moves(self, state: SearchState, depth_left: int,
                      first_action: Action | None = None
                      ) -> list[tuple[Action, SearchState | None]]:
        """Danh sách (nước đi, trạng thái con hoặc None) theo đúng thứ tự sẽ được xét.

        Không sắp xếp: thứ tự hàng/cột của candidate_actions, trạng thái con được
        tạo khi cần (None) để nhánh bị cắt tỉa không tốn công sinh. Có sắp xếp:
        phải sinh mọi con để lượng giá, nên trả kèm trạng thái con để dùng lại.
        """
        actions = state.candidates(self.radius)
        if self.move_ordering and depth_left >= 2:
            mover = state.current_player
            moves = [(action, state.apply(action)) for action in actions]
            moves.sort(key=lambda item: -self._evaluate(item[1], mover))
        else:
            moves = [(action, None) for action in actions]
        if first_action is not None:
            for index, (action, _) in enumerate(moves):
                if action == first_action:
                    moves.insert(0, moves.pop(index))
                    break
        return moves

    def search(self, state: SearchState, depth: int, *,
               first_action: Action | None = None) -> SearchResult:
        """Tìm nước tốt nhất cho state.current_player với độ sâu depth ply.

        first_action (nếu có) được xét đầu tiên ở nút gốc; Iterative Deepening
        dùng nó để đưa nước tốt nhất của độ sâu trước lên đầu.
        """
        depth = _validate_positive_int(depth, "Độ sâu")
        if state.is_terminal():
            raise ValueError("Không tìm kiếm từ trạng thái đã kết thúc")
        self.weights_for(state)  # báo lỗi trọng số trước khi bắt đầu duyệt
        self.nodes_visited, self.pruned_nodes = 1, 0
        root = state.current_player
        best_value, best_action = -_INF, None
        alpha = -_INF
        for action, child in self.ordered_moves(state, depth, first_action):
            child = child if child is not None else state.apply(action)
            value = self._minimax(child, depth - 1, 1, alpha, _INF, root)
            if value > best_value:
                best_value, best_action = value, action
            if self.use_alpha_beta:
                alpha = max(alpha, best_value)
        return SearchResult(best_value, best_action, depth, self.nodes_visited, self.pruned_nodes)

    def _minimax(self, state: SearchState, depth: int, ply: int,
                 alpha: int, beta: int, root: int) -> int:
        self.nodes_visited += 1
        if self.should_stop is not None and self.should_stop():
            raise SearchTimeout
        if state.is_terminal():
            score = terminal_score(state.result(), root)
            return score - ply if score > 0 else score + ply if score < 0 else 0
        if depth == 0:
            return self._evaluate(state, root)

        children = self.ordered_moves(state, depth)
        maximizing = state.current_player == root
        best = -_INF if maximizing else _INF
        for index, (action, child) in enumerate(children):
            child = child if child is not None else state.apply(action)
            value = self._minimax(child, depth - 1, ply + 1, alpha, beta, root)
            if maximizing:
                best = max(best, value)
                if self.use_alpha_beta:
                    alpha = max(alpha, best)
            else:
                best = min(best, value)
                if self.use_alpha_beta:
                    beta = min(beta, best)
            if self.use_alpha_beta and alpha >= beta:
                if index < len(children) - 1:
                    self.pruned_nodes += 1
                break
        return best


# Tác tử

class MinimaxAgent(BaseAgent):
    """Tác tử Minimax/Alpha-Beta theo hợp đồng BaseAgent.

    - Chế độ fixed: tìm đúng budget.depth ply.
    - Chế độ time: Iterative Deepening từ độ sâu 1, dừng chủ động trước deadline
      time_margin_s giây và trả nước của độ sâu đã hoàn tất gần nhất. Nước dự
      phòng (ứng viên đầu tiên) luôn sẵn sàng nếu chưa xong độ sâu 1.
    - stats ngoài các khóa chung còn có depth_reached và root_value của lượt đi.
    """

    def __init__(self, *, budget: SearchBudget | None = None, use_alpha_beta: bool = True,
                 move_ordering: bool = True, radius: int = 2,
                 weights: tuple[int, ...] | None = None,
                 time_margin_s: float = DEFAULT_TIME_MARGIN_S, name: str | None = None):
        if name is None:
            name = "Minimax Alpha-Beta" if use_alpha_beta else "Minimax"
        super().__init__(name, budget=budget)
        if (isinstance(time_margin_s, bool) or not isinstance(time_margin_s, (int, float))
                or not isfinite(time_margin_s) or time_margin_s < 0):
            raise ValueError("Biên an toàn thời gian phải là số hữu hạn không âm")
        self.time_margin_s = float(time_margin_s)
        self.searcher = MinimaxSearcher(use_alpha_beta=use_alpha_beta, move_ordering=move_ordering,
                                        radius=radius, weights=weights)
        # Tác tử luôn chơi luật của engine nên trọng số sai báo lỗi ngay khi tạo tác tử.
        self.searcher.weights_for(SearchState.initial())
        self.last_result: SearchResult | None = None

    def get_action(self, board: np.ndarray, player: int) -> Action:
        state = SearchState(board, player)
        self.last_result = None
        self.stats.update(depth_reached=0, root_value=None)
        candidates = state.candidates(self.searcher.radius)
        if not candidates:
            # Chỉ xảy ra khi bên gọi vi phạm hợp đồng (ván đã kết thúc); vẫn trả nước hợp lệ.
            row, col = np.argwhere(state.board == EMPTY)[0]
            return int(row), int(col)
        fallback = candidates[0]

        if self.budget.mode == "fixed":
            self.searcher.should_stop = None
            self.last_result = self.searcher.search(state, self.budget.depth)
            self._record(self.last_result)
        else:
            self.last_result = self._iterative_deepening(state)

        if self.last_result is None or self.last_result.action is None:
            return fallback
        self.stats.update(depth_reached=self.last_result.depth, root_value=self.last_result.value)
        return self.last_result.action

    def _record(self, counters: SearchResult | MinimaxSearcher) -> None:
        self.stats["nodes_visited"] += counters.nodes_visited
        self.stats["pruned_nodes"] += counters.pruned_nodes

    def _iterative_deepening(self, state: SearchState) -> SearchResult | None:
        if self.deadline is None:
            raise ValueError("Chế độ time cần gọi qua choose_action() để có deadline")
        soft_deadline = self.deadline - self.time_margin_s
        self.searcher.should_stop = lambda: perf_counter() >= soft_deadline
        best: SearchResult | None = None
        max_depth = int(np.count_nonzero(state.board == EMPTY))
        try:
            for depth in range(1, max_depth + 1):
                if self.searcher.should_stop is not None and self.searcher.should_stop():
                    break
                try:
                    result = self.searcher.search(
                        state, depth, first_action=best.action if best else None)
                except SearchTimeout:
                    self._record(self.searcher)
                    break
                self._record(result)
                best = result
                # Đã chứng minh thắng/thua chắc chắn trong tầm nhìn: tìm sâu hơn không đổi kết quả.
                if abs(result.value) >= WIN_SCORE - max_depth:
                    break
        finally:
            self.searcher.should_stop = None
        return best
