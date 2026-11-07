"""Minimax và Alpha-Beta cho cờ Caro 11×11.

Giai đoạn 1 — Trạng thái tìm kiếm và hàm lượng giá (file hiện tại).
Giai đoạn 2 — Minimax giới hạn độ sâu, cắt tỉa Alpha-Beta, Iterative Deepening.
Giai đoạn 3 — Đối chứng cắt tỉa và khảo sát độ sâu.

Quy ước góc nhìn: mọi hàm lượng giá nhận tham số `perspective` (1 hoặc -1) và
trả điểm theo góc nhìn bên đó: dương là có lợi, âm là bất lợi, 0 là cân bằng.
Đổi góc nhìn chỉ đổi dấu: evaluate(s, X) == -evaluate(s, O).
"""

from __future__ import annotations

from functools import lru_cache
from numbers import Integral

import numpy as np

from config.game_config import BOARD_SIZE, EMPTY, WIN_LENGTH
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
    counts = window_counts(board, win_length)
    score = int(weight_array @ counts[1]) - int(weight_array @ counts[-1])
    return perspective * score


def evaluate(state: SearchState, perspective: int, *,
             weights: tuple[int, ...] | None = None) -> int:
    """Lượng giá trạng thái: terminal dùng terminal_score, còn lại dùng heuristic.

    weights=None chọn bộ trọng số mặc định theo win_length và kích thước bàn
    của chính state, nên mọi trạng thái tạo được bằng SearchState đều lượng giá được.
    """
    if state.is_terminal():
        return terminal_score(state.result(), perspective)
    return heuristic_score(state.board, perspective, win_length=state.win_length, weights=weights)
