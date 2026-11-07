"""MCTS Rollout cho Caro 11×11 — Giai đoạn 1: trạng thái mô phỏng và cây tìm kiếm.

Module tách hai tầng:
- MCTSState: dữ liệu thuộc luật cờ (bàn cờ, lượt đi, nước cuối, số nước, kết quả).
  Engine không có API nạp bàn cờ tùy ý nên tầng này tự dựng trạng thái từ đầu vào
  của agent và được kiểm thử đối chiếu với CaroGame qua chuỗi nước đi hợp lệ.
- MCTSNode: dữ liệu thuộc quá trình tìm kiếm (cha/con, hành động dẫn tới nút,
  các nước chưa mở rộng, số lượt thăm N và giá trị tích lũy Q).

Quy ước giá trị (giữ nguyên ở mọi giai đoạn):
- Kết quả ván `result` theo góc nhìn X giống engine: 1 X thắng, -1 O thắng, 0 hòa.
- Q của một nút được cộng dồn theo góc nhìn của người vừa đi nước dẫn tới nút đó
  (`player_just_moved`): +1 thắng, -1 thua, 0 hòa. Nhờ vậy khi đứng ở nút cha,
  Q/N của mỗi nút con đã là giá trị theo góc nhìn của người đến lượt, không cần đổi dấu.
  Q/N chỉ là thành phần khai thác: khi tìm kiếm (Giai đoạn 2), nhánh được chọn theo điểm
  UCT = Q/N + c·√(ln N_cha / N_con), có thêm thành phần khám phá; nước chính thức trả về
  được chọn theo số lượt thăm N.

Chưa có vòng tìm kiếm UCT và lớp MCTSAgent; hai phần này thuộc Giai đoạn 2.
"""

from __future__ import annotations

from math import sqrt
from numbers import Integral

import numpy as np

from config.game_config import EMPTY, PLAYER_O, PLAYER_X, WIN_LENGTH
from core.game_engine import Action, CaroGame
from core.game_utils import candidate_actions

# Bán kính ứng viên mặc định, trùng với candidate_actions của dự án.
DEFAULT_RADIUS = 2
# Hệ số khám phá UCT dự kiến (c = √2); bắt đầu dùng ở Giai đoạn 2.
DEFAULT_EXPLORATION = sqrt(2)

_DIRECTIONS = CaroGame.DIRECTIONS


def outcome_value(result: int, player: int) -> float:
    """Đổi kết quả theo góc nhìn X sang giá trị theo góc nhìn `player`.

    Ví dụ: X thắng (result=1) cho X giá trị +1 và cho O giá trị -1;
    O thắng (result=-1) cho O +1 và cho X -1; hòa cho cả hai bên 0.
    """
    if isinstance(result, bool) or not isinstance(result, Integral) or result not in (-1, 0, 1):
        raise ValueError("Kết quả phải là -1, 0 hoặc 1")
    if isinstance(player, bool) or not isinstance(player, Integral) or player not in (PLAYER_X, PLAYER_O):
        raise ValueError("Người chơi phải có giá trị 1 hoặc -1")
    return float(int(result) * int(player))


class MCTSState:
    """Trạng thái mô phỏng độc lập với engine nhưng theo đúng luật của engine.

    `current_player` giữ đúng ngữ nghĩa của CaroGame: khi chưa kết thúc là người
    đến lượt; tại trạng thái kết thúc vẫn là người vừa đi nước cuối.
    """

    __slots__ = ("board", "current_player", "last_action", "move_count", "win_length", "_result")

    def __init__(self, board: np.ndarray, current_player: int, *, last_action: Action | None = None,
                 move_count: int = 0, result: int | None = None, win_length: int = WIN_LENGTH):
        # Hàm khởi tạo nội bộ; bên ngoài dùng from_board() hoặc from_game().
        self.board = board
        self.current_player = current_player
        self.last_action = last_action
        self.move_count = move_count
        self.win_length = win_length
        self._result = result

    @classmethod
    def from_board(cls, board: np.ndarray, player: int, win_length: int = WIN_LENGTH) -> "MCTSState":
        """Dựng trạng thái từ đầu vào của agent; luôn sao chép, không sửa `board`.

        Đầu vào chỉ có bàn cờ và người đến lượt nên `last_action` là None. Bàn cờ
        được quét toàn bộ một lần để nhận ra chuỗi thắng hoặc bàn đầy có sẵn.
        Khi chưa kết thúc, tin vào `player` do bên gọi cung cấp, không suy lượt từ số quân.
        Khi đã kết thúc, `current_player` theo ngữ nghĩa engine là người vừa đi nước cuối,
        suy từ bàn cờ: bên có chuỗi thắng; nếu hòa thì X khi X nhiều quân hơn O
        (X đi trước), ngược lại là O.
        """
        if (not isinstance(board, np.ndarray) or board.ndim != 2
                or board.shape[0] == 0 or board.shape[0] != board.shape[1]
                or not np.isin(board, (-1, 0, 1)).all()):
            raise ValueError("Bàn cờ phải là mảng vuông không rỗng, chỉ chứa -1, 0, 1")
        if isinstance(player, (bool, np.bool_)) or not isinstance(player, Integral) or player not in (PLAYER_X, PLAYER_O):
            raise ValueError("Người chơi phải có giá trị 1 hoặc -1")
        if (isinstance(win_length, bool) or not isinstance(win_length, Integral)
                or not 1 <= win_length <= board.shape[0]):
            raise ValueError("Độ dài thắng phải là số nguyên dương không vượt kích thước bàn")
        state = cls(board.astype(np.int8, copy=True), int(player),
                    move_count=int(np.count_nonzero(board)), win_length=int(win_length))
        winners = {int(state.board[r, c]) for r, c in np.argwhere(state.board != EMPTY)
                   if state._wins_at(int(r), int(c))}
        if len(winners) > 1:
            raise ValueError("Bàn cờ không hợp lệ: cả hai bên cùng có chuỗi thắng")
        if winners:
            state._result = winners.pop()
            state.current_player = state._result
        elif state.move_count == state.board.size:
            state._result = 0
            x_stones = int(np.count_nonzero(state.board == PLAYER_X))
            state.current_player = PLAYER_X if x_stones > state.move_count - x_stones else PLAYER_O
        return state

    @classmethod
    def from_game(cls, game: CaroGame) -> "MCTSState":
        """Sao chép đầy đủ trạng thái engine; dùng cho kiểm thử và notebook."""
        return cls(game.board.astype(np.int8, copy=True), int(game.current_player),
                   last_action=game.last_action, move_count=game.move_count,
                   result=game.result(), win_length=game.win_length)

    @property
    def size(self) -> int:
        return self.board.shape[0]

    def is_terminal(self) -> bool:
        return self._result is not None

    def result(self) -> int | None:
        """Kết quả theo góc nhìn X: None đang chơi, 1 X thắng, -1 O thắng, 0 hòa."""
        return self._result

    def legal_actions(self) -> list[Action]:
        """Toàn bộ ô trống (tập nước hợp lệ đầy đủ); rỗng khi đã kết thúc."""
        if self.is_terminal():
            return []
        return [(int(r), int(c)) for r, c in np.argwhere(self.board == EMPTY)]

    def candidate_actions(self, radius: int = DEFAULT_RADIUS) -> list[Action]:
        """Tập nước rút gọn quanh quân đã có; rỗng khi đã kết thúc."""
        if self.is_terminal():
            return []
        return candidate_actions(self.board, radius)

    def clone(self) -> "MCTSState":
        return MCTSState(self.board.copy(), self.current_player, last_action=self.last_action,
                         move_count=self.move_count, result=self._result, win_length=self.win_length)

    def step(self, action: Action) -> None:
        """Đi một nước tại chỗ theo đúng luật engine; nước sai không làm đổi trạng thái."""
        if self.is_terminal():
            raise ValueError("Không thể đi tiếp vì ván đấu đã kết thúc")
        if (not isinstance(action, tuple) or len(action) != 2
                or any(isinstance(x, (bool, np.bool_)) or not isinstance(x, Integral) for x in action)):
            raise ValueError("Nước đi phải là bộ tọa độ số nguyên (hàng, cột)")
        row, col = int(action[0]), int(action[1])
        if not (0 <= row < self.size and 0 <= col < self.size):
            raise ValueError("Nước đi nằm ngoài bàn cờ")
        if self.board[row, col] != EMPTY:
            raise ValueError("Ô cờ đã có quân")
        self.board[row, col] = self.current_player
        self.last_action = (row, col)
        self.move_count += 1
        if self._wins_at(row, col):
            self._result = self.current_player
        elif self.move_count == self.board.size:
            self._result = 0
        else:
            self.current_player = -self.current_player

    def play(self, action: Action) -> "MCTSState":
        """Trả về trạng thái mới sau nước đi; trạng thái hiện tại giữ nguyên."""
        child = self.clone()
        child.step(action)
        return child

    def _wins_at(self, row: int, col: int) -> bool:
        """Kiểm tra chuỗi qua ô (row, col) giống CaroGame.check_winner."""
        board, size = self.board, self.size
        player = board[row, col]
        for dr, dc in _DIRECTIONS:
            count = 1
            for sign in (-1, 1):
                r, c = row + sign * dr, col + sign * dc
                while 0 <= r < size and 0 <= c < size and board[r, c] == player:
                    count += 1
                    r, c = r + sign * dr, c + sign * dc
            if count >= self.win_length:
                return True
        return False


class MCTSNode:
    """Nút cây MCTS; chỉ giữ dữ liệu tìm kiếm, luật cờ nằm trong MCTSState.

    - `action`: nước dẫn từ nút cha tới nút này (None ở gốc).
    - `player_just_moved`: người đã đi `action`. Ở gốc chưa kết thúc là đối thủ của
      người đến lượt; ở gốc đã kết thúc là `current_player` (engine giữ người vừa đi).
    - `visits` (N) và `value` (Q) khởi tạo bằng 0; Q tính theo góc nhìn
      `player_just_moved`, xem outcome_value().
    - `untried_actions`: ứng viên bán kính `radius` chưa mở rộng; rỗng nếu terminal.
    """

    __slots__ = ("state", "parent", "action", "player_just_moved", "radius",
                 "children", "untried_actions", "visits", "value")

    def __init__(self, state: MCTSState, parent: "MCTSNode | None" = None,
                 action: Action | None = None, radius: int = DEFAULT_RADIUS):
        if (parent is None) != (action is None):
            raise ValueError("Nút gốc không có cha và hành động; nút con phải có cả hai")
        self.state = state
        self.parent = parent
        self.action = action
        if parent is not None:
            self.player_just_moved = parent.state.current_player
        elif state.is_terminal():
            # Tại terminal, current_player đã là người vừa đi nước cuối.
            self.player_just_moved = state.current_player
        else:
            self.player_just_moved = -state.current_player
        self.radius = radius
        self.children: dict[Action, MCTSNode] = {}
        self.untried_actions: list[Action] = state.candidate_actions(radius)
        self.visits = 0
        self.value = 0.0

    @classmethod
    def root(cls, board: np.ndarray, player: int, radius: int = DEFAULT_RADIUS) -> "MCTSNode":
        """Tạo gốc từ đầu vào của agent; bàn cờ được sao chép."""
        return cls(MCTSState.from_board(board, player), radius=radius)

    def is_terminal(self) -> bool:
        return self.state.is_terminal()

    def is_fully_expanded(self) -> bool:
        return not self.untried_actions

    def mean_value(self) -> float:
        """Q/N theo góc nhìn `player_just_moved`; nút chưa thăm trả về 0."""
        return self.value / self.visits if self.visits else 0.0

    def expand(self, action: Action) -> "MCTSNode":
        """Mở rộng một nước chưa thử; trạng thái của nút hiện tại không đổi.

        Nước chỉ bị xóa khỏi `untried_actions` sau khi nút con được tạo thành công,
        nên nước sai kiểu như (5.0, 5) bị từ chối mà không làm mất ứng viên (5, 5).
        """
        if self.is_terminal():
            raise ValueError("Không mở rộng trạng thái đã kết thúc")
        if action not in self.untried_actions:
            raise ValueError(f"Nước {action!r} không nằm trong các nước chưa mở rộng")
        # play() kiểm tra kiểu tọa độ và báo lỗi trước khi có bất kỳ thay đổi nào.
        child_state = self.state.play(action)
        key = child_state.last_action  # tọa độ đã chuẩn hóa về (int, int)
        child = MCTSNode(child_state, parent=self, action=key, radius=self.radius)
        self.untried_actions.remove(key)
        self.children[key] = child
        return child
