"""MCTS Rollout cho Caro 11×11 — trạng thái mô phỏng, cây tìm kiếm UCT và tác tử.

Module tách các tầng:
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

Bốn bước của một mô phỏng (Giai đoạn 2), mỗi bước là một hàm riêng để kiểm thử độc lập:
1. Lựa chọn — select_child(): đi từ gốc xuống khi nút chưa kết thúc và đã mở rộng hết,
   chọn con có điểm UCT cao nhất. Dừng tại nút terminal hoặc nút còn nước chưa mở rộng.
2. Mở rộng — MCTSNode.expand(): thêm đúng một con ứng với một nước chưa thử (chọn ngẫu nhiên
   theo seed). Bỏ qua nếu nút đã kết thúc.
3. Mô phỏng — rollout(): từ trạng thái của nút mới, hai bên luân phiên đi ngẫu nhiên đều trên
   các ô trống tới khi có người thắng hoặc bàn đầy. Không cắt ngắn theo độ sâu. Nút terminal
   lấy thẳng kết quả, không chạy rollout. Hết thời gian giữa chừng thì trả None: mô phỏng bị bỏ.
4. Cập nhật ngược — backpropagate(): cộng N += 1 và Q += outcome_value(kết quả, player_just_moved)
   cho mọi nút trên đường đi, từ nút mới lên gốc.
"""

from __future__ import annotations

from math import inf, isfinite, log, sqrt
from numbers import Integral, Real
from time import perf_counter
from typing import Callable

import numpy as np

from agents.base_agent import BaseAgent
from config.game_config import EMPTY, PLAYER_O, PLAYER_X, WIN_LENGTH
from config.search_budget import SearchBudget
from core.game_engine import Action, CaroGame
from core.game_utils import candidate_actions

# Bán kính ứng viên mặc định, trùng với candidate_actions của dự án.
DEFAULT_RADIUS = 2
# Hệ số khám phá UCT khởi đầu c = √2 (giá trị thường dùng của UCB1 cho phần thưởng trong [0, 1]).
# Phần thưởng ở đây nằm trong [-1, 1] nên c chỉ là điểm xuất phát, chưa phải giá trị đã hiệu chỉnh.
DEFAULT_EXPLORATION = sqrt(2)
# Biên an toàn ở chế độ time: dừng tìm kiếm trước deadline 0,1 giây (1,9 s trên hạn 2,0 s).
DEFAULT_TIME_MARGIN_S = 0.1
# Số nước rollout giữa hai lần đọc đồng hồ (đọc mỗi nước cũng được nhưng tốn thêm thời gian).
_CLOCK_CHECK_EVERY = 8

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
        """Tập nước rút gọn quanh quân đã có; rỗng khi đã kết thúc.

        Không rỗng khi ván chưa kết thúc: còn ô trống thì luôn có ô trống sát một quân nào đó.
        """
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


# ---------------------------------------------------------------- bốn bước MCTS

def uct_score(child: MCTSNode, parent_visits: int, exploration: float) -> float:
    """Điểm UCT của `child` nhìn từ người đến lượt ở nút cha.

    UCT = Q/N + c·√(ln N_cha / N_con). Q/N (khai thác) đã theo góc nhìn người đang chọn
    vì Q của con lưu theo `player_just_moved`; thành phần sau (khám phá) lớn khi con ít
    được thăm so với cha. Con chưa thăm (N = 0) được ưu tiên tuyệt đối: điểm vô cực.
    """
    if child.visits == 0:
        return inf
    return child.mean_value() + exploration * sqrt(log(parent_visits) / child.visits)


def select_child(node: MCTSNode, exploration: float, rng: np.random.Generator) -> MCTSNode:
    """Bước lựa chọn tại một nút: con có UCT cao nhất; đồng điểm thì chọn ngẫu nhiên theo seed."""
    if not node.children:
        raise ValueError("Nút chưa có con để lựa chọn")
    children = list(node.children.values())
    scores = [uct_score(child, node.visits, exploration) for child in children]
    best = max(scores)
    tied = [child for child, score in zip(children, scores) if score == best]
    return tied[0] if len(tied) == 1 else tied[int(rng.integers(len(tied)))]


def _line_wins(board: list[list[int]], size: int, win_length: int, row: int, col: int, player: int) -> bool:
    """Kiểm tra thắng qua ô vừa đi trên bàn cờ dạng list (nhanh hơn truy cập phần tử NumPy)."""
    for dr, dc in _DIRECTIONS:
        count = 1
        r, c = row + dr, col + dc
        while 0 <= r < size and 0 <= c < size and board[r][c] == player:
            count += 1
            r, c = r + dr, c + dc
        r, c = row - dr, col - dc
        while 0 <= r < size and 0 <= c < size and board[r][c] == player:
            count += 1
            r, c = r - dr, c - dc
        if count >= win_length:
            return True
    return False


def rollout(state: MCTSState, rng: np.random.Generator, stop_at: float | None = None,
            trace: list[Action] | None = None) -> int | None:
    """Bước mô phỏng: chơi ngẫu nhiên tới khi kết thúc, trả kết quả theo góc nhìn X.

    Chính sách: hai bên luân phiên, mỗi nước chọn ngẫu nhiên đều trong các ô trống còn lại
    (tương đương đi theo một hoán vị ngẫu nhiên của các ô trống). Dừng khi có chuỗi thắng
    (≥ 5 quân, kể cả bị chặn hai đầu) hoặc bàn đầy (hòa). Không cắt ngắn theo độ sâu.
    `state` không bị sửa. Trạng thái terminal trả thẳng kết quả, không đi thêm nước nào.
    Nếu `stop_at` (giá trị perf_counter) đã qua trong lúc chơi thì trả None: mô phỏng bị bỏ,
    không được đếm và không được cập nhật ngược. `trace` (nếu có) nhận các nước đã đi.
    """
    if state.is_terminal():
        return state.result()
    board = state.board.tolist()
    size, win_length = state.size, state.win_length
    empties = [(int(r), int(c)) for r, c in np.argwhere(state.board == EMPTY)]
    player = state.current_player
    for index, cell in enumerate(rng.permutation(len(empties))):
        if stop_at is not None and index % _CLOCK_CHECK_EVERY == 0 and perf_counter() >= stop_at:
            return None
        row, col = empties[cell]
        board[row][col] = player
        if trace is not None:
            trace.append((row, col))
        if _line_wins(board, size, win_length, row, col, player):
            return player
        player = -player
    return 0


def backpropagate(node: MCTSNode | None, result: int) -> None:
    """Bước cập nhật ngược: từ `node` lên gốc, N += 1 và Q += giá trị theo góc nhìn từng nút."""
    while node is not None:
        node.visits += 1
        node.value += outcome_value(result, node.player_just_moved)
        node = node.parent


def best_action(root: MCTSNode, rng: np.random.Generator) -> Action | None:
    """Quyết định cuối: con có N lớn nhất; đồng N thì Q/N cao hơn; vẫn đồng thì ngẫu nhiên theo seed.

    Trả None nếu chưa có con nào được thăm (chưa hoàn thành mô phỏng nào).
    """
    visited = [child for child in root.children.values() if child.visits > 0]
    if not visited:
        return None
    key = max((child.visits, child.mean_value()) for child in visited)
    tied = [child for child in visited if (child.visits, child.mean_value()) == key]
    chosen = tied[0] if len(tied) == 1 else tied[int(rng.integers(len(tied)))]
    return chosen.action


# ---------------------------------------------------------------- tác tử

class MCTSAgent(BaseAgent):
    """Tác tử MCTS + UCT + rollout ngẫu nhiên, tuân theo hợp đồng BaseAgent.

    - Ngân sách đọc từ `self.budget` tại mỗi lượt (runner gán lại budget sau khi tạo agent).
      `fixed`: chạy đúng `budget.simulations` mô phỏng hoàn thành.
      `time`: chạy tới `deadline − biên`, biên = min(time_margin_s, 0,5 × time_limit_s);
      không dùng `simulations` làm điều kiện dừng phụ.
    - Luôn có nước dự phòng (một ứng viên chọn ngẫu nhiên theo seed) phòng khi chưa hoàn thành
      mô phỏng nào trước mốc dừng.
    - Cây được dựng mới ở mỗi lượt (không tái sử dụng giữa các nước).
    - Thống kê mỗi lượt: `simulations` = mô phỏng hoàn thành; `nodes_visited` = số nút mới tạo
      (không tính gốc); `pruned_nodes` = 0.
    """

    def __init__(self, seed: int | None = None, *, budget: SearchBudget | None = None,
                 exploration: float = DEFAULT_EXPLORATION, radius: int = DEFAULT_RADIUS,
                 time_margin_s: float = DEFAULT_TIME_MARGIN_S, name: str = "MCTS"):
        super().__init__(name, budget=budget)
        if (isinstance(exploration, bool) or not isinstance(exploration, Real)
                or not isfinite(exploration) or exploration < 0):
            raise ValueError("Hệ số khám phá phải là số thực hữu hạn không âm")
        if isinstance(radius, bool) or not isinstance(radius, Integral) or radius < 1:
            raise ValueError("Bán kính phải là số nguyên dương")
        if (isinstance(time_margin_s, bool) or not isinstance(time_margin_s, Real)
                or not isfinite(time_margin_s) or time_margin_s < 0):
            raise ValueError("Biên an toàn thời gian phải là số thực hữu hạn không âm")
        self.exploration = float(exploration)
        self.radius = int(radius)
        self.time_margin_s = float(time_margin_s)
        self.rng = np.random.default_rng(seed)
        self.last_root: MCTSNode | None = None

    def _stop_at(self) -> float | None:
        """Mốc dừng hợp tác ở chế độ time; None ở chế độ fixed hoặc khi gọi ngoài choose_action."""
        if self.budget.mode != "time" or self.deadline is None:
            return None
        return self.deadline - min(self.time_margin_s, 0.5 * self.budget.time_limit_s)

    def search(self, board: np.ndarray, player: int,
               trace: Callable[[dict[str, object]], None] | None = None) -> tuple[MCTSNode, Action]:
        """Chạy tìm kiếm, trả (gốc cây, nước dự phòng); cập nhật `stats` và `last_root`.

        `trace` (nếu có) được gọi sau mỗi mô phỏng hoàn thành với dict mô tả bốn bước;
        notebook dùng nó để quan sát, không ảnh hưởng thuật toán.
        """
        if not isinstance(self.budget, SearchBudget):
            raise ValueError("Ngân sách phải là SearchBudget")
        root = MCTSNode.root(board, player, radius=self.radius)
        if root.is_terminal():
            raise ValueError("Không tìm nước đi cho trạng thái đã kết thúc")
        fallback = root.untried_actions[int(self.rng.integers(len(root.untried_actions)))]
        stop_at = self._stop_at()
        fixed = self.budget.mode == "fixed"
        completed = created = 0
        while not fixed or completed < self.budget.simulations:
            if stop_at is not None and perf_counter() >= stop_at:
                break
            # 1. Lựa chọn
            node, path = root, []
            while not node.is_terminal() and node.is_fully_expanded():
                node = select_child(node, self.exploration, self.rng)
                path.append(node.action)
            # 2. Mở rộng
            expanded = None
            if not node.is_terminal():
                expanded = node.untried_actions[int(self.rng.integers(len(node.untried_actions)))]
                node = node.expand(expanded)
                created += 1
            # 3. Mô phỏng
            moves: list[Action] | None = [] if trace is not None else None
            result = rollout(node.state, self.rng, stop_at, moves)
            if result is None:
                break  # hết giờ giữa rollout: bỏ mô phỏng, không cập nhật ngược
            # 4. Cập nhật ngược
            backpropagate(node, result)
            completed += 1
            if trace is not None:
                trace({"simulation": completed, "selection_path": path, "expanded": expanded,
                       "leaf_terminal": node.is_terminal(), "rollout_moves": moves, "result": result})
        self.stats["simulations"] = completed
        self.stats["nodes_visited"] = created
        self.stats["pruned_nodes"] = 0
        self.last_root = root
        return root, fallback

    def get_action(self, board: np.ndarray, player: int) -> Action:
        root, fallback = self.search(board, player)
        action = best_action(root, self.rng)
        return fallback if action is None else action
