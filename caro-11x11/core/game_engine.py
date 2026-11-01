"""Engine Caro freestyle xác định; chuỗi dài hơn năm quân vẫn thắng."""

from numbers import Integral

import numpy as np

from config.game_config import BOARD_SIZE, EMPTY, PLAYER_X, WIN_LENGTH

Action = tuple[int, int]


class CaroGame:
    DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))

    def __init__(self, board_size: int = BOARD_SIZE, win_length: int = WIN_LENGTH):
        for name, value in (("board_size", board_size), ("win_length", win_length)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} phải là số nguyên dương")
        if win_length > board_size:
            raise ValueError("Độ dài thắng không được lớn hơn kích thước bàn cờ")
        self.board_size = int(board_size)
        self.win_length = int(win_length)
        self.reset()

    def reset(self) -> None:
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = PLAYER_X
        self.last_action: Action | None = None
        self.move_count = 0
        self._result: int | None = None

    def legal_actions(self) -> list[Action]:
        if self.is_terminal():
            return []
        return [(int(r), int(c)) for r, c in np.argwhere(self.board == EMPTY)]

    def step(self, action: Action) -> None:
        if self.is_terminal():
            raise ValueError("Không thể đi tiếp vì ván đấu đã kết thúc")
        if not isinstance(action, tuple) or len(action) != 2:
            raise ValueError("Nước đi phải là một bộ tọa độ (hàng, cột)")
        if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, Integral) for x in action):
            raise ValueError("Tọa độ nước đi phải là số nguyên")
        row, col = map(int, action)
        if not (0 <= row < self.board_size and 0 <= col < self.board_size):
            raise ValueError("Nước đi nằm ngoài bàn cờ")
        if self.board[row, col] != EMPTY:
            raise ValueError("Ô cờ đã có quân")
        self.board[row, col] = self.current_player
        self.last_action = (row, col)
        self.move_count += 1
        winner = self.check_winner()
        if winner is not None:
            self._result = winner
        elif self.move_count == self.board_size ** 2:
            self._result = 0
        else:
            self.current_player = -self.current_player

    def check_winner(self) -> int | None:
        """Kiểm tra các đường qua nước gần nhất; chỉ đổi bàn cờ bằng step()."""
        if self.last_action is None:
            return None
        row, col = self.last_action
        player = int(self.board[row, col])
        for dr, dc in self.DIRECTIONS:
            count = 1
            for sign in (-1, 1):
                r, c = row + sign * dr, col + sign * dc
                while (0 <= r < self.board_size and 0 <= c < self.board_size
                       and self.board[r, c] == player):
                    count += 1
                    r, c = r + sign * dr, c + sign * dc
            if count >= self.win_length:
                return player
        return None

    def is_terminal(self) -> bool:
        return self._result is not None

    def result(self) -> int | None:
        return self._result

    def clone(self) -> "CaroGame":
        other = CaroGame(self.board_size, self.win_length)
        other.board = self.board.copy()
        other.current_player = self.current_player
        other.last_action = self.last_action
        other.move_count = self.move_count
        other._result = self._result
        return other

    def render(self) -> str:
        """In và trả về bàn cờ dạng văn bản cho terminal hoặc notebook."""
        symbols = {0: ".", 1: "X", -1: "O"}
        width = len(str(self.board_size - 1))
        lines = [" " * (width + 1) + " ".join(f"{c:>{width}}" for c in range(self.board_size))]
        lines.extend(f"{r:>{width}} " + " ".join(f"{symbols[int(v)]:>{width}}" for v in row)
                     for r, row in enumerate(self.board))
        text = "\n".join(lines)
        print(text)
        return text
