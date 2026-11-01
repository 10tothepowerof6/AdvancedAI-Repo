"""Lớp cơ sở cho tác tử; lớp con cần cài đặt get_action(board, player)."""

from abc import ABC, abstractmethod
from numbers import Integral
from time import perf_counter

import numpy as np

from core.game_engine import Action
from config.search_budget import SearchBudget


class BaseAgent(ABC):
    def __init__(self, name: str, *, budget: SearchBudget | None = None):
        if budget is not None and not isinstance(budget, SearchBudget):
            raise ValueError("Ngân sách phải là SearchBudget")
        self.name = name
        self.budget = budget if budget is not None else SearchBudget()
        self.deadline: float | None = None
        self.stats: dict[str, float | int] = {}
        self.reset_stats()

    def reset_stats(self) -> None:
        self.stats = {"thinking_time": 0.0, "nodes_visited": 0,
                      "simulations": 0, "pruned_nodes": 0}

    def choose_action(self, board: np.ndarray, player: int) -> Action:
        """Đo thời gian suy nghĩ, bảo vệ bàn cờ đầu vào và kiểm tra nước đi trả về.

        Chỉ yêu cầu tác tử chọn nước đi khi ván đấu chưa kết thúc.
        """
        start = perf_counter()
        self.deadline = (start + self.budget.time_limit_s
                         if self.budget.mode == "time" else None)
        self.reset_stats()
        if (not isinstance(board, np.ndarray) or board.ndim != 2
                or board.shape[0] == 0 or board.shape[0] != board.shape[1]
                or not np.isin(board, (-1, 0, 1)).all()):
            raise ValueError("Bàn cờ phải là mảng vuông không rỗng, chỉ chứa -1, 0, 1")
        if isinstance(player, (bool, np.bool_)) or not isinstance(player, Integral) or player not in (-1, 1):
            raise ValueError("Người chơi phải có giá trị 1 hoặc -1")
        if not np.any(board == 0):
            raise ValueError("Không còn nước đi hợp lệ")
        # Đối chiếu với bản sao riêng kể cả khi lớp con sửa dữ liệu nhận vào.
        snapshot = board.copy()
        try:
            action = self.get_action(snapshot.copy(), int(player))
        finally:
            self.stats["thinking_time"] = perf_counter() - start
        if (not isinstance(action, tuple) or len(action) != 2
                or any(isinstance(x, (bool, np.bool_)) or not isinstance(x, Integral) for x in action)):
            raise ValueError(f"{self.name} trả về nước đi sai định dạng: {action!r}")
        row, col = map(int, action)
        if not (0 <= row < board.shape[0] and 0 <= col < board.shape[1]) or snapshot[row, col] != 0:
            raise ValueError(f"{self.name} trả về nước đi không hợp lệ: {action!r}")
        return row, col

    def time_exhausted(self) -> bool:
        """Cho thuật toán chủ động dừng; không cưỡng chế ngắt hàm đang chạy."""
        return self.deadline is not None and perf_counter() >= self.deadline

    @abstractmethod
    def get_action(self, board: np.ndarray, player: int) -> Action:
        """Trả về nước hợp lệ (hàng, cột); bên gọi dùng choose_action()."""
        raise NotImplementedError
