import numpy as np

from agents.base_agent import BaseAgent
from core.game_engine import Action
from config.search_budget import SearchBudget


class RandomAgent(BaseAgent):
    def __init__(self, seed: int | None = None, *, budget: SearchBudget | None = None):
        super().__init__("Ngẫu nhiên", budget=budget)
        self.rng = np.random.default_rng(seed)

    def get_action(self, board: np.ndarray, player: int) -> Action:
        cells = np.argwhere(board == 0)
        if len(cells) == 0:
            raise ValueError("Không còn nước đi hợp lệ")
        row, col = cells[self.rng.integers(len(cells))]
        return int(row), int(col)
