import numpy as np

from agents.base_agent import BaseAgent
from core.game_engine import Action


class RandomAgent(BaseAgent):
    def __init__(self, seed: int | None = None):
        super().__init__("Random")
        self.rng = np.random.default_rng(seed)

    def get_action(self, board: np.ndarray, player: int) -> Action:
        cells = np.argwhere(board == 0)
        if len(cells) == 0:
            raise ValueError("No legal actions remain")
        row, col = cells[self.rng.integers(len(cells))]
        return int(row), int(col)
