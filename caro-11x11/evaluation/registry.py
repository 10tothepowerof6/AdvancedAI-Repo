"""Cấu hình tác tử có thể khởi tạo trong tiến trình con trên Windows."""

from dataclasses import dataclass

from agents.base_agent import BaseAgent
from config.search_budget import SearchBudget

MAIN_AGENTS = ("random", "rule", "minimax", "mcts", "neural")
AGENT_NAMES = {"random": "Ngẫu nhiên", "rule": "Rule-based có pattern",
               "rule-basic": "Rule-based cơ bản", "minimax": "Minimax Alpha-Beta",
               "minimax-plain": "Minimax không cắt tỉa", "mcts": "MCTS chuẩn", "neural": "CNN Policy"}


@dataclass(frozen=True)
class AgentSpec:
    kind: str
    seed: int = 0

    def __post_init__(self):
        if self.kind not in AGENT_NAMES:
            raise ValueError(f"Tác tử chưa được đăng ký: {self.kind}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("Seed phải là số nguyên không âm")

    def __call__(self) -> BaseAgent:
        if self.kind == "random":
            from agents.random_agent import RandomAgent
            return RandomAgent(seed=self.seed)
        if self.kind in ("rule", "rule-basic"):
            from agents.rule_based_agent import RuleBasedAgent
            return RuleBasedAgent(use_patterns=self.kind == "rule")
        if self.kind in ("minimax", "minimax-plain"):
            from agents.minimax_agent import MinimaxAgent
            return MinimaxAgent(use_alpha_beta=self.kind == "minimax")
        if self.kind == "mcts":
            from agents.mcts_agent import MCTSAgent
            return MCTSAgent(seed=self.seed)
        from agents.neural_agent import NeuralAgent
        return NeuralAgent()

    def create(self, budget: SearchBudget) -> BaseAgent:
        agent = self()
        agent.budget = budget
        return agent
