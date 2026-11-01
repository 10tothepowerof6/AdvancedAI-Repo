import numpy as np
import pytest

from agents.base_agent import BaseAgent
from agents.random_agent import RandomAgent
from config.search_budget import SearchBudget


@pytest.mark.parametrize("kwargs", [{"mode": "sai"}, {"simulations": 0}, {"depth": True},
                                    {"time_limit_s": 0}, {"time_limit_s": float("nan")}])
def test_invalid_budget(kwargs):
    with pytest.raises(ValueError):
        SearchBudget(**kwargs)


class ObservingAgent(BaseAgent):
    def get_action(self, board, player):
        assert self.deadline is not None
        assert self.time_exhausted()
        return (0, 0)


def test_deadline_and_fixed_mode(monkeypatch):
    clock = iter([10.0, 13.0, 13.0])
    monkeypatch.setattr("agents.base_agent.perf_counter", lambda: next(clock))
    agent = ObservingAgent("Quan sát", budget=SearchBudget(mode="time"))
    assert agent.choose_action(np.zeros((3, 3)), 1) == (0, 0)
    assert agent.deadline == 12.0
    assert agent.stats["thinking_time"] == 3.0
    agent.budget = SearchBudget()
    agent.deadline = None
    assert not agent.time_exhausted()


def test_random_accepts_common_budget():
    agent = RandomAgent(42, budget=SearchBudget(mode="time"))
    assert agent.choose_action(np.zeros((3, 3)), 1) is not None
    assert agent.deadline is not None
