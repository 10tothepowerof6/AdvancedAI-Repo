from time import sleep
from functools import partial

import pytest

from agents.base_agent import BaseAgent
from agents.random_agent import RandomAgent
from config.search_budget import SearchBudget
from core import CaroGame
from evaluation.match import AgentExecutionError, AgentWorker, play_match


class SlowAgent(BaseAgent):
    def __init__(self):
        super().__init__("Chậm")

    def get_action(self, board, player):
        sleep(10)
        return (0, 0)


class BrokenAgent(SlowAgent):
    def get_action(self, board, player):
        raise RuntimeError("Lỗi thử nghiệm")


class FailingFactory:
    def __call__(self):
        raise RuntimeError("Không tải được mô hình")


class FirstEmptyAgent(SlowAgent):
    def get_action(self, board, player):
        for r in range(board.shape[0]):
            for c in range(board.shape[1]):
                if board[r, c] == 0:
                    return (r, c)
        raise ValueError("Không còn ô trống")


@pytest.mark.parametrize("mode", ["fixed", "time"])
def test_timeout_forfeit_and_cleanup(mode):
    record, logs = play_match(SlowAgent, RandomAgent, agent_x="Chậm", agent_o="Ngẫu nhiên",
                              budget=SearchBudget(mode=mode, time_limit_s=0.1),
                              safety_timeout_s=0.1)
    assert record.winner == -1
    assert record.reason == "timeout"
    assert logs[0]["status"] == "timeout"


def test_errors_not_draws():
    record, logs = play_match(BrokenAgent, RandomAgent, agent_x="Lỗi", agent_o="B",
                              budget=SearchBudget())
    assert record.winner == -1 and record.reason == "agent_error"
    assert "Lỗi thử nghiệm" in logs[0]["error"]


def test_startup_error_is_not_match_result():
    with pytest.raises(AgentExecutionError, match="Không tải được mô hình"):
        AgentWorker(FailingFactory(), SearchBudget())


@pytest.mark.parametrize("mode", ["fixed", "time"])
def test_complete_match_preserves_initial_game(mode):
    game = CaroGame(3, 3)
    record, logs = play_match(FirstEmptyAgent, FirstEmptyAgent, agent_x="A", agent_o="B",
                              game=game, budget=SearchBudget(mode=mode))
    assert record.reason == "completed"
    assert record.winner == 1
    assert len(logs) == 7
    assert game.move_count == 0
    assert all(log["status"] == "move" for log in logs)


def test_worker_keeps_rng_state():
    import numpy as np
    worker = AgentWorker(partial(RandomAgent, seed=42), SearchBudget())
    try:
        board = np.zeros((11, 11), dtype=np.int8)
        actions = [worker.choose_action(board, 1, 5)[0] for _ in range(10)]
        reference = RandomAgent(seed=42)
        assert actions == [reference.choose_action(board, 1) for _ in range(10)]
    finally:
        worker.close()
    assert not worker.process.is_alive()
