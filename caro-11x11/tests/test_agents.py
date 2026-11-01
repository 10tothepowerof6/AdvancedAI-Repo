import numpy as np
import pytest

from agents import BaseAgent, RandomAgent
from core import CaroGame


def test_seed_reproducibility_and_last_cell():
    board = np.zeros((11, 11), dtype=np.int8)
    a, b = RandomAgent(42), RandomAgent(42)
    assert [a.choose_action(board, 1) for _ in range(50)] == [b.choose_action(board, 1) for _ in range(50)]
    board[:] = 1
    board[2, 7] = 0
    assert a.choose_action(board, -1) == (2, 7)


@pytest.mark.parametrize('seed', range(10))
def test_complete_random_games(seed):
    game = CaroGame()
    agents = {1: RandomAgent(seed), -1: RandomAgent(seed + 100)}
    while not game.is_terminal():
        original = game.board.copy()
        agent = agents[game.current_player]
        action = agent.choose_action(game.board, game.current_player)
        np.testing.assert_array_equal(game.board, original)
        assert action in game.legal_actions()
        assert agent.stats['thinking_time'] >= 0
        game.step(action)
    assert game.result() in (-1, 0, 1)
    assert game.move_count <= 121


class StubAgent(BaseAgent):
    def __init__(self, action=(0, 0), fail=False):
        super().__init__('Stub')
        self.action, self.fail = action, fail

    def get_action(self, board, player):
        board[:] = 0
        self.stats['nodes_visited'] += 1
        if self.fail:
            raise RuntimeError('Tìm kiếm thất bại')
        return self.action


def test_snapshot_validation_and_stats_reset():
    board = np.zeros((3, 3), dtype=np.int8)
    agent = StubAgent()
    agent.choose_action(board, 1)
    agent.choose_action(board, -1)
    assert agent.stats['nodes_visited'] == 1
    board[0, 0] = 1
    with pytest.raises(ValueError, match='không hợp lệ'):
        agent.choose_action(board, 1)
    assert board[0, 0] == 1


@pytest.mark.parametrize('action', [None, (-1, 0), (3, 0), (True, 0), (0.5, 0), [0, 0]])
def test_illegal_agent_output(action):
    with pytest.raises(ValueError):
        StubAgent(action).choose_action(np.zeros((3, 3)), 1)


def test_failure_propagates_with_timing():
    agent = StubAgent(fail=True)
    with pytest.raises(RuntimeError, match='Tìm kiếm thất bại'):
        agent.choose_action(np.zeros((3, 3)), 1)
    assert agent.stats['thinking_time'] >= 0


@pytest.mark.parametrize('board,player', [(np.ones((3, 3)), 1), (np.zeros((2, 3)), 1),
    (np.full((3, 3), 2), 1), (np.zeros((3, 3)), 0), (np.zeros((3, 3)), True)])
def test_invalid_agent_inputs(board, player):
    with pytest.raises(ValueError):
        RandomAgent().choose_action(board, player)
