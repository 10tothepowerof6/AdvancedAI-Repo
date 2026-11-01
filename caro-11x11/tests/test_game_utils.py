import numpy as np
import pytest

from core.game_utils import candidate_actions


def test_empty_full_and_edge():
    board = np.zeros((11, 11), dtype=np.int8)
    assert candidate_actions(board) == [(5, 5)]
    board[0, 0] = 1
    original = board.copy()
    actions = candidate_actions(board, radius=1)
    assert actions == [(0, 1), (1, 0), (1, 1)]
    np.testing.assert_array_equal(board, original)
    assert len(candidate_actions(board, radius=2)) == 8
    board[:] = 1
    assert candidate_actions(board) == []


@pytest.mark.parametrize('radius', [0, -1, True, 1.5])
def test_invalid_radius(radius):
    with pytest.raises(ValueError):
        candidate_actions(np.zeros((3, 3)), radius)
