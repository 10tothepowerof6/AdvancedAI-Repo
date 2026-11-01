import numpy as np
import pytest

from core.game_engine import CaroGame


def test_initial_reset():
    game = CaroGame()
    assert game.board.shape == (11, 11)
    assert game.board.dtype == np.int8
    assert game.current_player == 1
    assert len(game.legal_actions()) == 121
    assert game.result() is None
    assert game.check_winner() is None
    game.step((5, 5))
    assert game.current_player == -1
    assert game.move_count == 1
    game.reset()
    assert not game.board.any()
    assert game.last_action is None
    assert game.move_count == 0


@pytest.mark.parametrize('stones', [
    [(0, c) for c in range(5)],
    [(r, 10) for r in range(5)],
    [(r, r) for r in range(5)],
    [(r, 10-r) for r in range(5)],
])
@pytest.mark.parametrize('winner', [1, -1])
def test_all_winning_directions(stones, winner):
    game = CaroGame()
    fillers = [(10, 0), (10, 2), (10, 4), (10, 6), (10, 8)]
    if winner == -1:
        game.step(fillers.pop())
    for index, stone in enumerate(stones):
        game.step(stone)
        if index < 4:
            game.step(fillers[index])
    assert game.result() == winner
    assert game.check_winner() == winner
    assert game.current_player == winner
    assert game.is_terminal()
    assert game.legal_actions() == []
    with pytest.raises(ValueError):
        game.step((6, 6))


def test_overline_and_blocked_ends_win():
    game = CaroGame()
    # Lấp khoảng trống để nối sáu quân cùng lúc; hai đầu bị O chặn.
    for x, o in zip([1, 2, 3, 5, 6], [0, 7, 9, 10, 8]):
        game.step((0, x))
        game.step((0 if o in (0, 7) else 10, o))
    game.step((0, 4))
    assert game.result() == 1


def test_draw_and_clone_terminal():
    game = CaroGame(3, 3)
    for action in [(0, 0), (0, 1), (0, 2), (1, 1), (1, 0),
                   (1, 2), (2, 1), (2, 0), (2, 2)]:
        game.step(action)
    assert game.result() == 0
    assert game.is_terminal()
    clone = game.clone()
    assert clone.result() == 0
    assert clone.move_count == 9
    assert clone.last_action == game.last_action
    clone.reset()
    assert game.result() == 0


@pytest.mark.parametrize('action', [(-1, 0), (11, 0), (0, 11), (0.5, 1),
                                    (True, 0), (np.bool_(False), 0), [0, 0], (0,), None])
def test_invalid_action_is_atomic(action):
    game = CaroGame()
    with pytest.raises(ValueError):
        game.step(action)
    assert not game.board.any()
    assert game.move_count == 0
    assert game.current_player == 1


def test_occupied_clone_and_numpy_coordinates():
    game = CaroGame()
    game.step((np.int64(5), np.int64(5)))
    with pytest.raises(ValueError):
        game.step((5, 5))
    assert game.current_player == -1
    clone = game.clone()
    clone.step((0, 0))
    assert game.board[0, 0] == 0
    assert clone.current_player == 1
    assert game.current_player == -1


@pytest.mark.parametrize('size,k', [(0, 5), (11, 0), (3, 5), (True, 1), (3.5, 3)])
def test_invalid_configuration(size, k):
    with pytest.raises(ValueError):
        CaroGame(size, k)


def test_render(capsys):
    game = CaroGame(3, 3)
    game.step((0, 0))
    text = game.render()
    assert 'X' in text and '.' in text
    assert capsys.readouterr().out.strip() == text.strip()
