"""Kiểm thử Minimax — Giai đoạn 1: trạng thái tìm kiếm và hàm lượng giá.

Mỗi ca theo 3 bước Arrange / Act / Assert. Fixture được dựng bằng chuỗi nước
đi hợp lệ qua CaroGame.step() để bảo đảm trạng thái có thể xảy ra trong ván thật.
"""

import numpy as np
import pytest

from agents.minimax_agent import (
    WIN_SCORE,
    WINDOW_WEIGHTS,
    SearchState,
    default_weights,
    evaluate,
    find_winner,
    heuristic_bound,
    heuristic_score,
    terminal_score,
    window_counts,
)
from agents.random_agent import RandomAgent
from core.game_engine import CaroGame
from core.game_utils import candidate_actions

X, O = 1, -1
DIRECTION_STARTS = {"ngang": ((0, 1), (5, 1)), "dọc": ((1, 0), (1, 5)),
                    "chéo chính": ((1, 1), (1, 1)), "chéo phụ": ((1, -1), (1, 9))}


# Dựng fixture bằng chuỗi nước đi

def _interleave(x_moves, o_moves):
    """X đi trước; số nước O bằng hoặc ít hơn X một nước."""
    assert len(x_moves) - len(o_moves) in (0, 1)
    moves = []
    for i, x in enumerate(x_moves):
        moves.append(x)
        if i < len(o_moves):
            moves.append(o_moves[i])
    return moves


def _winning_sequence(winner, direction, length):
    """Chuỗi nước để winner tạo đường dài `length` theo `direction`.
    Với length 6, ô giữa được đi cuối để kiểm tra chuỗi dài hơn 5 vẫn thắng
    mà trước đó không có chuỗi 5 nào. Quân bên thua đặt trên các ô (chẵn, chẵn)
    cách nhau 2 ô nên không thể tạo chuỗi thắng.
    """
    (dr, dc), (r0, c0) = DIRECTION_STARTS[direction]
    line = [(r0 + k * dr, c0 + k * dc) for k in range(length)]
    if length == 6:
        line = line[:3] + line[4:] + [line[3]]
    filler = [(r, c) for r in range(0, 11, 2) for c in range(0, 11, 2) if (r, c) not in line]
    if winner == X:
        return _interleave(line, filler[:length - 1])
    return _interleave(filler[:length], line)


def _draw_sequence():
    """Bàn đầy 61 X / 60 O không có chuỗi 5: X tại ô có (c + r // 2) chẵn."""
    x_cells = [(r, c) for r in range(11) for c in range(11) if (c + r // 2) % 2 == 0]
    o_cells = [(r, c) for r in range(11) for c in range(11) if (c + r // 2) % 2 == 1]
    return _interleave(x_cells, o_cells)


def _fixture_cases():
    cases = {}
    for winner, name in ((X, "X"), (O, "O")):
        for direction in DIRECTION_STARTS:
            for length in (5, 6):
                cases[f"{name}-{direction}-{length}"] = (_winning_sequence(winner, direction, length), winner)
    cases["hòa-bàn-đầy"] = (_draw_sequence(), 0)
    return cases


FIXTURES = _fixture_cases()


def _game_after(moves):
    game = CaroGame()
    for move in moves:
        game.step(move)
    return game


def _assert_same(state, game):
    np.testing.assert_array_equal(state.board, game.board)
    assert state.current_player == game.current_player
    assert state.last_action == game.last_action
    assert state.move_count == game.move_count
    assert state.is_terminal() == game.is_terminal()
    assert state.result() == game.result()


def _random_board(seed, n_moves):
    """Bàn chưa kết thúc sau n_moves nước ngẫu nhiên hợp lệ; trả về (board, người đến lượt)."""
    game, agent = CaroGame(), RandomAgent(seed)
    for _ in range(n_moves):
        if game.is_terminal():
            break
        game.step(agent.choose_action(game.board, game.current_player))
    return game


# Trạng thái tìm kiếm — đối chiếu với engine

@pytest.mark.parametrize("name", FIXTURES)
def test_fixture_sequence_matches_engine_every_step(name):
    # Arrange
    moves, expected = FIXTURES[name]
    game, state = CaroGame(), SearchState.initial()
    _assert_same(state, game)
    # Act + Assert: so sánh sau từng nước
    for move in moves:
        game.step(move)
        state = state.apply(move)
        _assert_same(state, game)
    assert state.result() == expected


@pytest.mark.parametrize("seed", range(5))
def test_random_game_matches_engine_every_step(seed):
    # Arrange
    game, state = CaroGame(), SearchState.initial()
    agents = {X: RandomAgent(seed), O: RandomAgent(seed + 100)}
    # Act + Assert
    while not game.is_terminal():
        action = agents[game.current_player].choose_action(game.board, game.current_player)
        game.step(action)
        state = state.apply(action)
        _assert_same(state, game)


@pytest.mark.parametrize("name", FIXTURES)
def test_state_from_final_board_detects_result(name):
    # Arrange
    moves, expected = FIXTURES[name]
    game = _game_after(moves)
    # Act
    state = SearchState(game.board, game.current_player)
    # Assert
    assert state.is_terminal()
    assert state.result() == expected
    assert find_winner(game.board) == (expected or None)
    assert state.candidates() == []


def test_state_from_ongoing_board():
    # Arrange
    game = _game_after(_winning_sequence(X, "ngang", 5)[:-1])
    # Act
    state = SearchState(game.board, game.current_player)
    # Assert
    assert not state.is_terminal() and state.result() is None
    assert state.move_count == game.move_count
    assert state.candidates() == candidate_actions(game.board)


def test_apply_keeps_parent_and_input_board_unchanged():
    # Arrange
    game = _random_board(seed=7, n_moves=12)
    board = game.board.copy()
    parent = SearchState(board, game.current_player)
    before = (parent.board.copy(), parent.current_player, parent.last_action,
              parent.move_count, parent.result())
    # Act: duyệt mọi nước con, kể cả con của con
    for action in parent.candidates():
        child = parent.apply(action)
        for grandchild_action in child.candidates()[:5]:
            child.apply(grandchild_action)
    # Assert
    np.testing.assert_array_equal(parent.board, before[0])
    assert (parent.current_player, parent.last_action, parent.move_count, parent.result()) == before[1:]
    np.testing.assert_array_equal(board, game.board)


def test_state_copies_input_board():
    # Arrange
    board = np.zeros((11, 11), dtype=np.int8)
    state = SearchState(board, X)
    # Act
    board[5, 5] = X
    # Assert
    assert state.board[5, 5] == 0 and state.move_count == 0


def test_child_boards_are_independent():
    # Arrange
    parent = SearchState.initial()
    # Act
    left, right = parent.apply((5, 5)), parent.apply((0, 0))
    # Assert
    assert left.board[0, 0] == 0 and right.board[5, 5] == 0
    assert left.current_player == right.current_player == O


@pytest.mark.parametrize("action", [(5, 5), (-1, 0), (11, 0), (0, 11), (True, 0), (0.5, 0), [0, 0], None])
def test_apply_rejects_illegal_action(action):
    # Arrange
    state = SearchState.initial().apply((5, 5))
    # Act + Assert
    with pytest.raises(ValueError):
        state.apply(action)


def test_apply_rejects_terminal_state():
    # Arrange
    moves, _ = FIXTURES["X-ngang-5"]
    state = SearchState.initial()
    for move in moves:
        state = state.apply(move)
    # Act + Assert
    with pytest.raises(ValueError, match="kết thúc"):
        state.apply((10, 10))


@pytest.mark.parametrize("board,player", [
    (np.zeros((2, 3)), X), (np.full((3, 3), 2), X), (np.zeros((0, 0)), X),
    (np.zeros((11, 11)), 0), (np.zeros((11, 11)), True), ([[0]], X)])
def test_state_rejects_invalid_input(board, player):
    with pytest.raises(ValueError):
        SearchState(board, player)


def test_board_with_two_winners_is_invalid():
    # Arrange
    board = np.zeros((11, 11), dtype=np.int8)
    board[0, :5], board[2, :5] = X, O
    # Act + Assert
    with pytest.raises(ValueError, match="cả hai bên"):
        SearchState(board, X)

# Cửa sổ 5 ô
@pytest.mark.parametrize("cell,expected", [((5, 5), 20), ((0, 0), 3), ((0, 5), 8)])
def test_window_counts_single_stone(cell, expected):
    """Tính tay: ô tâm nằm trong 5 cửa sổ mỗi hướng (20); góc (0,0) chỉ 3 cửa sổ
    (ngang, dọc, chéo chính); ô (0,5) ở mép trên có 5 ngang + 1 dọc + 1 chéo chính
    + 1 chéo phụ = 8 (các hướng dọc/chéo chỉ còn cửa sổ bắt đầu tại chính ô đó)."""
    # Arrange
    board = np.zeros((11, 11), dtype=np.int8)
    board[cell] = X
    # Act
    counts = window_counts(board)
    # Assert
    assert counts[X][1] == expected
    assert counts[X][0] == 252 - expected
    assert counts[O][1] == 0 and counts[O][0] == 252 - expected


def test_window_counts_blocked_window_is_dead():
    # Arrange: X X X O trên cùng hàng — cửa sổ chứa cả hai màu không tính cho ai
    board = np.zeros((11, 11), dtype=np.int8)
    board[5, 2:5], board[5, 5] = X, O
    # Act
    counts = window_counts(board)
    # Assert: chỉ cửa sổ hàng 5 cột 0..4 chứa đủ 3 quân X mà không có O
    assert counts[X][3] == 1


# Hàm lượng giá
@pytest.mark.parametrize("result,perspective,expected", [
    (X, X, WIN_SCORE), (X, O, -WIN_SCORE), (O, O, WIN_SCORE), (O, X, -WIN_SCORE),
    (0, X, 0), (0, O, 0)])
def test_terminal_score_convention(result, perspective, expected):
    assert terminal_score(result, perspective) == expected


@pytest.mark.parametrize("result,perspective", [(2, X), (None, X), (True, X), (X, 0), (X, True)])
def test_terminal_score_rejects_invalid(result, perspective):
    with pytest.raises(ValueError):
        terminal_score(result, perspective)


def test_terminal_score_dominates_any_heuristic():
    # Cận trên lý thuyết: 252 cửa sổ × trọng số lớn nhất 1000 = 252 000
    assert heuristic_bound() == 252 * max(WINDOW_WEIGHTS)
    assert heuristic_bound() < WIN_SCORE // 10
    for seed in range(20):
        game = _random_board(seed, n_moves=20 + seed * 3)
        if not game.is_terminal():
            assert abs(heuristic_score(game.board, X)) <= heuristic_bound()


@pytest.mark.parametrize("name", FIXTURES)
def test_evaluate_terminal_uses_terminal_score(name):
    # Arrange
    moves, expected = FIXTURES[name]
    state = SearchState.initial()
    for move in moves:
        state = state.apply(move)
    # Act + Assert: điểm terminal, không phải heuristic
    assert evaluate(state, X) == WIN_SCORE * expected
    assert evaluate(state, O) == -WIN_SCORE * expected


def test_win_beats_strong_losing_position():
    # Arrange: O có nhiều cửa sổ 4 quân nhưng X vừa tạo chuỗi 5
    moves = _interleave([(10, 0), (10, 1), (10, 2), (10, 3), (10, 4)],
                        [(2, 2), (3, 3), (4, 4), (5, 5)])
    state = SearchState.initial()
    for move in moves:
        state = state.apply(move)
    # Act + Assert
    assert state.result() == X
    assert evaluate(state, X) == WIN_SCORE
    assert heuristic_score(state.board, O) > 0  # O có thế cờ tốt hơn về heuristic


@pytest.mark.parametrize("seed", range(10))
def test_perspective_symmetry(seed):
    # Arrange
    game = _random_board(seed, 15)
    state = SearchState(game.board, game.current_player)
    # Act + Assert
    assert evaluate(state, X) == -evaluate(state, O)


@pytest.mark.parametrize("seed", range(10))
def test_color_swap_symmetry(seed):
    """Đổi màu toàn bộ quân thì điểm của X trên bàn gốc bằng điểm của O trên bàn đảo."""
    board = _random_board(seed, 15).board
    assert heuristic_score(board, X) == heuristic_score(-board, O)


def test_empty_board_scores_zero():
    assert evaluate(SearchState.initial(), X) == 0
    assert evaluate(SearchState.initial(), O) == 0


@pytest.mark.parametrize("player", [X, O])
def test_longer_open_line_scores_higher(player):
    # Arrange: chuỗi mở dài 1..4 ở giữa hàng 5
    scores = []
    for length in range(1, 5):
        board = np.zeros((11, 11), dtype=np.int8)
        board[5, 3:3 + length] = player
        # Act
        scores.append(heuristic_score(board, player))
    # Assert: tăng chặt theo độ dài; đối thủ thấy điểm âm tương ứng
    assert scores == sorted(scores) and len(set(scores)) == 4
    assert all(s > 0 for s in scores)


def test_weights_strictly_increase_with_stones():
    """Nguyên tắc thiết kế: thêm một quân vào cửa sổ còn sống luôn làm điểm tăng
    (cửa sổ trống = 0; cửa sổ đủ 5 quân giao cho terminal_score nên bỏ qua)."""
    live = WINDOW_WEIGHTS[:-1]
    assert live[0] == 0
    assert all(a < b for a, b in zip(live, live[1:]))


@pytest.mark.parametrize("player", [X, O])
def test_connected_stones_beat_scattered_stones(player):
    # Arrange: cùng 4 quân, một bên nối thành hàng, một bên rải cách xa nhau
    connected = np.zeros((11, 11), dtype=np.int8)
    connected[5, 3:7] = player
    scattered = np.zeros((11, 11), dtype=np.int8)
    for cell in ((0, 0), (0, 10), (10, 0), (10, 10)):
        scattered[cell] = player
    # Act + Assert
    assert heuristic_score(connected, player) > heuristic_score(scattered, player) > 0


@pytest.mark.parametrize("player", [X, O])
def test_open_three_better_than_blocked_three(player):
    # Arrange
    open_three = np.zeros((11, 11), dtype=np.int8)
    open_three[5, 4:7] = player
    blocked = open_three.copy()
    blocked[5, 3] = -player
    blocked_far = open_three.copy()
    blocked_far[0, 0] = -player  # cùng số quân đối thủ nhưng không chặn
    # Act + Assert
    assert heuristic_score(open_three, player) > heuristic_score(blocked_far, player)
    assert heuristic_score(blocked_far, player) > heuristic_score(blocked, player)


@pytest.mark.parametrize("player", [X, O])
def test_open_four_better_than_blocked_four(player):
    # Arrange
    open_four = np.zeros((11, 11), dtype=np.int8)
    open_four[5, 3:7] = player
    blocked = open_four.copy()
    blocked[5, 2] = -player
    # Act + Assert
    assert heuristic_score(open_four, player) > heuristic_score(blocked, player) > 0


def test_evaluation_does_not_mutate_board():
    # Arrange
    game = _random_board(seed=3, n_moves=20)
    board = game.board.copy()
    state = SearchState(board, game.current_player)
    # Act
    evaluate(state, X)
    heuristic_score(board, O)
    window_counts(board)
    # Assert
    np.testing.assert_array_equal(board, game.board)
    np.testing.assert_array_equal(state.board, game.board)


@pytest.mark.parametrize("weights", [(1, 2, 3), (0, 1, 10, 100, 1000, -1), (0, 1, 10, 100, 1000, 0, 0),
                                     (0, 1.5, 10, 100, 1000, 0), (0, True, 10, 100, 1000, 0)])
def test_invalid_weights_rejected(weights):
    with pytest.raises(ValueError):
        heuristic_score(np.zeros((11, 11), dtype=np.int8), X, weights=weights)


# Cận heuristic phải nhỏ hơn WIN_SCORE (góp ý review PR #17)

# 252 cửa sổ: trọng số lớn nhất 39 682 cho cận 9 999 864 < WIN_SCORE, 39 683 cho 10 000 116.
LARGEST_ALLOWED = (WIN_SCORE - 1) // 252


def test_weights_with_bound_reaching_win_score_rejected():
    # Arrange: ví dụ trong review - một quân X ở tâm từng được 20 000 000 > WIN_SCORE
    weights = (0, 1_000_000, 0, 0, 0, 0)
    state = SearchState.initial().apply((5, 5))
    # Act + Assert
    with pytest.raises(ValueError, match="WIN_SCORE"):
        evaluate(state, X, weights=weights)
    with pytest.raises(ValueError, match="WIN_SCORE"):
        heuristic_score(state.board, X, weights=weights)
    with pytest.raises(ValueError, match="WIN_SCORE"):
        heuristic_bound(weights=weights)


def test_weights_bound_boundary():
    # Arrange
    allowed = (0, LARGEST_ALLOWED, 0, 0, 0, 0)
    too_large = (0, LARGEST_ALLOWED + 1, 0, 0, 0, 0)
    # Act + Assert
    assert heuristic_bound(weights=allowed) == 252 * LARGEST_ALLOWED < WIN_SCORE
    with pytest.raises(ValueError):
        heuristic_bound(weights=too_large)


def test_weights_bound_equal_to_win_score_rejected():
    # Arrange: bàn 4×4 luật 4 quân có 10 cửa sổ nên cận chạm đúng WIN_SCORE
    exact = (0, WIN_SCORE // 10, 0, 0, 0)
    below = (0, WIN_SCORE // 10 - 1, 0, 0, 0)
    # Act + Assert
    assert heuristic_bound(4, 4, below) == WIN_SCORE - 10
    with pytest.raises(ValueError):
        heuristic_bound(4, 4, exact)
    with pytest.raises(ValueError):
        evaluate(SearchState.initial(4, 4), X, weights=exact)


def test_won_state_beats_playing_states_with_largest_allowed_weights():
    # Arrange: trọng số lớn nhất được phép cho mọi loại cửa sổ còn sống
    weights = (0,) + (LARGEST_ALLOWED,) * 4 + (0,)
    moves, _ = FIXTURES["X-ngang-5"]
    won = SearchState.initial()
    for move in moves:
        won = won.apply(move)
    playing = [SearchState.initial().apply((5, 5))]
    for seed in range(10):
        game = _random_board(seed, n_moves=10 + seed * 4)
        if not game.is_terminal():
            playing.append(SearchState(game.board, game.current_player))
    # Act
    won_score = evaluate(won, X, weights=weights)
    # Assert
    assert won.result() == X and won_score == WIN_SCORE
    assert all(evaluate(s, p, weights=weights) < won_score for s in playing for p in (X, O))


# Trọng số mặc định theo win_length (góp ý review PR #17)

def test_default_weights_match_window_weights():
    assert default_weights() == default_weights(5, 11) == WINDOW_WEIGHTS


@pytest.mark.parametrize("size,win_length", [(3, 3), (6, 4), (11, 1), (11, 2), (11, 5), (11, 7), (11, 11)])
def test_default_weights_follow_win_length(size, win_length):
    # Act
    weights = default_weights(win_length, size)
    # Assert
    live = weights[:-1]
    assert len(weights) == win_length + 1 and weights[0] == weights[-1] == 0
    assert all(a < b for a, b in zip(live[1:], live[2:]))
    assert heuristic_bound(size, win_length) < WIN_SCORE


@pytest.mark.parametrize("size,win_length", [(3, 3), (6, 4), (11, 2), (11, 7), (11, 11)])
def test_evaluate_without_weights_on_other_rules(size, win_length):
    # Arrange
    empty = SearchState.initial(size, win_length)
    center = size // 2
    # Act
    child = empty.apply((center, center))
    # Assert: bàn trống cân bằng; một quân ở tâm có lợi cho bên vừa đi
    assert evaluate(empty, X) == 0
    assert evaluate(child, X) == -evaluate(child, O) > 0


def test_evaluate_3x3_full_game_without_weights():
    # Arrange: X thắng ở chéo chính sau 5 nước trên bàn 3×3 luật 3 quân
    moves = [(1, 1), (0, 1), (0, 0), (2, 0), (2, 2)]
    state = SearchState.initial(3, 3)
    scores = [evaluate(state, X)]
    # Act
    for move in moves:
        state = state.apply(move)
        scores.append(evaluate(state, X))
    # Assert
    assert state.result() == X and scores[-1] == WIN_SCORE
    assert all(abs(s) <= heuristic_bound(3, 3) for s in scores[:-1])


@pytest.mark.parametrize("size,win_length", [(0, 1), (3, 0), (3, 4), (3, True), (True, 1), (3, 2.0)])
def test_default_weights_rejects_invalid_rule(size, win_length):
    with pytest.raises(ValueError):
        default_weights(win_length, size)


# Bàn cờ chỉ đọc (góp ý review PR #17)

def _states_for_readonly():
    moves, _ = FIXTURES["X-ngang-5"]
    terminal = SearchState.initial()
    for move in moves:
        terminal = terminal.apply(move)
    game = _random_board(seed=2, n_moves=12)
    return {"initial": SearchState.initial(),
            "từ-mảng": SearchState(game.board, game.current_player),
            "con": SearchState.initial().apply((5, 5)),
            "kết-thúc": terminal}


@pytest.mark.parametrize("name", ["initial", "từ-mảng", "con", "kết-thúc"])
def test_board_is_read_only(name):
    # Arrange
    state = _states_for_readonly()[name]
    before = state.board.copy()
    # Act + Assert
    with pytest.raises(ValueError):
        state.board[0, 10] = X
    with pytest.raises(ValueError):
        state.board += 1
    np.testing.assert_array_equal(state.board, before)


def test_direct_five_in_row_write_rejected_and_state_consistent():
    # Arrange: tình huống trong review - ghi thẳng 5 quân X lên một hàng
    state = SearchState.initial()
    # Act
    with pytest.raises(ValueError):
        state.board[0, :5] = X
    # Assert
    assert not state.board.any()
    assert state.result() is None and state.move_count == 0


def test_apply_still_works_on_read_only_board():
    # Arrange
    parent = SearchState.initial()
    # Act
    child = parent.apply((5, 5))
    grandchild = child.apply((5, 6))
    # Assert
    assert child.board[5, 5] == X and grandchild.board[5, 6] == O
    assert not parent.board.any()
    assert not child.board.flags.writeable and not grandchild.board.flags.writeable


def test_input_board_stays_writable():
    # Arrange
    board = np.zeros((11, 11), dtype=np.int8)
    # Act
    SearchState(board, X)
    board[5, 5] = X
    # Assert
    assert board.flags.writeable and board[5, 5] == X
