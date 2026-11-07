"""Kiểm thử MCTS — Giai đoạn 1: trạng thái mô phỏng, nút cây và quy ước giá trị.

Giá trị kỳ vọng lấy từ engine CaroGame hoặc tính tay theo luật cờ,
không dùng chính hàm đang kiểm thử để sinh kết quả đối chiếu.

Các test dài chia ba phần: Chuẩn bị (dựng thế cờ) / Thực thi (gọi hàm cần kiểm tra)
/ Kiểm chứng (so với kỳ vọng). Test ngắn một hai dòng gộp chung ba phần.
"""

import numpy as np
import pytest

from agents.mcts_agent import DEFAULT_RADIUS, MCTSNode, MCTSState, outcome_value
from core.game_engine import CaroGame
from core.game_utils import candidate_actions

# Quân đệm ở hàng 10, cách xa các chuỗi thắng.
_FILL = [(10, 0), (10, 2), (10, 4), (10, 6), (10, 8)]

# Fixture: (tên, chuỗi nước đi hợp lệ, kích thước, độ dài thắng, kết quả kỳ vọng tính tay).
SEQUENCES = [
    ("x_thang_ngang", [(0, 0), (10, 0), (0, 1), (10, 2), (0, 2), (10, 4), (0, 3), (10, 6), (0, 4)], 11, 5, 1),
    ("x_thang_doc", [(0, 10), (10, 0), (1, 10), (10, 2), (2, 10), (10, 4), (3, 10), (10, 6), (4, 10)], 11, 5, 1),
    ("o_thang_cheo", [(10, 8)] + [m for i in range(5) for m in ((i, i), _FILL[i])][:-1], 11, 5, -1),
    ("o_thang_cheo_phu", [(10, 8)] + [m for i in range(5) for m in ((i, 10 - i), _FILL[i])][:-1], 11, 5, -1),
    # X lấp khoảng trống giữa nối sáu quân, hai đầu bị O chặn: vẫn thắng.
    ("x_thang_sau_quan_bi_chan", [(0, 1), (0, 0), (0, 2), (0, 7), (0, 3), (10, 9), (0, 5), (10, 10),
                                  (0, 6), (10, 8), (0, 4)], 11, 5, 1),
    ("hoa_3x3", [(0, 0), (0, 1), (0, 2), (1, 1), (1, 0), (1, 2), (2, 1), (2, 0), (2, 2)], 3, 3, 0),
    ("dang_choi", [(5, 5), (5, 6), (6, 5), (4, 4)], 11, 5, None),
]


def _assert_same_as_engine(state: MCTSState, game: CaroGame) -> None:
    np.testing.assert_array_equal(state.board, game.board)
    assert state.current_player == game.current_player
    assert state.last_action == game.last_action
    assert state.move_count == game.move_count
    assert state.result() == game.result()
    assert state.is_terminal() == game.is_terminal()
    assert state.legal_actions() == game.legal_actions()


# ---------------------------------------------------------------- quy ước giá trị

@pytest.mark.parametrize("result,player,expected", [
    (1, 1, 1.0), (1, -1, -1.0),    # X thắng: X +1, O -1
    (-1, -1, 1.0), (-1, 1, -1.0),  # O thắng: O +1, X -1
    (0, 1, 0.0), (0, -1, 0.0),     # hòa: cả hai 0
], ids=["x_thang_goc_x_+1", "x_thang_goc_o_-1", "o_thang_goc_o_+1", "o_thang_goc_x_-1",
        "hoa_goc_x_0", "hoa_goc_o_0"])
def test_outcome_value_convention(result, player, expected):
    assert outcome_value(result, player) == expected


@pytest.mark.parametrize("result,player", [(2, 1), (None, 1), (True, 1), (1, 0), (1, True), (0.5, 1)],
                         ids=["ket_qua_2", "ket_qua_None", "ket_qua_bool", "nguoi_choi_0",
                              "nguoi_choi_bool", "ket_qua_so_thuc"])
def test_outcome_value_rejects_invalid(result, player):
    with pytest.raises(ValueError):
        outcome_value(result, player)


# ---------------------------------------------------------------- trạng thái mô phỏng

@pytest.mark.parametrize("name,moves,size,k,expected", SEQUENCES, ids=[s[0] for s in SEQUENCES])
def test_state_matches_engine_each_step(name, moves, size, k, expected):
    # Chuẩn bị: engine và trạng thái MCTS cùng xuất phát từ bàn trống.
    game = CaroGame(size, k)
    state = MCTSState.from_board(game.board, game.current_player, win_length=k)
    _assert_same_as_engine(state, game)
    # Thực thi và kiểm chứng từng bước: đi cùng một nước trên cả hai rồi so sánh.
    for action in moves:
        game.step(action)
        state.step(action)
        _assert_same_as_engine(state, game)
    # Kiểm chứng: kết quả cuối khớp kết quả tính tay của fixture.
    assert state.result() == expected


@pytest.mark.parametrize("name,moves,size,k,expected", SEQUENCES, ids=[s[0] for s in SEQUENCES])
def test_from_board_matches_engine_without_history(name, moves, size, k, expected):
    # Agent chỉ nhận bàn cờ và người đến lượt, nên không biết nước cuối.
    # Chuẩn bị: engine bàn trống.
    game = CaroGame(size, k)
    for action in moves:
        # Thực thi: engine đi một nước, MCTS dựng lại trạng thái chỉ từ bàn cờ.
        # Tại terminal truyền đối thủ của người vừa đi (như thể "đến lượt") để chắc
        # from_board tự suy người vừa đi từ bàn cờ, không phụ thuộc tham số này.
        game.step(action)
        player = -game.current_player if game.is_terminal() else game.current_player
        state = MCTSState.from_board(game.board, player, win_length=k)
        # Kiểm chứng: khớp engine ở mọi thông tin suy ra được từ bàn cờ, kể cả terminal.
        np.testing.assert_array_equal(state.board, game.board)
        assert state.move_count == game.move_count
        assert state.result() == game.result()
        assert state.last_action is None
        assert state.current_player == game.current_player
        assert state.legal_actions() == game.legal_actions()


def test_from_board_empty_board_and_o_to_move():
    board = np.zeros((11, 11), dtype=np.int8)
    state = MCTSState.from_board(board, 1)
    assert (state.current_player, state.move_count, state.result()) == (1, 0, None)
    assert len(state.legal_actions()) == 121
    assert state.candidate_actions() == [(5, 5)]
    board[5, 5] = 1
    assert MCTSState.from_board(board, -1).current_player == -1


def test_from_board_copies_and_never_mutates_input():
    # Chuẩn bị: bàn cờ kiểu float có một quân X, lưu bản gốc để so.
    board = np.zeros((11, 11), dtype=np.float64)
    board[5, 5] = 1
    original = board.copy()
    # Thực thi: dựng trạng thái rồi đi nước và sửa trực tiếp trên trạng thái.
    state = MCTSState.from_board(board, -1)
    state.step((5, 6))
    state.board[0, 0] = 1
    # Kiểm chứng: trạng thái dùng int8 riêng; bàn cờ đầu vào không đổi.
    assert state.board.dtype == np.int8
    np.testing.assert_array_equal(board, original)


def test_from_board_rejects_two_winners():
    board = np.zeros((11, 11), dtype=np.int8)
    board[0, :5] = 1
    board[2, :5] = -1
    with pytest.raises(ValueError, match="cả hai bên"):
        MCTSState.from_board(board, 1)


@pytest.mark.parametrize("board,player", [
    (np.zeros((2, 3)), 1), (np.full((3, 3), 2), 1), (np.zeros(3), 1), ([[0]], 1),
    (np.zeros((3, 3)), 0), (np.zeros((3, 3)), True), (np.zeros((3, 3)), 1.0),
], ids=["ban_khong_vuong", "o_co_gia_tri_2", "mang_1_chieu", "khong_phai_numpy",
        "nguoi_choi_0", "nguoi_choi_bool", "nguoi_choi_so_thuc"])
def test_from_board_rejects_invalid_input(board, player):
    with pytest.raises(ValueError):
        MCTSState.from_board(board, player, win_length=3)


def test_play_returns_independent_child():
    # Chuẩn bị: trạng thái bàn trống, đến lượt X.
    parent = MCTSState.from_board(np.zeros((11, 11), dtype=np.int8), 1)
    # Thực thi: tạo hai tầng con bằng play().
    child = parent.play((5, 5))
    grandchild = child.play((5, 6))
    # Kiểm chứng: mỗi tầng chỉ có nước của mình; tầng trên giữ nguyên.
    assert parent.board[5, 5] == 0 and child.board[5, 5] == 1
    assert (parent.current_player, parent.move_count, parent.last_action) == (1, 0, None)
    assert (child.current_player, child.move_count, child.last_action) == (-1, 1, (5, 5))
    assert child.board[5, 6] == 0 and grandchild.board[5, 6] == -1


@pytest.mark.parametrize("action", [(5, 5), (-1, 0), (11, 0), (0.5, 1), (True, 0), [0, 0], (0,), None],
                         ids=["o_da_co_quan", "ngoai_ban_am", "ngoai_ban_vuot_canh", "toa_do_so_thuc",
                              "toa_do_bool", "list_khong_phai_tuple", "thieu_toa_do", "nuoc_None"])
def test_invalid_step_is_atomic(action):
    # Chuẩn bị: X đã đi (5,5), đến lượt O; lưu bản sao để so.
    state = MCTSState.from_board(np.zeros((11, 11), dtype=np.int8), 1)
    state.step((5, 5))
    before = state.clone()
    # Thực thi: đi nước sai phải báo lỗi.
    with pytest.raises(ValueError):
        state.step(action)
    # Kiểm chứng: trạng thái không bị thay đổi một phần.
    np.testing.assert_array_equal(state.board, before.board)
    assert (state.current_player, state.move_count, state.last_action) == (-1, 1, (5, 5))


def test_terminal_state_has_no_moves():
    # Chuẩn bị: ván X thắng hàng ngang chạy trên engine.
    game = CaroGame()
    for action in SEQUENCES[0][1]:
        game.step(action)
    # Thực thi: chép trạng thái engine sang MCTS.
    state = MCTSState.from_game(game)
    # Kiểm chứng: đã kết thúc, không còn nước, đi tiếp bị từ chối.
    assert state.is_terminal() and state.result() == 1
    assert state.current_player == 1  # giữ người vừa đi như engine
    assert state.legal_actions() == [] and state.candidate_actions() == []
    with pytest.raises(ValueError, match="kết thúc"):
        state.step((5, 5))


# ---------------------------------------------------------------- nút cây

def _small_board():
    # X ở (5,5), O ở (5,6); đến lượt X.
    board = np.zeros((11, 11), dtype=np.int8)
    board[5, 5], board[5, 6] = 1, -1
    return board


def test_root_node_initial_fields():
    board = _small_board()
    root = MCTSNode.root(board, 1)
    assert root.parent is None and root.action is None
    assert root.player_just_moved == -1  # gốc: đối thủ của người đến lượt
    assert (root.visits, root.value, root.mean_value()) == (0, 0.0, 0.0)
    assert root.children == {}
    assert root.untried_actions == candidate_actions(board, DEFAULT_RADIUS)
    assert not root.is_terminal() and not root.is_fully_expanded()


def test_root_player_just_moved_when_o_to_move():
    board = np.zeros((11, 11), dtype=np.int8)
    board[5, 5] = 1
    assert MCTSNode.root(board, -1).player_just_moved == 1


_TERMINAL = [s for s in SEQUENCES if s[4] is not None]


@pytest.mark.parametrize("name,moves,size,k,expected", _TERMINAL, ids=[s[0] for s in _TERMINAL])
def test_terminal_root_player_just_moved_is_last_mover(name, moves, size, k, expected):
    # Chuẩn bị: chạy ván tới terminal trên engine; engine giữ current_player = người vừa đi.
    game = CaroGame(size, k)
    for action in moves:
        game.step(action)
    last_mover = game.current_player
    # Thực thi: dựng gốc theo hai đường — chép từ engine và chỉ từ bàn cờ
    # (truyền đối thủ của người vừa đi như thể "đến lượt").
    from_game = MCTSNode(MCTSState.from_game(game), radius=DEFAULT_RADIUS)
    from_board = MCTSNode(MCTSState.from_board(game.board, -last_mover, win_length=k))
    # Kiểm chứng: gốc terminal lấy current_player làm player_just_moved, không đổi dấu.
    for root in (from_game, from_board):
        assert root.is_terminal() and root.state.result() == expected
        assert root.state.current_player == last_mover
        assert root.player_just_moved == last_mover
        assert root.untried_actions == []


def test_expand_links_child_and_keeps_parent_unchanged():
    # Chuẩn bị: gốc từ thế X(5,5), O(5,6), đến lượt X; lưu bàn cờ và ứng viên ban đầu.
    board = _small_board()
    root = MCTSNode.root(board, 1)
    parent_board = root.state.board.copy()
    untried_before = list(root.untried_actions)
    # Thực thi: X mở rộng nước (4,4).
    child = root.expand((4, 4))
    # Kiểm chứng: liên kết, góc nhìn, thống kê ban đầu của con; cha và đầu vào không đổi.
    assert child.parent is root and child.action == (4, 4)
    assert root.children == {(4, 4): child}
    assert child.player_just_moved == 1  # X đã đi (4,4)
    assert child.state.current_player == -1
    assert child.state.board[4, 4] == 1
    assert (child.visits, child.value) == (0, 0.0)
    assert child.untried_actions == candidate_actions(child.state.board, DEFAULT_RADIUS)
    np.testing.assert_array_equal(root.state.board, parent_board)
    np.testing.assert_array_equal(board, _small_board())
    assert root.untried_actions == [a for a in untried_before if a != (4, 4)]
    # Thực thi và kiểm chứng tầng tiếp theo: O mở rộng (6,6) từ nút con.
    grandchild = child.expand((6, 6))
    assert grandchild.player_just_moved == -1 and grandchild.state.board[6, 6] == -1
    assert child.state.board[6, 6] == 0


def test_expand_rejects_unknown_or_repeated_action():
    root = MCTSNode.root(_small_board(), 1)
    root.expand((4, 4))
    for action in [(4, 4), (5, 5), (0, 0)]:  # đã mở rộng, đã có quân, ngoài tập ứng viên
        with pytest.raises(ValueError):
            root.expand(action)


@pytest.mark.parametrize("bad_action", [(5.0, 5), (5, np.float64(5.0))],
                         ids=["hang_so_thuc_5.0", "cot_numpy_float"])
def test_expand_wrong_type_keeps_valid_candidate(bad_action):
    # Chuẩn bị: X(4,4), O(4,5), đến lượt X; (5,5) là ứng viên hợp lệ.
    board = np.zeros((11, 11), dtype=np.int8)
    board[4, 4], board[4, 5] = 1, -1
    root = MCTSNode.root(board, 1)
    untried_before = list(root.untried_actions)
    assert (5, 5) in untried_before
    # Thực thi: tọa độ sai kiểu bằng (5, 5) về giá trị nhưng không phải số nguyên.
    with pytest.raises(ValueError):
        root.expand(bad_action)
    # Kiểm chứng: không tạo con, không mất ứng viên; vẫn mở rộng được (5, 5) đúng kiểu.
    assert root.children == {}
    assert root.untried_actions == untried_before
    child = root.expand((5, 5))
    assert root.children == {(5, 5): child} and (5, 5) not in root.untried_actions


def test_expand_normalizes_numpy_integer_action():
    # Tọa độ kiểu số nguyên NumPy hợp lệ; khóa trong cây được chuẩn hóa về int.
    root = MCTSNode.root(_small_board(), 1)
    child = root.expand((np.int64(4), np.int64(4)))
    assert child.action == (4, 4) and all(type(x) is int for x in child.action)
    assert list(root.children) == [(4, 4)] and (4, 4) not in root.untried_actions


def test_fully_expanded_after_all_candidates():
    root = MCTSNode.root(_small_board(), 1)
    for action in list(root.untried_actions):
        root.expand(action)
    assert root.is_fully_expanded()
    assert len(root.children) == len(candidate_actions(_small_board()))


def test_terminal_node_is_not_expanded():
    # Chuẩn bị: X có 4 quân liền ở hàng 0, đến lượt X.
    board = np.zeros((11, 11), dtype=np.int8)
    board[0, :4] = 1
    board[10, :4] = -1
    root = MCTSNode.root(board, 1)
    # Thực thi: X mở rộng nước thứ 5.
    child = root.expand((0, 4))
    # Kiểm chứng: nút con là terminal X thắng, không còn gì để mở rộng.
    assert child.is_terminal() and child.state.result() == 1
    assert child.untried_actions == [] and child.is_fully_expanded()
    with pytest.raises(ValueError, match="kết thúc"):
        child.expand((5, 5))


def test_node_requires_parent_and_action_together():
    state = MCTSState.from_board(np.zeros((11, 11), dtype=np.int8), 1)
    with pytest.raises(ValueError):
        MCTSNode(state, action=(5, 5))
