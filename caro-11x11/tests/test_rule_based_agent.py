"""Kiểm thử Rule-based — Giai đoạn 1: luật thắng ngay, chặn thua và nước dự phòng.

Thế cờ được dựng bằng CaroGame.step() qua chuỗi nước hợp lệ (engine không có API nạp bàn cờ).
Nước thắng/chặn kỳ vọng được tính tay theo hình học của chuỗi; trong ván đầy đủ, kết quả
được đối chiếu với một hàm kiểm tra độc lập (_wins_after) chứ không dùng chính hàm đang kiểm thử.

Các test dài chia ba phần: Chuẩn bị / Thực thi / Kiểm chứng. Test ngắn gộp chung ba phần.
"""

import dataclasses
from functools import partial

import numpy as np
import pytest

from agents.random_agent import RandomAgent
from agents.rule_based_agent import (RULE_BLOCK, RULE_FALLBACK, RULE_WIN, Decision, RuleBasedAgent,
                                     has_winner, is_winning_move, line_length, position_score,
                                     winning_moves)
from config.search_budget import SearchBudget
from core.game_engine import CaroGame
from evaluation.match import play_match

X, O = 1, -1

# Quân đệm: mọi đoạn 5 ô liên tiếp chứa tối đa 2 ô trong nhóm này, nên quân đệm không tạo đe dọa.
_FILLER_POOL = [(10, 0), (10, 3), (10, 6), (10, 9), (8, 0), (8, 3), (8, 6), (8, 9)]


def _neighbors(cells):
    return {(r + dr, c + dc) for r, c in cells for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def build(x_cells, o_cells, to_move, *, avoid=(), size=11, k=5):
    """Dựng thế cờ bằng engine: bù quân đệm cho đúng lượt rồi đi xen kẽ X, O."""
    x_cells, o_cells = list(x_cells), list(o_cells)
    blocked = _neighbors(list(x_cells) + list(o_cells) + list(avoid))
    pool = [cell for cell in _FILLER_POOL if cell not in blocked]
    extra = 0 if to_move == X else 1           # X đi trước: X đến lượt khi hai bên bằng quân
    n_x = max(len(x_cells), len(o_cells) + extra)
    while len(x_cells) < n_x:
        x_cells.append(pool.pop(0))
    while len(o_cells) < n_x - extra:
        o_cells.append(pool.pop(0))
    game = CaroGame(size, k)
    for i in range(len(x_cells) + len(o_cells)):
        game.step(x_cells[i // 2] if i % 2 == 0 else o_cells[i // 2])
    assert not game.is_terminal() and game.current_player == to_move
    return game


def place(owner, cells, other_cells, to_move, avoid=()):
    x_cells, o_cells = (cells, other_cells) if owner == X else (other_cells, cells)
    return build(x_cells, o_cells, to_move, avoid=avoid)


def _wins_after(board, cell, player, k=5):
    """Kiểm tra độc lập: đặt quân lên bản sao rồi tìm chuỗi ≥ k trên mọi hàng, cột, đường chéo."""
    if board[cell] != 0:
        return False
    b = board.copy()
    b[cell] = player
    n = b.shape[0]
    lines = [b[i, :] for i in range(n)] + [b[:, j] for j in range(n)]
    lines += [np.diagonal(b, d) for d in range(-n + 1, n)]
    lines += [np.diagonal(np.fliplr(b), d) for d in range(-n + 1, n)]
    target = "1" * k
    return any(target in "".join("1" if v == player else "0" for v in line) for line in lines)


def _oracle_wins(board, player):
    return {(r, c) for r in range(board.shape[0]) for c in range(board.shape[1])
            if _wins_after(board, (r, c), player)}


# Fixture: (tên, quân của bên có đe dọa, quân chặn của bên kia, các ô thắng tính tay).
LINES = [
    ("ngang_giua", [(5, 3), (5, 4), (5, 5), (5, 6)], [], [(5, 2), (5, 7)]),
    ("doc_giua", [(3, 5), (4, 5), (5, 5), (6, 5)], [], [(2, 5), (7, 5)]),
    ("cheo_chinh_giua", [(3, 3), (4, 4), (5, 5), (6, 6)], [], [(2, 2), (7, 7)]),
    ("cheo_phu_giua", [(3, 7), (4, 6), (5, 5), (6, 4)], [], [(2, 8), (7, 3)]),
    ("ngang_sat_mep_trai", [(0, 0), (0, 1), (0, 2), (0, 3)], [], [(0, 4)]),
    ("doc_sat_mep_duoi", [(7, 10), (8, 10), (9, 10), (10, 10)], [], [(6, 10)]),
    ("cheo_chinh_tu_goc", [(0, 0), (1, 1), (2, 2), (3, 3)], [], [(4, 4)]),
    ("cheo_phu_tu_goc", [(0, 10), (1, 9), (2, 8), (3, 7)], [], [(4, 6)]),
    ("ngang_khoang_trong_giua", [(5, 2), (5, 3), (5, 5), (5, 6)], [], [(5, 4)]),
    ("doc_khoang_trong_sat_bien", [(0, 4), (1, 4), (2, 4), (4, 4)], [], [(3, 4)]),
    # Lấp khoảng trống tạo sáu quân liên tiếp: luật freestyle vẫn tính là thắng.
    ("noi_thanh_sau_quan", [(5, 1), (5, 2), (5, 3), (5, 5), (5, 6)], [], [(5, 4)]),
    # Hai đầu đã bị bên kia chặn, lấp giữa vẫn đủ năm quân nên vẫn thắng.
    ("bi_chan_hai_dau", [(5, 1), (5, 2), (5, 4), (5, 5)], [(5, 0), (5, 6)], [(5, 3)]),
]
LINE_IDS = [case[0] for case in LINES]
SIDES = pytest.mark.parametrize("owner", [X, O], ids=["X", "O"])


# ---------------------------------------------------------------- hàm nhận diện

@SIDES
@pytest.mark.parametrize("name,cells,blockers,expected", LINES, ids=LINE_IDS)
def test_winning_moves_match_hand_count(owner, name, cells, blockers, expected):
    game = place(owner, cells, blockers, to_move=owner, avoid=expected)
    assert winning_moves(game.board, owner) == sorted(expected)
    assert winning_moves(game.board, -owner) == []
    assert set(expected) == _oracle_wins(game.board, owner)


def test_line_length_counts_both_sides_and_stops_at_opponent():
    # Chuẩn bị: X ở (5,2),(5,3),(5,5); O chặn ở (5,6). Ô thử (5,4) đang trống.
    game = build([(5, 2), (5, 3), (5, 5)], [(5, 6), (0, 0)], to_move=O)
    board = game.board
    # Thực thi và kiểm chứng: ngang 3 + ô thử = 4, dừng ở quân O; dọc chỉ có ô thử.
    assert line_length(board, 5, 4, X, (0, 1)) == 4
    assert line_length(board, 5, 4, X, (1, 0)) == 1
    # Với O: ô thử (5,7) nối quân (5,6) thành 2, dừng ở quân X (5,5).
    assert line_length(board, 5, 7, O, (0, 1)) == 2


def test_is_winning_move_false_on_occupied_cell():
    game = build([(5, 3), (5, 4), (5, 5), (5, 6)], [(0, 0), (0, 2), (0, 4), (5, 7)], to_move=X)
    assert not is_winning_move(game.board, (5, 7), X)   # ô đã có quân O
    assert is_winning_move(game.board, (5, 2), X)


def test_four_without_space_is_not_a_threat():
    # Bốn quân X bị O chặn cả hai đầu: không còn ô nào tạo năm quân.
    game = build([(5, 3), (5, 4), (5, 5), (5, 6)], [(5, 2), (5, 7), (0, 0)], to_move=O)
    assert winning_moves(game.board, X) == []
    assert _oracle_wins(game.board, X) == set()


@pytest.mark.parametrize("moves,expected", [
    ([(5, 1), (0, 0), (5, 2), (0, 2), (5, 3), (0, 4), (5, 4), (0, 6), (5, 5)], True),
    ([(5, 1), (0, 0), (5, 2), (0, 2), (5, 3), (0, 4), (5, 4)], False),
    ([], False),
], ids=["x_da_co_nam_quan", "x_moi_bon_quan", "ban_trong"])
def test_has_winner_matches_engine(moves, expected):
    game = CaroGame()
    for move in moves:
        game.step(move)
    assert has_winner(game.board) == expected == (game.result() is not None)


def test_position_score_counts_neighbors():
    # X ở (5,5),(4,4); O ở (5,6). Ô (4,5) kề 2 quân X và 1 quân O.
    game = build([(5, 5), (4, 4)], [(5, 6)], to_move=O)
    assert position_score(game.board, (4, 5), X) == 2 * 2 + 1
    assert position_score(game.board, (4, 5), O) == 2 * 1 + 2
    assert position_score(game.board, (0, 10), X) == 0


# ---------------------------------------------------------------- luật thắng ngay / chặn thua

@SIDES
@pytest.mark.parametrize("name,cells,blockers,expected", LINES, ids=LINE_IDS)
def test_agent_takes_immediate_win(owner, name, cells, blockers, expected):
    # Chuẩn bị: bên có đe dọa đang đến lượt.
    game = place(owner, cells, blockers, to_move=owner, avoid=expected)
    agent = RuleBasedAgent()
    # Thực thi
    action = agent.choose_action(game.board, owner)
    # Kiểm chứng: nước nằm trong tập thắng tính tay, luật đúng và engine xác nhận thắng.
    decision = agent.explain_last_action()
    assert action in expected
    assert decision.rule == RULE_WIN and set(decision.acceptable) == set(expected)
    game.step(action)
    assert game.result() == owner


@SIDES
@pytest.mark.parametrize("name,cells,blockers,expected", LINES, ids=LINE_IDS)
def test_agent_blocks_opponent_threat(owner, name, cells, blockers, expected):
    # Chuẩn bị: bên kia đến lượt và không có nước thắng của riêng mình.
    defender = -owner
    game = place(owner, cells, blockers, to_move=defender, avoid=expected)
    agent = RuleBasedAgent()
    # Thực thi
    action = agent.choose_action(game.board, defender)
    # Kiểm chứng
    decision = agent.explain_last_action()
    assert action in expected
    assert decision.rule == RULE_BLOCK and set(decision.acceptable) == set(expected)
    game.step(action)
    if len(expected) == 1:
        # Đe dọa duy nhất: sau khi chặn, bên tấn công không còn nước thắng ngay.
        assert _oracle_wins(game.board, owner) == set()
    else:
        # Đe dọa kép (bốn mở): chặn được một đầu, đầu còn lại vẫn thắng — giới hạn tự nhiên của luật.
        assert "đe dọa kép" in decision.reason
        assert _oracle_wins(game.board, owner) == set(expected) - {action}


@pytest.mark.parametrize("to_move", [X, O], ids=["X_den_luot", "O_den_luot"])
def test_win_has_priority_over_block(to_move):
    # Chuẩn bị: bên đến lượt có bốn quân ở hàng 0 (thắng tại (0,4), sát mép); bên kia có bốn quân
    # ở hàng 5 (thắng tại (5,5), giữa bàn). Ô (5,5) xếp hạng vị trí tốt hơn, nên nếu luật chặn
    # chạy trước, tác tử sẽ chọn (5,5) thay vì nước thắng.
    mover = [(0, 0), (0, 1), (0, 2), (0, 3)]
    other = [(5, 3), (5, 4), (5, 6), (5, 7)]
    x_cells, o_cells = (mover, other) if to_move == X else (other, mover)
    game = build(x_cells, o_cells, to_move, avoid=[(0, 4), (5, 5)])
    assert winning_moves(game.board, to_move) == [(0, 4)]
    assert winning_moves(game.board, -to_move) == [(5, 5)]
    agent = RuleBasedAgent()
    # Thực thi
    action = agent.choose_action(game.board, to_move)
    # Kiểm chứng
    assert action == (0, 4) and agent.explain_last_action().rule == RULE_WIN
    game.step(action)
    assert game.result() == to_move


def test_open_three_is_not_an_immediate_threat():
    # O có ba quân mở (5,4)-(5,6): chưa phải đe dọa thắng ngay, Giai đoạn 1 dùng tầng dự phòng.
    game = build([], [(5, 4), (5, 5), (5, 6)], to_move=X)
    agent = RuleBasedAgent()
    agent.choose_action(game.board, X)
    assert agent.explain_last_action().rule == RULE_FALLBACK


def test_block_does_not_depend_on_candidate_radius():
    # Chuẩn bị: bán kính ứng viên 1; O có bốn quân ở hàng 0, ô thắng (0,4).
    # Tầng chặn quét toàn bộ ô trống, độc lập với bán kính ứng viên của tầng dự phòng.
    # (Ô thắng luôn kề chuỗi quân nên cũng nằm trong bán kính 1; quét toàn bàn là bảo đảm, không phải tối ưu.)
    game = build([(10, 0), (10, 3), (10, 6), (8, 0)], [(0, 0), (0, 1), (0, 2), (0, 3)], to_move=X,
                 avoid=[(0, 4)])
    agent = RuleBasedAgent(radius=1)
    assert agent.choose_action(game.board, X) == (0, 4)
    assert agent.explain_last_action().rule == RULE_BLOCK


def test_smaller_win_length_on_small_board():
    # Bàn 7×7, thắng 4: X có ba quân (3,1)-(3,3), ô thắng (3,0) và (3,4).
    game = CaroGame(7, 4)
    for move in [(3, 1), (0, 0), (3, 2), (0, 6), (3, 3), (6, 0)]:
        game.step(move)
    agent = RuleBasedAgent(win_length=4)
    action = agent.choose_action(game.board, X)
    assert action in {(3, 0), (3, 4)} and agent.explain_last_action().rule == RULE_WIN


# ---------------------------------------------------------------- trường hợp biên và hợp đồng API

def test_empty_board_plays_center_and_counts_nodes():
    agent = RuleBasedAgent()
    action = agent.choose_action(CaroGame().board, X)
    decision = agent.explain_last_action()
    assert action == (5, 5)
    assert decision.rule == RULE_FALLBACK and decision.acceptable == ((5, 5),)
    # 121 ô trống ở tầng thắng + 121 ở tầng chặn + 1 ứng viên (tâm) ở tầng dự phòng.
    assert agent.stats["nodes_visited"] == 121 + 121 + 1
    assert agent.stats["simulations"] == 0 and agent.stats["pruned_nodes"] == 0


def test_nodes_visited_stops_at_triggered_rule():
    # X có bốn quân (5,3)-(5,6); O chỉ có quân đệm rời rạc.
    agent = RuleBasedAgent()
    # X đến lượt: tầng thắng kích hoạt, chỉ quét toàn bộ ô trống một lần.
    game = build([(5, 3), (5, 4), (5, 5), (5, 6)], [(0, 0), (0, 2), (0, 4), (0, 6)], to_move=X)
    agent.choose_action(game.board, X)
    assert agent.stats["nodes_visited"] == int(np.count_nonzero(game.board == 0))
    # O đến lượt: tầng thắng không có nước, tầng chặn kích hoạt, quét toàn bộ ô trống hai lần.
    game = build([(5, 3), (5, 4), (5, 5), (5, 6)], [(0, 0), (0, 2), (0, 4)], to_move=O)
    agent.choose_action(game.board, O)
    assert agent.stats["nodes_visited"] == 2 * int(np.count_nonzero(game.board == 0))


def test_decide_called_twice_does_not_accumulate_stats():
    agent = RuleBasedAgent()
    board = CaroGame().board
    agent.decide(board, X)
    agent.decide(board, X)
    assert agent.stats["nodes_visited"] == 243


def _single_empty_cell_game():
    """Bàn còn đúng một ô trống, chưa ai thắng.

    Mẫu X khi (c + 2r) mod 4 < 2: hàng XXOO, cột xen kẽ, hai đường chéo tối đa hai quân liền nhau,
    nên không có chuỗi năm. Mẫu có 61 X và 60 O; bỏ trống một ô X để hai bên 60 quân, X đến lượt.
    """
    xs = [(r, c) for r in range(11) for c in range(11) if (c + 2 * r) % 4 < 2]
    os_ = [(r, c) for r in range(11) for c in range(11) if (c + 2 * r) % 4 >= 2]
    last = xs.pop()
    return build(xs, os_, to_move=X), last


def test_single_empty_cell_returns_it():
    # Chuẩn bị: 120 quân, một ô trống, ván chưa kết thúc.
    game, last = _single_empty_cell_game()
    assert game.legal_actions() == [last]
    agent = RuleBasedAgent()
    # Thực thi
    action = agent.choose_action(game.board, game.current_player)
    # Kiểm chứng: trả về ô duy nhất, ván kết thúc hòa.
    assert action == last
    game.step(action)
    assert game.result() == 0


def test_does_not_modify_board():
    # Gọi trực tiếp get_action (bỏ qua bản sao của BaseAgent) trên mảng chỉ đọc:
    # nếu có thao tác ghi, numpy sẽ báo lỗi.
    game = build([(5, 3), (5, 4), (5, 5)], [(4, 4), (6, 6), (0, 0)], to_move=X)
    board = game.board.copy()
    board.setflags(write=False)
    before = board.copy()
    agent = RuleBasedAgent()
    agent.get_action(board, X)
    winning_moves(board, O)
    has_winner(board)
    np.testing.assert_array_equal(board, before)


def test_returns_python_int_tuple():
    game = build([(5, 5)], [], to_move=O)
    action = RuleBasedAgent().choose_action(game.board.astype(np.int64), np.int64(O))
    assert isinstance(action, tuple) and len(action) == 2
    assert all(type(x) is int for x in action)


def test_rejects_finished_game():
    game = CaroGame()
    for move in [(5, 1), (0, 0), (5, 2), (0, 2), (5, 3), (0, 4), (5, 4), (0, 6), (5, 5)]:
        game.step(move)
    assert game.is_terminal()
    with pytest.raises(ValueError):
        RuleBasedAgent().choose_action(game.board, O)


@pytest.mark.parametrize("kwargs", [{"radius": 0}, {"radius": True}, {"radius": 1.5},
                                    {"win_length": 0}, {"win_length": False}],
                         ids=["ban_kinh_0", "ban_kinh_bool", "ban_kinh_so_thuc", "do_dai_0", "do_dai_bool"])
def test_rejects_invalid_config(kwargs):
    with pytest.raises(ValueError):
        RuleBasedAgent(**kwargs)


@pytest.mark.parametrize("board,player", [
    (np.zeros((4, 4), dtype=np.int8), X),        # độ dài thắng 5 lớn hơn bàn 4×4
    (np.zeros((11, 11), dtype=np.int8), 0),
    (np.zeros((11, 11), dtype=np.int8), True),
    (np.full((11, 11), 2, dtype=np.int8), X),
    (np.zeros((11, 10), dtype=np.int8), X),
], ids=["ban_nho_hon_do_dai_thang", "nguoi_choi_0", "nguoi_choi_bool", "gia_tri_la", "ban_khong_vuong"])
def test_decide_rejects_invalid_input(board, player):
    with pytest.raises(ValueError):
        RuleBasedAgent().decide(board, player)


def test_decision_is_immutable_copy():
    agent = RuleBasedAgent()
    assert agent.explain_last_action() is None
    action = agent.choose_action(CaroGame().board, X)
    decision = agent.explain_last_action()
    assert isinstance(decision, Decision) and decision.action == action
    assert isinstance(decision.acceptable, tuple)
    with pytest.raises(dataclasses.FrozenInstanceError):
        decision.action = (0, 0)


def test_deterministic_on_same_position():
    game = build([(5, 5), (4, 6)], [(5, 6), (6, 6)], to_move=X)
    actions = {RuleBasedAgent().choose_action(game.board, X) for _ in range(3)}
    assert len(actions) == 1


# ---------------------------------------------------------------- ván hoàn chỉnh

@pytest.mark.parametrize("seed", range(4))
@SIDES
def test_full_game_against_random_follows_rules(seed, owner):
    # Chuẩn bị: Rule-based cầm owner, Random cầm bên còn lại.
    game = CaroGame()
    rule, rand = RuleBasedAgent(), RandomAgent(seed=seed)
    while not game.is_terminal():
        player = game.current_player
        if player == owner:
            own = _oracle_wins(game.board, player)
            threats = _oracle_wins(game.board, -player)
            # Thực thi
            action = rule.choose_action(game.board, player)
            # Kiểm chứng từng nước bằng hàm kiểm tra độc lập.
            if own:
                assert action in own
            elif threats:
                assert action in threats
        else:
            action = rand.choose_action(game.board, player)
        game.step(action)
    assert game.result() in (-1, 0, 1)


def test_full_game_rule_based_self_play_completes():
    game = CaroGame()
    agents = {X: RuleBasedAgent("Rule X"), O: RuleBasedAgent("Rule O")}
    while not game.is_terminal():
        game.step(agents[game.current_player].choose_action(game.board, game.current_player))
    assert game.result() in (-1, 0, 1)


def test_play_match_runner():
    # Runner chạy mỗi agent ở tiến trình riêng; Rule-based không dùng mô phỏng.
    record, logs = play_match(partial(RuleBasedAgent), partial(RandomAgent, seed=3),
                              agent_x="Rule-based", agent_o="Ngẫu nhiên",
                              budget=SearchBudget(mode="time", time_limit_s=2.0))
    assert record.reason == "completed"
    rule_logs = [log for log in logs if log["player"] == X]
    assert rule_logs and all(log["status"] == "move" for log in rule_logs)
    assert all(log["stats"]["simulations"] == 0 and log["stats"]["nodes_visited"] > 0 for log in rule_logs)
