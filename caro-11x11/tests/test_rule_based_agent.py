"""Kiểm thử Rule-based - Giai đoạn 1 (thắng ngay, chặn thua, dự phòng) và Giai đoạn 2 (mẫu hình, thứ tự luật).

Các test Giai đoạn 1 dùng cấu hình mặc định ``use_patterns=False``; phần Giai đoạn 2 ở cuối file.

Thế cờ được dựng bằng CaroGame.step() qua chuỗi nước hợp lệ (engine không có API nạp bàn cờ).
Nước thắng/chặn kỳ vọng được tính tay theo hình học của chuỗi; trong ván đầy đủ, kết quả
được đối chiếu với một hàm kiểm tra độc lập (_wins_after) chứ không dùng chính hàm đang kiểm thử.

Các test dài chia ba phần: Chuẩn bị / Thực thi / Kiểm chứng. Test ngắn gộp chung ba phần.
"""

import dataclasses
import re
from functools import partial

import numpy as np
import pytest

from agents.random_agent import RandomAgent
from agents.rule_based_agent import (BASELINE_RULES, PATTERN_BROKEN_FOUR, PATTERN_FIVE, PATTERN_NONE,
                                     PATTERN_OPEN_FOUR, PATTERN_OPEN_THREE, RULE_ATTACK_BROKEN_FOUR,
                                     RULE_ATTACK_OPEN_FOUR, RULE_ATTACK_OPEN_THREE, RULE_BLOCK,
                                     RULE_BLOCK_OPEN_FOUR, RULE_BLOCK_OPEN_THREE, RULE_FALLBACK, RULE_WIN,
                                     RULES, Decision, RuleBasedAgent, classify_move, has_winner,
                                     is_winning_move, line_length, new_winning_cells, open_three_followups,
                                     position_score, rank_key, winning_moves)
from config.search_budget import SearchBudget
from core.game_engine import CaroGame
from core.game_utils import candidate_actions
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


# ---------------------------------------------------------------- ngân sách thời gian (đồng hồ giả)

class _FakeClock:
    """Đồng hồ giả: mỗi lần đọc tăng 1 ms, không phụ thuộc tốc độ máy."""

    def __init__(self, start=100.0, step=0.001):
        self.now, self.step = start, step

    def __call__(self):
        self.now += self.step
        return self.now


def _use_clock(monkeypatch, clock):
    # BaseAgent đặt deadline và tác tử kiểm tra mốc dừng bằng cùng một đồng hồ.
    monkeypatch.setattr("agents.base_agent.perf_counter", clock)
    monkeypatch.setattr("agents.rule_based_agent.perf_counter", clock)


def _time_agent(limit):
    return RuleBasedAgent(budget=SearchBudget(mode="time", time_limit_s=limit))


def test_time_mode_past_deadline_returns_prepared_fallback(monkeypatch):
    # Chuẩn bị: đồng hồ của tác tử luôn báo đã quá mốc dừng; X ở tâm, O ở (6,6).
    monkeypatch.setattr("agents.rule_based_agent.perf_counter", lambda: float("inf"))
    game = build([(5, 5)], [(6, 6)], to_move=X)
    agent = _time_agent(2.0)
    # Thực thi
    action = agent.choose_action(game.board, X)
    # Kiểm chứng: không xét ô nào; trả nước dự phòng chuẩn bị sẵn = ứng viên gần tâm nhất (tính tay:
    # tâm đã có quân, bốn ô kề thẳng cách Manhattan 1, hàng nhỏ nhất là (4,5)).
    decision = agent.explain_last_action()
    assert action == (4, 5)
    assert decision.rule == RULE_FALLBACK and decision.timed_out
    assert agent.stats["nodes_visited"] == 0 and "hết thời gian sau 0 ô" in decision.reason


def test_time_mode_stops_during_win_scan(monkeypatch):
    # Chuẩn bị: thế yên tĩnh; hạn 0,15 giây − biên 0,1 → mốc dừng sau khoảng 50 lần đọc đồng hồ giả,
    # tức là giữa lúc quét nước thắng (bàn còn 119 ô trống).
    clock = _FakeClock()
    _use_clock(monkeypatch, clock)
    game = build([(5, 5)], [(6, 6)], to_move=X)
    n_empty = int(np.count_nonzero(game.board == 0))
    agent = _time_agent(0.15)
    start = clock.now
    # Thực thi
    action = agent.choose_action(game.board, X)
    # Kiểm chứng: dừng trước deadline, số ô đã xét đúng bằng số ghi trong lý do và nhỏ hơn một lượt quét.
    decision = agent.explain_last_action()
    examined = agent.stats["nodes_visited"]
    assert clock.now < start + 0.15
    assert decision.timed_out and decision.rule == RULE_FALLBACK and action == (4, 5)
    assert 0 < examined < n_empty
    assert f"hết thời gian sau {examined} ô đã xét" in decision.reason


def test_time_mode_keeps_block_found_before_stop(monkeypatch):
    # Chuẩn bị: O có bốn quân ở hàng 0, ô thắng (0,4) là ô trống đầu tiên theo thứ tự quét.
    # Hạn 0,25 giây → mốc dừng sau khoảng 150 lần đọc: quét xong tầng thắng (113 ô), dừng giữa tầng chặn.
    clock = _FakeClock()
    _use_clock(monkeypatch, clock)
    game = build([], [(0, 0), (0, 1), (0, 2), (0, 3)], to_move=X, avoid=[(0, 4)])
    n_empty = int(np.count_nonzero(game.board == 0))
    agent = _time_agent(0.25)
    # Thực thi
    action = agent.choose_action(game.board, X)
    # Kiểm chứng: vẫn chặn đúng ô đã tìm thấy, ghi nhận hết giờ và số ô thực sự đã xét.
    decision = agent.explain_last_action()
    assert action == (0, 4) and decision.rule == RULE_BLOCK and decision.timed_out
    assert n_empty < agent.stats["nodes_visited"] < 2 * n_empty


def test_time_mode_stops_while_scoring_candidates(monkeypatch):
    # Chuẩn bị: thế yên tĩnh 119 ô trống; mốc dừng sau khoảng 238 + 10 lần đọc → dừng giữa lúc chấm điểm.
    clock = _FakeClock()
    _use_clock(monkeypatch, clock)
    game = build([(5, 5)], [(6, 6)], to_move=X)
    n_empty = int(np.count_nonzero(game.board == 0))
    candidates = candidate_actions(game.board, 2)
    agent = _time_agent(0.1 + (2 * n_empty + 10) * 0.001)
    # Thực thi
    action = agent.choose_action(game.board, X)
    # Kiểm chứng: chọn trong số ứng viên đã chấm, lý do ghi số ứng viên đã chấm trên tổng.
    decision = agent.explain_last_action()
    scored = agent.stats["nodes_visited"] - 2 * n_empty
    assert decision.rule == RULE_FALLBACK and decision.timed_out
    assert 0 < scored < len(candidates)
    assert f"chấm {scored}/{len(candidates)} ứng viên" in decision.reason
    assert action in candidates[:scored]


def test_time_mode_with_enough_time_matches_fixed(monkeypatch):
    clock = _FakeClock()
    _use_clock(monkeypatch, clock)
    game = build([(5, 3), (5, 4), (5, 5)], [(4, 4), (6, 6), (0, 0)], to_move=X)
    timed = _time_agent(2.0)
    fixed = RuleBasedAgent()
    assert timed.choose_action(game.board, X) == fixed.choose_action(game.board, X)
    assert not timed.explain_last_action().timed_out
    assert timed.stats["nodes_visited"] == fixed.stats["nodes_visited"]


def test_fixed_mode_ignores_clock(monkeypatch):
    # Chế độ fixed không có mốc dừng: đồng hồ báo quá hạn cũng không làm tác tử dừng sớm.
    monkeypatch.setattr("agents.rule_based_agent.perf_counter", lambda: float("inf"))
    agent = RuleBasedAgent()
    agent.choose_action(CaroGame().board, X)
    assert not agent.explain_last_action().timed_out
    assert agent.stats["nodes_visited"] == 243


@pytest.mark.parametrize("stale_deadline", [None, 0.0], ids=["deadline_None", "deadline_cu_da_qua"])
def test_direct_decide_in_time_mode_sets_its_own_deadline(monkeypatch, stale_deadline):
    # Gọi decide() trực tiếp: tự đặt deadline cho lần gọi, không dùng deadline cũ của lượt trước.
    clock = _FakeClock()
    _use_clock(monkeypatch, clock)
    agent = _time_agent(2.0)
    agent.deadline = stale_deadline
    decision = agent.decide(CaroGame().board, X)
    assert decision.action == (5, 5) and not decision.timed_out


def test_internal_decide_without_deadline_in_time_mode_raises():
    with pytest.raises(ValueError, match="deadline"):
        _time_agent(2.0)._decide(CaroGame().board, X, None)


@pytest.mark.parametrize("margin", [-0.1, True, float("inf"), "0.1"],
                         ids=["am", "bool", "vo_cuc", "chuoi"])
def test_rejects_invalid_time_margin(margin):
    with pytest.raises(ValueError):
        RuleBasedAgent(time_margin_s=margin)


# ---Giai đoạn 2: mẫu hình và thứ tự luật

def _pattern_agent(**kwargs):
    return RuleBasedAgent(use_patterns=True, **kwargs)


def _oracle_new_wins(board, action, player):
    """Kiểm tra độc lập: ô trước nước thử chưa thắng, sau nước thử (đặt trên bản sao) thì thắng."""
    after = board.copy()
    after[action] = player
    return {(r, c) for r in range(board.shape[0]) for c in range(board.shape[1])
            if (r, c) != action and _wins_after(after, (r, c), player) and not _wins_after(board, (r, c), player)}


# Bốn hướng; điểm neo chọn sao cho các ô thứ -1..7 của đường đều nằm trong bàn.
DIRS = {"ngang": ((3, 2), (0, 1)), "doc": ((2, 3), (1, 0)),
        "cheo_chinh": ((2, 2), (1, 1)), "cheo_phu": ((2, 8), (1, -1))}
# Điểm neo sát mép: ô thứ -1 của đường nằm ngoài bàn.
EDGE_DIRS = {"ngang": ((5, 0), (0, 1)), "doc": ((0, 5), (1, 0)),
             "cheo_chinh": ((0, 3), (1, 1)), "cheo_phu": ((0, 7), (1, -1))}
DIR_IDS = list(DIRS)


def _cells(anchor, step, idx):
    (r0, c0), (dr, dc) = anchor, step
    return [(r0 + i * dr, c0 + i * dc) for i in idx]


# Mẫu trên một đường: (tên, ô của mình, ô đối thủ, ô nước thử, mẫu kỳ vọng, ô kỳ vọng) theo chỉ số trên đường.
# Ô kỳ vọng là ô thắng mới (bốn) hoặc nước tiếp theo (ba mở), tính tay theo hình học của đoạn 5 ô.
LINE_PATTERNS = [
    ("bon_mo_hai_dau", [1, 2, 3], [], 4, PATTERN_OPEN_FOUR, [0, 5]),
    ("bon_bi_chan_mot_dau", [1, 2, 3], [0], 4, PATTERN_BROKEN_FOUR, [5]),
    ("bon_co_khe", [1, 2, 4], [], 5, PATTERN_BROKEN_FOUR, [3]),
    ("doi_thu_ngat_giua", [1, 2], [3], 4, PATTERN_NONE, []),
    ("ba_mo_lien", [2, 3], [], 4, PATTERN_OPEN_THREE, [1, 5]),
    ("ba_mo_co_khe", [2, 3], [], 5, PATTERN_OPEN_THREE, [4]),
    ("ba_bi_chan_mot_dau", [1, 2], [0], 3, PATTERN_NONE, []),
    ("hai_quan_cach_xa", [0], [], 4, PATTERN_NONE, []),
]
EDGE_PATTERNS = [
    ("bon_sat_mep", [0, 1, 2], [], 3, PATTERN_BROKEN_FOUR, [4]),
    ("ba_sat_mep", [0, 1], [], 2, PATTERN_NONE, []),
]


def _line_case(owner, dirs, direction, mine, theirs):
    anchor, step = dirs[direction]
    own = _cells(anchor, step, mine)
    opp = _cells(anchor, step, theirs)
    game = place(owner, own, opp, to_move=owner, avoid=_cells(anchor, step, range(-1, 8)))
    return game, anchor, step


@SIDES
@pytest.mark.parametrize("direction", DIR_IDS)
@pytest.mark.parametrize("name,mine,theirs,move,pattern,expected", LINE_PATTERNS,
                         ids=[case[0] for case in LINE_PATTERNS])
def test_line_patterns_four_directions(owner, direction, name, mine, theirs, move, pattern, expected):
    # Chuẩn bị
    game, anchor, step = _line_case(owner, DIRS, direction, mine, theirs)
    action = _cells(anchor, step, [move])[0]
    want = set(_cells(anchor, step, expected))
    # Thực thi
    got = classify_move(game.board, action, owner)
    # Kiểm chứng: loại mẫu, ô thắng mới (đối chiếu cả hàm kiểm tra độc lập) và nước tiếp theo của ba mở.
    assert got == pattern
    if pattern in (PATTERN_OPEN_FOUR, PATTERN_BROKEN_FOUR):
        assert set(new_winning_cells(game.board, action, owner)) == want == _oracle_new_wins(game.board, action, owner)
    if pattern == PATTERN_OPEN_THREE:
        assert set(open_three_followups(game.board, action, owner)) == want
        assert not new_winning_cells(game.board, action, owner)


@SIDES
@pytest.mark.parametrize("direction", DIR_IDS)
@pytest.mark.parametrize("name,mine,theirs,move,pattern,expected", EDGE_PATTERNS,
                         ids=[case[0] for case in EDGE_PATTERNS])
def test_edge_patterns_four_directions(owner, direction, name, mine, theirs, move, pattern, expected):
    game, anchor, step = _line_case(owner, EDGE_DIRS, direction, mine, theirs)
    action = _cells(anchor, step, [move])[0]
    assert classify_move(game.board, action, owner) == pattern
    assert set(new_winning_cells(game.board, action, owner)) == set(_cells(anchor, step, expected))


def test_double_threat_across_two_lines_counts_distinct_cells():
    # O có ba quân ở hàng 0 và ba quân ở cột 0; O đi (0,0) tạo hai ô thắng ở hai hướng khác nhau.
    game = build([(5, 5), (6, 7), (9, 9)], [(0, 1), (0, 2), (0, 3), (1, 0), (2, 0), (3, 0)], to_move=O)
    assert new_winning_cells(game.board, (0, 0), O) == ((0, 4), (4, 0))
    assert classify_move(game.board, (0, 0), O) == PATTERN_OPEN_FOUR


def test_cell_already_winning_is_not_new():
    # X đã có bốn quân (5,1)-(5,4), ô (5,5) thắng sẵn. Nước (5,6) cũng dẫn tới (5,5) nhưng ô đó không mới.
    game = build([(5, 1), (5, 2), (5, 3), (5, 4)], [(0, 0), (0, 2), (0, 4), (0, 6)], to_move=X)
    assert is_winning_move(game.board, (5, 5), X)
    assert new_winning_cells(game.board, (5, 6), X) == ()


def test_existing_open_four_move_does_not_make_neighbor_an_open_three():
    # X có ba mở (5,3)-(5,5): (5,2) và (5,6) vốn đã tạo được bốn mở. Nước (4,6) không góp gì vào đó,
    # nên không được tính là ba mở chỉ vì sau nó vẫn còn các nước bốn mở cũ.
    game = build([(5, 3), (5, 4), (5, 5)], [(0, 0), (0, 2), (0, 4)], to_move=X)
    assert len(new_winning_cells(game.board, (5, 6), X)) == 2
    assert open_three_followups(game.board, (4, 6), X) == ()
    assert classify_move(game.board, (4, 6), X) == PATTERN_NONE


def test_classify_five_and_occupied():
    game = build([(5, 1), (5, 2), (5, 3), (5, 4)], [(0, 0), (0, 2), (0, 4), (0, 6)], to_move=X)
    assert classify_move(game.board, (5, 5), X) == PATTERN_FIVE
    assert classify_move(game.board, (5, 1), X) == PATTERN_NONE
    assert new_winning_cells(game.board, (5, 1), X) == () and open_three_followups(game.board, (5, 1), X) == ()


@pytest.mark.parametrize("seed", range(3))
def test_new_winning_cells_matches_oracle_on_random_positions(seed):
    # Chuẩn bị: thế cờ giữa ván từ hai RandomAgent.
    game, rand = CaroGame(), RandomAgent(seed=seed)
    for _ in range(30):
        game.step(rand.choose_action(game.board, game.current_player))
        if game.is_terminal():
            pytest.skip("ván kết thúc sớm")
    board = game.board
    # Thực thi và kiểm chứng trên mọi ứng viên bán kính 1, cho cả hai bên.
    for action in candidate_actions(board, 1):
        for player in (X, O):
            assert set(new_winning_cells(board, action, player)) == _oracle_new_wins(board, action, player)


# ---- thứ tự luật và xung đột

def _decide(owner, mine, theirs):
    game = place(owner, mine, theirs, to_move=owner)
    return game, _pattern_agent().decide(game.board, owner)


@SIDES
def test_win_beats_everything_with_patterns(owner):
    _, decision = _decide(owner, [(5, 3), (5, 4), (5, 5), (5, 6)], [(0, 0), (0, 1), (0, 2), (2, 9)])
    assert decision.rule == RULE_WIN and set(decision.acceptable) == {(5, 2), (5, 7)}


@SIDES
def test_block_urgent_beats_attack_open_four(owner):
    # Mình có ba mở (tạo được bốn mở) nhưng đối thủ có bốn quân, ô thắng (0,4): phải chặn.
    _, decision = _decide(owner, [(5, 3), (5, 4), (5, 5)], [(0, 0), (0, 1), (0, 2), (0, 3)])
    assert decision.rule == RULE_BLOCK and decision.action == (0, 4)


@SIDES
def test_attack_open_four_beats_block_open_four(owner):
    # Mình: ba mở ở hàng 5; đối thủ: ba quân dọc ở cột 8, có nước (4,8) tạo bốn mở.
    # Nước bốn mở của mình (tính tay): (5,2) và (5,6); cùng 2 ô thắng, (5,6) gần tâm hơn.
    _, decision = _decide(owner, [(5, 3), (5, 4), (5, 5)], [(1, 8), (2, 8), (3, 8)])
    assert decision.rule == RULE_ATTACK_OPEN_FOUR
    assert set(decision.acceptable) == {(5, 2), (5, 6)} and decision.action == (5, 6)


@SIDES
def test_block_open_four_beats_attack_broken_four(owner):
    # Mình: ba quân bị chặn trái ở (5,2) nên chỉ tạo được bốn một ô thắng; đối thủ có nước bốn mở (4,8).
    # Nước làm đối thủ hết bốn mở (tính tay): (0,8), (4,8), (5,8); (4,8) có điểm vị trí 1 và gần tâm nhất.
    _, decision = _decide(owner, [(5, 3), (5, 4), (5, 5)], [(5, 2), (1, 8), (2, 8), (3, 8)])
    assert decision.rule == RULE_BLOCK_OPEN_FOUR
    assert set(decision.acceptable) == {(0, 8), (4, 8), (5, 8)} and decision.action == (4, 8)


@SIDES
def test_attack_broken_four_beats_open_three_rules(owner):
    _, decision = _decide(owner, [(5, 3), (5, 4), (5, 5)], [(5, 2)])
    assert decision.rule == RULE_ATTACK_BROKEN_FOUR
    assert set(decision.acceptable) == {(5, 6), (5, 7)} and decision.action == (5, 6)


@SIDES
def test_attack_open_three_beats_block_open_three(owner):
    # Mình: hai quân (5,4),(5,5). Nước ba mở (tính tay): (5,2),(5,3),(5,6),(5,7); (5,3) và (5,6) có
    # hai nước tiếp theo, (5,2) và (5,7) chỉ có một; giữa (5,3) và (5,6) thì (5,6) gần tâm hơn.
    _, decision = _decide(owner, [(5, 4), (5, 5)], [(2, 8), (3, 8)])
    assert decision.rule == RULE_ATTACK_OPEN_THREE
    assert set(decision.acceptable) == {(5, 2), (5, 3), (5, 6), (5, 7)} and decision.action == (5, 6)


@SIDES
def test_block_open_three_takes_cell_leaving_fewest(owner):
    # Đối thủ: hai quân (5,4),(5,5), nước ba mở (5,2),(5,3),(5,6),(5,7). Chiếm (5,3) hoặc (5,6) thì đối thủ
    # hết nước ba mở trên hàng này; chiếm (5,2) hoặc (5,7) thì vẫn còn.
    _, decision = _decide(owner, [], [(5, 4), (5, 5)])
    assert decision.rule == RULE_BLOCK_OPEN_THREE
    assert set(decision.acceptable) == {(5, 3), (5, 6)} and decision.action == (5, 6)


def test_tie_break_strength_comes_before_position_score():
    # Quy tắc phá hòa đã công bố: độ mạnh của mẫu (số ô thắng mới hoặc số nước tiếp theo) xét trước rank_key.
    game = build([(5, 5)], [(6, 6)], to_move=X)
    board = game.board
    near, far = (4, 5), (0, 0)
    assert rank_key(board, near, X) < rank_key(board, far, X)
    strength = {near: 1, far: 2}.get
    decision = RuleBasedAgent._pick(board, X, RULE_ATTACK_OPEN_THREE, [near, far], "thử", False, 0, strength)
    assert decision.action == far
    assert RuleBasedAgent._pick(board, X, RULE_FALLBACK, [near, far], "thử", False).action == near


def test_quiet_position_falls_back_like_baseline():
    game = build([(5, 5)], [(6, 6)], to_move=X)
    full, base = _pattern_agent().decide(game.board, X), RuleBasedAgent().decide(game.board, X)
    assert full.rule == base.rule == RULE_FALLBACK and full.action == base.action


@pytest.mark.parametrize("mine,theirs", [
    ([(5, 3), (5, 4), (5, 5)], [(1, 8), (2, 8), (3, 8)]),
    ([(5, 3), (5, 4), (5, 5)], [(5, 2), (1, 8), (2, 8), (3, 8)]),
    ([(5, 4), (5, 5)], [(2, 8), (3, 8)]),
    ([], [(5, 4), (5, 5)]),
], ids=["bon_mo", "chan_bon_mo", "ba_mo", "chan_ba_mo"])
def test_use_patterns_false_keeps_baseline_rules(mine, theirs):
    # Cùng thế cờ: tắt pattern chỉ còn ba tầng của Giai đoạn 1.
    game = place(X, mine, theirs, to_move=X)
    base = RuleBasedAgent(use_patterns=False)
    assert base.decide(game.board, X).rule in BASELINE_RULES and base.rules == BASELINE_RULES
    assert _pattern_agent().rules == RULES


def test_opponent_double_threat_is_reported_not_solved():
    # Đối thủ đã có bốn mở (hai ô thắng độc lập): một lượt chỉ chặn được một ô, cả hai cấu hình như nhau.
    game = build([(0, 0), (0, 10), (10, 0), (10, 10)], [(5, 3), (5, 4), (5, 5), (5, 6)], to_move=X)
    for agent in (RuleBasedAgent(), _pattern_agent()):
        decision = agent.decide(game.board, X)
        assert decision.rule == RULE_BLOCK and set(decision.acceptable) == {(5, 2), (5, 7)}
        assert "đe dọa kép" in decision.reason


def _regression_game():
    # Thế hồi quy ở docs/Ngoc-Phuc.md mục 14.5: xen kẽ X, O, X đi trước; sau 12 nước X đến lượt.
    xs = [(1, 4), (2, 5), (9, 10), (8, 8), (10, 8), (7, 10)]
    os_ = [(0, 1), (1, 0), (0, 2), (2, 0), (0, 3), (3, 0)]
    game = CaroGame()
    for x, o in zip(xs, os_):
        game.step(x)
        game.step(o)
    return game


def test_multi_direction_defense_regression():
    # Chuẩn bị
    board = _regression_game().board
    # Thực thi
    decision = _pattern_agent().decide(board, X)
    # Kiểm chứng: chặn (0,0) làm O hết nước tạo đe dọa kép; chặn (0,4) thì O còn (4,0) tạo (0,0) và (5,0).
    assert decision.rule == RULE_BLOCK_OPEN_FOUR and decision.action == (0, 0)
    assert decision.acceptable == ((0, 0),)
    after = board.copy()
    after[0, 4] = X
    assert new_winning_cells(after, (4, 0), O) == ((0, 0), (5, 0))
    after = board.copy()
    after[0, 0] = X
    assert all(len(new_winning_cells(after, m, O)) < 2 for m in candidate_actions(after, 2))


@pytest.mark.parametrize("make", [
    _regression_game,
    lambda: place(X, [(5, 3), (5, 4), (5, 5)], [(5, 2), (1, 8), (2, 8), (3, 8)], to_move=X),
], ids=["hoi_quy", "chan_bon_mo"])
def test_block_open_four_matches_full_recount(make):
    # Tác tử chỉ kiểm tra lại các nước bốn mở đã tìm thấy; đối chiếu với đếm lại trên toàn bộ ứng viên.
    board = make().board
    decision = _pattern_agent().decide(board, X)
    remaining = {}
    for action in candidate_actions(board, 2):
        after = board.copy()
        after[action] = X
        remaining[action] = sum(len(new_winning_cells(after, m, O)) >= 2 for m in candidate_actions(board, 2)
                                if m != action)
    fewest = min(remaining.values())
    assert set(decision.acceptable) == {a for a, n in remaining.items() if n == fewest}


# ----- API, truy vết, ván hoàn chỉnh

@pytest.mark.parametrize("value", ["yes", 1, None], ids=["chuoi", "so_nguyen", "none"])
def test_rejects_invalid_use_patterns(value):
    with pytest.raises(ValueError):
        RuleBasedAgent(use_patterns=value)


def test_pattern_agent_keeps_board_and_returns_python_ints():
    game = _regression_game()
    before = game.board.copy()
    action = _pattern_agent().choose_action(game.board, X)
    assert np.array_equal(game.board, before)
    assert type(action) is tuple and all(type(v) is int for v in action)


def test_pattern_trace_is_independent_and_counts_nodes():
    agent = _pattern_agent()
    agent.choose_action(_regression_game().board, X)
    decision = agent.explain_last_action()
    assert decision.examined == agent.stats["nodes_visited"] > 0
    assert agent.stats["simulations"] == 0 and agent.stats["pruned_nodes"] == 0
    with pytest.raises(dataclasses.FrozenInstanceError):
        decision.rule = RULE_FALLBACK
    # Lượt sau không làm đổi bản truy vết đã lấy ra.
    agent.choose_action(CaroGame().board, X)
    assert decision.action == (0, 0) and agent.explain_last_action().action == (5, 5)


def test_pattern_agent_is_deterministic():
    game = place(X, [(5, 4), (5, 5)], [(2, 8), (3, 8)], to_move=X)
    decisions = {_pattern_agent().decide(game.board, X) for _ in range(3)}
    assert len(decisions) == 1


@pytest.mark.parametrize("seed", range(3))
@SIDES
def test_pattern_agent_full_game_against_random(seed, owner):
    # Bật pattern vẫn phải giữ thắng ngay và chặn thua (đối chiếu hàm kiểm tra độc lập).
    game = CaroGame()
    rule, rand = _pattern_agent(), RandomAgent(seed=seed)
    while not game.is_terminal():
        player = game.current_player
        if player == owner:
            own, threats = _oracle_wins(game.board, player), _oracle_wins(game.board, -player)
            action = rule.choose_action(game.board, player)
            if own:
                assert action in own
            elif threats:
                assert action in threats
            assert rule.explain_last_action().rule in RULES
        else:
            action = rand.choose_action(game.board, player)
        game.step(action)
    assert game.result() in (-1, 0, 1)


def test_pattern_vs_baseline_game_completes():
    game = CaroGame()
    agents = {X: _pattern_agent(name="Rule X"), O: RuleBasedAgent("Rule O", use_patterns=False)}
    while not game.is_terminal():
        game.step(agents[game.current_player].choose_action(game.board, game.current_player))
    assert game.result() in (-1, 0, 1)


def test_play_match_runner_with_patterns():
    record, logs = play_match(partial(RuleBasedAgent, use_patterns=True), partial(RandomAgent, seed=5),
                              agent_x="Rule-based pattern", agent_o="Ngẫu nhiên",
                              budget=SearchBudget(mode="time", time_limit_s=2.0))
    assert record.reason == "completed"
    rule_logs = [log for log in logs if log["player"] == X]
    assert rule_logs and all(log["status"] == "move" and log["stats"]["nodes_visited"] > 0 for log in rule_logs)


# ---- ngân sách thời gian khi bật pattern

def _pattern_time_agent(limit):
    return _pattern_agent(budget=SearchBudget(mode="time", time_limit_s=limit))


class _RecordingClock(_FakeClock):
    """Đồng hồ giả ghi lại riêng các lần tác tử Rule-based đọc giờ (BaseAgent đọc qua __call__)."""

    def __init__(self, start=100.0, step=0.001):
        super().__init__(start, step)
        self.agent_reads = []

    def agent_read(self):
        value = self()
        self.agent_reads.append(value)
        return value


def _use_recording_clock(monkeypatch, clock):
    monkeypatch.setattr("agents.base_agent.perf_counter", clock)
    monkeypatch.setattr("agents.rule_based_agent.perf_counter", clock.agent_read)


def _reads_past_stop(clock, agent):
    # Số lần tác tử đọc giờ từ mốc dừng hợp tác trở đi; dừng đúng lúc thì chỉ có 1 (lần phát hiện hết giờ).
    stop_at = agent.deadline - min(agent.time_margin_s, 0.5 * agent.budget.time_limit_s)
    return sum(value >= stop_at for value in clock.agent_reads)


def test_time_mode_stops_during_pattern_scan(monkeypatch):
    # Chuẩn bị: thế yên tĩnh; mốc dừng sau 2 lượt quét toàn bàn + khoảng 10 lần đọc: giữa lúc tìm bốn mở.
    clock = _RecordingClock()
    _use_recording_clock(monkeypatch, clock)
    game = build([(5, 5)], [(6, 6)], to_move=X)
    n_empty = int(np.count_nonzero(game.board == 0))
    candidates = candidate_actions(game.board, 2)
    agent = _pattern_time_agent(0.1 + (2 * n_empty + 10) * 0.001)
    start = clock.now
    # Thực thi
    action = agent.choose_action(game.board, X)
    # Kiểm chứng: dừng trước deadline, trả nước dự phòng chuẩn bị sẵn, số nước thử khớp lý do.
    decision = agent.explain_last_action()
    examined = agent.stats["nodes_visited"]
    assert clock.now < start + agent.budget.time_limit_s
    assert decision.timed_out and decision.rule == RULE_FALLBACK and action == (4, 5)
    assert 0 < examined - 2 * n_empty < len(candidates)
    assert "dừng khi đang tìm bốn mở của mình" in decision.reason
    assert f"hết thời gian sau {examined} nước thử" in decision.reason
    assert _reads_past_stop(clock, agent) == 1


def test_time_mode_stops_while_trying_blocks(monkeypatch):
    # Chuẩn bị: thế hồi quy; mốc dừng sau 2 lượt quét toàn bàn, 2 lượt quét ứng viên và khoảng 30 lần đọc
    # nữa: giữa lúc thử các nước chặn.
    clock = _RecordingClock()
    _use_recording_clock(monkeypatch, clock)
    game = _regression_game()
    n_empty = int(np.count_nonzero(game.board == 0))
    candidates = candidate_actions(game.board, 2)
    agent = _pattern_time_agent(0.1 + (2 * n_empty + 2 * len(candidates) + 30) * 0.001)
    # Thực thi
    action = agent.choose_action(game.board, X)
    # Kiểm chứng: vẫn là tầng chặn bốn mở, chỉ chọn trong các nước đã thử xong, ghi đúng số nước đã thử.
    decision = agent.explain_last_action()
    match = re.search(r"thử (\d+)/(\d+) nước chặn", decision.reason)
    assert decision.rule == RULE_BLOCK_OPEN_FOUR and decision.timed_out and match
    tried = int(match.group(1))
    assert 0 < tried < len(candidates) == int(match.group(2))
    assert action in candidates[:tried] and set(decision.acceptable) <= set(candidates[:tried])
    assert _reads_past_stop(clock, agent) == 1


def test_time_mode_stops_inside_nested_open_three_scan(monkeypatch):
    # Chuẩn bị: mốc dừng khoảng 15 lần đọc sau khi bắt đầu tìm ba mở; ứng viên đầu tiên đã cần vài chục
    # nước thử lồng nhau nên phải dừng ngay trong vòng lồng.
    clock = _RecordingClock()
    _use_recording_clock(monkeypatch, clock)
    game = place(X, [(5, 4), (5, 5)], [(2, 8), (3, 8)], to_move=X)
    n_empty = int(np.count_nonzero(game.board == 0))
    candidates = candidate_actions(game.board, 2)
    full = _pattern_agent()
    full.decide(game.board, X)
    agent = _pattern_time_agent(0.1 + (2 * n_empty + 2 * len(candidates) + 15) * 0.001)
    # Thực thi
    agent.choose_action(game.board, X)
    # Kiểm chứng: dừng khi chưa tìm được ba mở nào (các nước ba mở ở hàng 5, nằm sau trong thứ tự quét).
    decision = agent.explain_last_action()
    in_scan = agent.stats["nodes_visited"] - 2 * n_empty - 2 * len(candidates)
    assert decision.timed_out and decision.rule == RULE_FALLBACK
    assert "dừng khi đang tìm ba mở của mình" in decision.reason
    assert 0 < in_scan < 15 and agent.stats["nodes_visited"] < full.stats["nodes_visited"]
    # Dừng ngay ở lần đọc giờ đầu tiên quá mốc, không chạy nốt vòng lồng của ứng viên đang xét.
    assert _reads_past_stop(clock, agent) == 1


@pytest.mark.parametrize("make", [
    _regression_game,
    lambda: place(X, [(5, 4), (5, 5)], [(2, 8), (3, 8)], to_move=X),
    lambda: place(X, [], [(5, 4), (5, 5)], to_move=X),
], ids=["chan_bon_mo", "ba_mo", "chan_ba_mo"])
def test_every_trial_move_checks_the_clock_first(monkeypatch, make):
    # Đủ giờ: mỗi nước thử được đếm vào nodes_visited đều có một lần kiểm tra mốc dừng ngay trước nó,
    # kể cả các nước thử lồng nhau, nên số lần đọc giờ không nhỏ hơn số nút.
    clock = _RecordingClock(step=1e-6)
    _use_recording_clock(monkeypatch, clock)
    agent = _pattern_time_agent(2.0)
    agent.choose_action(make().board, X)
    assert not agent.explain_last_action().timed_out
    assert len(clock.agent_reads) >= agent.stats["nodes_visited"] > 0


def test_pattern_time_mode_with_enough_time_matches_fixed(monkeypatch):
    clock = _FakeClock(step=1e-6)
    _use_clock(monkeypatch, clock)
    board = _regression_game().board
    timed, fixed = _pattern_time_agent(2.0), _pattern_agent()
    assert timed.choose_action(board, X) == fixed.choose_action(board, X)
    assert not timed.explain_last_action().timed_out
    assert timed.stats["nodes_visited"] == fixed.stats["nodes_visited"]


def test_pattern_fixed_mode_ignores_clock(monkeypatch):
    monkeypatch.setattr("agents.rule_based_agent.perf_counter", lambda: float("inf"))
    agent = _pattern_agent()
    assert agent.choose_action(_regression_game().board, X) == (0, 0)
    assert not agent.explain_last_action().timed_out
