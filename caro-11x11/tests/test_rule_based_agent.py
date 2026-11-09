"""Bộ kiểm thử tự động cho RuleBasedAgent — Giai đoạn 1.

Cấu trúc mỗi ca kiểm thử theo 3 bước:
  Arrange — Thiết lập trạng thái bàn cờ ban đầu.
  Act     — Gọi hàm cần kiểm tra để sinh hành vi/nước đi.
  Assert  — Đối chiếu kết quả thu được với giá trị kỳ vọng.

Bàn cờ được dựng bằng CaroGame.step() để đảm bảo tính hợp lệ.
Không sửa các file ngoài phạm vi: core/, config/, agents/base_agent.py, agents/__init__.py.
"""

import numpy as np
import pytest

import agents.rule_based_agent as rule_based_module
from agents.random_agent import RandomAgent
from agents.rule_based_agent import (
    RuleBasedAgent,
    find_winning_actions,
    is_winning_move,
    positional_score,
)
from config.game_config import EMPTY, PLAYER_O, PLAYER_X
from core.game_engine import CaroGame


# ===========================================================================
# Fixture helper — dựng bàn cờ trực tiếp bằng numpy (linh hoạt hơn)
# ===========================================================================

def _board_from_stones(
    x_stones: list[tuple[int, int]],
    o_stones: list[tuple[int, int]],
    size: int = 11,
) -> np.ndarray:
    """Tạo bàn cờ numpy trực tiếp từ danh sách tọa độ.

    Không phụ thuộc vào thứ tự đặt quân của CaroGame — dùng cho các
    trạng thái cần kiểm soát chính xác số quân mỗi bên.
    """
    board = np.zeros((size, size), dtype=np.int8)
    for r, c in x_stones:
        board[r, c] = PLAYER_X
    for r, c in o_stones:
        board[r, c] = PLAYER_O
    return board


def _build_board(moves_x: list[tuple[int, int]], moves_o: list[tuple[int, int]]) -> np.ndarray:
    """Dựng bàn cờ 11×11 bằng cách xen kẽ nước X và O qua CaroGame.step().

    X đi trước. Độ dài moves_o phải = len(moves_x) hoặc = len(moves_x) - 1.
    Trả về bản sao board (không tham chiếu object nội bộ của engine).
    """
    game = CaroGame()
    for i, x_move in enumerate(moves_x):
        game.step(x_move)
        if i < len(moves_o):
            game.step(moves_o[i])
    return game.board.copy()


# ===========================================================================
# PHẦN 1 — is_winning_move: kiểm tra hàm cấp thấp
# ===========================================================================

class TestIsWinningMove:
    """Kiểm tra hàm is_winning_move theo từng hướng và trường hợp biên."""

    def test_ngang_thang_nguyen_x(self):
        """X có 4 quân ngang [0,0]–[0,3], đặt vào [0,4] → thắng."""
        # Arrange
        board = _board_from_stones([(0, 0), (0, 1), (0, 2), (0, 3)], [(1, 0), (1, 1)])
        # Act & Assert
        assert is_winning_move(board, 0, 4, PLAYER_X), "X phải thắng ngang tại (0,4)"

    def test_doc_thang_nguyen_x(self):
        """X có 4 quân dọc [0,5]–[3,5], đặt vào [4,5] → thắng."""
        # Arrange
        board = _board_from_stones([(0, 5), (1, 5), (2, 5), (3, 5)], [(0, 0)])
        # Act & Assert
        assert is_winning_move(board, 4, 5, PLAYER_X), "X phải thắng dọc tại (4,5)"

    def test_cheo_chinh_x(self):
        """X có 4 quân chéo chính [0,0]–[3,3], đặt vào [4,4] → thắng."""
        # Arrange
        board = _board_from_stones([(0, 0), (1, 1), (2, 2), (3, 3)], [(0, 1)])
        # Act & Assert
        assert is_winning_move(board, 4, 4, PLAYER_X), "X phải thắng chéo chính tại (4,4)"

    def test_cheo_phu_x(self):
        """X có 4 quân chéo phụ [0,10]–[3,7], đặt vào [4,6] → thắng."""
        # Arrange
        board = _board_from_stones([(0, 10), (1, 9), (2, 8), (3, 7)], [(0, 0)])
        # Act & Assert
        assert is_winning_move(board, 4, 6, PLAYER_X), "X phải thắng chéo phụ tại (4,6)"

    def test_ngang_thang_nguyen_o(self):
        """O có 4 quân ngang [2,0]–[2,3], đặt vào [2,4] → thắng."""
        # Arrange
        board = _board_from_stones([(0, 0), (0, 1)], [(2, 0), (2, 1), (2, 2), (2, 3)])
        # Act & Assert
        assert is_winning_move(board, 2, 4, PLAYER_O), "O phải thắng ngang tại (2,4)"

    def test_doc_thang_nguyen_o(self):
        """O có 4 quân dọc [0,3]–[3,3], đặt vào [4,3] → thắng."""
        # Arrange
        board = _board_from_stones([(0, 0), (0, 1)], [(0, 3), (1, 3), (2, 3), (3, 3)])
        # Act & Assert
        assert is_winning_move(board, 4, 3, PLAYER_O), "O phải thắng dọc tại (4,3)"

    def test_cheo_chinh_o(self):
        """O có 4 quân chéo chính [1,1]–[4,4], đặt vào [5,5] → thắng."""
        # Arrange
        board = _board_from_stones([(0, 0), (0, 1)], [(1, 1), (2, 2), (3, 3), (4, 4)])
        # Act & Assert
        assert is_winning_move(board, 5, 5, PLAYER_O), "O phải thắng chéo chính tại (5,5)"

    def test_cheo_phu_o(self):
        """O có 4 quân chéo phụ [1,9]–[4,6], đặt vào [5,5] → thắng."""
        # Arrange
        board = _board_from_stones([(0, 0), (0, 1)], [(1, 9), (2, 8), (3, 7), (4, 6)])
        # Act & Assert
        assert is_winning_move(board, 5, 5, PLAYER_O), "O phải thắng chéo phụ tại (5,5)"

    def test_o_da_co_quan_khong_thang(self):
        """Ô đã có quân → is_winning_move phải trả về False."""
        # Arrange
        board = _board_from_stones([(5, 5)], [])
        # Act & Assert
        assert not is_winning_move(board, 5, 5, PLAYER_X), "Ô đã có quân không được là nước thắng"

    def test_chuoi_qua_bien_khong_thang(self):
        """Ô (0,5) không nối thành chuỗi 5 với X tại (0,0)–(0,3) vì có khoảng trống."""
        # Arrange: X tại (0,0),(0,1),(0,2),(0,3) — ô (0,4) trống, (0,5) không liền kề
        board = _board_from_stones([(0, 0), (0, 1), (0, 2), (0, 3)], [(1, 0)])
        # Act & Assert: đặt tại (0,5) không thắng (có khoảng trống ở (0,4))
        assert not is_winning_move(board, 0, 5, PLAYER_X), "(0,5) cách biệt, không thắng"

    def test_chuoi_hon_5_quan_van_la_thang(self):
        """Nối chuỗi thành 6 quân liên tiếp vẫn thắng (freestyle Caro)."""
        # Arrange: X tại (0,1)(0,2)(0,3)(0,5)(0,6), ô (0,4) trống → nối 6
        board = _board_from_stones([(0, 1), (0, 2), (0, 3), (0, 5), (0, 6)], [(1, 0)])
        # Act & Assert
        assert is_winning_move(board, 0, 4, PLAYER_X), "Nối 6 quân phải thắng"

    def test_chuoi_bi_chan_2_dau_van_thang(self):
        """O X X . X X O — đặt vào ô trống ở giữa tạo 5 quân liên tiếp → thắng."""
        # Arrange: X tại (0,2)(0,3)(0,5)(0,6); O chặn (0,1) và (0,7)
        board = _board_from_stones(
            [(0, 2), (0, 3), (0, 5), (0, 6)],
            [(0, 1), (0, 7)],
        )
        # Act & Assert: đặt X tại (0,4) tạo chuỗi 5 (0,2)–(0,6), bị O chặn 2 đầu → thắng
        assert is_winning_move(board, 0, 4, PLAYER_X), "Chuỗi 5 bị chặn 2 đầu vẫn thắng"


# ===========================================================================
# PHẦN 1B — positional_score và find_winning_actions: kiểm tra hàm hỗ trợ
# ===========================================================================

class TestPositionalScoreAndHelpers:
    """Kiểm tra độc lập hàm positional_score và find_winning_actions."""

    def test_positional_score_uu_tien_tam(self):
        """Trên bàn trống, điểm tại tâm (5, 5) > điểm tại (3, 3) > điểm tại góc (0, 0)."""
        # Arrange
        board = np.zeros((11, 11), dtype=np.int8)

        # Act
        score_center = positional_score(board, 5, 5, PLAYER_X)
        score_mid = positional_score(board, 3, 3, PLAYER_X)
        score_corner = positional_score(board, 0, 0, PLAYER_X)

        # Assert
        assert score_center > score_mid > score_corner, (
            f"Điểm vị trí phải giảm dần từ tâm ra góc: "
            f"{score_center} > {score_mid} > {score_corner}"
        )

    def test_positional_score_adjacency_bonus(self):
        """Khi có quân đồng minh tại (5, 5), ô kề sát bán kính 1 là (5, 6) có điểm cao hơn ô cách xa (5, 8)."""
        # Arrange
        board = _build_board([(5, 5)], [])

        # Act
        score_adj = positional_score(board, 5, 6, PLAYER_X)
        score_far = positional_score(board, 5, 8, PLAYER_X)

        # Assert
        assert score_adj > score_far, (
            f"Ô kề sát đồng minh (5,6) phải có điểm cao hơn (5,8): "
            f"{score_adj} > {score_far}"
        )

    def test_find_winning_actions_hai_dau_mo(self):
        """Chuỗi 4 quân [(5, 1), (5, 2), (5, 3), (5, 4)] trả về đúng [(5, 0), (5, 5)]."""
        # Arrange
        board = _build_board(
            [(5, 1), (5, 2), (5, 3), (5, 4)],
            [(0, 0), (0, 2), (0, 4), (0, 6)],
        )

        # Act
        win_actions = find_winning_actions(board, PLAYER_X)

        # Assert
        assert win_actions == [(5, 0), (5, 5)], (
            f"Chuỗi 4 hai đầu mở phải trả về [(5, 0), (5, 5)], nhận {win_actions}"
        )

    def test_find_winning_actions_khong_co_nuoc_thang(self):
        """Khi mới có 2 quân, trả về danh sách rỗng []."""
        # Arrange
        board = _build_board([(5, 5), (5, 6)], [(0, 0), (0, 2)])

        # Act
        win_actions = find_winning_actions(board, PLAYER_X)

        # Assert
        assert win_actions == [], f"Khi mới có 2 quân phải trả về [], nhận {win_actions}"


# ===========================================================================
# PHẦN 2 — Thắng ngay (win_now): agent chọn nước thắng
# ===========================================================================

class TestWinNow:
    """Agent phải chọn nước thắng ngay khi có cơ hội, ở cả 4 hướng và 2 phía."""

    def test_thang_ngang_x(self):
        """X có 4 quân ngang hàng 5, cột 1–4; nước thắng tại (5,0) hoặc (5,5)."""
        # Arrange: đến lượt X -> len(moves_x) == len(moves_o) == 4
        board = _build_board(
            [(5, 1), (5, 2), (5, 3), (5, 4)],  # X: 4 quân ngang
            [(0, 0), (0, 2), (0, 4), (0, 6)],  # O: 4 quân phân tán
        )
        agent = RuleBasedAgent()

        # Act
        action = agent.choose_action(board, PLAYER_X)
        info = agent.explain_last_action()

        # Assert
        assert info["rule_triggered"] == "win_now", "Phải kích hoạt luật win_now"
        win_set = set(find_winning_actions(board, PLAYER_X))
        assert action in win_set, f"Nước {action} phải là nước thắng ngang"

    def test_thang_ngang_o(self):
        """O có 4 quân ngang; không có nước thắng ngay cho X."""
        # Arrange: đến lượt O -> len(moves_x) == len(moves_o) + 1 (5 và 4)
        board = _build_board(
            [(0, 0), (0, 2), (0, 4), (2, 0), (2, 2)],  # X: 5 quân phân tán
            [(7, 1), (7, 2), (7, 3), (7, 4)],          # O: 4 quân ngang
        )
        agent = RuleBasedAgent()

        # Act
        action = agent.choose_action(board, PLAYER_O)
        info = agent.explain_last_action()

        # Assert
        assert info["rule_triggered"] == "win_now"
        win_set = set(find_winning_actions(board, PLAYER_O))
        assert action in win_set, f"Nước {action} phải là nước thắng ngang cho O"

    def test_thang_doc_x(self):
        """X có 4 quân dọc cột 5 hàng 0–3, nước thắng tại (4,5)."""
        # Arrange: đến lượt X -> len(moves_x) == len(moves_o) == 4
        board = _build_board(
            [(0, 5), (1, 5), (2, 5), (3, 5)],
            [(0, 0), (0, 2), (0, 4), (0, 6)],
        )
        agent = RuleBasedAgent()

        # Act
        action = agent.choose_action(board, PLAYER_X)
        info = agent.explain_last_action()

        # Assert
        assert info["rule_triggered"] == "win_now"
        assert action in find_winning_actions(board, PLAYER_X)

    def test_thang_doc_o(self):
        """O có 4 quân dọc cột 3, hàng 1–4; thắng tại (0,3) hoặc (5,3)."""
        # Arrange: đến lượt O -> len(moves_x) == len(moves_o) + 1 (5 và 4)
        board = _build_board(
            [(0, 0), (0, 2), (0, 4), (2, 0), (2, 2)],
            [(1, 3), (2, 3), (3, 3), (4, 3)],
        )
        agent = RuleBasedAgent()

        # Act
        action = agent.choose_action(board, PLAYER_O)
        info = agent.explain_last_action()

        # Assert
        assert info["rule_triggered"] == "win_now"
        assert action in find_winning_actions(board, PLAYER_O)

    def test_thang_cheo_chinh_x(self):
        """X có 4 quân chéo chính (0,0)–(3,3), thắng tại (4,4)."""
        # Arrange: đến lượt X -> len(moves_x) == len(moves_o) == 4
        board = _build_board(
            [(0, 0), (1, 1), (2, 2), (3, 3)],
            [(0, 2), (0, 4), (2, 0), (4, 0)],
        )
        agent = RuleBasedAgent()

        # Act
        action = agent.choose_action(board, PLAYER_X)
        info = agent.explain_last_action()

        # Assert
        assert info["rule_triggered"] == "win_now"
        assert action in find_winning_actions(board, PLAYER_X)

    def test_thang_cheo_phu_o(self):
        """O có 4 quân chéo phụ (1,9)–(4,6), thắng tại (5,5) hoặc (0,10)."""
        # Arrange: đến lượt O -> len(moves_x) == len(moves_o) + 1 (5 và 4)
        board = _build_board(
            [(0, 0), (0, 2), (0, 4), (2, 0), (2, 2)],
            [(1, 9), (2, 8), (3, 7), (4, 6)],
        )
        agent = RuleBasedAgent()

        # Act
        action = agent.choose_action(board, PLAYER_O)
        info = agent.explain_last_action()

        # Assert
        assert info["rule_triggered"] == "win_now"
        assert action in find_winning_actions(board, PLAYER_O)

    def test_thang_sat_bien_trai_x(self):
        """X có 4 quân sát cột 0, nước thắng tại (0,4) — nằm trong biên."""
        # Arrange: đến lượt X -> len(moves_x) == len(moves_o) == 4
        board = _build_board(
            [(0, 0), (0, 1), (0, 2), (0, 3)],
            [(2, 5), (2, 7), (4, 5), (4, 7)],
        )
        agent = RuleBasedAgent()

        # Act
        action = agent.choose_action(board, PLAYER_X)
        info = agent.explain_last_action()

        # Assert
        assert info["rule_triggered"] == "win_now"
        assert action == (0, 4), f"Nước thắng phải là (0,4), nhận được {action}"

    def test_thang_chuoi_hon_5_quan(self):
        """X có 5 quân + 1 khoảng trống ở giữa tạo chuỗi 6 → vẫn thắng."""
        # Arrange: đến lượt X -> len(moves_x) == len(moves_o) == 5
        board = _build_board(
            [(0, 1), (0, 2), (0, 3), (0, 5), (0, 6)],
            [(2, 0), (2, 2), (4, 0), (4, 2), (6, 0)],
        )
        agent = RuleBasedAgent()

        # Act
        action = agent.choose_action(board, PLAYER_X)
        info = agent.explain_last_action()

        # Assert: (0,4) nối thành chuỗi 6 → thắng
        assert info["rule_triggered"] == "win_now"
        assert action == (0, 4), f"Nước nối chuỗi >5 phải tại (0,4), nhận {action}"

    def test_thang_chuoi_bi_chan_2_dau(self):
        """O X X . X X O — đặt vào giữa tạo 5 quân liên tiếp (bị chặn 2 đầu) → vẫn thắng."""
        # Arrange: đến lượt X -> len(moves_x) == len(moves_o) == 4
        board = _build_board(
            [(0, 2), (0, 3), (0, 5), (0, 6)],
            [(0, 1), (0, 7), (2, 0), (2, 2)],
        )
        agent = RuleBasedAgent()

        # Act
        action = agent.choose_action(board, PLAYER_X)
        info = agent.explain_last_action()

        # Assert: (0,4) nối thành 5 quân liên tiếp (bị chặn 2 đầu → vẫn thắng)
        assert info["rule_triggered"] == "win_now"
        assert action == (0, 4), f"Phải thắng tại (0,4), nhận {action}"


# ===========================================================================
# PHẦN 3 — Chặn thua (block_urgent): agent chặn đúng vị trí đe dọa
# ===========================================================================
# Lưu ý thiết kế fixture block_urgent:
#   - Bên đang đến lượt (người bảo vệ) KHÔNG có chuỗi 4 quân liên tiếp.
#   - Đối thủ có đúng 4 quân liên tiếp tạo đe dọa thắng ngay.
# ===========================================================================

class TestBlockUrgent:
    """Agent phải chặn đối thủ khi không có nước thắng ngay."""

    def test_x_chan_o_ngang(self):
        """O có 4 quân ngang tại hàng 7; X phân tán, không thắng ngay → X chặn."""
        # Arrange: đến lượt X -> len(moves_x) == len(moves_o) == 4
        board = _build_board(
            [(0, 0), (0, 2), (2, 0), (2, 2)],   # X: 4 quân phân tán
            [(7, 3), (7, 4), (7, 5), (7, 6)],   # O: 4 quân ngang → đe dọa (7,2) và (7,7)
        )
        agent = RuleBasedAgent()

        # Act
        action = agent.choose_action(board, PLAYER_X)
        info = agent.explain_last_action()

        # Assert
        assert info["rule_triggered"] == "block_urgent", (
            f"Phải kích hoạt block_urgent, nhận {info['rule_triggered']}"
        )
        block_set = set(find_winning_actions(board, PLAYER_O))
        assert action in block_set, f"Nước chặn {action} phải nằm trong tập đe dọa"

    def test_o_chan_x_ngang(self):
        """X có 4 quân ngang tại hàng 2; O phân tán, không thắng ngay → O chặn."""
        # Arrange: đến lượt O -> len(moves_x) == len(moves_o) + 1 (4 và 3)
        board = _build_board(
            [(2, 1), (2, 2), (2, 3), (2, 4)],   # X: 4 quân ngang
            [(0, 0), (0, 2), (4, 0)],            # O: 3 quân phân tán
        )
        agent = RuleBasedAgent()

        # Act
        action = agent.choose_action(board, PLAYER_O)
        info = agent.explain_last_action()

        # Assert
        assert info["rule_triggered"] == "block_urgent"
        block_set = set(find_winning_actions(board, PLAYER_X))
        assert action in block_set

    def test_x_chan_o_doc(self):
        """O có 4 quân dọc cột 7 hàng 0–3; X không thắng → X phải chặn."""
        # Arrange: đến lượt X -> len(moves_x) == len(moves_o) == 4
        board = _build_board(
            [(5, 0), (5, 2), (7, 0), (7, 2)],   # X: 4 quân phân tán
            [(0, 7), (1, 7), (2, 7), (3, 7)],   # O: 4 quân dọc
        )
        agent = RuleBasedAgent()

        # Act
        action = agent.choose_action(board, PLAYER_X)
        info = agent.explain_last_action()

        # Assert
        assert info["rule_triggered"] == "block_urgent"
        assert action in find_winning_actions(board, PLAYER_O)

    def test_o_chan_x_doc(self):
        """X có 4 quân dọc cột 3 hàng 2–5; O không thắng → O phải chặn."""
        # Arrange: đến lượt O -> len(moves_x) == len(moves_o) + 1 (4 và 3)
        board = _build_board(
            [(2, 3), (3, 3), (4, 3), (5, 3)],   # X: 4 quân dọc
            [(0, 0), (0, 2), (8, 8)],            # O: 3 quân phân tán
        )
        agent = RuleBasedAgent()

        # Act
        action = agent.choose_action(board, PLAYER_O)
        info = agent.explain_last_action()

        # Assert
        assert info["rule_triggered"] == "block_urgent"
        assert action in find_winning_actions(board, PLAYER_X)

    def test_x_chan_o_cheo_chinh(self):
        """O có 4 quân chéo chính (1,1)–(4,4); X không thắng → X phải chặn."""
        # Arrange: đến lượt X -> len(moves_x) == len(moves_o) == 4
        board = _build_board(
            [(0, 2), (0, 4), (2, 0), (8, 8)],   # X: 4 quân phân tán
            [(1, 1), (2, 2), (3, 3), (4, 4)],   # O: 4 quân chéo chính
        )
        agent = RuleBasedAgent()

        # Act
        action = agent.choose_action(board, PLAYER_X)
        info = agent.explain_last_action()

        # Assert
        assert info["rule_triggered"] == "block_urgent"
        assert action in find_winning_actions(board, PLAYER_O)

    def test_x_chan_o_cheo_phu(self):
        """O có 4 quân chéo phụ (1,9)–(4,6); X không thắng → X phải chặn."""
        # Arrange: đến lượt X -> len(moves_x) == len(moves_o) == 4
        board = _build_board(
            [(0, 0), (0, 2), (2, 0), (8, 8)],   # X: 4 quân phân tán
            [(1, 9), (2, 8), (3, 7), (4, 6)],   # O: 4 quân chéo phụ
        )
        agent = RuleBasedAgent()

        # Act
        action = agent.choose_action(board, PLAYER_X)
        info = agent.explain_last_action()

        # Assert
        assert info["rule_triggered"] == "block_urgent"
        assert action in find_winning_actions(board, PLAYER_O)

    def test_chan_sat_bien_duoi(self):
        """O có 4 quân sát hàng 10 (biên dưới); X không thắng → X chặn."""
        # Arrange: đến lượt X -> len(moves_x) == len(moves_o) == 4
        board = _build_board(
            [(5, 0), (5, 2), (7, 0), (7, 2)],       # X: 4 quân phân tán
            [(10, 0), (10, 1), (10, 2), (10, 3)],   # O: sát biên dưới
        )
        agent = RuleBasedAgent()

        # Act
        action = agent.choose_action(board, PLAYER_X)
        info = agent.explain_last_action()

        # Assert
        assert info["rule_triggered"] == "block_urgent"
        assert action in find_winning_actions(board, PLAYER_O)

    def test_chan_sat_bien_trai(self):
        """O có 4 quân sát cột 0 (biên trái); X không thắng → X chặn."""
        # Arrange: đến lượt X -> len(moves_x) == len(moves_o) == 4
        board = _build_board(
            [(5, 5), (5, 7), (7, 5), (7, 7)],   # X: 4 quân phân tán
            [(0, 0), (1, 0), (2, 0), (3, 0)],   # O: sát biên trái
        )
        agent = RuleBasedAgent()

        # Act
        action = agent.choose_action(board, PLAYER_X)
        info = agent.explain_last_action()

        # Assert
        assert info["rule_triggered"] == "block_urgent"
        assert action in find_winning_actions(board, PLAYER_O)


# ===========================================================================
# PHẦN 4 — Xung đột ưu tiên: thắng ngay > chặn thua
# ===========================================================================

class TestWinOverBlock:
    """Khi đồng thời có nước thắng ngay và đe dọa của đối thủ,
    agent phải ưu tiên thắng ngay (win_now > block_urgent).
    """

    def test_x_co_the_thang_va_o_dang_de_doa(self):
        """X có cơ hội thắng ngay VÀ O cũng có 4 quân đe dọa → X chọn thắng."""
        # Arrange: đến lượt X -> len(moves_x) == len(moves_o) == 4
        board = _build_board(
            [(5, 1), (5, 2), (5, 3), (5, 4)],   # X: 4 ngang → thắng ngay
            [(0, 9), (1, 9), (2, 9), (3, 9)],   # O: 4 dọc → đe dọa
        )
        agent = RuleBasedAgent()

        # Act
        action = agent.choose_action(board, PLAYER_X)
        info = agent.explain_last_action()

        # Assert
        assert info["rule_triggered"] == "win_now", (
            f"Phải ưu tiên win_now dù O đang đe dọa, nhận {info['rule_triggered']}"
        )
        win_set = set(find_winning_actions(board, PLAYER_X))
        assert action in win_set, f"Nước {action} phải là nước thắng ngay"

    def test_o_co_the_thang_va_x_dang_de_doa(self):
        """O có cơ hội thắng ngay VÀ X cũng có 4 quân đe dọa → O chọn thắng."""
        # Arrange: đến lượt O -> len(moves_x) == len(moves_o) + 1 (5 và 4)
        board = _build_board(
            [(8, 2), (8, 3), (8, 4), (8, 5), (10, 10)],  # X: 4 ngang + 1 quân rời
            [(0, 3), (1, 3), (2, 3), (3, 3)],            # O: 4 dọc → thắng ngay
        )
        agent = RuleBasedAgent()

        # Act
        action = agent.choose_action(board, PLAYER_O)
        info = agent.explain_last_action()

        # Assert
        assert info["rule_triggered"] == "win_now", (
            f"O phải ưu tiên win_now, nhận {info['rule_triggered']}"
        )
        win_set = set(find_winning_actions(board, PLAYER_O))
        assert action in win_set

    @pytest.mark.parametrize("player", [PLAYER_X, PLAYER_O])
    def test_uu_tien_thang_hon_chan_tham_so_hoa(self, player):
        """Với cả X và O: win_now luôn được ưu tiên hơn block_urgent."""
        # Arrange: player có 4 quân thẳng hàng; đối thủ cũng có 4 quân đe dọa
        if player == PLAYER_X:
            board = _build_board(
                [(2, 0), (2, 1), (2, 2), (2, 3)],   # X: 4 quân → thắng tại (2,4)
                [(4, 0), (4, 1), (4, 2), (4, 3)],   # O: 4 quân → đe dọa tại (4,4)
            )
        else:
            board = _build_board(
                [(4, 0), (4, 1), (4, 2), (4, 3), (10, 10)],  # X: 4 quân đe dọa + 1 quân rời
                [(2, 0), (2, 1), (2, 2), (2, 3)],            # O: 4 quân → thắng tại (2,4)
            )
        agent = RuleBasedAgent()

        # Act
        action = agent.choose_action(board, player)
        info = agent.explain_last_action()

        # Assert: phải chọn win_now
        assert info["rule_triggered"] == "win_now"
        win_set = set(find_winning_actions(board, player))
        assert action in win_set


# ===========================================================================
# PHẦN 5 — Trường hợp biên và hợp đồng API
# ===========================================================================

class TestEdgeCasesAndContract:
    """Kiểm tra các trường hợp đặc biệt và hợp đồng giao tiếp BaseAgent."""

    def test_ban_co_trong_tra_ve_tam(self):
        """Bàn cờ trống → agent dùng positional_fallback, chọn vị trí gần tâm."""
        # Arrange
        board = np.zeros((11, 11), dtype=np.int8)
        agent = RuleBasedAgent()

        # Act
        action = agent.choose_action(board, PLAYER_X)
        info = agent.explain_last_action()

        # Assert
        assert info["rule_triggered"] == "positional_fallback"
        # Tâm bàn 11×11 là (5,5); với tất định và bàn trống, kỳ vọng (5,5)
        assert action == (5, 5), f"Bàn trống phải chọn tâm (5,5), nhận {action}"

    def test_ban_co_1_o_trong(self):
        """Bàn cờ chỉ còn 1 ô trống → agent phải trả về đúng ô đó."""
        # Arrange: lấp đầy tất cả trừ (10, 10) bằng X (bỏ qua luật game vì dùng numpy)
        board = np.ones((11, 11), dtype=np.int8)
        board[10, 10] = EMPTY

        # Act
        agent = RuleBasedAgent()
        action = agent.choose_action(board, PLAYER_X)

        # Assert
        assert action == (10, 10), f"Phải trả về ô duy nhất (10,10), nhận {action}"

    def test_board_bat_bien(self):
        """Mảng board đầu vào không bị sửa đổi sau khi gọi choose_action."""
        # Arrange
        board = _board_from_stones(
            [(5, 1), (5, 2), (5, 3), (5, 4)],
            [(0, 0), (0, 2), (0, 4)],
        )
        snapshot = board.copy()
        agent = RuleBasedAgent()

        # Act
        agent.choose_action(board, PLAYER_X)

        # Assert
        np.testing.assert_array_equal(
            board, snapshot,
            err_msg="Mảng board bị sửa đổi sau khi choose_action"
        )

    def test_nodes_visited_duong(self):
        """stats['nodes_visited'] phải > 0 sau mỗi lượt đi."""
        # Arrange
        board = np.zeros((11, 11), dtype=np.int8)
        agent = RuleBasedAgent()

        # Act
        agent.choose_action(board, PLAYER_X)

        # Assert
        assert agent.stats["nodes_visited"] > 0, (
            "nodes_visited phải > 0 sau khi agent quyết định nước đi"
        )

    def test_nodes_visited_reset_moi_luot(self):
        """stats['nodes_visited'] được đặt lại về 0 ở đầu mỗi lượt đi."""
        # Arrange
        board = np.zeros((11, 11), dtype=np.int8)
        agent = RuleBasedAgent()

        # Act: gọi 2 lần liên tiếp
        agent.choose_action(board, PLAYER_X)
        visited_first = agent.stats["nodes_visited"]
        agent.choose_action(board, PLAYER_O)
        visited_second = agent.stats["nodes_visited"]

        # Assert: lần 2 không cộng dồn từ lần 1
        assert visited_second > 0, "nodes_visited lần 2 phải > 0"
        # Trên bàn trống, 2 lần có cùng số ô trống → kỳ vọng bằng nhau
        assert visited_first == visited_second, (
            "stats phải được reset, không cộng dồn giữa các lượt"
        )

    def test_explain_last_action_cac_truong(self):
        """explain_last_action() phải trả về đầy đủ 5 trường cần thiết."""
        # Arrange
        board = np.zeros((11, 11), dtype=np.int8)
        agent = RuleBasedAgent()

        # Act
        agent.choose_action(board, PLAYER_X)
        info = agent.explain_last_action()

        # Assert
        required_keys = {
            "rule_triggered", "selected_action",
            "candidate_count", "matching_actions", "reason"
        }
        assert required_keys.issubset(info.keys()), (
            f"Thiếu trường trong explain_last_action: {required_keys - set(info.keys())}"
        )
        assert info["rule_triggered"] in ("win_now", "block_urgent", "positional_fallback")
        assert isinstance(info["selected_action"], tuple)
        assert isinstance(info["candidate_count"], int) and info["candidate_count"] > 0
        assert isinstance(info["matching_actions"], list)
        assert isinstance(info["reason"], str) and len(info["reason"]) > 0

        # Kiểm tra matching_actions ở positional_fallback chỉ lưu top_group điểm cao nhất
        board_one_stone = _build_board([(5, 5)], [])
        agent.choose_action(board_one_stone, PLAYER_O)
        info_fb = agent.explain_last_action()
        assert info_fb["candidate_count"] == 24
        assert len(info_fb["matching_actions"]) == 8, (
            "matching_actions ở positional_fallback chỉ nên giữ nhóm điểm vị trí cao nhất"
        )

    def test_explain_la_ban_sao_doc_lap(self):
        """explain_last_action() trả về bản sao, không ảnh hưởng last_decision_info."""
        # Arrange
        board = np.zeros((11, 11), dtype=np.int8)
        agent = RuleBasedAgent()
        agent.choose_action(board, PLAYER_X)

        # Act
        info = agent.explain_last_action()
        info["rule_triggered"] = "TAMPERED"  # sửa bản sao

        # Assert: nội bộ không bị ảnh hưởng
        info2 = agent.explain_last_action()
        assert info2["rule_triggered"] != "TAMPERED", (
            "explain_last_action phải trả về bản sao, không tham chiếu nội bộ"
        )

    def test_van_co_hoan_chinh_voi_random(self):
        """RuleBasedAgent đấu trọn vẹn ván cờ với RandomAgent không phát sinh lỗi."""
        # Arrange
        game = CaroGame()
        rule_agent = RuleBasedAgent()
        random_agent = RandomAgent(seed=42)
        agents = {PLAYER_X: rule_agent, PLAYER_O: random_agent}

        # Act: đấu đến khi kết thúc
        moves = 0
        while not game.is_terminal():
            current = game.current_player
            action = agents[current].choose_action(game.board, current)
            game.step(action)
            moves += 1

        # Assert
        assert game.is_terminal(), "Ván đấu phải kết thúc"
        assert game.result() in (-1, 0, 1), "Kết quả phải là -1, 0 hoặc 1"
        assert moves <= 121, f"Số nước không được vượt quá 121, thực tế: {moves}"

    def test_thinking_time_duong(self):
        """stats['thinking_time'] được đo đúng (>= 0) qua BaseAgent.choose_action."""
        # Arrange
        board = np.zeros((11, 11), dtype=np.int8)
        agent = RuleBasedAgent()

        # Act
        agent.choose_action(board, PLAYER_X)

        # Assert
        assert agent.stats["thinking_time"] >= 0, (
            "thinking_time phải >= 0 (do BaseAgent đo)"
        )


# ===========================================================================
# PHẦN 6 — Tie-breaking tất định và theo seed
# ===========================================================================

class TestTieBreaking:
    """Kiểm tra cơ chế phá hòa tất định và ngẫu nhiên có kiểm soát."""

    def test_tat_dinh_tra_ve_ket_qua_on_dinh(self):
        """Không có seed → kết quả luôn giống nhau khi gọi nhiều lần."""
        # Arrange
        board = np.zeros((11, 11), dtype=np.int8)
        agent1 = RuleBasedAgent(seed=None)
        agent2 = RuleBasedAgent(seed=None)

        # Act
        actions1 = [agent1.choose_action(board, PLAYER_X) for _ in range(5)]
        actions2 = [agent2.choose_action(board, PLAYER_X) for _ in range(5)]

        # Assert
        assert actions1 == actions2, (
            "Tie-breaking tất định phải trả về cùng kết quả với các agent khác nhau"
        )

    def test_seed_tai_lap_duoc(self):
        """Hai agent cùng seed cho kết quả giống hệt nhau."""
        # Arrange
        board = _board_from_stones([(5, 5)], [])
        agent1 = RuleBasedAgent(seed=7)
        agent2 = RuleBasedAgent(seed=7)

        # Act
        actions1 = [agent1.choose_action(board, PLAYER_O) for _ in range(10)]
        actions2 = [agent2.choose_action(board, PLAYER_O) for _ in range(10)]

        # Assert
        assert actions1 == actions2, "Hai agent cùng seed phải cho kết quả giống nhau"



# ===========================================================================
# GIAI ĐOẠN 2 — Nhận diện pattern và thứ tự ưu tiên (PHA RED của TDD)
# ===========================================================================
# Hợp đồng API Giai đoạn 2 trong agents/rule_based_agent.py (chưa cài đặt ở pha RED):
#   count_winning_threats(board, row, col, player, win_length=5) -> int
#   creates_open_four(board, row, col, player, win_length=5)     -> bool
#   creates_broken_four(board, row, col, player, win_length=5)   -> bool
#   creates_open_three(board, row, col, player, win_length=5)    -> bool
#
# Thứ tự ưu tiên khi use_patterns=True (cao -> thấp):
#   win_now > block_urgent > attack_open_four > block_open_four
#   > attack_broken_four > attack_open_three > block_open_three > positional_fallback
#
# Các hàm pattern được truy cập qua getattr để khi chưa cài đặt, test báo FAILED
# với thông điệp rõ ràng (không làm hỏng việc thu thập 48 test Giai đoạn 1).
# ===========================================================================

# Bảng hướng: tên -> (vector hướng, ô gốc). Mỗi đường có đủ chỗ cho ô k = -1..4
# nên khi đặt quân tại k = 3 sẽ tạo chuỗi 4 hở cả hai đầu (k = -1 và k = 4).
_LINE_ORIGINS: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {
    "ngang": ((0, 1), (5, 3)),
    "doc": ((1, 0), (3, 5)),
    "cheo_chinh": ((1, 1), (3, 3)),
    "cheo_phu": ((1, -1), (3, 7)),
}

# Ô đệm đặt ở góc/biên xa vùng thử nghiệm, dùng để cân bằng số quân theo lượt đi.
_FILLER_CELLS: list[tuple[int, int]] = [
    (0, 0), (0, 10), (10, 0), (10, 10), (0, 5), (10, 5),
]

# Bộ thế cờ theo nhãn luật: nhãn -> (quân của bên đến lượt, quân của đối thủ).
# Mọi thế đều dựng qua CaroGame.step() với số quân hợp lệ nhờ _build_turn_board.
_PATTERN_SCENARIOS: dict[str, tuple[list[tuple[int, int]], list[tuple[int, int]]]] = {
    # Bên đi có 3 quân hở hai đầu -> đánh thêm 1 quân tạo open four.
    "attack_open_four": ([(8, 3), (8, 4), (8, 5)], []),
    # Đối thủ có 3 quân hở -> phải chặn trước khi đối thủ tạo open four.
    "block_open_four": ([(8, 3), (8, 4)], [(2, 3), (2, 4), (2, 5)]),
    # Bên đi có 3 quân bị đối thủ chặn 1 đầu (8,2) -> chỉ tạo được broken four.
    "attack_broken_four": ([(8, 3), (8, 4), (8, 5), (2, 3), (2, 4)], [(8, 2)]),
    # Hai bên chỉ có 2 quân -> bên đi tự tạo open three.
    "attack_open_three": ([(8, 3), (8, 4)], [(2, 3), (2, 4)]),
    # Bên đi không có quân liên kết, đối thủ có 2 quân -> chặn open three.
    "block_open_three": ([], [(2, 3), (2, 4)]),
}


def _pattern_api(name: str):
    """Lấy hàm pattern Giai đoạn 2; báo FAILED rõ ràng nếu chưa được cài đặt (pha RED)."""
    func = getattr(rule_based_module, name, None)
    if func is None:
        pytest.fail(
            f"[RED] Chưa cài đặt `{name}(board, row, col, player, win_length=5)` "
            "trong agents/rule_based_agent.py (Giai đoạn 2).",
            pytrace=False,
        )
    return func


def _board_with_stones(
    player: int,
    own_stones: list[tuple[int, int]],
    opponent_stones: list[tuple[int, int]] = (),
) -> np.ndarray:
    """Dựng nhanh bàn cờ numpy: quân của `player` và (tùy chọn) quân đối thủ."""
    if player == PLAYER_X:
        return _board_from_stones(list(own_stones), list(opponent_stones))
    return _board_from_stones(list(opponent_stones), list(own_stones))


def _line_cells(
    origin: tuple[int, int], direction: tuple[int, int], steps: tuple[int, ...]
) -> list[tuple[int, int]]:
    """Trả về các ô origin + k * direction cho từng k trong steps."""
    return [
        (origin[0] + k * direction[0], origin[1] + k * direction[1]) for k in steps
    ]


def _build_turn_board(
    mover: int,
    mover_stones: list[tuple[int, int]],
    opponent_stones: list[tuple[int, int]],
) -> np.ndarray:
    """Dựng bàn cờ hợp lệ qua `_build_board` (CaroGame.step()) để đến lượt `mover`.

    Điều kiện số quân: |X| = |O| khi đến lượt X; |X| = |O| + 1 khi đến lượt O.
    Bên thiếu quân được bù bằng ô đệm ở góc/biên (không nằm trong vùng thử nghiệm).
    """
    if mover == PLAYER_X:
        x_stones, o_stones = list(mover_stones), list(opponent_stones)
    else:
        x_stones, o_stones = list(opponent_stones), list(mover_stones)
    wanted_gap = 0 if mover == PLAYER_X else 1
    used = set(x_stones) | set(o_stones)
    fillers = [cell for cell in _FILLER_CELLS if cell not in used]
    while len(x_stones) - len(o_stones) < wanted_gap:
        x_stones.append(fillers.pop(0))
    while len(x_stones) - len(o_stones) > wanted_gap:
        o_stones.append(fillers.pop(0))
    return _build_board(x_stones, o_stones)


def _scenario_board(label: str, mover: int) -> np.ndarray:
    """Dựng thế cờ của nhãn luật `label` cho bên `mover` đến lượt."""
    mover_stones, opponent_stones = _PATTERN_SCENARIOS[label]
    return _build_turn_board(mover, mover_stones, opponent_stones)


def _action_satisfies_label(
    board: np.ndarray, action: tuple[int, int], mover: int, label: str
) -> bool:
    """Kiểm tra nước `action` đúng bản chất của nhãn luật (dựa trên hàm pattern)."""
    row, col = action
    if label == "attack_open_four":
        return _pattern_api("creates_open_four")(board, row, col, mover)
    if label == "block_open_four":
        return _pattern_api("creates_open_four")(board, row, col, -mover)
    if label == "attack_broken_four":
        return _pattern_api("creates_broken_four")(board, row, col, mover)
    if label == "attack_open_three":
        return _pattern_api("creates_open_three")(board, row, col, mover)
    if label == "block_open_three":
        return _pattern_api("creates_open_three")(board, row, col, -mover)
    raise AssertionError(f"Nhãn luật không hỗ trợ: {label}")


# ===========================================================================
# PHẦN 7 — Nhận diện pattern (hàm cấp thấp)
# ===========================================================================

class TestPatternRecognition:
    """Kiểm tra độc lập các hàm nhận diện pattern, kể cả ca âm tính và bất biến board."""

    @pytest.mark.parametrize("player", [PLAYER_X, PLAYER_O])
    @pytest.mark.parametrize("direction_name", list(_LINE_ORIGINS))
    def test_creates_open_four_bon_huong_hai_phia(self, direction_name, player):
        """Đặt quân thứ 4 vào chuỗi 3 hở hai đầu tạo open four (2 ô thắng độc lập)."""
        # Arrange: ba quân tại k = 0, 1, 2; nước thử tại k = 3; hai đầu k = -1 và k = 4 trống
        direction, origin = _LINE_ORIGINS[direction_name]
        board = _board_with_stones(player, _line_cells(origin, direction, (0, 1, 2)))
        row, col = _line_cells(origin, direction, (3,))[0]

        # Act
        is_open_four = _pattern_api("creates_open_four")(board, row, col, player)
        is_broken_four = _pattern_api("creates_broken_four")(board, row, col, player)
        threats = _pattern_api("count_winning_threats")(board, row, col, player)

        # Assert
        assert is_open_four, f"Hướng {direction_name}: phải là open four"
        assert not is_broken_four, "Open four không được đồng thời là broken four"
        assert threats == 2, f"Open four phải có đúng 2 ô thắng, nhận {threats}"

    @pytest.mark.parametrize("player", [PLAYER_X, PLAYER_O])
    def test_creates_open_four_bi_chan_mot_dau_la_broken_four(self, player):
        """O X X X . -> đặt vào tạo O X X X X .: chỉ còn 1 đầu mở nên là broken four."""
        # Arrange: quân đối thủ chặn đầu trái tại (5,1)
        board = _board_with_stones(player, [(5, 2), (5, 3), (5, 4)], [(5, 1)])

        # Act
        is_open_four = _pattern_api("creates_open_four")(board, 5, 5, player)
        is_broken_four = _pattern_api("creates_broken_four")(board, 5, 5, player)

        # Assert
        assert not is_open_four, "Chuỗi 4 bị chặn 1 đầu không phải open four"
        assert is_broken_four, "Chuỗi 4 bị chặn 1 đầu phải là broken four"

    @pytest.mark.parametrize("player", [PLAYER_X, PLAYER_O])
    def test_creates_open_four_sat_bien_la_broken_four(self, player):
        """| X X X . (cột 0..3) -> chuỗi 4 sát mép biên chỉ có 1 đầu mở: broken four."""
        # Arrange: ba quân tại cột 0, 1, 2; nước thử tại cột 3
        board = _board_with_stones(player, [(5, 0), (5, 1), (5, 2)])

        # Act
        is_open_four = _pattern_api("creates_open_four")(board, 5, 3, player)
        is_broken_four = _pattern_api("creates_broken_four")(board, 5, 3, player)

        # Assert
        assert not is_open_four, "Chuỗi 4 sát biên không phải open four"
        assert is_broken_four, "Chuỗi 4 sát biên phải là broken four"

    @pytest.mark.parametrize("player", [PLAYER_X, PLAYER_O])
    def test_creates_open_four_khoang_trong_la_broken_four(self, player):
        """X X . X -> đặt thêm thành X X . X X: chỉ có 1 điểm lấp ở giữa nên là broken four."""
        # Arrange: quân tại (5,2), (5,3), (5,5); nước thử tại (5,6); ô lấp thắng là (5,4)
        board = _board_with_stones(player, [(5, 2), (5, 3), (5, 5)])

        # Act
        is_open_four = _pattern_api("creates_open_four")(board, 5, 6, player)
        is_broken_four = _pattern_api("creates_broken_four")(board, 5, 6, player)
        threats = _pattern_api("count_winning_threats")(board, 5, 6, player)

        # Assert
        assert not is_open_four, "X X . X X chỉ có 1 điểm thắng, không phải open four"
        assert is_broken_four, "X X . X X phải là broken four"
        assert threats == 1, f"Phải có đúng 1 ô thắng (5,4), nhận {threats}"

    @pytest.mark.parametrize("player", [PLAYER_X, PLAYER_O])
    def test_chuoi_bi_ngat_boi_doi_thu_khong_phai_four(self, player):
        """X X O X X: chuỗi bị quân đối thủ ngắt -> không phải open four lẫn broken four."""
        # Arrange: quân đối thủ nằm giữa tại (5,4); nước thử tại (5,6)
        board = _board_with_stones(player, [(5, 2), (5, 3), (5, 5)], [(5, 4)])

        # Act
        is_open_four = _pattern_api("creates_open_four")(board, 5, 6, player)
        is_broken_four = _pattern_api("creates_broken_four")(board, 5, 6, player)

        # Assert
        assert not is_open_four, "Chuỗi bị ngắt không phải open four"
        assert not is_broken_four, "Chuỗi bị ngắt không phải broken four"

    @pytest.mark.parametrize("player", [PLAYER_X, PLAYER_O])
    @pytest.mark.parametrize("direction_name", list(_LINE_ORIGINS))
    def test_creates_open_three_bon_huong(self, direction_name, player):
        """Đặt quân thứ 3 vào chuỗi 2 hở (. . X X X . .) tạo open three ở cả 4 hướng."""
        # Arrange: hai quân tại k = 0, 1; nước thử tại k = 2; còn chỗ để lên open four
        direction, origin = _LINE_ORIGINS[direction_name]
        board = _board_with_stones(player, _line_cells(origin, direction, (0, 1)))
        row, col = _line_cells(origin, direction, (2,))[0]

        # Act
        is_open_three = _pattern_api("creates_open_three")(board, row, col, player)
        is_open_four = _pattern_api("creates_open_four")(board, row, col, player)
        is_broken_four = _pattern_api("creates_broken_four")(board, row, col, player)

        # Assert
        assert is_open_three, f"Hướng {direction_name}: phải là open three"
        assert not is_open_four and not is_broken_four, (
            "Open three chưa tạo nước thắng ngay nên không phải open/broken four"
        )

    @pytest.mark.parametrize("player", [PLAYER_X, PLAYER_O])
    def test_creates_open_three_chuoi_khoang_trong(self, player):
        """. X . X . -> đánh vào thành . X . X X . vẫn là open three (lấp ô giữa lên open four)."""
        # Arrange: quân tại (5,3), (5,5); nước thử tại (5,6)
        board = _board_with_stones(player, [(5, 3), (5, 5)])

        # Act
        is_open_three = _pattern_api("creates_open_three")(board, 5, 6, player)

        # Assert
        assert is_open_three, ". X . X X . phải là open three"

    @pytest.mark.parametrize("player", [PLAYER_X, PLAYER_O])
    def test_creates_open_three_chuoi_khoang_trong_thu_hai(self, player):
        """. X X . X . -> đánh quân rời thành chuỗi có khoảng trống vẫn là open three."""
        # Arrange: quân tại (5,3), (5,4); nước thử tại (5,6); ô lấp (5,5) tạo open four
        board = _board_with_stones(player, [(5, 3), (5, 4)])

        # Act
        is_open_three = _pattern_api("creates_open_three")(board, 5, 6, player)

        # Assert
        assert is_open_three, ". X X . X . phải là open three"

    @pytest.mark.parametrize("player", [PLAYER_X, PLAYER_O])
    def test_creates_open_three_bi_chan_mot_dau_khong_phai(self, player):
        """O X X . . -> đặt vào thành O X X X . .: bị chặn một đầu nên KHÔNG phải open three."""
        # Arrange: quân đối thủ chặn đầu trái tại (5,2); nước thử tại (5,5)
        board = _board_with_stones(player, [(5, 3), (5, 4)], [(5, 2)])

        # Act
        is_open_three = _pattern_api("creates_open_three")(board, 5, 5, player)

        # Assert
        assert not is_open_three, "Chuỗi 3 bị chặn 1 đầu không đủ để lên open four"

    @pytest.mark.parametrize("player", [PLAYER_X, PLAYER_O])
    def test_creates_open_three_sat_bien_khong_phai(self, player):
        """| X X . -> sát mép biên không đủ không gian 5 ô hở hai đầu nên KHÔNG phải open three."""
        # Arrange: hai quân tại cột 0, 1; nước thử tại cột 2
        board = _board_with_stones(player, [(5, 0), (5, 1)])

        # Act
        is_open_three = _pattern_api("creates_open_three")(board, 5, 2, player)

        # Assert
        assert not is_open_three, "Chuỗi 3 sát biên không thể lên open four"

    def test_count_winning_threats_khong_co_nuoc_thang(self):
        """Đặt quân thứ 3 chưa tạo ô thắng nào -> count_winning_threats == 0."""
        # Arrange
        board = _board_with_stones(PLAYER_X, [(5, 4), (5, 5)])

        # Act
        threats = _pattern_api("count_winning_threats")(board, 5, 6, PLAYER_X)

        # Assert
        assert threats == 0, f"Chưa có ô thắng nào, nhận {threats}"

    def test_pattern_khong_sua_board(self):
        """Các hàm pattern không được làm biến đổi mảng board đầu vào."""
        # Arrange: thế cờ có cả quân X và O, thử trên nhiều ô khác nhau
        board = _board_with_stones(
            PLAYER_X, [(5, 2), (5, 3), (5, 4), (3, 3)], [(5, 1), (7, 7), (2, 8)]
        )
        snapshot = board.copy()
        names = (
            "count_winning_threats",
            "creates_open_four",
            "creates_broken_four",
            "creates_open_three",
        )

        # Act: gọi mọi hàm trên nhiều ô và cả hai phía
        for name in names:
            func = _pattern_api(name)
            for row, col in [(5, 5), (5, 0), (4, 4), (0, 0)]:
                for player in (PLAYER_X, PLAYER_O):
                    func(board, row, col, player)

        # Assert
        np.testing.assert_array_equal(
            board, snapshot, err_msg="Hàm pattern đã sửa đổi mảng board đầu vào"
        )

    def test_kieu_tra_ve_cua_ham_pattern(self):
        """count_winning_threats trả về int; các hàm creates_* trả về bool thuần."""
        # Arrange
        board = _board_with_stones(PLAYER_X, [(5, 3), (5, 4), (5, 5)])

        # Act
        threats = _pattern_api("count_winning_threats")(board, 5, 6, PLAYER_X)
        flags = [
            _pattern_api(name)(board, 5, 6, PLAYER_X)
            for name in ("creates_open_four", "creates_broken_four", "creates_open_three")
        ]

        # Assert
        assert isinstance(threats, int) and not isinstance(threats, bool)
        assert all(isinstance(flag, bool) for flag in flags), (
            f"Các hàm creates_* phải trả về bool thuần, nhận {[type(f) for f in flags]}"
        )


# ===========================================================================
# PHẦN 8 — Thứ tự ưu tiên và xung đột giữa các tầng luật (use_patterns=True)
# ===========================================================================

class TestPatternPriorityAndConflict:
    """Kiểm tra bảng ưu tiên: tầng cao hơn phải thắng khi nhiều luật cùng kích hoạt."""

    def test_win_now_hon_attack_open_four(self):
        """X vừa có nước thắng ngay, vừa có nước tạo open four -> chọn win_now."""
        # Arrange: hàng 5 có 4 quân (thắng tại (5,0)/(5,5)); cột 3 có 3 quân hở (đánh (8,3) tạo open four)
        board = _build_turn_board(
            PLAYER_X,
            [(5, 1), (5, 2), (5, 3), (5, 4), (6, 3), (7, 3)],
            [],
        )
        agent = RuleBasedAgent(use_patterns=True)

        # Act
        action = agent.choose_action(board, PLAYER_X)
        info = agent.explain_last_action()

        # Assert
        assert info["rule_triggered"] == "win_now"
        assert action in {(5, 0), (5, 5)}, f"Phải thắng ngay tại đầu chuỗi, nhận {action}"

    def test_block_urgent_hon_attack_open_four(self):
        """X có thể tạo open four nhưng O đang đe dọa thắng ngay -> chọn block_urgent."""
        # Arrange: X có 3 quân hở hàng 8; O có 4 quân hàng 2 (thắng tại (2,0)/(2,5))
        board = _build_turn_board(
            PLAYER_X,
            [(8, 3), (8, 4), (8, 5)],
            [(2, 1), (2, 2), (2, 3), (2, 4)],
        )
        agent = RuleBasedAgent(use_patterns=True)

        # Act
        action = agent.choose_action(board, PLAYER_X)
        info = agent.explain_last_action()

        # Assert
        assert info["rule_triggered"] == "block_urgent"
        assert action in {(2, 0), (2, 5)}, f"Phải chặn một đầu chuỗi 4 của O, nhận {action}"

    def test_attack_open_four_hon_block_open_four(self):
        """Hai bên đều có 3 quân hở: tự tạo open four kết liễu ưu tiên hơn chặn đối thủ."""
        # Arrange: X 3 quân hàng 8, O 3 quân hàng 2; X đến lượt
        board = _build_turn_board(
            PLAYER_X,
            [(8, 3), (8, 4), (8, 5)],
            [(2, 3), (2, 4), (2, 5)],
        )
        agent = RuleBasedAgent(use_patterns=True)

        # Act
        action = agent.choose_action(board, PLAYER_X)
        info = agent.explain_last_action()

        # Assert
        assert info["rule_triggered"] == "attack_open_four"
        assert _pattern_api("creates_open_four")(board, *action, PLAYER_X), (
            f"Nước {action} phải tạo open four cho X"
        )

    def test_block_open_four_hon_attack_broken_four(self):
        """X chỉ tạo được broken four (bị chặn 1 đầu) nhưng O sắp tạo open four -> phải chặn trước."""
        # Arrange: X 3 quân hàng 8 bị O chặn tại (8,2); O có 3 quân hở hàng 2
        board = _build_turn_board(
            PLAYER_X,
            [(8, 3), (8, 4), (8, 5)],
            [(8, 2), (2, 3), (2, 4), (2, 5)],
        )
        assert _pattern_api("creates_broken_four")(board, 8, 6, PLAYER_X), (
            "Điều kiện thế cờ: X phải có nước tạo broken four tại (8,6)"
        )
        agent = RuleBasedAgent(use_patterns=True)

        # Act
        action = agent.choose_action(board, PLAYER_X)
        info = agent.explain_last_action()

        # Assert
        assert info["rule_triggered"] == "block_open_four"
        assert _pattern_api("creates_open_four")(board, *action, PLAYER_O), (
            f"Nước {action} phải là ô O dùng để tạo open four"
        )

    def test_block_open_four_hon_attack_open_three(self):
        """X có thể tự tạo open three nhưng O sắp tạo open four -> phải chặn trước."""
        # Arrange: X 2 quân hàng 8; O 3 quân hở hàng 2
        board = _scenario_board("block_open_four", PLAYER_X)
        assert _pattern_api("creates_open_three")(board, 8, 5, PLAYER_X), (
            "Điều kiện thế cờ: X phải có nước tạo open three tại (8,5)"
        )
        agent = RuleBasedAgent(use_patterns=True)

        # Act
        action = agent.choose_action(board, PLAYER_X)
        info = agent.explain_last_action()

        # Assert
        assert info["rule_triggered"] == "block_open_four"
        assert _pattern_api("creates_open_four")(board, *action, PLAYER_O)

    def test_attack_broken_four_hon_attack_open_three(self):
        """X vừa tạo được broken four vừa tạo được open three -> ưu tiên broken four."""
        # Arrange: hàng 8 có 3 quân X bị O chặn tại (8,2); hàng 2 có 2 quân X hở
        board = _scenario_board("attack_broken_four", PLAYER_X)
        assert _pattern_api("creates_open_three")(board, 2, 5, PLAYER_X), (
            "Điều kiện thế cờ: X phải có nước tạo open three tại (2,5)"
        )
        agent = RuleBasedAgent(use_patterns=True)

        # Act
        action = agent.choose_action(board, PLAYER_X)
        info = agent.explain_last_action()

        # Assert
        assert info["rule_triggered"] == "attack_broken_four"
        assert _pattern_api("creates_broken_four")(board, *action, PLAYER_X)

    def test_attack_open_three_hon_block_open_three(self):
        """Hai bên đều có 2 quân hở: tự tạo open three ưu tiên hơn chặn open three của đối thủ."""
        # Arrange
        board = _scenario_board("attack_open_three", PLAYER_X)
        assert _pattern_api("creates_open_three")(board, 2, 5, PLAYER_O), (
            "Điều kiện thế cờ: O phải có nước tạo open three tại (2,5)"
        )
        agent = RuleBasedAgent(use_patterns=True)

        # Act
        action = agent.choose_action(board, PLAYER_X)
        info = agent.explain_last_action()

        # Assert
        assert info["rule_triggered"] == "attack_open_three"
        assert _pattern_api("creates_open_three")(board, *action, PLAYER_X)

    def test_block_open_three_hon_positional_fallback(self):
        """X không có đòn tấn công nào, O sắp tạo open three -> chặn thay vì chọn theo vị trí."""
        # Arrange
        board = _scenario_board("block_open_three", PLAYER_X)
        agent = RuleBasedAgent(use_patterns=True)

        # Act
        action = agent.choose_action(board, PLAYER_X)
        info = agent.explain_last_action()

        # Assert
        assert info["rule_triggered"] == "block_open_three"
        assert _pattern_api("creates_open_three")(board, *action, PLAYER_O)

    @pytest.mark.parametrize("player", [PLAYER_X, PLAYER_O])
    @pytest.mark.parametrize(
        "label",
        [
            "attack_open_four",
            "block_open_four",
            "attack_broken_four",
            "attack_open_three",
            "block_open_three",
        ],
    )
    def test_doi_xung_goc_nhin_x_o(self, label, player):
        """Đảo vai X/O trên cùng hình thái cờ: tác tử phải chọn cùng loại luật, không thiên lệch."""
        # Arrange: cùng hình thái, bên `player` đến lượt (số quân hợp lệ theo lượt đi)
        board = _scenario_board(label, player)
        snapshot = board.copy()
        agent = RuleBasedAgent(use_patterns=True)

        # Act
        action = agent.choose_action(board, player)
        info = agent.explain_last_action()

        # Assert
        assert info["rule_triggered"] == label, (
            f"player={player}: kỳ vọng {label}, nhận {info['rule_triggered']}"
        )
        assert _action_satisfies_label(board, action, player, label), (
            f"player={player}: nước {action} không đúng bản chất luật {label}"
        )
        np.testing.assert_array_equal(board, snapshot)


# ===========================================================================
# PHẦN 9 — Đe dọa kép, truy vết và tương thích ngược
# ===========================================================================

class TestDoubleThreatAndPatternContract:
    """Kiểm tra đe dọa kép, hợp đồng truy vết, tương thích ngược và phá hòa ở tầng pattern."""

    def test_de_doa_kep_hai_dau_van_kich_hoat_block_urgent(self):
        """. O O O O . (2 ô thắng độc lập): agent chặn 1 trong 2 đầu hợp lệ, không phát sinh lỗi."""
        # Arrange: O có 4 quân hở hai đầu hàng 2; X (đến lượt) không có đòn nào
        board = _build_turn_board(PLAYER_X, [], [(2, 1), (2, 2), (2, 3), (2, 4)])
        agent = RuleBasedAgent(use_patterns=True)

        # Act
        action = agent.choose_action(board, PLAYER_X)
        info = agent.explain_last_action()

        # Assert
        assert info["rule_triggered"] == "block_urgent"
        assert action in {(2, 0), (2, 5)}, f"Phải chặn một trong hai đầu, nhận {action}"
        assert board[action] == EMPTY, "Nước chặn phải nằm trên ô trống"

    @pytest.mark.parametrize(
        "label",
        [
            "attack_open_four",
            "block_open_four",
            "attack_broken_four",
            "attack_open_three",
            "block_open_three",
        ],
    )
    def test_explain_tang_pattern_day_du_khoa(self, label):
        """Tầng pattern phải ghi đủ 5 khóa truy vết (reason tiếng Việt), action là tuple[int, int]."""
        # Arrange
        board = _scenario_board(label, PLAYER_X)
        agent = RuleBasedAgent(use_patterns=True)

        # Act
        action = agent.choose_action(board, PLAYER_X)
        info = agent.explain_last_action()

        # Assert: kiểu trả về của choose_action
        assert isinstance(action, tuple) and len(action) == 2
        assert all(type(value) is int for value in action), (
            f"Hành động phải gồm số nguyên Python thuần, nhận {action!r}"
        )
        # Assert: nhãn luật và đủ 5 khóa truy vết
        required_keys = {
            "rule_triggered", "selected_action",
            "candidate_count", "matching_actions", "reason",
        }
        assert required_keys.issubset(info.keys()), (
            f"Thiếu khóa: {required_keys - set(info.keys())}"
        )
        assert info["rule_triggered"] == label
        assert info["selected_action"] == action
        assert isinstance(info["candidate_count"], int) and info["candidate_count"] > 0
        assert isinstance(info["matching_actions"], list)
        assert action in info["matching_actions"], (
            "Nước đã chọn phải nằm trong matching_actions của tầng kích hoạt"
        )
        assert isinstance(info["reason"], str) and info["reason"], "reason không được rỗng"
        assert any(ord(char) > 127 for char in info["reason"]), (
            "reason phải bằng tiếng Việt (có ký tự có dấu)"
        )

    @pytest.mark.parametrize(
        "label", ["attack_open_four", "attack_open_three", "block_open_three"]
    )
    def test_use_patterns_false_bo_qua_tang_pattern(self, label):
        """use_patterns=False bỏ qua tầng pattern và rơi xuống positional_fallback; True thì kích hoạt."""
        # Arrange: thế cờ chưa có win_now/block_urgent nhưng có pattern tiềm năng
        board = _scenario_board(label, PLAYER_X)
        agent_off = RuleBasedAgent(use_patterns=False)
        agent_on = RuleBasedAgent(use_patterns=True)

        # Act
        agent_off.choose_action(board, PLAYER_X)
        agent_on.choose_action(board, PLAYER_X)

        # Assert
        assert agent_off.explain_last_action()["rule_triggered"] == "positional_fallback", (
            "use_patterns=False phải giữ nguyên hành vi baseline Giai đoạn 1"
        )
        assert agent_on.explain_last_action()["rule_triggered"] == label, (
            "use_patterns=True phải kích hoạt đúng tầng pattern"
        )

    def test_mac_dinh_use_patterns_la_false(self):
        """Mặc định use_patterns=False để không làm đổi hành vi baseline Giai đoạn 1."""
        # Arrange & Act
        agent = RuleBasedAgent()

        # Assert
        assert agent.use_patterns is False

    @pytest.mark.parametrize("label", ["attack_open_four", "attack_open_three"])
    def test_pha_hoa_tat_dinh_o_tang_pattern(self, label):
        """seed=None: nhiều nước cùng loại pattern vẫn cho kết quả tất định, ổn định giữa các agent."""
        # Arrange
        board = _scenario_board(label, PLAYER_X)
        agent_a = RuleBasedAgent(use_patterns=True, seed=None)
        agent_b = RuleBasedAgent(use_patterns=True, seed=None)

        # Act
        actions_a = [agent_a.choose_action(board, PLAYER_X) for _ in range(5)]
        actions_b = [agent_b.choose_action(board, PLAYER_X) for _ in range(5)]

        # Assert
        assert agent_a.explain_last_action()["rule_triggered"] == label
        assert len(set(actions_a)) == 1, "Tất định: 5 lần gọi liên tiếp phải cùng một nước"
        assert actions_a == actions_b, "Tất định: hai agent khác nhau phải cùng kết quả"

    @pytest.mark.parametrize("label", ["attack_open_four", "attack_open_three"])
    def test_pha_hoa_seed_tai_lap_o_tang_pattern(self, label):
        """seed=42: hai instance cùng seed tái lập kết quả; mọi nước đều đúng bản chất luật."""
        # Arrange
        board = _scenario_board(label, PLAYER_X)
        agent_a = RuleBasedAgent(use_patterns=True, seed=42)
        agent_b = RuleBasedAgent(use_patterns=True, seed=42)

        # Act
        actions_a = [agent_a.choose_action(board, PLAYER_X) for _ in range(10)]
        actions_b = [agent_b.choose_action(board, PLAYER_X) for _ in range(10)]

        # Assert
        assert agent_a.explain_last_action()["rule_triggered"] == label
        assert actions_a == actions_b, "Cùng seed phải tái lập đúng dãy nước đi"
        assert all(
            _action_satisfies_label(board, action, PLAYER_X, label) for action in actions_a
        ), "Mọi nước chọn ngẫu nhiên phải đúng bản chất luật pattern"

    @pytest.mark.parametrize("label", list(_PATTERN_SCENARIOS))
    def test_tang_pattern_khong_sua_board(self, label):
        """choose_action với use_patterns=True không được sửa mảng board đầu vào."""
        # Arrange
        board = _scenario_board(label, PLAYER_X)
        snapshot = board.copy()
        agent = RuleBasedAgent(use_patterns=True)

        # Act
        agent.choose_action(board, PLAYER_X)

        # Assert
        np.testing.assert_array_equal(
            board, snapshot, err_msg="Tầng pattern đã sửa đổi mảng board đầu vào"
        )


# ===========================================================================
# PHẦN 10 — Điều chỉnh theo góp ý của Reviewer
# ===========================================================================

class TestReviewerFeedback:
    """Kiểm tra 3 điều chỉnh từ Reviewer: sao chép sâu, chặn tối ưu, mốc dừng thời gian."""

    def test_explain_last_action_sao_chep_sau_matching_actions(self):
        """Sửa list trả về từ explain_last_action() không được làm hỏng dữ liệu nội bộ."""
        # Arrange
        board = _scenario_board("attack_open_four", PLAYER_X)
        agent = RuleBasedAgent(use_patterns=True)
        agent.choose_action(board, PLAYER_X)
        info = agent.explain_last_action()
        expected = list(info["matching_actions"])
        assert expected, "Điều kiện tiên quyết: matching_actions phải khác rỗng"

        # Act
        info["matching_actions"].clear()
        info_again = agent.explain_last_action()

        # Assert
        assert info_again["matching_actions"] == expected, (
            "matching_actions nội bộ đã bị sửa qua bản sao trả về (copy nông)"
        )
        assert info_again["matching_actions"] is not info["matching_actions"]

    def test_chan_open_four_chon_nuoc_de_lai_it_co_hoi_nhat(self):
        """O có (0,1)-(0,3) và (1,0)-(3,0): X phải chọn (0,0) chặn cả hai hướng."""
        # Arrange: quân X rời rạc, không tạo đe dọa riêng; (1, 5) làm (0, 4) có
        # điểm vị trí cao hơn (0, 0) để phép chọn chỉ theo điểm vị trí sẽ chọn sai
        o_stones = [(0, 1), (0, 2), (0, 3), (1, 0), (2, 0), (3, 0)]
        x_stones = [(6, 6), (10, 9), (1, 5), (9, 2), (5, 3), (8, 10)]
        board = _build_turn_board(PLAYER_X, x_stones, o_stones)
        snapshot = board.copy()
        agent = RuleBasedAgent(use_patterns=True)

        # Act
        action = agent.choose_action(board, PLAYER_X)
        info = agent.explain_last_action()

        # Assert
        assert info["rule_triggered"] == "block_open_four"
        assert (0, 4) in info["matching_actions"] and (4, 0) in info["matching_actions"]
        assert action == (0, 0), (
            f"Phải chọn (0, 0) để chặn cả hai hướng, nhận {action}"
        )
        np.testing.assert_array_equal(board, snapshot)

    def test_het_gio_giua_chung_luc_quet_pattern_tra_nuoc_du_phong(self, monkeypatch):
        """Hết giờ giữa chừng: dừng quét ngay và trả về nước dự phòng đã chuẩn bị sẵn."""
        # Arrange: thế cờ không có pattern nào → quét hết sẽ rơi xuống fallback
        board = _build_turn_board(PLAYER_X, [(5, 5)], [(5, 6)])
        total_empty = int(np.count_nonzero(board == EMPTY))
        budget = rule_based_module.SearchBudget(mode="time", time_limit_s=2.0)
        agent_full = RuleBasedAgent(use_patterns=True)
        agent_limited = RuleBasedAgent(use_patterns=True, budget=budget)
        calls = {"n": 0}

        def fake_time_exhausted() -> bool:
            calls["n"] += 1
            return calls["n"] > 12

        monkeypatch.setattr(agent_limited, "time_exhausted", fake_time_exhausted)

        # Act
        full_action = agent_full.choose_action(board, PLAYER_X)
        limited_action = agent_limited.choose_action(board, PLAYER_X)
        info = agent_limited.explain_last_action()

        # Assert
        full_nodes = agent_full.stats["nodes_visited"]
        limited_nodes = agent_limited.stats["nodes_visited"]
        assert limited_nodes < full_nodes, "Hết giờ phải quét ít ứng viên hơn"
        assert limited_nodes - 2 * total_empty <= 12, (
            "Chỉ được đếm node cho ứng viên thực sự đã kiểm tra"
        )
        assert info["rule_triggered"] == "positional_fallback"
        assert limited_action == full_action
        assert board[limited_action] == EMPTY


# ===========================================================================
# PHẦN 11 — Tự rà soát chuyên sâu (Self-Audit & Hardening)
# ===========================================================================

class TestSelfAuditAndHardening:
    """Kiểm toán toàn diện: Kiểu dữ liệu thuần, chặn tối ưu block_open_three và block_urgent, dừng thời gian."""

    def test_explain_last_action_pure_python_ints_and_deep_copy(self):
        """Mọi tọa độ trong explain_last_action() phải là int thuần Python và bất biến với sửa đổi ngoài."""
        # Arrange
        board = _scenario_board("attack_open_four", PLAYER_X)
        agent = RuleBasedAgent(use_patterns=True)
        agent.choose_action(board, PLAYER_X)

        # Act
        info = agent.explain_last_action()
        sel = info["selected_action"]
        matches = info["matching_actions"]

        # Assert: kiểu int thuần (không phải numpy int)
        assert isinstance(sel, tuple) and len(sel) == 2
        assert type(sel[0]) is int and type(sel[1]) is int
        for r, c in matches:
            assert isinstance((r, c), tuple) and len((r, c)) == 2
            assert type(r) is int and type(c) is int

        # Assert: sửa đổi dict và list trả về không đổi dữ liệu nội bộ
        matches.clear()
        info["selected_action"] = (-99, -99)
        fresh_info = agent.explain_last_action()
        assert len(fresh_info["matching_actions"]) > 0
        assert fresh_info["selected_action"] == sel

    def test_block_open_three_uu_tien_nuoc_de_lai_it_co_hoi_nhat(self):
        """O có hai đường 2 quân (3,4)-(3,5) và (4,3)-(5,3) giao nhau tại (3,3).
        X phải chọn (3,3) để triệt tiêu cả hai hướng tạo open-three của O,
        thay vì chọn (3,6) dù (3,6) có quân đồng minh X ở (3,8) kéo điểm vị trí cao hơn.
        """
        # Arrange
        board = np.zeros((11, 11), dtype=np.int8)
        board[3, 4] = PLAYER_O
        board[3, 5] = PLAYER_O
        board[4, 3] = PLAYER_O
        board[5, 3] = PLAYER_O
        # X có quân đồng minh ở (3,8) để (3,6) có điểm vị trí cao hơn (3,3)
        board[3, 8] = PLAYER_X
        board[8, 2] = PLAYER_X
        board[9, 9] = PLAYER_X
        board[10, 10] = PLAYER_X

        agent = RuleBasedAgent(use_patterns=True)

        # Act
        action = agent.choose_action(board, PLAYER_X)
        info = agent.explain_last_action()

        # Assert
        assert info["rule_triggered"] == "block_open_three"
        assert action == (3, 3), f"Phải chọn (3, 3) để triệt tiêu cả hai hướng open-three, nhận {action}"

    def test_block_urgent_uu_tien_nuoc_chan_triet_tieu_nhieu_de_doa_nhat(self):
        """O có hai đường 4 quân: đường ngang (1,1)-(1,4) (threats tại (1,0) và (1,5))
        và đường dọc (2,5)-(5,5) bị chặn ở (6,5) (threat duy nhất tại (1,5)).
        X phải chọn (1,5) vì để lại 0 nước thắng cho O, thay vì (1,0) dù (1,0) có điểm vị trí cao hơn.
        """
        # Arrange
        board = np.zeros((11, 11), dtype=np.int8)
        for c in range(1, 5):
            board[1, c] = PLAYER_O
        for r in range(2, 6):
            board[r, 5] = PLAYER_O
        board[6, 5] = PLAYER_X  # chặn đầu dưới của đường dọc

        # Quân X gần (1,0) để (1,0) có điểm vị trí cao hơn (1,5)
        board[0, 0] = PLAYER_X
        board[2, 0] = PLAYER_X
        board[0, 1] = PLAYER_X
        board[2, 1] = PLAYER_X
        board[0, 2] = PLAYER_X
        board[9, 9] = PLAYER_X
        board[10, 10] = PLAYER_X

        o_threats = find_winning_actions(board, PLAYER_O)
        assert set(o_threats) == {(1, 0), (1, 5)}

        agent = RuleBasedAgent(use_patterns=True)

        # Act
        action = agent.choose_action(board, PLAYER_X)
        info = agent.explain_last_action()

        # Assert
        assert info["rule_triggered"] == "block_urgent"
        assert action == (1, 5), f"Phải chọn (1, 5) để triệt tiêu hoàn toàn nước thắng của O, nhận {action}"

    def test_least_counterplay_dung_thoi_gian_an_toan(self, monkeypatch):
        """Khi _should_stop_scanning kích hoạt trong _least_counterplay_actions, agent không bị treo."""
        # Arrange
        o_stones = [(0, 1), (0, 2), (0, 3), (1, 0), (2, 0), (3, 0)]
        x_stones = [(6, 6), (10, 9), (1, 5), (9, 2), (5, 3), (8, 10)]
        board = _build_turn_board(PLAYER_X, x_stones, o_stones)
        agent = RuleBasedAgent(use_patterns=True)

        # Giả lập hết giờ sau 1 lần kiểm tra
        call_count = {"cnt": 0}

        def mock_should_stop():
            call_count["cnt"] += 1
            return call_count["cnt"] > 3

        monkeypatch.setattr(agent, "_should_stop_scanning", mock_should_stop)

        # Act & Assert: gọi bình thường không sinh lỗi và trả về nước hợp lệ
        action = agent.choose_action(board, PLAYER_X)
        assert isinstance(action, tuple) and len(action) == 2
        assert board[action] == EMPTY
