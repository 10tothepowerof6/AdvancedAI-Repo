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
