"""Tác tử Rule-based cho cờ Caro 11×11.

Giai đoạn 1 — Luật thắng ngay và chặn thua (baseline chiến thuật).
Giai đoạn 2 — Mở rộng nhận diện pattern nâng cao (use_patterns=True).
Giai đoạn 3 — Đánh giá tác dụng pattern và bàn giao.

Mọi phương thức công khai đều tuân thủ hợp đồng BaseAgent:
  - Không sửa mảng board đầu vào.
  - Trả về tuple (row, col) số nguyên Python thuần.
  - Phương thức gọi từ bên ngoài: choose_action(board, player).
"""

from __future__ import annotations

import numpy as np

from agents.base_agent import BaseAgent
from config.game_config import BOARD_SIZE, EMPTY, WIN_LENGTH
from config.search_budget import SearchBudget
from core.game_engine import Action
from core.game_utils import candidate_actions

# 4 hướng kiểm tra: ngang, dọc, chéo chính, chéo phụ
DIRECTIONS: tuple[tuple[int, int], ...] = ((0, 1), (1, 0), (1, 1), (1, -1))


# ---------------------------------------------------------------------------
# Hàm hỗ trợ cấp module — có thể import riêng lẻ trong notebook
# ---------------------------------------------------------------------------

def is_winning_move(
    board: np.ndarray,
    row: int,
    col: int,
    player: int,
    win_length: int = WIN_LENGTH,
) -> bool:
    """Kiểm tra nếu player đặt quân vào (row, col) thì thắng ngay hay không.

    Điều kiện:
      - Ô (row, col) đang trống.
      - Sau khi đặt quân player tại đó, ít nhất 1 trong 4 hướng có chuỗi
        liên tiếp >= win_length (kể cả bị chặn cả hai đầu).

    Hàm KHÔNG sửa mảng board gốc — dùng biến đếm trực tiếp trên board.
    """
    if board[row, col] != EMPTY:
        return False

    board_size = board.shape[0]

    for dr, dc in DIRECTIONS:
        # Đếm quân liên tiếp của player về 2 phía, tính cả ô trung tâm
        count = 1  # tính ô (row, col) sẽ đặt quân
        for sign in (-1, 1):
            r, c = row + sign * dr, col + sign * dc
            while (
                0 <= r < board_size
                and 0 <= c < board_size
                and board[r, c] == player
            ):
                count += 1
                r += sign * dr
                c += sign * dc
        if count >= win_length:
            return True

    return False


def find_winning_actions(
    board: np.ndarray,
    player: int,
    win_length: int = WIN_LENGTH,
) -> list[Action]:
    """Tìm tất cả nước đi trên TOÀN BỘ ô trống giúp player thắng ngay.

    Quét toàn bàn (không giới hạn bán kính) để tránh bỏ sót đe dọa.
    Kết quả sắp theo (row, col) tăng dần để ổn định kết quả.
    """
    winning: list[Action] = []
    for r, c in np.argwhere(board == EMPTY):
        row, col = int(r), int(c)
        if is_winning_move(board, row, col, player, win_length):
            winning.append((row, col))
    return winning


def positional_score(
    board: np.ndarray,
    row: int,
    col: int,
    player: int,
) -> float:
    """Điểm ưu tiên vị trí dùng làm nước đi dự phòng (positional fallback).

    Thành phần:
      1. Điểm gần tâm: giảm dần theo khoảng cách Chebyshev tới tâm bàn cờ.
      2. Điểm thưởng lân cận: cộng thêm cho mỗi quân cùng màu trong bán kính 2
         (giúp ưu tiên kéo dài chuỗi ngay cả khi chưa có đe dọa cụ thể).
    """
    board_size = board.shape[0]
    center = board_size // 2

    # Điểm gần tâm (tối đa = center, bàn 11×11 → tối đa 5)
    chebyshev = max(abs(row - center), abs(col - center))
    score = float(center - chebyshev)

    # Điểm thưởng quân lân cận bán kính 2
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            if dr == 0 and dc == 0:
                continue
            nr, nc = row + dr, col + dc
            if 0 <= nr < board_size and 0 <= nc < board_size:
                if board[nr, nc] == player:
                    # Quân cùng màu trong bán kính 1 được thưởng nhiều hơn
                    bonus = 0.3 if max(abs(dr), abs(dc)) == 1 else 0.1
                    score += bonus
                elif board[nr, nc] == -player:
                    # Thưởng nhỏ cho gần quân đối thủ (kiểm soát không gian)
                    score += 0.05

    return score


# ---------------------------------------------------------------------------
# Lớp RuleBasedAgent
# ---------------------------------------------------------------------------

class RuleBasedAgent(BaseAgent):
    """Tác tử dựa trên tập luật chiến thuật cho cờ Caro 11×11.

    Thứ tự ưu tiên quyết định nước đi:
      1. win_now      — Thắng ngay lập tức nếu có thể.
      2. block_urgent — Chặn đối thủ thắng ngay ở lượt sau.
      3. positional_fallback — Chọn vị trí tốt nhất từ tập ứng viên.

    Tham số:
      name             : Tên hiển thị của tác tử.
      budget           : Ngân sách tìm kiếm (SearchBudget).
      candidate_radius : Bán kính tập ứng viên cho fallback (mặc định 2).
      use_patterns     : Bật nhận diện pattern nâng cao (Giai đoạn 2).
      seed             : Seed cho tie-breaking ngẫu nhiên; None = tất định.
    """

    def __init__(
        self,
        name: str = "RuleBased",
        *,
        budget: SearchBudget | None = None,
        candidate_radius: int = 2,
        use_patterns: bool = False,
        seed: int | None = None,
    ) -> None:
        super().__init__(name, budget=budget)
        self.candidate_radius = candidate_radius
        self.use_patterns = use_patterns
        self.seed = seed
        # rng chỉ khởi tạo khi seed được cung cấp
        self._rng: np.random.Generator | None = (
            np.random.default_rng(seed) if seed is not None else None
        )
        # Lưu thông tin truy vết nước đi gần nhất (phục vụ explain_last_action)
        self.last_decision_info: dict[str, object] = {}

    # ------------------------------------------------------------------
    # Tie-breaking và chọn nước tốt nhất từ danh sách ứng viên
    # ------------------------------------------------------------------

    def _tiebreak_key(
        self,
        board: np.ndarray,
        row: int,
        col: int,
        player: int,
    ) -> tuple[float, float, int, int]:
        """Khóa sắp xếp tất định: (điểm_vị_trí, -khoảng_cách_tâm, -row, -col).

        Dùng âm để max trở thành min khi sắp xếp giảm dần.
        """
        board_size = board.shape[0]
        center = board_size // 2
        pos_score = positional_score(board, row, col, player)
        manhattan = abs(row - center) + abs(col - center)
        # Trả về tuple sắp xếp giảm dần: điểm cao → ưu tiên
        return (pos_score, -manhattan, -row, -col)

    def _choose_best(
        self,
        actions: list[Action],
        board: np.ndarray,
        player: int,
    ) -> Action:
        """Chọn nước tốt nhất từ danh sách theo cơ chế tie-breaking đã công bố.

        - seed=None: tất định (deterministic) theo khóa sắp xếp.
        - seed!=None: ngẫu nhiên có kiểm soát trong nhóm điểm cao nhất.
        """
        if len(actions) == 1:
            return actions[0]

        # Tính điểm cho tất cả ứng viên
        scored = [
            (self._tiebreak_key(board, r, c, player), (r, c))
            for r, c in actions
        ]

        if self._rng is None:
            # Tất định: lấy phần tử có key lớn nhất
            best_key = max(k for k, _ in scored)
            # Nếu nhiều nước có cùng key tốt nhất → lấy phần tử đầu tiên (ổn định)
            for key, action in scored:
                if key == best_key:
                    return action
        else:
            # Ngẫu nhiên có kiểm soát: lọc nhóm nước có positional_score cao nhất
            best_pos = max(k[0] for k, _ in scored)
            top_group = [(r, c) for k, (r, c) in scored if k[0] == best_pos]
            idx = int(self._rng.integers(len(top_group)))
            return top_group[idx]

        # Dự phòng (không bao giờ chạy đến đây nếu actions không rỗng)
        return actions[0]

    # ------------------------------------------------------------------
    # Giải thích quyết định
    # ------------------------------------------------------------------

    def explain_last_action(self) -> dict[str, object]:
        """Trả về bản sao thông tin truy vết của nước đi gần nhất.

        Các trường:
          rule_triggered  : "win_now" | "block_urgent" | "positional_fallback"
          selected_action : (row, col) đã chọn
          candidate_count : số ứng viên được xét ở tầng này
          matching_actions: danh sách các nước cùng tầng (đã tìm được)
          reason          : mô tả tiếng Việt về lý do chọn nước đó
        """
        return dict(self.last_decision_info)

    # ------------------------------------------------------------------
    # get_action — điểm vào thuật toán (gọi qua choose_action của BaseAgent)
    # ------------------------------------------------------------------

    def get_action(self, board: np.ndarray, player: int) -> Action:
        """Chọn nước đi theo thứ tự ưu tiên luật chiến thuật.

        Tuyệt đối không sửa mảng board đầu vào.
        Cập nhật stats["nodes_visited"]: mỗi ô trống được kiểm tra = 1 node.
        """
        # Tổng số ô trống — dùng cho thống kê nodes_visited
        empty_cells: list[tuple[int, int]] = [
            (int(r), int(c)) for r, c in np.argwhere(board == EMPTY)
        ]
        total_empty = len(empty_cells)

        # ----------------------------------------------------------------
        # TẦNG 1: Thắng ngay (win_now)
        # ----------------------------------------------------------------
        win_actions = find_winning_actions(board, player)
        # nodes_visited: đã quét toàn bộ ô trống để tìm nước thắng
        self.stats["nodes_visited"] += total_empty

        if win_actions:
            best = self._choose_best(win_actions, board, player)
            self.last_decision_info = {
                "rule_triggered": "win_now",
                "selected_action": best,
                "candidate_count": len(win_actions),
                "matching_actions": list(win_actions),
                "reason": (
                    f"Thắng ngay tại {best} — tìm thấy {len(win_actions)} "
                    "nước thắng, chọn theo tie-breaking."
                ),
            }
            return best

        # ----------------------------------------------------------------
        # TẦNG 2: Chặn đối thủ thắng ngay (block_urgent)
        # ----------------------------------------------------------------
        opponent = -player
        block_actions = find_winning_actions(board, opponent)
        # nodes_visited: đã quét thêm toàn bộ ô trống để tìm đe dọa đối thủ
        self.stats["nodes_visited"] += total_empty

        if block_actions:
            best = self._choose_best(block_actions, board, player)
            self.last_decision_info = {
                "rule_triggered": "block_urgent",
                "selected_action": best,
                "candidate_count": len(block_actions),
                "matching_actions": list(block_actions),
                "reason": (
                    f"Chặn thua tại {best} — đối thủ có {len(block_actions)} "
                    "đe dọa thắng ngay, chặn theo tie-breaking."
                ),
            }
            return best

        # ----------------------------------------------------------------
        # TẦNG 3: Dự phòng vị trí (positional_fallback)
        # ----------------------------------------------------------------
        candidates = candidate_actions(board, radius=self.candidate_radius)

        # Nếu candidate_actions trả về rỗng (bàn đầy) → fallback toàn bàn
        # (trường hợp này BaseAgent đã lọc trước, nhưng phòng thủ chắc chắn)
        if not candidates:
            candidates = empty_cells

        # nodes_visited: đếm thêm số ứng viên được xét
        self.stats["nodes_visited"] += len(candidates)

        best = self._choose_best(candidates, board, player)
        scored_candidates = [
            (positional_score(board, r, c, player), (r, c))
            for r, c in candidates
        ]
        best_pos_score = max(score for score, _ in scored_candidates)
        top_group = [
            action
            for score, action in scored_candidates
            if score == best_pos_score
        ]
        self.last_decision_info = {
            "rule_triggered": "positional_fallback",
            "selected_action": best,
            "candidate_count": len(candidates),
            "matching_actions": top_group,
            "reason": (
                f"Dự phòng vị trí tại {best} — không có đe dọa trực tiếp, "
                f"chọn từ {len(candidates)} ứng viên bán kính "
                f"{self.candidate_radius} theo tie-breaking."
            ),
        }
        return best

