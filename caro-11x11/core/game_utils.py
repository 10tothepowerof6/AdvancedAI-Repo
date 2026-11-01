"""Tiện ích sinh ứng viên cho tìm kiếm; không thay đổi luật của engine."""

from numbers import Integral

import numpy as np

from core.game_engine import Action


def candidate_actions(board: np.ndarray, radius: int = 2) -> list[Action]:
    """Lấy ô trống cách quân đã có tối đa radius hàng và cột.

    Bàn trống trả về tâm; bàn đầy trả về danh sách rỗng. Kết quả sắp theo
    hàng/cột để có thể tái lập. Bên gọi phải kiểm tra trạng thái kết thúc trước.
    Đây là phép giới hạn không gian tìm kiếm, không phải tập nước hợp lệ đầy đủ.
    """
    if (not isinstance(board, np.ndarray) or board.ndim != 2
            or board.shape[0] == 0 or board.shape[0] != board.shape[1]
            or not np.isin(board, (-1, 0, 1)).all()):
        raise ValueError("Bàn cờ phải là mảng vuông không rỗng, chỉ chứa -1, 0, 1")
    if isinstance(radius, bool) or not isinstance(radius, Integral) or radius < 1:
        raise ValueError("Bán kính phải là số nguyên dương")
    size = board.shape[0]
    occupied = np.argwhere(board != 0)
    if not len(occupied):
        return [(size // 2, size // 2)]
    mask = np.zeros(board.shape, dtype=bool)
    for row, col in occupied:
        mask[max(0, row-radius):min(size, row+radius+1),
             max(0, col-radius):min(size, col+radius+1)] = True
    return [(int(r), int(c)) for r, c in np.argwhere(mask & (board == 0))]
