# Caro AI 11x11

Trò chơi dùng luật Caro tự do: X (1) đi trước, O (-1) đi sau, ô trống là 0.
Nối từ năm quân liên tiếp trở lên là thắng, kể cả khi bị chặn hai đầu.
Bàn cờ đầy mà chưa có người thắng thì hòa.

## Cài đặt và chạy

Dùng Python 3.11.9 và cài các gói trong `requirements.txt` vào môi trường ảo.
Chạy lệnh sau từ thư mục gốc dự án:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -v
```

## API dùng chung

`CaroGame` exposes `reset()`, `legal_actions()`, `step((row, col))`,
`is_terminal()`, `result()`, `check_winner()`, `clone()`, and `render()`.
Tọa độ bắt đầu từ 0. `step()` cập nhật trạng thái và không trả về giá trị.
Nước đi sai sẽ phát sinh `ValueError` và không làm thay đổi trạng thái.
`result()` trả về `None` khi đang chơi, 1/-1 khi có người thắng, hoặc 0 nếu hòa.
Người chơi chỉ đổi lượt sau nước đi hợp lệ nếu ván chưa kết thúc.
`check_winner()` kiểm tra quanh nước đi gần nhất; không sửa trực tiếp bàn cờ.
Dùng `clone()` để tạo trạng thái riêng cho tìm kiếm; hiện chưa có hàm undo.

Subclass `BaseAgent` and implement `get_action(board, player)`.
Chỉ gọi `choose_action(board, player)` khi ván chưa kết thúc. Hàm này giữ nguyên
bàn cờ đầu vào, đặt lại thống kê mỗi lượt, đo thời gian suy nghĩ và kiểm tra
nước đi được chọn. Tác tử có thể cập nhật `nodes_visited`, `simulations` và
`pruned_nodes`. `RandomAgent(seed=42)` chọn đều trong các ô trống bằng bộ sinh
số ngẫu nhiên riêng. Khởi tạo lại cùng seed để lặp lại chuỗi nước đi.

```python
from core import CaroGame
from agents import RandomAgent

game = CaroGame()
agent = RandomAgent(seed=42)
while not game.is_terminal():
    game.step(agent.choose_action(game.board, game.current_player))
game.render()
print(game.result())
```

Notebook nên import các module Python dùng chung thay vì chép lại thuật toán.
Các phần Rule-based, Minimax, MCTS, huấn luyện mạng, giải đấu và giao diện sẽ
được triển khai ở các giai đoạn tiếp theo.
