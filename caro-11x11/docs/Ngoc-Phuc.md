# Ngọc Phúc — Minimax và Alpha-Beta

## 1. Mục tiêu và tổ chức sản phẩm

Mục tiêu chính là hoàn thiện tác tử **Minimax** kết hợp kỹ thuật cắt tỉa **Alpha-Beta** cho cờ Caro 11×11, sẵn sàng tích hợp vào giải đấu chung và ứng dụng demo của dự án.

Mỗi giai đoạn thực hiện yêu cầu đầy đủ 4 thành phần cấu thành:
- **Module mã nguồn:** Cài đặt thuật toán trong `agents/minimax_agent.py`.
- **Kiểm thử tự động:** Xây dựng bộ test riêng trong `tests/test_minimax_agent.py`.
- **Thực nghiệm và trực quan hóa:** Trình bày quá trình đo lường, phân tích trong `notebooks/02_minimax.ipynb`.
- **Tài liệu theo dõi:** Cập nhật thiết kế, tiến độ và các hạn chế kỹ thuật trong `docs/Ngoc-Phuc.md`.

> **Lưu ý cốt lõi:** Notebook phải trực tiếp import và sử dụng tác tử từ module `agents/minimax_agent.py`. Tuyệt đối không duy trì hai phiên bản thuật toán song song giữa notebook và mã nguồn chính. Tài liệu này đóng vai trò định hướng kiến trúc, yêu cầu kỹ thuật và tiêu chí nghiệm thu; không cung cấp mã mẫu hay mã giả.

---

## 2. Phạm vi file và quy ước chung

### Phân công trách nhiệm theo file
| File thao tác | Trách nhiệm chính |
| :--- | :--- |
| `agents/minimax_agent.py` | Lớp `MinimaxAgent` và các hàm logic nội bộ |
| `tests/test_minimax_agent.py` | Bộ kiểm thử tự động riêng cho thuật toán |
| `notebooks/02_minimax.ipynb` | Kịch bản thí nghiệm, đo đạc số liệu và phân tích kết quả |
| `docs/Ngoc-Phuc.md` | Ghi nhận quyết định thiết kế, tiến độ thực hiện và giới hạn kỹ thuật |

### Quy ước phát triển và làm việc với Git
- **Thư mục làm việc:** Mọi thao tác lệnh và đường dẫn tương đối đều tính từ thư mục gốc `caro-11x11`.
- **Phạm vi chỉnh sửa:** Các thành phần nền tảng (`core/`, `config/`, `BaseAgent`, `RandomAgent`, `agents/__init__.py`, `requirements.txt`, các bài test chung, `README.md` gốc) thuộc quyền quản lý tích hợp của nhóm trưởng. Mọi đề xuất thay đổi cần thông qua thảo luận hoặc PR riêng. Tác tử cá nhân có thể được import trực tiếp từ file module mà không cần sửa `agents/__init__.py`.
- **Ngôn ngữ sử dụng:** Đầu ra console, chú thích code (comment), giải thích trong notebook, thông điệp commit và nội dung PR đều sử dụng tiếng Việt chuẩn. Tên biến, hàm, lớp tuân thủ quy chuẩn kỹ thuật tiếng Anh (PEP 8).
- **Quy chuẩn thông điệp commit:** Sử dụng tiền tố chuẩn theo Conventional Commits: `feat`, `fix`, `test`, `docs`, `chore` (ví dụ: `fix(minimax): sửa xử lý trạng thái kết thúc`). Phần mô tả nêu rõ nội dung thay đổi và nguyên nhân; chỉ xác nhận "đã kiểm thử" khi các lệnh test thực sự đã chạy thành công.

### Quy định tổ chức kiểm thử
- Toàn bộ test đặt trong thư mục `tests/`; tên file bắt đầu bằng tiền tố `test_` để pytest tự động nhận diện.
- Không tự ý chỉnh sửa các bài kiểm thử nền tảng (`tests/test_engine.py`, `tests/test_agents.py`, `tests/test_game_utils.py`) nhằm làm cho kết quả test cá nhân đạt chuẩn.
- Cấu trúc mỗi ca kiểm thử nên rõ ràng theo 3 bước:
  1. **Chuẩn bị (Arrange):** Thiết lập trạng thái bàn cờ ban đầu.
  2. **Thực thi (Act):** Gọi hàm cần kiểm tra để sinh hành vi/nước đi.
  3. **Kiểm chứng (Assert):** Đối chiếu kết quả thu được với giá trị kỳ vọng.

**Lệnh chạy kiểm thử trong môi trường ảo:**
```powershell
# Chạy riêng bộ kiểm thử của thuật toán
.\.venv\Scripts\python.exe -m pytest tests/test_minimax_agent.py -v

# Chạy toàn bộ các bài kiểm thử của dự án
.\.venv\Scripts\python.exe -m pytest tests -v
```

---

## 3. Hợp đồng tích hợp

### Quy tắc bàn cờ và luật chơi Caro 11×11
- Bàn cờ kích thước cố định 11×11.
- Ký hiệu người chơi: Quân $X = 1$ (đi trước), Quân $O = -1$ (đi sau), Ô trống $= 0$.
- Điều kiện thắng: Đạt từ 5 quân liên tiếp trở lên theo hàng ngang, dọc hoặc chéo (kể cả trường hợp bị chặn cả hai đầu).
- Điều kiện hòa: Bàn cờ được lấp đầy mà không bên nào tạo được chuỗi thắng.

### Hợp đồng giao tiếp tác tử (`BaseAgent`)
- Kế thừa lớp `BaseAgent`, cài đặt phương thức `get_action(board, player)` nhận vào mảng hai chiều `board` và `player` đến lượt đi.
- Đầu ra trả về là một tuple tọa độ số nguyên `(row, col)`, chỉ số tính từ 0. Phía gọi bên ngoài sử dụng phương thức `choose_action(board, player)`.
- Tác tử chỉ đưa ra quyết định khi ván đấu chưa kết thúc. Tuyệt đối không chỉnh sửa trực tiếp mảng `board` nhận vào.
- Lớp cha `BaseAgent` tự động đo thời gian suy nghĩ `thinking_time` (giây) và xác thực tính hợp lệ của nước đi trả về.

### Quản lý thống kê và tập ứng viên
- Các chỉ số `nodes_visited`, `simulations`, `pruned_nodes` trong từ điển `stats` được đặt lại về 0 ở đầu mỗi lượt đi. Chỉ cập nhật các chỉ số có ý nghĩa và nêu rõ phương pháp đếm trong notebook.
- Hàm `candidate_actions` mặc định tìm kiếm trong phạm vi bán kính $radius = 2$ quanh các quân cờ hiện có trên bàn. Đây là tập nước đi rút gọn nhằm tối ưu hiệu năng. Tác tử ngẫu nhiên (Random) vẫn chọn đều trên toàn bộ các ô trống. Mọi điều chỉnh về bán kính hay giới hạn số lượng ứng viên đều phải được công bố rõ ràng.
- Game Engine không có API nạp bàn cờ tùy ý hay đi lùi nước (undo). Không tự ý gán đè mảng `board` rồi giả định các thuộc tính `last_action`, `move_count` và kết quả tự động đồng bộ. Nếu xây dựng tầng mô phỏng riêng, cần đối chiếu với engine thông qua chuỗi nước đi hợp lệ.
- Kết quả ván đấu (`result`) quy ước theo góc nhìn X: mang giá trị `None` khi đang diễn ra, `1` khi X thắng, `-1` khi O thắng, `0` khi hòa. Tại trạng thái kết thúc (terminal), `current_player` vẫn lưu người vừa thực hiện nước đi cuối. Thang điểm đánh giá nội bộ của thuật toán phải được định nghĩa và giải thích riêng.

### Quy chuẩn ngân sách và kiểm soát thời gian (Toàn dự án)
Bảng cấu hình ngân sách tìm kiếm áp dụng chung cho tất cả các tác tử:

| Chế độ | Cấu hình khởi đầu | Yêu cầu thực hiện |
| :--- | :--- | :--- |
| **fixed** | MCTS: 1.000 mô phỏng/nước<br>Minimax: độ sâu 2 ply | Hoàn thành đúng ngân sách quy định; khảo sát mở rộng ở các mức 100 / 500 / 1.000 mô phỏng và độ sâu 1 / 2 / 3 ply |
| **time** | Giới hạn tối đa 2,0 giây/nước cho tất cả agent | MCTS tận dụng thời gian cho đến deadline; Minimax áp dụng tìm kiếm sâu dần (Iterative Deepening) và bảo lưu kết quả của độ sâu đã hoàn tất |

- **Báo cáo độc lập:** Hai chế độ phải được đánh giá riêng biệt; chế độ `fixed` không phản ánh tính công bằng về thời gian thực thi. Ở chế độ `time`, số mô phỏng hoặc độ sâu của chế độ `fixed` không được dùng làm điều kiện dừng bổ sung.
- **Hỗ trợ API:** Lớp `SearchBudget` (`config/search_budget.py`) lưu trữ các trường `mode`, `simulations`, `depth`, `time_limit_s`. Truyền `budget` xuống `BaseAgent` khi khởi tạo subclass; không thay đổi chữ ký phương thức chung.
- **Cơ chế dừng hợp tác (Cooperative Stopping):** Tác tử phải chủ động dừng tìm kiếm trước deadline để kịp hoàn tất và gửi kết quả. `self.time_exhausted()` chỉ trả về đúng khi đã tới deadline, không tự trừ biên an toàn. Mốc dừng sớm khởi đầu đề xuất là 1,9 giây trên hạn 2,0 giây (chừa khoảng 100 ms); thuật toán tự quản lý, công khai và kiểm chứng biên này trên máy chạy. Đây không phải bảo đảm tuyệt đối tránh timeout; hạn cứng của runner vẫn là 2,0 giây, không được cộng thêm 100 ms. Minimax lưu lại kết quả của độ sâu sâu dần đã hoàn tất; luôn có sẵn nước đi dự phòng ban đầu đề phòng trường hợp chưa kịp xong độ sâu đầu tiên.
- **Cơ chế cưỡng chế dừng (Hard Timeout Enforcement):** Điều phối bởi `evaluation/match.py` qua `play_match` và `AgentWorker`. Mỗi tác tử chạy trên tiến trình riêng suốt cả ván. Nếu quá giờ, tiến trình bị ngắt và xử thua với lý do `timeout`. Nước đi sai luật hoặc ngoại lệ cũng bị xử thua ngay lập tức kèm ghi nhận nguyên nhân.
- **Tính toán độ trễ:** Phân biệt `thinking_time` (đo nội bộ trong `choose_action`) và `elapsed_s` (đo bởi runner, tính cả chi phí giao tiếp IPC và kiểm tra nước đi). Dùng `elapsed_s` làm chuẩn độ trễ thi đấu; không tự ý cộng thêm dung sai ẩn. Khởi tạo/nạp mô hình có hạn mức riêng tối đa 30 giây.
- **Giám sát chống treo (Watchdog Timer):** Ở chế độ `fixed`, watchdog mặc định đặt mức 30 giây/nước nhằm chống treo tiến trình (tùy biến qua `safety_timeout_s`). Khi kích hoạt, hệ thống ghi nhận `timeout`. Ở chế độ `time`, giới hạn 2,0 giây được sử dụng trực tiếp.
- **Yêu cầu đa tiến trình:** Factory tạo agent phải đặt trong module `.py` import được. Script Windows cần có `if __name__ == "__main__":`. Trong notebook, cần import factory từ module ngoài để tránh lỗi khi spawn tiến trình.

### Chỉ số đánh giá và module tổng hợp dùng chung
- Nhóm trưởng sở hữu `evaluation/metrics.py` và `tests/test_metrics.py`. Các thành viên gọi trực tiếp các hàm này, không tự viết lại logic tổng hợp.
- `MatchRecord` bao gồm: `agent_x`, `agent_o`, `winner`, `reason`. Hai bên thi đấu bắt buộc phải có định danh khác nhau. Hàm `summarize_results(records, agent)` tự động quy đổi kết quả theo góc nhìn của agent đang được tổng hợp.
- **Công thức tính điểm:** $N = W + D + L$; $\text{score} = \frac{W + 0.5 \times D}{N}$. Ví dụ: 6 thắng, 2 hòa, 2 thua đạt điểm số 0,7. Chưa có ván đấu nào thì điểm là `None`.
- Các ván xử thua do `timeout`, `agent_error`, `illegal_action` vẫn tính vào $W/D/L$ như một ván thua; trường `forfeits` thống kê riêng lỗi do chính agent gây ra.
- Hàm `summarize_latencies(seconds)` nhận danh sách thời gian hoàn thành của các nước đi hợp lệ, trả về `count`, `mean_s`, `p95_s` (nội suy tuyến tính). Mẫu rỗng trả về `None`. Số lần và thời gian lỗi/timeout được báo cáo riêng biệt.

### Quy định kiểm thử trước khi merge
- Toàn bộ test đặt trực tiếp trong `tests/`, đặt tên theo chức năng (`test_<tên_module>.py`). Không chia theo thư mục người phụ trách.
- Không tạo bài test có phụ thuộc vào tác tử chưa hoàn thiện của người khác.
- Khi có lỗi ở bộ test chung: Xác định rõ lỗi đã tồn tại trên `main` hay mới xuất hiện do PR hiện tại. Tuyệt đối không sửa test của phần khác để làm cho bài test của mình vượt qua; không tùy tiện bỏ qua lỗi.
- Không đưa tiêu chí tốc độ cố định hay điều kiện phải thắng tuyệt đối mọi ván vào unit test.

---

## 4. Khung tổ chức thuật toán

Cấu trúc thuật toán Minimax và Alpha-Beta cần phân định ranh giới trách nhiệm rõ ràng giữa các thành phần:

| Thành phần | Yêu cầu tổ chức và kỹ thuật |
| :--- | :--- |
| **Trạng thái tìm kiếm** | Quản lý bàn cờ, lượt đi và trạng thái kết thúc (terminal); các nhánh tìm kiếm con tuyệt đối không làm thay đổi trạng thái cha. |
| **Hàm lượng giá** | Tách biệt đánh giá trạng thái terminal (thắng/thua/hòa) khỏi đánh giá heuristic; thống nhất góc nhìn và ý nghĩa thang điểm. |
| **Thuật toán Minimax** | Quy định độ sâu tìm kiếm theo ply (một ply = một nước đi của một phía), phân định vai trò Max/Min và điều kiện dừng tìm kiếm. |
| **Cắt tỉa Alpha-Beta** | Tích hợp kỹ thuật cắt tỉa trên cùng nền tảng Minimax, có cấu hình bật/tắt (toggle) để phục vụ đối chứng thực nghiệm. |
| **Quyết định và thống kê** | Ghi nhận số nút đã duyệt, số lần cắt tỉa, độ sâu đạt được; xác định rõ quy tắc phá hòa khi có nhiều nước đồng điểm. |

> **Lưu ý thiết kế:** Mỗi thành phần cần có ca kiểm thử độc lập; không chỉ thử nghiệm trên tác tử hoàn chỉnh cuối cùng. Tuyệt đối không đưa các chức năng phụ như vẽ biểu đồ, đọc file notebook hay vận hành giải đấu vào phương thức lựa chọn nước đi.

---

## 5. Khung lập trình toàn bộ notebook

File `notebooks/02_minimax.ipynb` được tổ chức tuần tự theo 10 phần dưới đây. Mỗi phần cần có cell Markdown mô tả mục tiêu, cell thực thi mã nguồn và phần phân tích, nhận xét kết quả:

1. **Câu hỏi nghiên cứu và phạm vi:** Alpha-Beta giúp giảm bao nhiêu phần trăm số nút duyệt và thời gian tính toán so với Minimax thuần túy khi giữ nguyên chất lượng nước đi ở cùng độ sâu?
2. **Môi trường và cấu hình:** Ghi nhận độ sâu theo ply, bán kính ứng viên, hệ số trọng số heuristic, thứ tự xét nước đi và cấu hình phần cứng thử nghiệm.
3. **Kiểm chứng trạng thái:** Đối chiếu chuyển trạng thái và phát hiện terminal với Game Engine trên bộ vị trí bàn cờ cố định.
4. **Giải thích hàm lượng giá:** Trình bày ý nghĩa thang điểm, đối chiếu các vị trí tốt/xấu cho cả hai người chơi X và O.
5. **Quan sát cây Minimax:** Trực quan hóa quyết định trên cây tìm kiếm kích thước nhỏ, thể hiện điểm số tại các nút lá và kết quả lựa chọn theo góc nhìn đã thống nhất.
6. **Đối chứng thực nghiệm Alpha-Beta:** So sánh trực tiếp trên cùng vị trí, cùng độ sâu, cùng thứ tự nước: giá trị tìm kiếm ở nút gốc, nước đi lựa chọn, số lượng nút đã thăm và số lần cắt tỉa.
7. **Khảo sát độ sâu tìm kiếm:** Đánh giá các mức độ sâu 1 / 2 / 3 ply (nếu khả thi); ghi nhận chi phí tính toán và giới hạn thời gian thực tế.
8. **Thi đấu thử nghiệm:** Chạy đối đầu với `RandomAgent` theo từng cặp ván đổi bên đi trước/sau; ghi nhận kết quả $W/D/L$, điểm số và độ trễ.
9. **Phân tích các trường hợp thất bại:** Phân tích nguyên nhân lỗi (do độ sâu hạn chế, heuristic chưa bao quát hay do tập ứng viên bị thu hẹp); không gộp chung các nguyên nhân.
10. **Tổng kết và hướng dẫn tái lập:** Đưa ra kết luận có căn cứ từ số liệu thực nghiệm, nêu rõ các hạn chế kỹ thuật, cấu hình bàn giao và thứ tự chạy lại notebook.

### Nguyên tắc thực nghiệm trong Notebook
- Tập trung toàn bộ cấu hình ở phần đầu notebook; ghi lại cấu hình cụ thể đi kèm mỗi thí nghiệm khi có sự thay đổi.
- Sử dụng chung bộ vị trí bàn cờ và phương pháp ghi nhận kết quả để đảm bảo tính công bằng khi so sánh các biến thể. Luôn cố định `seed` nếu có yếu tố ngẫu nhiên.
- Thực thi thuật toán trực tiếp từ module `agents/minimax_agent.py`; notebook chỉ đóng vai trò tổ chức thí nghiệm và trình bày kết quả.
- Mỗi tình huống thử nghiệm cần thể hiện rõ: trạng thái bàn cờ, bên đến lượt, kỳ vọng trước khi chạy và quan sát thực tế sau khi chạy.
- Luôn chạy tuần tự từ một kernel mới (Restart Kernel & Run All); tránh sử dụng các biến còn sót lại từ các cell đã chạy trước đó.
- Không thay thế bộ unit test bằng việc chỉ quan sát bàn cờ trong notebook.
- Chưa có số liệu thì ghi nhận là chưa thực hiện. Tuyệt đối không điền bảng dữ liệu giả định, không đưa ra kết luận vội vã chỉ dựa trên một ván thắng đơn lẻ.
- Kết quả thử nghiệm cục bộ không thay thế giải đấu chính thức (tối thiểu 100 ván/cặp, đổi bên, nhiều seed). Tuyệt đối không dùng các ván đánh giá cuối cùng để điều chỉnh tham số.

---

## 6. Giai đoạn 0 — Khởi tạo môi trường và xác nhận nền tảng

- Đọc kỹ `README.md` gốc của repository, cài đặt môi trường ảo và dependencies từ `requirements.txt`.
- Chạy toàn bộ test nền tảng và kiểm tra hoạt động của `notebooks/00_game_engine_test.ipynb`.
- Mở `notebooks/02_minimax.ipynb`, xác nhận Jupyter đang sử dụng đúng kernel `.venv` và các ví dụ tích hợp mẫu chạy thành công.
- Kiểm tra trạng thái Git (`git status`), commit hoặc lưu trữ (stash) các công việc đang dở trước khi chuyển nhánh.
- Giai đoạn 0 không tạo commit rỗng hay PR riêng. Các nhánh của từng giai đoạn tiếp theo được tạo tuần tự từ `main` sau khi PR của giai đoạn trước đó đã được merge hoàn tất.
- Nếu nhánh tương ứng đã tồn tại sẵn, chuyển sang nhánh đó thay vì tạo mới. Kiểm tra kỹ lịch sử Git trước khi đổi tên nhánh.

---

## 7. Giai đoạn 1 — Xây dựng trạng thái và hàm lượng giá

### Mục tiêu và nhiệm vụ
1. Xác định cách thức biểu diễn trạng thái tìm kiếm từ bàn cờ đầu vào; đảm bảo việc duyệt các nước đi con không làm thay đổi trạng thái cha.
2. Thống nhất góc nhìn đánh giá trong hàm lượng giá: phân định rõ đánh giá từ phía nào, ý nghĩa của điểm số dương/âm và mức độ ưu tiên tuyệt đối của các trạng thái kết thúc (terminal).
3. Mô tả chi tiết các thành phần cấu thành điểm heuristic, lý giải nguyên nhân lựa chọn và các giới hạn; không sử dụng các hệ số điểm số mà thiếu giải thích.
4. Hoàn thiện nội dung các phần 1–4 trong notebook và bộ unit test nền tảng; giai đoạn này chưa được công bố như một tác tử hoàn chỉnh.

### Kiểm thử và tiêu chí nghiệm thu
- Đối chiếu quá trình chuyển trạng thái và kết quả ván cờ với Game Engine trên toàn bộ tập dữ liệu mẫu (cho cả X và O, các hướng thắng ngang/dọc/chéo, chuỗi dài hơn năm quân và ván hòa).
- Đảm bảo điểm số của trạng thái terminal không bị điểm heuristic của các trạng thái chưa kết thúc lấn át.
- Đảm bảo tính nhất quán khi chuyển đổi góc nhìn đánh giá; việc sao chép hoặc hoàn tác trạng thái không làm biến dạng bàn cờ gốc.

> **Mốc hoàn thành (Milestone):** Hàm lượng giá có định nghĩa rõ ràng, bộ test kiểm tra trạng thái vượt qua 100% và notebook trực quan hóa được ý nghĩa thang điểm.

### Hướng dẫn thiết kế ca kiểm thử
1. Tạo các fixture kiểm thử bằng chuỗi nước đi hợp lệ từ engine; chuẩn bị đầy đủ các cặp vị trí cho X và O, trạng thái terminal (thắng/thua/hòa) và trạng thái đang chơi.
2. Kiểm thử riêng biệt tầng lượng giá với các kỳ vọng thứ tự điểm số tốt/xấu theo đúng quy ước góc nhìn đã định; không gán ghép các hằng số tùy tiện làm chuẩn đúng.
3. Kiểm tra tính độc lập của trạng thái: sau khi duyệt và đánh giá các nước con, bàn cờ, lượt chơi và các trường thông tin của trạng thái cha phải được giữ nguyên vẹn.

**Tiêu chí đạt chuẩn trước khi yêu cầu Review:**
- [ ] Chuyển trạng thái và phát hiện terminal khớp hoàn toàn với Game Engine trên toàn bộ fixture.
- [ ] Trạng thái terminal thắng/thua có điểm số ưu tiên vượt trội; trạng thái hòa nhận điểm số đúng quy ước.
- [ ] Tính đối xứng về góc nhìn đánh giá được đảm bảo; trạng thái cha và mảng bàn cờ gốc không bị biến đổi.

**Bằng chứng bắt buộc trong PR:** Lệnh kiểm thử đã chạy, số lượng test đạt/thất bại, danh sách fixture đã thử nghiệm và các hạn chế kỹ thuật đã biết. Không chỉ gửi ảnh chụp bàn cờ mà thiếu giải thích hành vi được kiểm tra.

### Quy trình nhánh, commit và đẩy mã nguồn
- **Tên nhánh:** `feature/minimax-foundation` *(tạo từ main sau khi nền tảng chung sẵn sàng)*.

<details>
<summary>Lệnh thao tác Git (PowerShell)</summary>

```powershell
git status
git switch main
git pull --ff-only origin main
git switch -c feature/minimax-foundation

# Thực hiện code và kiểm thử trước khi commit
.\.venv\Scripts\python.exe -m pytest tests -v

git add -- agents/minimax_agent.py notebooks/02_minimax.ipynb tests/test_minimax_agent.py docs/Ngoc-Phuc.md
git diff --cached --stat
git diff --cached
git commit -m "feat(minimax): xây dựng trạng thái tìm kiếm và hàm lượng giá"
git push -u origin feature/minimax-foundation
```

</details>

### Tiêu đề và mô tả PR Giai đoạn 1
- **Tiêu đề PR:** `feat(minimax): xây dựng trạng thái và lượng giá cho Minimax`
- **Mô tả nội dung mẫu:**
  > Bổ sung tầng trạng thái tìm kiếm và hàm lượng giá với quy ước góc nhìn thống nhất. Notebook trình bày chi tiết thang điểm terminal, các thành phần heuristic và ví dụ minh họa trên bàn cờ. PR tập trung vào nền tảng trạng thái và lượng giá, chưa bao gồm thuật toán tìm kiếm hoàn chỉnh.
- **Nội dung bắt buộc điền trong PR:**
  - **Phạm vi:** Danh sách các phần đã thực hiện, các file thay đổi và những điểm chưa hoàn thiện.
  - **Kiểm chứng:** Lệnh kiểm thử đã chạy, số lượng test pass/fail, xác nhận notebook đã chạy trơn tru từ kernel mới.
  - **Kết quả:** Cấu hình thử nghiệm, các vị trí bàn cờ đã khảo sát; nếu chưa có thực nghiệm thì ghi rõ là chưa thực hiện.
  - **Giới hạn và phụ thuộc:** Các lỗi tồn đọng đã biết, đề xuất điều chỉnh API chung (nếu có) và tài liệu tham khảo.

---

## 8. Giai đoạn 2 — Hoàn thiện Minimax và Alpha-Beta

### Mục tiêu và nhiệm vụ
1. Cài đặt hoàn chỉnh thuật toán Minimax giới hạn độ sâu; quy ước rõ ràng một đơn vị độ sâu (1 ply) tương ứng với một nước đi của một phía người chơi.
2. Tích hợp kỹ thuật cắt tỉa Alpha-Beta với cơ chế bật/tắt (toggle) linh hoạt trên cùng tầng trạng thái, hàm lượng giá và tập nước đi ứng viên.
3. Thống nhất thứ tự xét các nước đi và cơ chế giải quyết khi có nhiều nước đồng điểm số; nếu áp dụng sắp xếp nước đi (move ordering), cần mô tả riêng cấu hình này.
4. Tích hợp tác tử với `BaseAgent`, cập nhật số liệu thống kê: số nút đã duyệt (`nodes_visited`), số lần cắt tỉa (`pruned_nodes`). Lưu ý: số lần cắt tỉa không đồng nhất với tổng số nút con chưa duyệt trên nhánh bị cắt.
5. Hoàn thiện các phần 5–6 trong notebook, chứng minh tính nhất quán giữa Minimax và Alpha-Beta trên cây tìm kiếm nhỏ ở chế độ `fixed`. Bổ sung cơ chế kiểm soát deadline và tìm kiếm sâu dần (Iterative Deepening) để sẵn sàng tham gia chế độ `time`.

### Kiểm thử và tiêu chí nghiệm thu
- Kiểm tra các tình huống chiến thuật then chốt: nước thắng ngay lập tức, nước chặn đối thủ thắng trong tầm nhìn của độ sâu tìm kiếm và phản ứng chuẩn xác ở lượt của đối phương.
- So sánh giá trị đánh giá ở nút gốc giữa hai chế độ (bật và tắt Alpha-Beta) trên cùng trạng thái, cùng độ sâu và cùng thứ tự duyệt: giá trị điểm số trả về phải hoàn toàn trùng khớp (nước đi có thể khác nhau nếu có nhiều nước đồng điểm).
- Kiểm tra điểm dừng độ sâu chính xác theo quy ước, trạng thái terminal không bị mở rộng thêm, nước đi trả về hợp lệ và bàn cờ gốc không bị biến đổi.
- Sử dụng bài test cố định để chứng minh: việc áp dụng Alpha-Beta không làm tăng số nút duyệt so với Minimax thuần túy khi mọi điều kiện so sánh là như nhau.

> **Mốc hoàn thành (Milestone):** Tác tử thi đấu trọn vẹn ván cờ, hai chế độ Minimax và Alpha-Beta cho kết quả nhất quán về mặt giá trị, bộ test hợp đồng API vượt qua thành công.

### Hướng dẫn thiết kế ca kiểm thử
1. Xây dựng cây tìm kiếm nhỏ hoặc các thế cờ gần kết thúc có giá trị tính tay để kiểm tra vai trò Max/Min và điều kiện dừng độ sâu.
2. Chạy đối chứng bật/tắt Alpha-Beta với cùng tham số (`depth`, `heuristic`, `candidate set`, `ordering`); so sánh trực tiếp giá trị tại nút gốc thay vì chỉ so sánh tọa độ nước đi.
3. Khi xuất hiện nhiều nước đi có điểm số bằng nhau, kiểm tra nước đi trả về thuộc tập hợp các nước tối ưu chấp nhận được, không ép buộc một tọa độ duy nhất.
4. Ghi nhận số nút duyệt và số lần cắt tỉa trong bài test; không dùng thời gian thực thi làm căn cứ duy nhất để chứng minh hiệu quả của cắt tỉa.

**Tiêu chí đạt chuẩn trước khi yêu cầu Review:**
- [ ] Hai chế độ (bật/tắt cắt tỉa) cho cùng giá trị đánh giá tại nút gốc trên toàn bộ tập đối chứng; nước đi trả về đạt giá trị tối ưu đó trong tầm tìm kiếm.
- [ ] Độ sâu tính theo ply hoạt động đúng quy ước; trạng thái terminal không bị mở rộng; các nước thắng hoặc chặn thua trong tầm nhìn được xử lý chính xác.
- [ ] Cắt tỉa Alpha-Beta không làm tăng số nút duyệt khi cùng điều kiện; có ít nhất một ca kiểm thử thực tế kích hoạt được thao tác cắt tỉa.
- [ ] Nước đi sinh ra luôn hợp lệ, bàn cờ gốc giữ nguyên vẹn và các chỉ số thống kê được đặt lại chuẩn xác giữa các lượt đi.

**Bằng chứng bắt buộc trong PR:** Lệnh kiểm thử đã chạy, số lượng test đạt/thất bại, danh sách fixture đã thử nghiệm và các hạn chế kỹ thuật đã biết.

### Quy trình nhánh, commit và đẩy mã nguồn
- **Tên nhánh:** `feature/minimax-search` *(chỉ tạo sau khi PR Giai đoạn 1 đã được merge vào main)*.

<details>
<summary>Lệnh thao tác Git (PowerShell)</summary>

```powershell
git status
git switch main
git pull --ff-only origin main
git switch -c feature/minimax-search

# Thực hiện code và kiểm thử trước khi commit
.\.venv\Scripts\python.exe -m pytest tests -v

git add -- agents/minimax_agent.py notebooks/02_minimax.ipynb tests/test_minimax_agent.py docs/Ngoc-Phuc.md
git diff --cached --stat
git diff --cached
git commit -m "feat(minimax): hoàn thiện Minimax và cắt tỉa Alpha-Beta"
git push -u origin feature/minimax-search
```

</details>

### Tiêu đề và mô tả PR Giai đoạn 2
- **Tiêu đề PR:** `feat(minimax): hoàn thiện Minimax và chế độ Alpha-Beta`
- **Mô tả nội dung mẫu:**
  > Bổ sung thuật toán tìm kiếm Minimax giới hạn độ sâu kết hợp kỹ thuật cắt tỉa Alpha-Beta có thể bật/tắt linh hoạt. Cả hai chế độ sử dụng chung hàm lượng giá và cách thức sinh nước đi để phục vụ đối chứng. Notebook và các ca kiểm thử xác nhận tính nhất quán về giá trị và năng lực xử lý chiến thuật.

---

## 9. Giai đoạn 3 — Đánh giá cắt tỉa và khảo sát độ sâu

### Mục tiêu và nhiệm vụ
1. Cố định bộ vị trí bàn cờ thử nghiệm; so sánh chi tiết hiệu quả khi bật và tắt cắt tỉa Alpha-Beta trong cùng điều kiện (cùng độ sâu, bán kính, heuristic và thứ tự duyệt nước đi).
2. Đo đạc số lượng nút duyệt, số lần cắt tỉa, thời gian thực thi và giá trị tìm kiếm; khảo sát mở rộng ở các độ sâu 1, 2 và 3 ply nếu ngân sách tính toán cho phép.
3. Tổ chức thi đấu thử nghiệm với `RandomAgent`, đảm bảo cân bằng quyền đi trước/sau giữa hai bên; ghi nhận rõ các cấu hình có hoặc không áp dụng sắp xếp nước đi.
4. Hoàn thiện nội dung các phần 7–10 trong notebook. Đối với chế độ `time`, bắt buộc hỗ trợ kiểm tra deadline và tìm kiếm sâu dần (Iterative Deepening), ghi nhận độ sâu thực tế hoàn thành. Cần hoàn thiện cả hai chế độ `fixed` và `time` trước khi bàn giao.

### Kiểm thử và tiêu chí nghiệm thu
- Chạy lại toàn bộ test tự động và notebook từ một kernel mới hoàn toàn.
- Đảm bảo bảng đối chứng không vô tình thay đổi đồng thời nhiều tham số; thời gian thực tế phải được đo đạc trực tiếp, không dùng ước lượng qua số nút.
- Đối chiếu số lượng ván đấu và danh sách `seed`; phân biệt rõ ràng giữa bài đo cắt tỉa (ablation study) ở cùng độ sâu với bài đánh giá thi đấu (benchmark) ở cùng quỹ thời gian.

> **Mốc hoàn thành (Milestone):** Có bảng số liệu thực nghiệm đo đạc thực tế về cắt tỉa và độ sâu, phân tích nguyên nhân lỗi rõ ràng; không khẳng định tính công bằng về thời gian với MCTS khi chỉ so sánh ở cùng độ sâu.

### Hướng dẫn thiết kế ca kiểm thử
1. Trong `tests/test_minimax_agent.py`, kiểm tra cấu trúc log của tác tử và gọi module đánh giá dùng chung (`evaluation/metrics.py`) trên tập dữ liệu mẫu tính tay. Không tự viết lại bộ tổng hợp số liệu; `tests/test_metrics.py` do nhóm trưởng quản lý. Tách biệt kiểm tra tính chính xác của số liệu khỏi các bài benchmark tốn thời gian.
2. Đối chiếu tệp kê khai (manifest) của từng lần chạy để đảm bảo việc bật/tắt cắt tỉa không vô tình làm thay đổi thứ tự duyệt nước đi hay tập ứng viên.
3. Chạy lại notebook từ kernel mới để đảm bảo toàn bộ số liệu và bảng biểu được sinh ra từ kết quả thực tế, không dùng dữ liệu nhập tay.

**Tiêu chí đạt chuẩn trước khi yêu cầu Review:**
- [ ] Bảng đối chứng cắt tỉa có đầy đủ thông số cấu hình; số lượng nút và giá trị tìm kiếm nhất quán với kết quả ở Giai đoạn 2.
- [ ] Kết quả khảo sát các độ sâu 1 / 2 / 3 ply ghi nhận rõ độ sâu thực tế hoàn thành; các cấu hình không khả thi do vượt thời gian phải được báo cáo rõ giới hạn.
- [ ] Thời gian và tỷ lệ thắng được trình bày như kết quả thực nghiệm khách quan; không dùng ngưỡng thời gian chạy cố định làm điều kiện đánh giá pass/fail trong pytest.

**Bằng chứng bắt buộc trong PR:** Lệnh kiểm thử đã chạy, số lượng test đạt/thất bại, danh sách fixture đã thử nghiệm và các hạn chế kỹ thuật đã biết.

### Quy trình nhánh, commit và đẩy mã nguồn
- **Tên nhánh:** `feature/minimax-evaluation` *(chỉ tạo sau khi PR Giai đoạn 2 đã được merge vào main)*.

<details>
<summary>Lệnh thao tác Git (PowerShell)</summary>

```powershell
git status
git switch main
git pull --ff-only origin main
git switch -c feature/minimax-evaluation

# Thực hiện code và kiểm thử trước khi commit
.\.venv\Scripts\python.exe -m pytest tests -v

git add -- agents/minimax_agent.py notebooks/02_minimax.ipynb tests/test_minimax_agent.py docs/Ngoc-Phuc.md
git diff --cached --stat
git diff --cached
git commit -m "test(minimax): bổ sung đối chứng cắt tỉa và khảo sát độ sâu"
git push -u origin feature/minimax-evaluation
```

</details>

### Tiêu đề và mô tả PR Giai đoạn 3
- **Tiêu đề PR:** `test(minimax): bổ sung đánh giá Alpha-Beta và độ sâu tìm kiếm`
- **Mô tả nội dung mẫu:**
  > Bổ sung thực nghiệm đối chứng bật/tắt cắt tỉa Alpha-Beta và khảo sát các mức độ sâu tìm kiếm trong điều kiện cấu hình được kiểm soát chặt chẽ. Notebook trình bày chi tiết số lượng nút duyệt, thời gian thực thi, kết quả thi đấu thử nghiệm và phân tích hạn chế do hàm heuristic hoặc giới hạn độ sâu.

---

## 10. Quy trình tạo PR và xử lý phản hồi Review

### Thiết lập thông tin bắt buộc trên GitHub
- **Reviewer:** @10tothepowerof6 (Nhóm trưởng). Chọn trực tiếp trong mục Reviewers trên giao diện PR khi đã sẵn sàng review; chỉ nhắc tên trong mô tả sẽ không tạo thông báo chính thức.
- **Assignee:** @PhucLee23. Người phụ trách có trách nhiệm theo dõi và hoàn thiện PR; reviewer chịu trách nhiệm kiểm tra và đánh giá thay đổi.
- **Milestone:**
  - Task 1–2 / PR tương ứng: [Báo cáo tiến độ nửa dự án](https://github.com/10tothepowerof6/AdvancedAI-Repo/milestone/2), hạn 09/11/2026.
  - Task 3: [milestone tổng](https://github.com/10tothepowerof6/AdvancedAI-Repo/milestone/1).
- **Liên kết Issue:**
  - Giai đoạn 1: [Task 1 (Minimax)](https://github.com/10tothepowerof6/AdvancedAI-Repo/issues/4)
  - Giai đoạn 2: [Task 2 (Minimax)](https://github.com/10tothepowerof6/AdvancedAI-Repo/issues/5)
  - Giai đoạn 3: [Task 3 (Minimax)](https://github.com/10tothepowerof6/AdvancedAI-Repo/issues/6)
  *(Không tạo issue cho Giai đoạn 0).*
- **Đóng issue:** Trong phần mô tả của PR hoàn thành task, sử dụng cú pháp `Closes #<số_issue>` (ví dụ: `Closes #4`). PR nháp hoặc PR chỉ hoàn thành một phần thì chỉ gắn liên kết tham chiếu thông thường, không dùng từ khóa `Closes`.
- **Quy tắc bảo vệ nhánh:** Nhánh `main` bắt buộc phải có ít nhất một phê duyệt (approval) từ thành viên khác trước khi merge (áp dụng cho mọi tài khoản).

### Các bước tạo PR và xử lý góp ý
1. Sau khi đẩy nhánh (`git push`), truy cập repository trên GitHub, chọn **Pull requests** → **New pull request**.
2. Chọn nhánh đích (`base`) là `main` và nhánh so sánh (`compare`) là nhánh tính năng tương ứng của giai đoạn. Kiểm tra kỹ danh sách commit và mục **Files changed**, đảm bảo chỉ chứa các file thuộc phạm vi được phân công.
3. Sử dụng tiêu đề và nội dung mô tả chuẩn của từng giai đoạn; điền đầy đủ số liệu kiểm thử thực tế. Chọn chế độ **Draft Pull Request** nếu công việc chưa hoàn thành toàn bộ tiêu chí.
4. Gán Reviewer cho nhóm trưởng. Khi nhận được phản hồi góp ý, thực hiện chỉnh sửa trực tiếp trên cùng nhánh, chạy lại bộ kiểm thử, đưa file vào vùng chờ (`git add`) và tạo commit mới (`fix`, `docs`, `test`) với thông điệp bằng tiếng Việt.
5. Push lại nhánh đó lên GitHub; PR sẽ tự động cập nhật. Phản hồi các nhận xét trong PR và cập nhật kết quả kiểm thử mới.
6. Sau khi PR được duyệt và merge vào `main`, tiến hành cập nhật lại nhánh `main` cục bộ trước khi tạo nhánh mới cho giai đoạn tiếp theo.

### Xử lý đồng bộ và xung đột nhánh (Merge Conflict)
Nếu nhánh `main` có cập nhật mới trong quá trình mở PR:
1. Commit toàn bộ công việc đang làm dở trên nhánh tính năng.
2. Chuyển sang nhánh `main`, kéo thay đổi mới nhất: `git pull --ff-only origin main`.
3. Chuyển lại nhánh tính năng và hợp nhất thay đổi: `git merge main`.
4. Nếu phát sinh xung đột (conflict): Mở từng file bị xung đột, kiểm tra kỹ lưỡng nội dung của cả hai bên, tiến hành giải quyết, chạy lại bộ test và hoàn tất commit hợp nhất: `chore(minimax): đồng bộ thay đổi từ main`. Tuyệt đối không tự ý xóa bỏ phần code của thành viên khác.

---

## 11. Checklist bàn giao cuối cùng

- [ ] Thuật toán triển khai đúng phạm vi phân công, tuân thủ nghiêm ngặt API chung và không sửa đổi mảng `board` gốc.
- [ ] Bộ unit test riêng bao phủ đầy đủ các thành phần; toàn bộ test chung của dự án vẫn vượt qua thành công.
- [ ] File notebook hoàn thiện đầy đủ 10 mục, có thể chạy trơn tru từ kernel mới; số liệu có cấu hình và seed rõ ràng.
- [ ] Thông điệp commit, nội dung mô tả PR, chú thích code và đầu ra hiển thị đều sử dụng tiếng Việt chuẩn.
- [ ] Mỗi giai đoạn được tổ chức trên một nhánh và PR riêng; giai đoạn trước đã được merge hoàn tất trước khi bắt đầu giai đoạn sau.
- [ ] PR cung cấp đầy đủ tài liệu tham khảo, các hạn chế kỹ thuật và số liệu kiểm chứng thực tế.
- [ ] Không đưa vào Git các file không liên quan, thư mục môi trường ảo (`.venv`) hoặc dữ liệu có dung lượng lớn.
- [ ] Nhóm trưởng xác nhận kiểm thử tích hợp thành công trước khi chính thức nghiệm thu phần việc.

*Trạng thái tài liệu: Kế hoạch triển khai kỹ thuật; chưa xác nhận tác tử, thực nghiệm hay PR đã hoàn thành.*
