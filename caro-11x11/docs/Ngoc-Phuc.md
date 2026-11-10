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

*Trạng thái tài liệu: Giai đoạn 1 đã merge (PR #17). Giai đoạn 2 đã mở PR #20 và sửa theo review, chờ duyệt lại. Giai đoạn 3 chưa thực hiện.*

---

## 12. Nhật ký thực hiện — Giai đoạn 1

### Trạng thái: Đã merge (PR #17)

### Thành phần đã cài đặt (`agents/minimax_agent.py`)
| Thành phần | Vai trò |
| :--- | :--- |
| `SearchState` | Trạng thái tìm kiếm: bản sao bàn cờ, người đến lượt, nước cuối, số nước, kết quả. `apply(action)` trả về trạng thái **mới**, cha giữ nguyên. |
| `find_winner`, `window_counts` | Phát hiện thắng trên toàn bàn và đếm cửa sổ 5 ô còn sống cho mỗi bên. |
| `terminal_score` | Điểm trạng thái kết thúc: thắng `+WIN_SCORE`, thua `−WIN_SCORE`, hòa `0`. |
| `heuristic_score`, `heuristic_bound` | Điểm heuristic cho trạng thái chưa kết thúc và cận trên trị tuyệt đối của nó. |
| `evaluate` | Điểm tổng hợp: kiểm tra terminal trước, sau đó mới dùng heuristic. |

### Quyết định thiết kế chính
- **Trạng thái bất biến qua `apply()`:** mỗi nước con sao chép bàn cờ (121 byte) thay vì đặt/gỡ quân tại chỗ. Ưu tiên tính đúng ở Giai đoạn 1; nếu Giai đoạn 2 đo thấy chi phí sao chép đáng kể sẽ cân nhắc lại và ghi nhận.
- **Khớp quy ước engine:** phát hiện thắng chỉ xét 4 đường qua nước vừa đi; hòa khi bàn đầy; ở trạng thái kết thúc `current_player` giữ người vừa đi. Khi dựng `SearchState` từ bàn cờ bất kỳ, kết quả được xác định bằng quét toàn bàn (`find_winner`).
- **Góc nhìn:** mọi hàm lượng giá nhận `perspective`; điểm dương có lợi cho bên đó; `evaluate(s, X) == -evaluate(s, O)`.
- **Heuristic cửa sổ 5 ô:** 252 cửa sổ trên bàn 11×11. Cửa sổ chỉ chứa quân một bên được cộng `WINDOW_WEIGHTS[k] = (0, 1, 10, 100, 1000, 0)[k]` với `k` là số quân. Lý do: cửa sổ bị đối thủ chiếm không thể thành chuỗi thắng (phân biệt chuỗi mở/bị chặn tự nhiên); mỗi quân thêm tăng điểm 10 lần; nhận diện cả chuỗi có khe; tự tạo ưu tiên trung tâm.
- **Terminal vượt trội:** `WIN_SCORE = 10 000 000`, gấp ~39,7 lần cận trên heuristic `252 × 1000 = 252 000`.

### Kiểm thử (`tests/test_minimax_agent.py`)
- Fixture dựng bằng chuỗi nước hợp lệ qua `CaroGame.step()`: X và O thắng theo 4 hướng với chuỗi 5 và chuỗi 6 (16 chuỗi), ván hòa bàn đầy 61 X / 60 O, và 5 ván ngẫu nhiên (seed 0–4). So sánh với engine **sau từng nước**.
- Tính độc lập trạng thái: cha và mảng đầu vào giữ nguyên sau khi duyệt nước con/cháu; các nước con độc lập với nhau.
- Lượng giá: quy ước điểm terminal, terminal vượt cận heuristic, đối xứng góc nhìn và đối xứng màu, thứ tự điểm các mẫu cho cả X và O, đếm cửa sổ tính tay (tâm 20, góc 3, mép 8), không thay đổi bàn cờ.
- Đã thử cố ý cài 8 lỗi (không đổi lượt, sửa bàn cha, bỏ góc nhìn, chuỗi 6 không thắng, bỏ phát hiện hòa, bỏ ưu tiên terminal, sai trọng số, không sao chép đầu vào): bộ test đều phát hiện.

### Notebook (`notebooks/02_minimax.ipynb`)
- Đã hoàn thiện phần 1–4; phần 5–10 ghi rõ "Chưa thực hiện".
- Phần 3: 37 chuỗi nước (17 fixture + 20 ván ngẫu nhiên seed 0–19), 1 768 bước so sánh, 0 bước lệch so với engine.
- Phần 4: bảng điểm các mẫu cho cả X và O (2 mở 72 < 3 bị chặn 143 < 3 mở 367 < 4 bị chặn 1 158 < 4 mở 2 279) và heatmap điểm một quân theo vị trí.

### Hạn chế kỹ thuật đã biết
- Heuristic không xét bên đến lượt: hai bên cùng có thế "4 mở" cho điểm bằng nhau dù bên đến lượt thắng ngay. Giai đoạn 2 giao việc này cho tìm kiếm.
- Không nhận diện đe dọa kép (ví dụ 4 + 3 mở) như một mẫu riêng; điểm là tổng tuyến tính.
- Trọng số cấp số 10 có lý do thiết kế nhưng chưa tinh chỉnh bằng thực nghiệm.
- Vùng 3×3 quanh tâm có cùng điểm vị trí; heuristic không phân biệt các ô này.
- Mỗi lần lượng giá quét lại toàn bộ 252 cửa sổ (chưa cập nhật tăng dần).

### Sửa theo review PR #17
| Góp ý | Cách sửa | Test bổ sung |
| :--- | :--- | :--- |
| Trọng số `(0, 1 000 000, 0, 0, 0, 0)` cho một quân ở tâm 20 000 000 điểm, vượt `WIN_SCORE`. | `_validate_weights` tính cận `số cửa sổ × trọng số lớn nhất` và từ chối khi cận `>= WIN_SCORE`; áp dụng cho `heuristic_score`, `evaluate`, `heuristic_bound`. | Ví dụ trong review bị từ chối; biên 39 682 được nhận, 39 683 bị từ chối (bàn 11×11); bàn 4×4 luật 4 quân với cận đúng bằng `WIN_SCORE` bị từ chối; với trọng số lớn nhất được phép, ván đã thắng vẫn cao điểm hơn mọi thế đang chơi. |
| `SearchState.initial(3, 3)` hợp lệ nhưng `evaluate()` báo lỗi vì trọng số mặc định chỉ dành cho luật 5 quân. | Thêm `default_weights(win_length, board_size)`: cửa sổ `k` quân được `10^(k-1)`, bằng đúng `WINDOW_WEIGHTS` với luật 5 quân trên bàn 11×11. Chỉ khi chuỗi thắng quá dài làm cận chạm `WIN_SCORE` mới giảm cơ số (tối thiểu 2). `weights=None` trong `evaluate`/`heuristic_score`/`heuristic_bound` dùng bộ này theo `win_length` và cạnh bàn. | `evaluate` không truyền `weights` trên các luật 3×3, 6×6/4, 11×11 với 2, 7, 11 quân; trọn một ván 3×3 đến khi X thắng; tham số luật không hợp lệ bị từ chối. |
| Ghi trực tiếp 5 quân X vào `state.board` tạo chuỗi thắng nhưng `result()` vẫn là `None`, `move_count` vẫn là 0. | Mảng `board` chuyển sang chỉ đọc sau khi khởi tạo trạng thái gốc và sau khi `apply()` ghi nước đi vào bản sao. Mảng đầu vào của người gọi vẫn ghi được. | Ghi trực tiếp bị từ chối ở trạng thái gốc, trạng thái dựng từ mảng, trạng thái con và trạng thái kết thúc; tình huống 5 quân trong review giữ nguyên `result`/`move_count`; `apply()` vẫn tạo được con và cháu. |

- Toàn bộ test của dự án: 242 pass (trước khi sửa: 209).
- Đã cố ý cài lại từng lỗi (bỏ kiểm tra cận, dùng `>` thay `>=`, bỏ chỉ đọc ở `__init__`, bỏ chỉ đọc ở `apply`, `evaluate` dùng lại `WINDOW_WEIGHTS`): cả 5 đều bị test bắt.
- Notebook chạy lại cho kết quả giống hệt bản đã commit, vì trọng số mặc định cho luật 5 quân không đổi.

---

## 13. Nhật ký thực hiện - Giai đoạn 2

### Trạng thái: Đã mở PR #20, đã sửa theo review, chờ duyệt lại

### Thành phần đã cài đặt (`agents/minimax_agent.py`)
| Thành phần | Vai trò |
| :--- | :--- |
| `MinimaxSearcher` | Minimax giới hạn độ sâu với cờ `use_alpha_beta` (bật/tắt cắt tỉa) và `move_ordering` (bật/tắt sắp xếp nước). |
| `SearchResult` | Kết quả một lần tìm kiếm: giá trị gốc, nước đi, độ sâu, số nút duyệt, số lần cắt tỉa. |
| `MinimaxAgent` | Tác tử theo hợp đồng `BaseAgent`: chế độ `fixed` và `time` (Iterative Deepening). |

### Quyết định thiết kế chính
- **Độ sâu và vai trò:** một ply là một nước của một bên. Nút gốc là Max (bên đến lượt ở gốc); nút mà đối thủ đến lượt là Min. Mọi điểm tính theo góc nhìn bên ở gốc.
- **Terminal:** không mở rộng. Thắng ở ply `p` được `WIN_SCORE - p`, thua được `-(WIN_SCORE - p)`: ưu tiên thắng sớm và kéo dài khi buộc phải thua.
- **Alpha-Beta bật/tắt trên cùng nền tảng:** hai chế độ dùng chung trạng thái, hàm lượng giá, tập ứng viên (bán kính 2) và thứ tự xét nước. Khi tắt, alpha/beta không được cập nhật nên không có lần cắt nào.
- **Thứ tự xét nước:** mặc định hàng rồi cột của `candidate_actions`. Tùy chọn `move_ordering=True` (mặc định của tác tử) sắp nước con theo heuristic của bên đi, tốt trước, chỉ ở nút còn ít nhất 2 ply; sort ổn định nên tất định.
- **Phá hòa:** giữ nước đồng điểm gặp đầu tiên (so sánh `>` chặt). Nhờ vậy bật/tắt Alpha-Beta trả về cùng một nước khi cùng thứ tự xét.
- **Thống kê:** `nodes_visited` đếm mọi nút được duyệt, tính cả gốc. `pruned_nodes` đếm số lần `alpha >= beta` mà còn ít nhất một nước con chưa xét; đây là số lần cắt, không phải số nút bị bỏ qua. Tác tử ghi thêm `depth_reached` và `root_value` vào `stats`.
- **Chế độ `time`:** Iterative Deepening từ độ sâu 1. Dừng chủ động ở `deadline - 0,1 s` (kiểm tra ở mọi nút); độ sâu đang dở bị bỏ, giữ kết quả độ sâu hoàn tất gần nhất. Nước tốt nhất của độ sâu trước được xét đầu tiên ở gốc. Dừng sớm khi đã chứng minh thắng/thua chắc chắn. Nước dự phòng là ứng viên đầu tiên nếu chưa xong độ sâu 1. Độ sâu của chế độ `fixed` không được dùng làm điều kiện dừng.
- **Trọng số theo luật của trạng thái:** `weights=None` dùng `default_weights` theo `win_length` và cỡ bàn của trạng thái đang tìm, giống `evaluate()` sau review PR #17. Vì điểm thắng bị trừ theo ply, cận heuristic phải nhỏ hơn `WIN_SCORE - số ô của bàn` (11×11: cận 252 000, còn dư khoảng 9,75 triệu). `MinimaxSearcher` kiểm tra khi bắt đầu tìm kiếm; `MinimaxAgent` kiểm tra ngay khi tạo vì tác tử luôn chơi luật của engine.
- **Tối ưu không đổi kết quả:** trong vòng tìm kiếm dùng lõi heuristic không kiểm tra đầu vào; khi không sắp xếp, trạng thái con chỉ được sinh khi thực sự xét nên nhánh bị cắt không tốn công. Đo trên máy nhóm: nhanh hơn 2-4 lần, giá trị và nước đi không đổi.

### Kiểm thử bổ sung (`tests/test_minimax_agent.py`)
- Bản Minimax tham chiếu và Alpha-Beta tham chiếu viết độc lập trong file test (duyệt hết, thứ tự hàng/cột) để đối chiếu.
- Chiến thuật: thắng ngay cho X và O ở độ sâu 1/2/3, chặn chuỗi 4 cho X và O ở độ sâu 2, ưu tiên thắng thay vì chặn khi cả hai cùng có 4.
- Giá trị gốc khớp tham chiếu ở độ sâu 1, 2 (và 3 trên cây nhỏ), cho cả 4 tổ hợp bật/tắt cắt tỉa và sắp xếp; nước trả về đạt đúng giá trị gốc.
- Bật/tắt Alpha-Beta: cùng giá trị, cùng nước, số nút không tăng (6 vị trí × độ sâu 1, 2 × 2 thứ tự xét = 24 ca); số nút và số lần cắt khớp chính xác bản tham chiếu, kể cả trường hợp `alpha >= beta` ở nước con cuối (không tính là cắt).
- Số nút khi tắt cắt tỉa đúng bằng cây duyệt hết, chứng minh dừng đúng độ sâu và không mở rộng nút terminal.
- Trọng số: trọng số sai bị từ chối ở cả searcher và tác tử; bàn 4×4 luật 4 quân với cận `WIN_SCORE - 15` được `evaluate()` nhận nhưng search từ chối vì không chừa đủ biên ply; trọng số mặc định cho đúng nước thắng trên các luật 3×3, 6×6 luật 4, 11×11 luật 5; mặc định cho cùng kết quả với truyền `WINDOW_WEIGHTS`.
- Tác tử: nước hợp lệ, bàn cờ gốc giữ nguyên, thống kê đặt lại mỗi lượt, chơi trọn ván với Random (độ sâu 1 và 2), chế độ `time` trả đúng giá trị của độ sâu đã hoàn tất, dùng nước dự phòng khi hết giờ, dừng sớm khi đã thắng chắc.
- Thử cố ý cài 11 lỗi (đảo vai Max/Min, bỏ trừ ply, lệch độ sâu, sai phá hòa, sai điều kiện cắt, không cập nhật beta, đếm cắt sai, Iterative Deepening dùng độ sâu dở, không dừng khi thắng chắc, không cộng dồn thống kê...): bộ test phát hiện 10 lỗi. Lỗi còn lại ("cắt khi tắt Alpha-Beta") không làm đổi hành vi vì khi tắt, alpha/beta luôn là vô cực. Thêm 2 lỗi cho phần trọng số (bỏ biên ply, searcher dùng lại `WINDOW_WEIGHTS`): cả 2 đều bị bắt.
- Toàn bộ test của dự án (kể cả MCTS và mạng nơ-ron của nhóm trên `main`): 491 pass.

### Notebook (`notebooks/02_minimax.ipynb`)
- Phần 5: cây nhỏ độ sâu 2 (bán kính 1) in toàn bộ giá trị Max/Min, giá trị tính tay trùng `MinimaxSearcher` (-85 tại (4,5)); Alpha-Beta trên cùng cây cho cùng giá trị và cùng nước, duyệt 64/187 nút với 10 lần cắt; bản đồ giá trị từng nước gốc; so sánh tầm nhìn độ sâu 1 và 2 trên thế "X phải chặn" có in bàn cờ và bên đến lượt (71/71 nước không chặn dẫn tới thua ở ply 2).
- Phần 6: 6 vị trí × độ sâu 1, 2 × có/không sắp xếp: giá trị gốc và nước đi trùng khớp 100%. Ở độ sâu 2, Alpha-Beta giảm trung bình 71,2% số nút khi không sắp xếp và 84,6% khi có sắp xếp. Ở độ sâu 1 không có cắt tỉa (nút gốc không có cận trên).
- Phần 6.1: 4 ván qua `play_match` ở chế độ `time` 2,0 s (Minimax cầm X và O, Random seed 0 và 1): 23 nước, không timeout, `elapsed_s` lớn nhất 1,902 s, p95 1,901 s; độ sâu hoàn tất 3-5 ở 19/23 nước, 4 nước còn lại dừng ở độ sâu 1. Các số này phụ thuộc tốc độ máy nên mỗi lần chạy lệch đôi chút.

### Hạn chế kỹ thuật đã biết
- Độ sâu 3 không cắt tỉa mất hàng chục giây mỗi nước nên không dùng thực tế; tác tử chế độ `fixed` mặc định độ sâu 2.
- Biên 0,1 s mới kiểm chứng trên một máy (số liệu phần 6.1); máy chậm hơn có thể cần biên lớn hơn.
- Sắp xếp nước dùng heuristic tĩnh; chưa có bảng chuyển vị (transposition table) hay killer move.
- Tập ứng viên bán kính 2 có thể bỏ sót nước xa cụm quân.

### Sửa theo review PR #20
| Góp ý | Cách sửa | Test bổ sung |
| :--- | :--- | :--- |
| Khi bật sắp xếp nước, `ordered_moves` tạo và chấm điểm cả nhóm ứng viên rồi mới kiểm tra hạn dừng (đo được: hết giờ ngay từ đầu vẫn chấm 99/99 nước gốc). | Kiểm tra `should_stop()` trước khi sinh và chấm từng nước con; hết giờ báo `SearchTimeout` ngay. Phần kiểm tra được gom vào `_check_stop()` dùng chung với `_minimax`. Thứ tự sắp xếp và kết quả tìm kiếm không đổi. | Hạn dừng báo hết giờ ở lần kiểm tra thứ 3 thì chỉ chấm 2 nước; hết giờ từ đầu thì không chấm nước gốc nào; chưa tới hạn thì kết quả, số nút, số lần cắt giữ nguyên; đồng hồ giả cho tác tử chế độ `time` dừng giữa lúc chấm nước gốc ở độ sâu 2 và giữ kết quả độ sâu 1. |
| Ván đã có người thắng nhưng bàn còn ô trống thì tác tử vẫn trả về một ô trống. | `get_action` báo `ValueError` ngay khi trạng thái đã kết thúc; bỏ nhánh tự chọn ô trống. Ván chưa kết thúc luôn có ô trống kề một quân nên tập ứng viên không rỗng. | Gọi qua `choose_action()` trên ván X thắng và ván O thắng, cho cả hai bên, ở cả chế độ `fixed` và `time`: đều báo `ValueError`, bàn cờ đầu vào giữ nguyên. |

- Toàn bộ test của dự án: 501 pass (trước khi sửa: 491).
- Đã cố ý cài lại từng lỗi (bỏ kiểm tra hạn khi chấm nước, chỉ kiểm tra sau khi đã chấm hết, ván kết thúc vẫn trả ô trống): cả 3 đều bị test bắt.
- Notebook chạy lại với code mới cho giá trị, nước đi, số nút và số lần cắt giống hệt bản đã commit; chỉ các số đo thời gian lệch vài mili giây như mọi lần chạy, nên không commit lại notebook.

---

## 14. Phân công bổ sung — Task 2 Rule-based

**Phụ trách.** Ngọc Phúc (`@PhucLee23`) · [Issue #8 — Hoàn thiện pattern và thứ tự ưu tiên](https://github.com/10tothepowerof6/AdvancedAI-Repo/issues/8).

Đây là đầu việc bổ sung, thực hiện trên nhánh riêng với Minimax. Mục tiêu là mở rộng `RuleBasedAgent` với nhận diện mẫu hình, xử lý xung đột luật và truy vết đủ rõ để Task 3 so sánh bản đầy đủ với baseline trên cùng điều kiện.

### 14.1. Điều kiện bắt đầu và phạm vi chỉnh sửa

Chỉ tạo nhánh `feature/rule-based-task-2` sau khi [Task 1 Rule-based của Phi Hùng](https://github.com/10tothepowerof6/AdvancedAI-Repo/issues/7) đã merge. Cập nhật `main`, chạy test Rule-based trên baseline và đọc API khởi tạo, cách phá hòa, thống kê cùng notebook phần 1–4 đã bàn giao. Trong thời gian chờ có thể chuẩn bị bảng ưu tiên, kỳ vọng cho từng thế cờ và ca kiểm thử trên giấy.

| File | Phần phụ trách |
| :--- | :--- |
| `agents/rule_based_agent.py` | Mở rộng baseline với pattern, xung đột luật, phòng thủ nhiều hướng, dừng theo ngân sách và truy vết. |
| `tests/test_rule_based_agent.py` | Giữ test baseline, bổ sung test pattern, chiến thuật, deadline và tích hợp hai cấu hình. |
| `notebooks/01_rule_based.ipynb` | Giữ phần 1–4 của baseline, thêm phần 5–6. Các phần 7–10 dành cho Task 3. |
| `docs/Ngoc-Phuc.md` | Ghi quyết định thiết kế, cấu hình, kết quả kiểm thử và giới hạn thực tế tại mục này. |

`core/`, `config/`, `BaseAgent`, `RandomAgent`, `agents/__init__.py`, `evaluation/`, test chung và `requirements.txt` do nhóm trưởng quản lý. Chú thích, đầu ra, notebook và phần diễn giải trong commit hoặc PR dùng tiếng Việt.

### 14.2. Hợp đồng tích hợp và nhận diện mẫu hình

- **Luật cờ và đầu vào.** Bàn 11×11, X = `1`, O = `-1`, ô trống = `0`. Từ 5 quân liên tiếp theo bốn hướng là thắng, kể cả khi bị chặn hai đầu. Agent kế thừa `BaseAgent`, nhận `SearchBudget` và được gọi qua `choose_action(board, player)` trên ván đang diễn ra. Giữ đầu ra là `(row, col)` gồm hai số nguyên Python, chỉ số từ 0. Đánh thử trên bản sao, giữ bàn cờ gốc nguyên vẹn.
- **Hai cấu hình đối chứng.** `use_patterns=False` giữ hành vi ba tầng của Task 1. `use_patterns=True` bật các tầng mẫu hình. Khai báo cờ tường minh khi so sánh hai biến thể. Dùng cùng hàm điểm vị trí, cách phá hòa, seed và bán kính cho hai cấu hình.
- **Nhận diện theo nước tiếp theo.** Tách hàm kiểm tra từng mẫu khỏi việc chọn nước. Một nước đã thắng ngay được xử lý ở tầng `win_now`. Đếm các ô thắng mới do nước thử tạo ra theo tọa độ riêng biệt, kể cả khi nhiều đường cùng dẫn tới một ô. Phân biệt quân mình, quân đối thủ, ô trống và biên bàn.

| Mẫu hình | Điều kiện và ca đối chứng |
| :--- | :--- |
| Bốn mở hoặc đe dọa kép | Có ít nhất hai ô thắng độc lập ở lượt sau. Kiểm tra chuỗi bốn hở hai đầu và giao điểm tạo đe dọa ở hai hướng. Ghi riêng hình dạng chuỗi và số ô thắng khi giải thích. |
| Bốn nửa mở hoặc có khe | Có đúng một ô thắng mới, ở đầu còn mở hoặc ở khe giữa. Mép bàn và quân đối thủ chặn một đầu phải được xử lý đúng. Một quân đối thủ ngắt giữa chuỗi có thể khiến mẫu không còn ô thắng. |
| Ba mở | Nước thử chưa tạo ô thắng ngay nhưng mở đường cho ít nhất một nước tiếp theo tạo bốn mở hoặc đe dọa kép. Kiểm tra cả ba liền và ba có khe. Dùng mẫu sát biên hoặc bị chặn không thể phát triển để đối chứng. |

### 14.3. Thứ tự luật, phòng thủ và ngân sách

| Thứ tự | Luật | Phạm vi và cách quyết định |
| :--- | :--- | :--- |
| 1 | `win_now` | Xét toàn bộ ô trống và thắng ngay nếu có thể. |
| 2 | `block_urgent` | Xét toàn bộ ô trống để chặn đối thủ thắng ở lượt kế. |
| 3 | `attack_open_four` | Tạo bốn mở hoặc đe dọa kép của bên đang đi. |
| 4 | `block_open_four` | Ưu tiên nước làm giảm tối đa số nước tạo bốn mở hoặc đe dọa kép còn lại của đối thủ. |
| 5 | `attack_broken_four` | Tạo một đe dọa thắng buộc đối thủ phải đáp trả. |
| 6 | `attack_open_three` | Tạo ba mở để chuẩn bị đe dọa ở lượt sau. |
| 7 | `block_open_three` | Ngăn đối thủ tạo ba mở. |
| 8 | `positional_fallback` | Chọn nước dự phòng theo điểm vị trí của baseline. |

- **Phạm vi ứng viên.** Tầng 3–8 dùng `candidate_actions` với bán kính mặc định 2. Khi `use_patterns=False`, bỏ tầng 3–7. Công bố mọi thay đổi phạm vi, trọng số và quy tắc phá hòa. Ghi rõ giới hạn nếu tập rút gọn bỏ sót nước xa cụm quân.
- **Phòng thủ nhiều hướng.** Khi có nhiều phương án ở `block_open_four`, thử từng nước trên bản sao, tính lại cơ hội tạo đe dọa kép của đối thủ trong cùng phạm vi rồi mới dùng điểm vị trí để phá hòa. Phân biệt phòng thủ trước khi đe dọa kép hình thành với trường hợp đối thủ đã có hai ô thắng độc lập và một lượt chỉ chặn được một ô.
- **Ngân sách.** `fixed` hoàn tất các bước của tầng luật đang xét. Ở `time`, giữ sẵn một nước dự phòng hợp lệ và kiểm tra mốc dừng giữa các ứng viên, gồm quét pattern lồng nhau và thử lại phương án phòng thủ. Có thể bắt đầu với mốc 1,9 giây trên hạn cứng 2,0 giây, sau đó kiểm chứng trên máy chạy và điều chỉnh biên cho ngân sách ngắn. `time_exhausted()` kiểm tra hạn cứng. Khi dừng sớm, trả nước hợp lệ đã có và ghi đúng lý do. Runner cưỡng chế quá giờ, ghi `timeout` và xử thua.
- **Truy vết và thống kê.** `explain_last_action()` ghi luật kích hoạt, tọa độ chọn, số ứng viên thực sự đã xét, danh sách nước khớp và lý do. Dữ liệu trả về phải độc lập với agent. Công bố đơn vị đếm `nodes_visited`, gồm cách đếm lượt đánh thử phòng thủ, và chỉ đếm công việc thực hiện khi bị dừng sớm. `simulations` và `pruned_nodes` để 0. Khi chạy qua `play_match`, dùng `elapsed_s` từ runner để mô tả độ trễ thi đấu.

### 14.4. Khung notebook phần 5–6

| Phần | Nội dung cần có |
| :--- | :--- |
| 5. Pattern và độ ưu tiên | Ghi bảng luật, mẫu dương/âm, cách tính điểm, phạm vi ứng viên và cách phá hòa. Dựng các thế qua `CaroGame.step()`, ghi kỳ vọng trước khi chạy rồi gọi các hàm nhận diện từ module để đối chiếu. |
| 6. Giải thích nước đi | Dùng agent từ module để chạy các ca xung đột, đổi X/O, phòng thủ nhiều hướng và đe dọa kép đã hình thành. Hiển thị lý do và nước khớp lấy trực tiếp từ truy vết, rồi ghi rõ giới hạn. |

Giữ cùng cấu hình và fixture khi đối chiếu. Chạy notebook từ kernel mới, ghi kết quả thật. Phân tích ablation, thi đấu nhiều ván và bảng W/D/L thuộc Task 3.

### 14.5. Cách viết test và tiêu chí đạt

- [ ] Mỗi pattern có ca dương và ca âm. Bao phủ bốn hướng, hai phía, đầu bị chặn, biên, chuỗi có khe và quân đối thủ ngắt giữa. Tính tay tập nước thắng hoặc kiểm tra bằng `CaroGame.step()` trên ván hợp lệ để lấy kỳ vọng độc lập.
- [ ] Kiểm tra từng tầng và các cặp luật xung đột. Thắng ngay luôn đứng đầu, chặn thắng ngay đứng trên pattern. Đổi X/O, kiểm tra phá hòa theo quy tắc đã công bố và xác nhận cấu hình tắt pattern giữ hành vi baseline.
- [ ] Kiểm tra thế có một nước chặn được nhiều hướng và thế đã có hai ô thắng độc lập. Với nhiều nước tương đương, dùng tập nước đúng hoặc tiêu chí phá hòa đã công bố.
- [ ] Dùng đồng hồ giả cho ca hết giờ trước khi quét pattern, trong lúc quét và khi đang thử phương án phòng thủ. Kiểm tra vòng quét dừng, nước trả về hợp lệ, truy vết và số ứng viên khớp công việc thực tế.
- [ ] Kiểm tra API, bàn cờ đầu vào nguyên vẹn, bản truy vết độc lập. Chạy test riêng, toàn bộ suite và notebook từ kernel mới. Ghi số test đạt/thất bại, fixture, cấu hình, kết quả thực chạy và các giới hạn đã biết trong PR.

**Thế hồi quy phòng thủ nhiều hướng.** Dựng bằng `CaroGame.step()`, xen kẽ X và O theo bảng, X đi trước. Sau 12 nước, X đi `(0, 0)` thì O hết nước tạo đe dọa kép. Nếu X đi `(0, 4)`, O còn nước `(4, 0)` tạo hai ô thắng `(0, 0)` và `(5, 0)`.

| Bên | Tọa độ theo thứ tự |
| :--- | :--- |
| X | `(1, 4)`, `(2, 5)`, `(9, 10)`, `(8, 8)`, `(10, 8)`, `(7, 10)` |
| O | `(0, 1)`, `(1, 0)`, `(0, 2)`, `(2, 0)`, `(0, 3)`, `(3, 0)` |

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_rule_based_agent.py -v
.\.venv\Scripts\python.exe -m pytest tests -v
```

### 14.6. Nhánh, cột mốc commit và PR

Sau khi Task 1 đã merge, từ thư mục `caro-11x11` kiểm tra `git status`, lưu công việc đang làm trên nhánh hiện tại rồi chuyển sang `main`, chạy `git pull --ff-only origin main` và chạy `git switch -c feature/rule-based-task-2`.

| Cột mốc | File được stage | Commit đề xuất |
| :--- | :--- | :--- |
| Nhận diện mẫu và bảng luật chạy được | `agents/rule_based_agent.py`, `tests/test_rule_based_agent.py` | `feat(rule-based): bổ sung mẫu hình và thứ tự ưu tiên` |
| Phòng thủ nhiều hướng, dừng theo ngân sách và truy vết | `agents/rule_based_agent.py`, `tests/test_rule_based_agent.py` | `feat(rule-based): xử lý phòng thủ nhiều hướng và kiểm soát thời gian` |
| Notebook phần 5–6 và ghi chú bàn giao | `notebooks/01_rule_based.ipynb`, `docs/Ngoc-Phuc.md` | `docs(rule-based): giải thích quyết định và bàn giao cấu hình pattern` |

Stage đúng thay đổi của từng cột mốc, kiểm tra `git diff --cached`, commit rồi push. Lần đầu dùng `git push -u origin feature/rule-based-task-2`, các lần sau dùng `git push` trên cùng nhánh. Nếu cần review sớm, mở PR nháp và chỉ tham chiếu `#8`.

- **Tiêu đề PR.** `feat(rule-based): hoàn thiện nhận diện mẫu hình và ưu tiên chiến thuật`.
- **Mô tả PR.** Mở rộng baseline với các tầng mẫu hình, giải quyết xung đột, phòng thủ nhiều hướng và giới hạn thời gian. Giải thích quyết định bằng notebook phần 5–6. Kèm danh sách file, lệnh chạy, số test thực tế, fixture, hành vi hai cấu hình và giới hạn kỹ thuật.
- **Thông tin GitHub.** Base `main`, compare `feature/rule-based-task-2`, assignee `@PhucLee23`, reviewer `@10tothepowerof6`, [milestone Báo cáo tiến độ nửa dự án](https://github.com/10tothepowerof6/AdvancedAI-Repo/milestone/2). Thêm `Closes #8` khi đủ tiêu chí nghiệm thu. Kiểm tra diff chỉ chứa phần mở rộng Task 2 trước khi yêu cầu review.
- **Bàn giao sau merge.** Ghi rõ cách khởi tạo hai cấu hình, điểm vị trí, quy tắc phá hòa, thống kê, mốc dừng và thế cờ kiểm chứng. Đây là thông tin đầu vào để Task 3 chạy đối chứng và thi đấu thử.
