# Phi Hùng — MCTS Rollout

## 1. Mục tiêu và tổ chức sản phẩm

Mục tiêu chính là hoàn thiện tác tử **MCTS** (Tìm kiếm cây Monte Carlo kết hợp Rollout ngẫu nhiên) cho cờ Caro 11×11, sẵn sàng tích hợp vào giải đấu chung và ứng dụng demo của dự án.

Mỗi giai đoạn thực hiện yêu cầu đầy đủ 4 thành phần cấu thành:
- **Module mã nguồn:** Cài đặt thuật toán trong `agents/mcts_agent.py`.
- **Kiểm thử tự động:** Xây dựng bộ test riêng trong `tests/test_mcts_agent.py`.
- **Thực nghiệm và trực quan hóa:** Trình bày quá trình đo lường, phân tích trong `notebooks/03_mcts.ipynb`.
- **Tài liệu theo dõi:** Cập nhật thiết kế, tiến độ và các hạn chế kỹ thuật trong `docs/Phi-Hung.md`.

> **Lưu ý cốt lõi:** Notebook phải trực tiếp import và sử dụng tác tử từ module `agents/mcts_agent.py`. Tuyệt đối không duy trì hai phiên bản thuật toán song song giữa notebook và mã nguồn chính. Tài liệu này đóng vai trò định hướng kiến trúc, yêu cầu kỹ thuật và tiêu chí nghiệm thu; không cung cấp mã mẫu hay mã giả.

---

## 2. Phạm vi file và quy ước chung

### Phân công trách nhiệm theo file
| File thao tác | Trách nhiệm chính |
| :--- | :--- |
| `agents/mcts_agent.py` | Lớp `MCTSAgent` và các cấu trúc cây, nút nội bộ |
| `tests/test_mcts_agent.py` | Bộ kiểm thử tự động riêng cho thuật toán |
| `notebooks/03_mcts.ipynb` | Kịch bản thí nghiệm, đo đạc số liệu và phân tích kết quả |
| `docs/Phi-Hung.md` | Ghi nhận quyết định thiết kế, tiến độ thực hiện và giới hạn kỹ thuật |

### Quy ước phát triển và làm việc với Git
- **Thư mục làm việc:** Mọi thao tác lệnh và đường dẫn tương đối đều tính từ thư mục gốc `caro-11x11`.
- **Phạm vi chỉnh sửa:** Các thành phần nền tảng (`core/`, `config/`, `BaseAgent`, `RandomAgent`, `agents/__init__.py`, `requirements.txt`, các bài test chung, `README.md` gốc) thuộc quyền quản lý tích hợp của nhóm trưởng. Mọi đề xuất thay đổi cần thông qua thảo luận hoặc PR riêng. Tác tử cá nhân có thể được import trực tiếp từ file module mà không cần sửa `agents/__init__.py`.
- **Ngôn ngữ sử dụng:** Đầu ra console, chú thích code (comment), giải thích trong notebook, thông điệp commit và nội dung PR đều sử dụng tiếng Việt chuẩn. Tên biến, hàm, lớp tuân thủ quy chuẩn kỹ thuật tiếng Anh (PEP 8).
- **Quy chuẩn thông điệp commit:** Sử dụng tiền tố chuẩn theo Conventional Commits: `feat`, `fix`, `test`, `docs`, `chore` (ví dụ: `fix(mcts): sửa xử lý trạng thái kết thúc`). Phần mô tả nêu rõ nội dung thay đổi và nguyên nhân; chỉ xác nhận "đã kiểm thử" khi các lệnh test thực sự đã chạy thành công.

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
.\.venv\Scripts\python.exe -m pytest tests/test_mcts_agent.py -v

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
- **Cơ chế dừng hợp tác (Cooperative Stopping):** Tác tử phải chủ động dừng tìm kiếm trước deadline để kịp hoàn tất và gửi kết quả. `self.time_exhausted()` chỉ trả về đúng khi đã tới deadline, không tự trừ biên an toàn. Mốc dừng sớm khởi đầu đề xuất là 1,9 giây trên hạn 2,0 giây (chừa khoảng 100 ms); thuật toán tự quản lý, công khai và kiểm chứng biên này trên máy chạy. Đây không phải bảo đảm tuyệt đối tránh timeout; hạn cứng của runner vẫn là 2,0 giây, không được cộng thêm 100 ms. MCTS chủ động kiểm tra deadline trong quá trình rollout; luôn có sẵn nước đi dự phòng ban đầu đề phòng trường hợp chưa kịp xong mô phỏng đầu tiên.
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
- Tái lập kết quả MCTS: Chạy ở chế độ `fixed` kết hợp truyền `seed` cố định; không ép buộc hai lần chạy ở chế độ `time` phải chọn nước đi giống hệt nhau.

---

## 4. Khung tổ chức thuật toán

Cấu trúc thuật toán MCTS Rollout cần phân định ranh giới trách nhiệm rõ ràng giữa các thành phần:

| Thành phần | Yêu cầu tổ chức và kỹ thuật |
| :--- | :--- |
| **Trạng thái mô phỏng** | Biểu diễn bàn cờ, người đến lượt, trạng thái kết thúc và kết quả; thống nhất phương thức khởi tạo trạng thái từ đầu vào của agent. |
| **Cây tìm kiếm** | Phân định rõ nút gốc, mối liên kết cha/con, hành động dẫn đến nút, số lượt thăm ($N$) và giá trị tích lũy ($Q$). Nêu rõ nút lưu giá trị theo góc nhìn của người chơi nào. |
| **Lựa chọn và mở rộng** | Mô tả công thức UCT (Upper Confidence Bound for Trees), cách xử lý các nút chưa thăm, các hành động chưa được mở rộng và điều kiện dừng duyệt cây. |
| **Mô phỏng và lan truyền ngược** | Tách biệt chính sách mô phỏng (rollout policy) khỏi việc cập nhật thống kê để kiểm thử độc lập; xử lý rõ ràng các kết quả thắng, thua, hòa và cách lan truyền giá trị ngược lên cây (backpropagation). |
| **Quyết định cuối cùng** | Phân biệt việc chọn nhánh trong quá trình tìm kiếm với việc chọn nước đi chính thức trả về; thiết lập ngân sách mô phỏng và cơ chế phá hòa. |

> **Lưu ý thiết kế:** Mỗi thành phần cần có ca kiểm thử độc lập; không chỉ thử nghiệm trên tác tử hoàn chỉnh cuối cùng. Tuyệt đối không đưa các chức năng phụ như vẽ biểu đồ, đọc file notebook hay vận hành giải đấu vào phương thức lựa chọn nước đi.

---

## 5. Khung lập trình toàn bộ notebook

File `notebooks/03_mcts.ipynb` được tổ chức tuần tự theo 10 phần dưới đây. Mỗi phần cần có cell Markdown mô tả mục tiêu, cell thực thi mã nguồn và phần phân tích, nhận xét kết quả:

1. **Câu hỏi nghiên cứu và giới hạn:** MCTS thay đổi chất lượng nước đi và chi phí tính toán như thế nào khi tăng số lượt mô phỏng? Nêu rõ các giới hạn về lọc ứng viên, chính sách rollout và thời gian thực thi.
2. **Môi trường và cấu hình:** Ghi nhận `seed`, bán kính ứng viên, ngân sách mô phỏng, hệ số khám phá ($c$), chính sách rollout, phần cứng và phiên bản thư viện.
3. **Kiểm tra trạng thái:** Minh họa trực quan trạng thái bàn cờ trước/sau nước đi, lượt chơi và kết quả; đối chiếu tính chính xác với Game Engine.
4. **Quan sát cấu trúc cây:** Trình bày số nút con, lượt thăm và giá trị đánh giá của các nhánh tại một thế cờ nhỏ; giải thích rõ quy ước góc nhìn giá trị.
5. **Theo dõi bốn bước MCTS:** Trực quan hóa quy trình 4 bước: Lựa chọn (Selection), Mở rộng (Expansion), Mô phỏng (Simulation), Cập nhật ngược (Backpropagation) trong một lượt tìm kiếm qua bảng hoặc đồ thị.
6. **Kiểm tra năng lực quyết định:** Đánh giá trên bộ tình huống chiến thuật cố định để quan sát phân bố chọn nước và các lỗi chiến thuật; ghi nhận đầy đủ các seed thất bại.
7. **Khảo sát ngân sách:** Tách hai phần kết quả: `fixed` so sánh 100 / 500 / 1.000 mô phỏng trên cùng tập vị trí và seed; `time` chạy giới hạn 2,0 giây/nước, ghi số mô phỏng thực hoàn thành, độ trễ và timeout. Không gộp hai chế độ thành cùng điều kiện thí nghiệm.
8. **Thi đấu thử nghiệm:** Chạy đối đầu với `RandomAgent` theo từng cặp ván đổi bên đi trước/sau; ghi nhận kết quả $W/D/L$, điểm số và chi phí thời gian. Kết quả thử nghiệm không thay thế giải đấu chính thức.
9. **Phân tích các trường hợp thất bại:** Chọn các ván thua điển hình để phân tích: đe dọa bị bỏ qua, tác động của rollout ngẫu nhiên và độ sâu thực tế của cây tìm kiếm.
10. **Tổng kết và hướng dẫn tái lập:** Đưa ra câu trả lời dựa trên số liệu thực nghiệm, nêu rõ hạn chế kỹ thuật, cấu hình đề xuất bàn giao và thứ tự chạy lại notebook.

### Nguyên tắc thực nghiệm trong Notebook
- Tập trung toàn bộ cấu hình ở phần đầu notebook; ghi lại cấu hình cụ thể đi kèm mỗi thí nghiệm khi có sự thay đổi.
- Sử dụng chung bộ vị trí bàn cờ và phương pháp ghi nhận kết quả để đảm bảo tính công bằng khi so sánh các biến thể. Luôn cố định `seed` nếu có yếu tố ngẫu nhiên.
- Thực thi thuật toán trực tiếp từ module `agents/mcts_agent.py`; notebook chỉ đóng vai trò tổ chức thí nghiệm và trình bày kết quả.
- Mỗi tình huống thử nghiệm cần thể hiện rõ: trạng thái bàn cờ, bên đến lượt, kỳ vọng trước khi chạy và quan sát thực tế sau khi chạy.
- Luôn chạy tuần tự từ một kernel mới (Restart Kernel & Run All); tránh sử dụng các biến còn sót lại từ các cell đã chạy trước đó.
- Không thay thế bộ unit test bằng việc chỉ quan sát bàn cờ trong notebook.
- Chưa có số liệu thì ghi nhận là chưa thực hiện. Tuyệt đối không điền bảng dữ liệu giả định, không đưa ra kết luận vội vã chỉ dựa trên một ván thắng đơn lẻ.
- Kết quả thử nghiệm cục bộ không thay thế giải đấu chính thức (tối thiểu 100 ván/cặp, đổi bên, nhiều seed). Tuyệt đối không dùng các ván đánh giá cuối cùng để điều chỉnh tham số.

---

## 6. Giai đoạn 0 — Khởi tạo môi trường và xác nhận nền tảng

- Đọc kỹ `README.md` gốc của repository, cài đặt môi trường ảo và dependencies từ `requirements.txt`.
- Chạy toàn bộ test nền tảng và kiểm tra hoạt động của `notebooks/00_game_engine_test.ipynb`.
- Mở `notebooks/03_mcts.ipynb`, xác nhận Jupyter đang sử dụng đúng kernel `.venv` và các ví dụ tích hợp mẫu chạy thành công.
- Kiểm tra trạng thái Git (`git status`), commit hoặc lưu trữ (stash) các công việc đang dở trước khi chuyển nhánh.
- Giai đoạn 0 không tạo commit rỗng hay PR riêng. Các nhánh của từng giai đoạn tiếp theo được tạo tuần tự từ `main` sau khi PR của giai đoạn trước đó đã được merge hoàn tất.
- Nếu nhánh tương ứng đã tồn tại sẵn, chuyển sang nhánh đó thay vì tạo mới. Kiểm tra kỹ lịch sử Git trước khi đổi tên nhánh.

---

## 7. Giai đoạn 1 — Thiết kế trạng thái và cây tìm kiếm

### Mục tiêu và nhiệm vụ
1. Xác định dữ liệu cần lưu giữ trong trạng thái mô phỏng và nút cây; phân định rõ dữ liệu nào thuộc về luật cờ và dữ liệu nào phục vụ quá trình tìm kiếm MCTS.
2. Thống nhất quy ước giá trị của nút ($Q$) và người chơi tương ứng. Giải thích bằng ví dụ cụ thể các trường hợp thắng, thua, hòa cho cả hai phía X và O; đảm bảo tính nhất quán xuyên suốt các giai đoạn.
3. Thiết lập các thông số ban đầu: ngân sách mô phỏng, `seed` và bán kính ứng viên; định nghĩa phương pháp thu thập thống kê trước khi đo đạc.
4. Hoàn thiện nội dung các phần 1–4 trong notebook; giai đoạn này tập trung vào tầng cấu trúc nền tảng, chưa được công bố như một tác tử MCTS hoàn chỉnh.

### Kiểm thử và tiêu chí nghiệm thu
- Đối chiếu các chuỗi nước đi hợp lệ với Game Engine: bàn cờ, người đến lượt, kết quả thắng/thua/hòa phải khớp hoàn toàn.
- Kiểm tra tính độc lập của các bản sao trạng thái: việc mở rộng nút con không làm biến đổi nút cha; nút mới được tạo có giá trị thống kê ban đầu chính xác ($N=0, Q=0$).
- Đảm bảo trạng thái terminal không tiếp tục mở rộng nhánh con hay kích hoạt rollout; mảng bàn cờ gốc không bị thay đổi.

> **Mốc hoàn thành (Milestone):** Tầng biểu diễn trạng thái và cấu trúc cây có bài test độc lập vượt qua 100%; quy ước giá trị được giải thích minh bạch; chưa tuyên bố tác tử có khả năng thi đấu.

### Hướng dẫn thiết kế ca kiểm thử
1. Tạo fixture kiểm thử từ các chuỗi nước đi hợp lệ của engine, lưu trữ trạng thái bàn cờ và lượt đi hiện tại làm đầu vào; chuẩn bị các tình huống thắng/thua/hòa đã biết trước kết quả. Ưu tiên các trường hợp thế cờ nhỏ, dễ tính toán kết quả kỳ vọng bằng tay.
2. Mỗi ca kiểm thử chỉ tập trung xác minh một trách nhiệm duy nhất: tính độc lập của trạng thái con, lượt đi chính xác, nút gốc không có nút cha, liên kết cha/con chính xác và chỉ số thống kê khởi tạo đúng chuẩn.
3. Giá trị kỳ vọng phải căn cứ trên luật cờ hoặc phép tính tay; không sử dụng chính hàm mô phỏng đang kiểm thử để sinh kết quả đối chiếu.

**Tiêu chí đạt chuẩn trước khi yêu cầu Review:**
- [ ] Trạng thái mô phỏng khớp hoàn toàn với Game Engine ở từng bước trên toàn bộ fixture cho cả hai phía X và O.
- [ ] Việc mở rộng hoặc thay đổi trạng thái con không làm biến đổi trạng thái cha và mảng bàn cờ đầu vào.
- [ ] Trạng thái terminal không bị mở rộng thêm và các trường thống kê của nút được khởi tạo đúng quy ước.

**Bằng chứng bắt buộc trong PR:** Lệnh kiểm thử đã chạy, số lượng test đạt/thất bại, danh sách fixture đã thử nghiệm và các hạn chế kỹ thuật đã biết.

### Quy trình nhánh, commit và đẩy mã nguồn
- **Tên nhánh:** `feature/mcts-foundation` *(tạo từ main sau khi nền tảng chung sẵn sàng)*.

<details>
<summary>Lệnh thao tác Git (PowerShell)</summary>

```powershell
git status
git switch main
git pull --ff-only origin main
git switch -c feature/mcts-foundation

# Thực hiện code và kiểm thử trước khi commit
.\.venv\Scripts\python.exe -m pytest tests -v

git add -- agents/mcts_agent.py notebooks/03_mcts.ipynb tests/test_mcts_agent.py docs/Phi-Hung.md
git diff --cached --stat
git diff --cached
git commit -m "feat(mcts): xây dựng trạng thái mô phỏng và cấu trúc cây"
git push -u origin feature/mcts-foundation
```

</details>

### Tiêu đề và mô tả PR Giai đoạn 1
- **Tiêu đề PR:** `feat(mcts): xây dựng nền tảng trạng thái và cây tìm kiếm`
- **Mô tả nội dung mẫu:**
  > Bổ sung mô hình biểu diễn trạng thái mô phỏng và cấu trúc nút cây phục vụ thuật toán MCTS. Notebook trình bày quy ước giá trị, các tham số cấu hình và đối chiếu trạng thái với Game Engine. PR tập trung vào tầng nền tảng trạng thái và cây; thuật toán tìm kiếm UCT hoàn chỉnh sẽ được triển khai ở giai đoạn tiếp theo.

---

## 8. Giai đoạn 2 — Hoàn thiện tìm kiếm UCT và Agent

### Mục tiêu và nhiệm vụ
1. Cài đặt hoàn chỉnh chu trình 4 bước của MCTS (Selection, Expansion, Simulation, Backpropagation); phân định rõ ràng điều kiện dừng của từng bước. Theo dõi chính xác số lượt mô phỏng đã chạy thực tế.
2. Xây dựng công thức UCT cân bằng tối ưu giữa việc khám phá (exploration) và khai thác (exploitation); đảm bảo phía đối thủ chọn nhánh hợp lý, không để đối thủ luôn chọn nhánh có lợi cho người chơi ở gốc.
3. Cài đặt quá trình Rollout luân phiên lượt đi giữa hai bên và kết thúc đúng luật cờ. Nếu áp dụng cơ chế cắt ngắn rollout (depth cutoff), cần mô tả riêng hàm đánh giá tại điểm cắt, không xem đây là một ván hòa thực sự.
4. Tích hợp quyết định chọn nước đi cuối cùng vào `BaseAgent`. Thống nhất tiêu chí chọn nước đi (theo số lượt thăm cao nhất), quy tắc phá hòa và xử lý các trường hợp ngân sách không hợp lệ.
5. Hoàn thiện các phần 5–6 trong notebook; các bước tìm kiếm trình bày trong notebook phải là dữ liệu quan sát trực tiếp từ module, không duy trì bản cài đặt thuật toán thứ hai.

### Kiểm thử và tiêu chí nghiệm thu
- Kiểm tra việc cập nhật số lượt thăm và giá trị tích lũy trên ví dụ cây nhỏ có kết quả terminal biết trước; kiểm tra tính đúng đắn của góc nhìn giá trị tại tầng của đối thủ.
- Ở chế độ `fixed`: kiểm tra số lượt mô phỏng thực hiện khớp chính xác với ngân sách quy định và khả năng tái lập kết quả khi giữ nguyên `seed`. Ở chế độ `time`: kiểm tra cơ chế dừng hợp tác theo deadline, ghi nhận số mô phỏng thực chạy.
- Kiểm tra các trường hợp đặc biệt: bàn cờ chỉ còn một ô hợp lệ, nhiều nước đi đồng điểm số, tính hợp lệ của đầu ra và tính bất biến của bàn cờ gốc.
- Các thế cờ chiến thuật được dùng để đo lường xác suất chọn nước qua nhiều seed ngẫu nhiên; không ép buộc MCTS ngẫu nhiên phải luôn thắng ở mức ngân sách mô phỏng nhỏ.

> **Mốc hoàn thành (Milestone):** Tác tử thi đấu trọn vẹn ván cờ, các chỉ số thống kê đáng tin cậy, bộ test hợp đồng API và test từng thành phần đều vượt qua thành công.

### Hướng dẫn thiết kế ca kiểm thử
1. Xây dựng cây tìm kiếm nhỏ với số lượt thăm và giá trị tích lũy biết trước để kiểm tra công thức UCT, cách xử lý nút chưa thăm và hành vi ở tầng đối thủ; tính toán giá trị kỳ vọng trước khi chạy test.
2. Kiểm tra hàm lan truyền ngược (backpropagation) bằng kết quả terminal cố định để phát hiện lỗi sai dấu reward. Kiểm thử hàm rollout trên các vị trí chỉ còn một nước đi và các vị trí kết thúc rõ ràng.
3. Chạy với cùng `seed` và ngân sách mô phỏng cố định để kiểm tra tính tái lập; không dùng chế độ `time` cho các bài test đòi hỏi quyết định đầu ra phải giống hệt nhau.
4. Tách biệt bộ test kiểm tra hợp đồng API khỏi các bài đo đánh giá chiến thuật đa seed. Không đặt điều kiện phải thắng tuyệt đối mọi ván vào unit test.

**Tiêu chí đạt chuẩn trước khi yêu cầu Review:**
- [ ] Tác tử luôn trả về nước đi hợp lệ trên bộ fixture; bàn cờ gốc không bị chỉnh sửa; số lượt thăm và reward cập nhật khớp kỳ vọng tính trước.
- [ ] Ở chế độ `fixed`, hoàn thành đủ số mô phỏng đã cấu hình khi runner không ngắt vì lỗi hoặc watchdog 30 giây. Nếu watchdog ngắt, ghi ván `timeout`, không báo hoàn thành ngân sách. Ở chế độ `time`, ghi số mô phỏng hoàn thành trước mốc dừng; không yêu cầu đủ ngân sách fixed. Thống kê được reset mỗi lượt.
- [ ] Giữ nguyên seed và cấu hình cố định mang lại kết quả tái lập 100%; trạng thái terminal được xử lý chính xác trên cây.
- [ ] Các hạn chế chiến thuật ở ngân sách nhỏ được ghi nhận bằng số liệu thực tế; không để test ngẫu nhiên bị fail chỉ vì chưa chọn được nước đi tối ưu nhất.

**Bằng chứng bắt buộc trong PR:** Lệnh kiểm thử đã chạy, số lượng test đạt/thất bại, danh sách fixture đã thử nghiệm và các hạn chế kỹ thuật đã biết.

**Phân biệt các kiểu dừng:** Watchdog ngắt toàn bộ lượt vì quá giờ là `timeout`. Dừng theo deadline ở chế độ `time` là điều kiện kết thúc tìm kiếm bình thường nếu trả nước trước hạn cứng. Cắt ngắn một rollout theo độ sâu là lựa chọn thuật toán riêng, phải có cách đánh giá đã công bố, không tự ghi là hòa. Mô phỏng bị bỏ dở do hết thời gian không được đếm như mô phỏng hoàn thành hoặc cập nhật reward giả.

### Quy trình nhánh, commit và đẩy mã nguồn
- **Tên nhánh:** `feature/mcts-uct` *(chỉ tạo sau khi PR Giai đoạn 1 đã được merge vào main)*.

<details>
<summary>Lệnh thao tác Git (PowerShell)</summary>

```powershell
git status
git switch main
git pull --ff-only origin main
git switch -c feature/mcts-uct

# Thực hiện code và kiểm thử trước khi commit
.\.venv\Scripts\python.exe -m pytest tests -v

git add -- agents/mcts_agent.py notebooks/03_mcts.ipynb tests/test_mcts_agent.py docs/Phi-Hung.md
git diff --cached --stat
git diff --cached
git commit -m "feat(mcts): hoàn thiện tìm kiếm UCT và rollout"
git push -u origin feature/mcts-uct
```

</details>

### Tiêu đề và mô tả PR Giai đoạn 2
- **Tiêu đề PR:** `feat(mcts): hoàn thiện agent MCTS với UCT và rollout`
- **Mô tả nội dung mẫu:**
  > Hoàn thiện chu trình tìm kiếm MCTS bao gồm lựa chọn (Selection), mở rộng (Expansion), mô phỏng (Simulation) và cập nhật ngược (Backpropagation). Tác tử tuân thủ hợp đồng API chung, quản lý ngân sách tìm kiếm và ghi nhận thống kê thực tế. Notebook minh họa cấu trúc cây, các tình huống xử lý chiến thuật và các giới hạn quan sát được.

---

## 9. Giai đoạn 3 — Khảo sát ngân sách và bàn giao kết quả

### Mục tiêu và nhiệm vụ
1. Cố định bộ vị trí bàn cờ, tập hợp `seed` và bán kính ứng viên trước khi khảo sát. Bắt buộc đánh giá riêng hai chế độ: `fixed` với 100 / 500 / 1.000 mô phỏng và `time` với giới hạn 2,0 giây/nước trên cùng điều kiện phần cứng. Ở `time`, không dùng 1.000 mô phỏng làm điều kiện dừng bổ sung.
2. Ghi nhận chi tiết kết quả theo từng vị trí và tổng hợp các chỉ số: thời gian thực thi, số mô phỏng thực tế, số nút trên cây và độ biến động kết quả qua các seed.
3. Tổ chức thi đấu thử nghiệm với `RandomAgent` riêng cho cả `fixed` và `time`, cân bằng đi trước/sau và công khai số ván, seed, cấu hình. Đây là nghiệm thu phát triển; giải đấu cuối tối thiểu 100 ván/cặp do nhóm trưởng điều phối. Nếu khảo sát rollout có heuristic, tách thành biến thể có tên riêng và giải thích phần logic dùng chung.
4. Hoàn thiện nội dung các phần 7–10 trong notebook; lựa chọn cấu hình đề xuất bàn giao và ghi nhận cụ thể các phần chưa được đánh giá.

### Kiểm thử và tiêu chí nghiệm thu
- Chạy lại toàn bộ bộ test và notebook từ một kernel mới; đảm bảo kết quả không bị phụ thuộc vào thứ tự thực thi cell.
- Kiểm tra tính đầy đủ của số liệu: tổng số ván đấu, kết quả $W/D/L$, danh sách seed và cấu hình từng lượt; không loại bỏ các ván thua hay ván lỗi khỏi báo cáo mà không có lý do.
- Trong khảo sát `fixed`, chỉ thay số mô phỏng, giữ các yếu tố khác cố định. Ở `time`, kiểm tra hạn cứng 2,0 giây, ghi số mô phỏng thực hoàn thành và mọi lần timeout; không đòi số mô phỏng bằng nhau giữa các lần chạy.

> **Mốc hoàn thành (Milestone):** Notebook cung cấp đầy đủ số liệu thực tế, phân tích các hạn chế kỹ thuật, công bố cấu hình bàn giao và hướng dẫn chạy lại thực nghiệm; không đánh đồng kết quả thử nghiệm cục bộ với giải đấu chính thức.

### Hướng dẫn thiết kế ca kiểm thử
1. Trong `tests/test_mcts_agent.py`, kiểm tra định dạng log của tác tử và gọi hàm tính chỉ số chung (`evaluation/metrics.py`) trên tập dữ liệu nhỏ đã biết trước kết quả. Module tổng hợp số liệu do nhóm trưởng cài đặt; không tự viết lại logic tính toán. Không đưa các giải đấu lớn vào kiểm thử pytest.
2. Kiểm tra tính toàn vẹn: mỗi cấu hình phải có đúng vị trí, seed, số ván và chế độ. Với `fixed`, đối chiếu số mô phỏng yêu cầu và số hoàn thành; với `time`, đối chiếu giới hạn 2,0 giây và trạng thái timeout, chỉ ghi số mô phỏng như kết quả đo.
3. Chạy notebook từ kernel mới để kiểm tra tính ổn định của luồng thực nghiệm; lưu trữ các trường hợp thất bại và phân tích nguyên nhân thay vì loại bỏ khỏi bảng kết quả.

**Tiêu chí đạt chuẩn trước khi yêu cầu Review:**
- [ ] Tổng số ván $W/D/L$ khớp chính xác với số ván hợp lệ, điểm số được tính đúng công thức và các lỗi thực thi được báo cáo riêng biệt.
- [ ] Có bảng kết quả riêng cho `fixed` và `time`. Bảng fixed chỉ thay số mô phỏng; bảng time ghi cấu hình 2,0 giây, số mô phỏng hoàn thành, độ trễ và số lần timeout; đủ dữ liệu đối chiếu seed/config.
- [ ] Không đặt ngưỡng cứng cho số mili giây hoặc tỷ lệ thắng cố định khi chưa có baseline ổn định; mọi kết luận phải căn cứ trên số liệu thực tế.

**Bằng chứng bắt buộc trong PR:** Lệnh kiểm thử đã chạy, số lượng test đạt/thất bại, danh sách fixture đã thử nghiệm và các hạn chế kỹ thuật đã biết.

### Quy trình nhánh, commit và đẩy mã nguồn
- **Tên nhánh:** `feature/mcts-evaluation` *(chỉ tạo sau khi PR Giai đoạn 2 đã được merge vào main)*.

<details>
<summary>Lệnh thao tác Git (PowerShell)</summary>

```powershell
git status
git switch main
git pull --ff-only origin main
git switch -c feature/mcts-evaluation

# Thực hiện code và kiểm thử trước khi commit
.\.venv\Scripts\python.exe -m pytest tests -v

git add -- agents/mcts_agent.py notebooks/03_mcts.ipynb tests/test_mcts_agent.py docs/Phi-Hung.md
git diff --cached --stat
git diff --cached
git commit -m "test(mcts): bổ sung khảo sát ngân sách và phân tích MCTS"
git push -u origin feature/mcts-evaluation
```

</details>

### Tiêu đề và mô tả PR Giai đoạn 3
- **Tiêu đề PR:** `test(mcts): bổ sung thí nghiệm ngân sách và đánh giá MCTS`
- **Mô tả nội dung mẫu:**
  > Bổ sung các bài khảo sát về số lượt mô phỏng, thi đấu thử nghiệm và phân tích các trường hợp lỗi của MCTS. Các cấu hình sử dụng chung bộ vị trí và danh sách seed để xác định chính xác tác động của ngân sách tính toán. Notebook ghi nhận đầy đủ số liệu thực tế và phạm vi kết luận.

---

## 10. Quy trình tạo PR và xử lý phản hồi Review

### Thiết lập thông tin bắt buộc trên GitHub
- **Reviewer:** @10tothepowerof6 (Nhóm trưởng). Chọn trực tiếp trong mục Reviewers trên giao diện PR khi đã sẵn sàng review; chỉ nhắc tên trong mô tả sẽ không tạo thông báo chính thức.
- **Assignee:** @phi10hung. Người phụ trách có trách nhiệm theo dõi và hoàn thiện PR; reviewer chịu trách nhiệm kiểm tra và đánh giá thay đổi.
- **Milestone:**
  - Task 1–2 / PR tương ứng: [Báo cáo tiến độ nửa dự án](https://github.com/10tothepowerof6/AdvancedAI-Repo/milestone/2), hạn 09/11/2026.
  - Task 3: [milestone tổng](https://github.com/10tothepowerof6/AdvancedAI-Repo/milestone/1).
- **Liên kết Issue:**
  - Giai đoạn 1: [Task 1 (MCTS)](https://github.com/10tothepowerof6/AdvancedAI-Repo/issues/1)
  - Giai đoạn 2: [Task 2 (MCTS)](https://github.com/10tothepowerof6/AdvancedAI-Repo/issues/2)
  - Giai đoạn 3: [Task 3 (MCTS)](https://github.com/10tothepowerof6/AdvancedAI-Repo/issues/3)
  *(Không tạo issue cho Giai đoạn 0).*
- **Đóng issue:** Trong phần mô tả của PR hoàn thành task, sử dụng cú pháp `Closes #<số_issue>` (ví dụ: `Closes #1`). PR nháp hoặc PR chỉ hoàn thành một phần thì chỉ gắn liên kết tham chiếu thông thường, không dùng từ khóa `Closes`.
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
4. Nếu phát sinh xung đột (conflict): Mở từng file bị xung đột, kiểm tra kỹ lưỡng nội dung của cả hai bên, tiến hành giải quyết, chạy lại bộ test và hoàn tất commit hợp nhất: `chore(mcts): đồng bộ thay đổi từ main`. Tuyệt đối không tự ý xóa bỏ phần code của thành viên khác.

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
