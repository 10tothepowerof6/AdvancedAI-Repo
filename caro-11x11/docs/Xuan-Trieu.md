# Đỗ Xuân Triệu — Kế hoạch phát triển Neural Network và Tích hợp hệ thống

## 1. Thông tin chung và phân công trách nhiệm

- **Người phụ trách:** @10tothepowerof6 (Nhóm trưởng).
- **Phạm vi chuyên môn:** Xây dựng mô hình học sâu (CNN policy), chuẩn bị dữ liệu, huấn luyện, tích hợp hệ thống, tổ chức giải đấu và hoàn thiện sản phẩm báo cáo/demo.
- **Theo dõi tiến độ:** Các giai đoạn tiếp theo được quản lý qua ba issue GitHub thuộc mảng Neural Network: Task 1, Task 2 và Task 3. Nền tảng Game Engine và RandomAgent đã hoàn thành từ trước được xem là dữ liệu đầu vào, không tạo task mới hay issue cho Giai đoạn 0.
- **Quy định vị trí test:**
  - Test nền tảng: `tests/test_engine.py`, `tests/test_agents.py`, `tests/test_game_utils.py`.
  - Test dữ liệu, huấn luyện và NeuralAgent: `tests/test_neural_agent.py`.
  - Toàn bộ test đặt trực tiếp trong thư mục `tests/`, không chia thư mục con theo tên cá nhân.

---

## 2. Quy chuẩn ngân sách và kiểm soát thời gian (Toàn dự án)

Bảng cấu hình ngân sách tìm kiếm áp dụng chung cho tất cả các tác tử trong dự án:

| Chế độ | Cấu hình khởi đầu | Yêu cầu thực hiện |
| :--- | :--- | :--- |
| **fixed** | MCTS: 1.000 mô phỏng/nước<br>Minimax: độ sâu 2 ply | Hoàn thành đúng ngân sách quy định; khảo sát mở rộng ở các mức 100 / 500 / 1.000 mô phỏng và độ sâu 1 / 2 / 3 ply |
| **time** | Giới hạn tối đa 2,0 giây/nước cho tất cả agent | MCTS tận dụng thời gian cho đến deadline; Minimax áp dụng tìm kiếm sâu dần (Iterative Deepening) và bảo lưu kết quả của độ sâu đã hoàn tất |

> **Lưu ý:** Đây là cấu hình khởi đầu phục vụ triển khai, chưa phải thông số tối ưu qua benchmark. Kết quả ở hai chế độ phải được báo cáo độc lập; chế độ `fixed` không đồng nghĩa với việc tiêu tốn thời gian ngang nhau. Ở chế độ `time`, số mô phỏng hoặc độ sâu của chế độ `fixed` không được dùng làm điều kiện dừng bổ sung. Các tác tử có tốc độ tính toán tức thời (Random, Rule-based, Neural) được phép trả về ngay khi có quyết định, không cần chờ hết 2 giây.

### Giao diện lập trình (API) hiện có
- Lớp `SearchBudget` (`config/search_budget.py`) quản lý 4 thuộc tính: `mode`, `simulations`, `depth`, `time_limit_s`.
- Lớp `BaseAgent` tiếp nhận đối tượng `budget` qua tham số khởi tạo (mặc định là `fixed`).
- Giữ nguyên chữ ký của hai phương thức nền tảng: `get_action(board, player)` và `choose_action(board, player)`.
- Thuật toán truy xuất cấu hình qua `self.budget`, `self.deadline` và phương thức kiểm tra `self.time_exhausted()`.
- Biến `self.deadline` được thiết lập ở mỗi nước đi dựa trên `time.perf_counter()` khi ở chế độ `time`, và mang giá trị `None` ở chế độ `fixed`.
- Khi khởi tạo lớp con kế thừa từ `BaseAgent`, cần truyền `budget` xuống lớp cha; tuyệt đối không thay đổi chữ ký phương thức dùng chung.

### Cơ chế dừng hợp tác (Cooperative Stopping)
- Tác tử cần chủ động dừng tìm kiếm trước deadline để hoàn tất hàm và truyền dữ liệu. `time_exhausted()` chỉ trả về đúng khi đã tới deadline; hàm không tự trừ biên an toàn. Mốc dừng sớm khởi đầu đề xuất là 1,9 giây trên hạn 2,0 giây (chừa khoảng 100 ms). Biên này do thuật toán quản lý, cần công khai và kiểm chứng trên máy chạy; không bảo đảm tuyệt đối tránh timeout. Runner vẫn nhận kết quả trong tối đa 2,0 giây, không cộng biên này vào hạn cứng.
- MCTS: Kiểm tra thời gian định kỳ ngay trong các vòng lặp rollout.
- Minimax: Duy trì kết quả của vòng tìm kiếm sâu dần (Iterative Deepening) đã hoàn tất gần nhất.
- Luôn chuẩn bị sẵn một nước đi hợp lệ dự phòng đề phòng trường hợp chưa kịp hoàn thành độ sâu đầu tiên; bộ chạy trận đấu (runner) không tự ý đi thay bằng nước ngẫu nhiên.

### Cơ chế cưỡng chế dừng (Hard Timeout Enforcement)
- Được điều phối bởi `evaluation/match.py` thông qua hàm `play_match` và lớp `AgentWorker`.
- Mỗi tác tử được thực thi trên một tiến trình riêng biệt xuyên suốt cả ván đấu nhằm duy trì trạng thái của bộ sinh số ngẫu nhiên (RNG).
- Khi vượt quá thời gian quy định: Tiến trình bị ngắt cưỡng bức, tác tử vi phạm bị xử thua ván đấu với nguyên nhân ghi nhận là `timeout`.
- Khi phát sinh ngoại lệ (Exception) hoặc trả về nước đi sai luật: Tác tử bị xử thua ngay lập tức kèm nguyên nhân chi tiết; hệ thống không tự ý chuyển thành ván hòa hay âm thầm bỏ qua ván đấu.

### Cách tính toán độ trễ (Latency)
- Giới hạn thời gian cứng được tính từ thời điểm runner gửi yêu cầu cho đến khi nhận được phản hồi, bao gồm chi phí giao tiếp liên tiến trình (IPC) và bước kiểm tra hợp lệ.
- Phân biệt hai loại chỉ số thời gian:
  - `thinking_time` (trong trường `stats`): Đo thuần túy từ lúc bắt đầu `choose_action` đến khi kết thúc `get_action`.
  - `elapsed_s` (do runner đo lường): Bao gồm `thinking_time`, chi phí IPC và thời gian kiểm tra nước đi. Dùng `elapsed_s` làm thước đo độ trễ chính thức trong thi đấu.
- Tuyệt đối không tự ý cộng thêm dung sai thời gian ngoài quy định.
- Quá trình khởi tạo hoặc nạp trọng số mô hình (load model) có hạn mức riêng tối đa 30 giây (tách biệt khỏi thời gian suy nghĩ mỗi nước). Nếu xảy ra lỗi khởi tạo, ngoại lệ sẽ được ném ra ngoài và không tính ván đấu vào kết quả thi đấu.

### Giám sát chống treo (Watchdog Timer)
- Ở chế độ `fixed`: Bộ giám sát mặc định thiết lập mức 30 giây/nước nhằm chống treo tiến trình (tùy biến qua tham số `safety_timeout_s`). Khi kích hoạt, hệ thống ghi nhận kết quả `timeout`, không ghi nhận đã hoàn thành ngân sách.
- Cần công khai thông số watchdog trong các báo cáo thực nghiệm và thiết lập ngưỡng đủ lớn để các bài đo cắt tỉa hoàn thành trọn vẹn.
- Ở chế độ `time`: Mốc 2,0 giây được sử dụng trực tiếp thay thế cho watchdog.

### Yêu cầu kỹ thuật khi thực thi đa tiến trình
- Hàm khởi tạo tác tử (factory) phải nằm trong module `.py` có thể import được và tạo mới đối tượng agent ở mỗi ván đấu.
- Trên môi trường Windows, mọi script thực thi bắt buộc phải có khối bảo vệ `if __name__ == "__main__":`.
- Trong Jupyter Notebook, cần import factory từ module bên ngoài thay vì định nghĩa lớp/hàm trực tiếp trong cell (tránh lỗi khi hệ thống spawn tiến trình con).
- Nếu sử dụng GPU, phải đảm bảo hoàn tất toàn bộ tiến trình tính toán cần thiết trước khi trả về kết quả.

### Cấu trúc dữ liệu trả về từ Match Runner
- Hàm `play_match` trả về đối tượng `MatchRecord` kèm nhật ký chi tiết cho từng lượt đi: `player`, `ply`, `mode`, `limit_s`, `budget`, `elapsed_s`, `status`.
- Nước đi hợp lệ lưu kèm `action` và `stats`; trường hợp lỗi lưu kèm `error`.
- Đây là bộ điều khiển từng ván đơn lẻ (single-match runner), chưa bao gồm cơ chế thi đấu vòng tròn (round-robin) hay tự động xuất file CSV/JSON. Lịch thi đấu, tệp kê khai seed/config, ghi log và tổng hợp báo cáo chung sẽ do nhóm trưởng hoàn thiện ở giai đoạn tích hợp.

---

## 3. Chỉ số đánh giá và module tổng hợp dùng chung

Nhóm trưởng trực tiếp quản lý hai file: `evaluation/metrics.py` và `tests/test_metrics.py`. Các thành viên gọi hàm trực tiếp từ module dùng chung này vào notebook hoặc test riêng; không tự ý sao chép hay viết lại logic tính toán.

Mọi đề xuất chỉnh sửa module chung cần được thảo luận hoặc tạo PR riêng để đảm bảo tính đồng bộ trên toàn dự án.

### Định dạng bản ghi trận đấu (`MatchRecord`)
Đối tượng `MatchRecord` bao gồm các trường: `agent_x`, `agent_o`, `winner`, `reason`. Hai bên thi đấu bắt buộc phải có định danh cấu hình khác nhau, kể cả khi cùng thuộc một loại tác tử.

Hàm `summarize_results(records, agent)` tự động chuyển đổi góc nhìn theo bên X hoặc O của tác tử đang được tổng hợp, không cố định kết quả theo người chơi X.

### Công thức tính điểm số (Score)
- **Tổng số ván đấu:** $N = W + D + L$ (Thắng + Hòa + Thua)
- **Điểm số tiêu chuẩn:**
  $$\text{score} = \frac{W + 0.5 \times D}{N}$$
- *Ví dụ:* Thành tích 6 thắng, 2 hòa, 2 thua đạt điểm số $0.7$. Khi chưa có ván đấu nào, điểm số mang giá trị `None`.

### Quy tắc xử lý các ván xử thua
- Các ván kết thúc do `timeout`, `agent_error` hoặc `illegal_action` vẫn được tính vào $W/D/L$ tương tự một ván thua thông thường.
- Mục `forfeits` thống kê riêng số lượng lỗi do chính tác tử đó trực tiếp gây ra.
- Lỗi khởi tạo không sinh ra `MatchRecord`, phía gọi hàm cần báo cáo riêng như một sự cố hạ tầng.
- Lỗi định dạng đầu ra bị `BaseAgent` chặn bằng `ValueError` được runner ghi nhận là `agent_error` kèm thông báo gốc; `illegal_action` là lớp kiểm tra tính hợp lệ bổ sung từ engine.

### Thống kê độ trễ (`summarize_latencies`)
- Tiếp nhận danh sách thời gian thực thi của các nước đi hợp lệ (đơn vị: giây).
- Trả về: `count` (số lượng mẫu), `mean_s` (thời gian trung bình), `p95_s` (phân vị 95 sử dụng phương pháp nội suy tuyến tính). Mẫu rỗng trả về `None`.
- Hệ thống từ chối các giá trị không hợp lệ (số âm, NaN, vô cực).
- Số lần và thời gian của các nước lỗi/timeout được báo cáo riêng biệt, không gộp chung vào độ trễ của các nước đi hợp lệ.
- Khoảng tin cậy và mức độ sử dụng tài nguyên phần cứng sẽ được bổ sung ở giai đoạn tích hợp cuối cùng.

### Quy định kiểm thử ở Giai đoạn 3
- Bài test ở giai đoạn 3 của các thành viên tập trung kiểm tra cấu trúc log/cấu hình do tác tử sinh ra và gọi hàm `metrics` chung trên tập dữ liệu mẫu nhỏ đã biết trước kết quả.
- Không phân công lại việc viết hàm tính toán thống kê cho từng người. Nếu phát hiện lỗi trong module chung, cần lập issue và đề xuất cập nhật bộ test chung.

---

## 4. Quy định và phạm vi kiểm thử tự động

- Toàn bộ file kiểm thử đặt trực tiếp trong thư mục `tests/`, đặt tên theo chức năng hoặc thuật toán (`test_<tên_module>.py`). Không chia thư mục con theo tên thành viên.
- Chạy một file test cụ thể: pytest chỉ thu thập các ca kiểm thử trong file đó.
- Chạy toàn bộ thư mục `tests`: pytest sẽ thực thi tất cả các bài kiểm thử của dự án.
- Không tạo bài test có phụ thuộc (import) vào tác tử của thành viên khác khi tác tử đó chưa hoàn thiện.
- Các hàm bổ trợ (helper/factory) riêng đặt tại module tương ứng; helper dùng chung cần được thống nhất trước khi đưa vào phạm vi chia sẻ.
- Các file test thuật toán ban đầu là file rỗng. Khi chạy riêng, pytest sẽ trả về mã thoát 5 (không tìm thấy test) – điều này chưa đồng nghĩa với việc kiểm thử đã đạt. Cần bổ sung test thực tế theo từng mốc tiến độ.
- Khi xuất hiện lỗi ở bộ test chung: Cần xác định rõ lỗi đã tồn tại sẵn trên nhánh `main` hay mới phát sinh từ PR hiện tại.
- Tuyệt đối không chỉnh sửa test của module khác chỉ để làm cho PR của mình vượt qua; không tùy tiện bỏ qua (skip) lỗi.
- Lỗi có sẵn từ trước do nhóm trưởng phân loại và xử lý; PR làm phát sinh lỗi tích hợp mới sẽ không đủ điều kiện để merge.
- Không đưa các tiêu chuẩn không ổn định (như giới hạn thời gian chạy cố định hay điều kiện phải thắng tuyệt đối mọi ván) vào unit test.
- Kiểm thử deadline hợp tác: Áp dụng cơ chế giả lập đồng hồ (mock clock) để đảm bảo kết quả ổn định.
- Kiểm thử cưỡng chế timeout: Cho tác tử chạy thử với độ trễ vượt ngưỡng rõ rệt để xác nhận hệ thống dọn dẹp tiến trình và xử thua chính xác.
- Tái lập kết quả MCTS: Chạy ở chế độ `fixed` kết hợp truyền `seed` cố định; không ép buộc hai lần chạy ở chế độ `time` phải chọn nước đi giống hệt nhau.

**Lệnh chạy kiểm thử phục vụ phát triển Neural Network và trước khi merge:**
```powershell
# Kiểm thử riêng module Neural Network
.\.venv\Scripts\python.exe -m pytest tests/test_neural_agent.py -v

# Kiểm thử toàn bộ hệ thống
.\.venv\Scripts\python.exe -m pytest tests -v
```

---

## 5. Giai đoạn 1 — Chuẩn bị dữ liệu và biểu diễn trạng thái cho Neural Network

- **Issue theo dõi:** [Task 1 (Neural Network)](https://github.com/10tothepowerof6/AdvancedAI-Repo/issues/10)
- **Tên nhánh (Branch):** `feature/neural-data`
- **Phạm vi file chính:** `training/generate_data.py`, `training/dataset.py`, `tests/test_neural_agent.py`, `notebooks/04_neural_network.ipynb`

### Mục tiêu và yêu cầu kỹ thuật
- Chuẩn hóa biểu diễn trạng thái bàn cờ theo góc nhìn của người chơi hiện tại, cấu trúc nhãn nước đi và tổ chức dữ liệu theo từng ván đấu.
- Sinh nhãn dữ liệu từ các tác tử chuyên gia đã được nghiệm thu; lưu trữ đầy đủ `seed`, cấu hình, nguồn gốc nhãn và định danh ván đấu.
- Phân chia dữ liệu theo ván đấu (train / val / test) trước khi áp dụng tăng cường dữ liệu đối xứng (data augmentation), nhằm ngăn ngừa triệt để tình trạng rò rỉ dữ liệu giữa các tập.
- Lưu ý: Không gọi phương pháp học bắt chước từ chuyên gia (Imitation Learning) là học tăng cường (Reinforcement Learning).

### Kiểm thử và tiêu chí nghiệm thu
- Kiểm thử bộ mã hóa (encoder) cho cả hai phía người chơi, tính hợp lệ của nhãn, tính nhất quán khi biến đổi trạng thái và hành động.
- Đảm bảo việc phân chia dữ liệu không bị rò rỉ ván đấu hoặc bản sao tăng cường sang tập kiểm định.
- Nghiệm thu khi có tập dữ liệu thực tế kèm tệp kê khai (manifest) và pipeline có thể chạy lại tự động từ đầu.

### Thông tin Commit và PR
- **Tiêu đề commit / PR:** `feat(neural): bổ sung dữ liệu chuyên gia và biểu diễn trạng thái`
- **Nội dung mô tả PR:** Nêu rõ nguồn dữ liệu chuyên gia, số lượng ván và số mẫu thực tế, tỷ lệ phân chia tập dữ liệu, phương pháp khử trùng lặp (deduplication), thiết lập seed và kết quả chạy kiểm thử.

---

## 6. Giai đoạn 2 — Huấn luyện CNN Policy và lưu Checkpoint

- **Issue theo dõi:** [Task 2 (Neural Network)](https://github.com/10tothepowerof6/AdvancedAI-Repo/issues/11)
- **Tên nhánh (Branch):** `feature/neural-training` *(chỉ tạo sau khi Task 1 đã được nghiệm thu và merge vào main)*
- **Phạm vi file chính:** `training/model.py`, `training/train.py`, `tests/test_neural_agent.py`, `models/README.md`, `notebooks/04_neural_network.ipynb`

### Mục tiêu và yêu cầu kỹ thuật
- Huấn luyện mạng CNN Policy trên tập dữ liệu chuyên gia, theo dõi chặt chẽ hàm mất mát (loss) trên tập train và tập validation; lựa chọn checkpoint dựa trên kết quả validation.
- Lưu trữ đầy đủ siêu tham số (hyperparameters), seed, thời gian huấn luyện và thông tin phần cứng sử dụng.
- Tuyệt đối không dùng tập test hoặc các ván đánh giá cuối cùng để điều chỉnh siêu tham số.
- Cung cấp liên kết tải về kèm mã băm kiểm tra (checksum) nếu file trọng số được lưu trữ bên ngoài kho Git.

### Kiểm thử và tiêu chí nghiệm thu
- Kiểm thử đầu ra của mạng nơ-ron tương ứng với 121 vị trí bàn cờ (11×11), giá trị loss là số hữu hạn, trọng số được cập nhật thực tế qua quá trình huấn luyện và đảm bảo tính nhất quán khi lưu/nạp checkpoint.
- Nghiệm thu khi có file checkpoint đã hoàn thành huấn luyện, biểu đồ quá trình học tập (learning curves) và số liệu đánh giá thực tế trên tập dữ liệu held-out.

### Thông tin Commit và PR
- **Tiêu đề commit / PR:** `feat(neural): huấn luyện CNN policy và lưu checkpoint`
- **Nội dung mô tả PR:** Nêu rõ kiến trúc mạng, nguồn dữ liệu sử dụng, cấu hình huấn luyện chi tiết, tiêu chí lựa chọn checkpoint và chi phí tài nguyên tính toán.

---

## 7. Giai đoạn 3 — Tích hợp NeuralAgent, giải đấu và demo đồ án

- **Issue theo dõi:** [Task 3 (Neural Network)](https://github.com/10tothepowerof6/AdvancedAI-Repo/issues/12)
- **Tên nhánh (Branch):** `feature/neural-integration` *(chỉ tạo sau khi Task 2 và các tác tử còn lại đã sẵn sàng trên main)*
- **Phạm vi file chính:** `agents/neural_agent.py`, `evaluation/`, `app/`, `tests/test_neural_agent.py`, notebook Neural Network / giải đấu và tài liệu hướng dẫn tái lập thực nghiệm.

### Mục tiêu và yêu cầu kỹ thuật
- Tích hợp mô hình đã huấn luyện vào `NeuralAgent`, áp dụng mặt nạ lọc nước đi không hợp lệ (legal move mask), thông báo lỗi rõ ràng khi không tìm thấy checkpoint.
- Hoàn thiện hệ thống giải đấu vòng tròn giữa 5 tác tử với quy mô tối thiểu 100 ván cho mỗi cặp đối đầu, đảm bảo đổi bên đi trước/sau và khảo sát trên nhiều seed; lưu trữ cấu hình và log thô đầy đủ.
- Lập báo cáo tổng hợp: Ngân sách tìm kiếm, thống kê $W/D/L$, điểm số ($score$), độ trễ trung bình và phân vị 95 ($mean\_s / p95\_s$), số lượng nút/mô phỏng, chi phí huấn luyện, khoảng tin cậy và tối thiểu 2 bài thực nghiệm đối chứng (ablation study) có ý nghĩa.
- Hoàn thiện ứng dụng demo, báo cáo tổng kết, slide thuyết trình, poster và phân tích các giới hạn còn tồn tại.

### Kiểm thử và tiêu chí nghiệm thu
- Kiểm thử tính chính xác của legal mask, đảm bảo không làm biến đổi bàn cờ gốc, nạp checkpoint thành công, lịch thi đấu cân bằng, số liệu thống kê khớp hoàn toàn với log và ứng dụng demo kết thúc ván đúng luật.
- Nghiệm thu khi toàn bộ kết quả có bằng chứng thực nghiệm đầy đủ kèm tài liệu hướng dẫn tái lập; tuyệt đối không điền số liệu giả định.

### Thông tin Commit và PR
- **Tiêu đề commit / PR:** `feat(integration): tích hợp NeuralAgent và hệ thống đánh giá`
- **Nội dung mô tả PR:** Nêu rõ danh sách các tác tử đã tích hợp, cấu hình giải đấu chi tiết, kết quả thực nghiệm, hướng dẫn vận hành demo và các hạn chế kỹ thuật còn tồn đọng.

---

## 8. Các module nền tảng do nhóm trưởng phụ trách

Nhóm trưởng trực tiếp quản lý các module dùng chung sau:
- `config/search_budget.py`
- `evaluation/match.py`
- `evaluation/metrics.py`
- `tests/test_search_budget.py`
- `tests/test_match.py`
- `tests/test_metrics.py`

Đây là nền tảng cốt lõi của toàn dự án, cần duy trì tính tương thích ngược và đảm bảo toàn bộ bộ test nền tảng vượt qua thành công trước khi tích hợp các tính năng giải đấu và demo.

---

## 9. Quy trình tạo Pull Request và quy chuẩn Git

- Mỗi giai đoạn được phát triển trên một nhánh mới tạo từ `main` (sau khi đã cập nhật các thay đổi mới nhất từ PR trước đó).
- Chỉ đưa vào vùng chờ (`git add`) các file liên quan đến nhiệm vụ, kiểm tra kỹ sai khác (`git diff`), viết commit tóm tắt bằng tiếng Việt rồi push nhánh lên remote `origin`.
- Tạo PR trên GitHub với nhánh đích (`base`) là `main` và nhánh so sánh (`compare`) là nhánh tính năng tương ứng.

### Thiết lập thông tin trên GitHub
- **Assignee:** @10tothepowerof6
- **Reviewer:** @phi10hung (tác giả PR không tự gán quyền review cho chính mình).
- **Milestone:**
  - Task 1–2 / PR tương ứng: [Báo cáo tiến độ nửa dự án](https://github.com/10tothepowerof6/AdvancedAI-Repo/milestone/2), hạn 09/11/2026.
  - Task 3: [milestone tổng](https://github.com/10tothepowerof6/AdvancedAI-Repo/milestone/1).
- **Liên kết Issue:** Gắn đúng mã Task 1–3 thuộc mảng Neural Network. Chỉ dùng từ khóa `Closes` kèm số issue GitHub (`#10`, `#11`, `#12`) khi đã hoàn thành toàn bộ yêu cầu của task; không dùng cú pháp "Task 1/2/3".
- **Nội dung mô tả PR:** Bắt buộc phải có phạm vi triển khai, lệnh kiểm thử đã chạy, số liệu thực nghiệm, tài liệu tham khảo và các hạn chế kỹ thuật.
- **Lưu ý cấu hình GitHub:** Các trường thông tin (Reviewer, Assignee, Milestone) phải được chọn trực tiếp trong thanh bên (sidebar) của giao diện GitHub; việc chỉ đề cập tên trong phần mô tả sẽ không tự động kích hoạt gán người hay mốc. Nhánh `main` đã thiết lập quy tắc bảo vệ bắt buộc phải có ít nhất một phê duyệt (approval) từ thành viên khác trước khi merge (áp dụng cho cả quyền quản trị viên).
