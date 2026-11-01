# Tấn Phát — Rule-based

## 1. Mục tiêu và tổ chức sản phẩm

Mục tiêu chính là hoàn thiện tác tử dựa trên tập luật chiến thuật và nhận diện mẫu hình (**Rule-based**) cho cờ Caro 11×11, sẵn sàng tích hợp vào giải đấu chung và ứng dụng demo của dự án.

Mỗi giai đoạn thực hiện yêu cầu đầy đủ 4 thành phần cấu thành:
- **Module mã nguồn:** Cài đặt thuật toán trong `agents/rule_based_agent.py`.
- **Kiểm thử tự động:** Xây dựng bộ test riêng trong `tests/test_rule_based_agent.py`.
- **Thực nghiệm và trực quan hóa:** Trình bày quá trình đo lường, phân tích trong `notebooks/01_rule_based.ipynb`.
- **Tài liệu theo dõi:** Cập nhật thiết kế, tiến độ và các hạn chế kỹ thuật trong `docs/Tan-Phat.md`.

> **Lưu ý cốt lõi:** Notebook phải trực tiếp import và sử dụng tác tử từ module `agents/rule_based_agent.py`. Tuyệt đối không duy trì hai phiên bản thuật toán song song giữa notebook và mã nguồn chính. Tài liệu này đóng vai trò định hướng kiến trúc, yêu cầu kỹ thuật và tiêu chí nghiệm thu; không cung cấp mã mẫu hay mã giả.

---

## 2. Phạm vi file và quy ước chung

### Phân công trách nhiệm theo file
| File thao tác | Trách nhiệm chính |
| :--- | :--- |
| `agents/rule_based_agent.py` | Lớp `RuleBasedAgent` và các hàm nhận diện luật, pattern nội bộ |
| `tests/test_rule_based_agent.py` | Bộ kiểm thử tự động riêng cho thuật toán |
| `notebooks/01_rule_based.ipynb` | Kịch bản thí nghiệm, đo đạc số liệu và phân tích kết quả |
| `docs/Tan-Phat.md` | Ghi nhận quyết định thiết kế, tiến độ thực hiện và giới hạn kỹ thuật |

### Quy ước phát triển và làm việc với Git
- **Thư mục làm việc:** Mọi thao tác lệnh và đường dẫn tương đối đều tính từ thư mục gốc `caro-11x11`.
- **Phạm vi chỉnh sửa:** Các thành phần nền tảng (`core/`, `config/`, `BaseAgent`, `RandomAgent`, `agents/__init__.py`, `requirements.txt`, các bài test chung, `README.md` gốc) thuộc quyền quản lý tích hợp của nhóm trưởng. Mọi đề xuất thay đổi cần thông qua thảo luận hoặc PR riêng. Tác tử cá nhân có thể được import trực tiếp từ file module mà không cần sửa `agents/__init__.py`.
- **Ngôn ngữ sử dụng:** Đầu ra console, chú thích code (comment), giải thích trong notebook, thông điệp commit và nội dung PR đều sử dụng tiếng Việt chuẩn. Tên biến, hàm, lớp tuân thủ quy chuẩn kỹ thuật tiếng Anh (PEP 8).
- **Quy chuẩn thông điệp commit:** Sử dụng tiền tố chuẩn theo Conventional Commits: `feat`, `fix`, `test`, `docs`, `chore` (ví dụ: `fix(rule-based): sửa xử lý trạng thái kết thúc`). Phần mô tả nêu rõ nội dung thay đổi và nguyên nhân; chỉ xác nhận "đã kiểm thử" khi các lệnh test thực sự đã chạy thành công.

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
.\.venv\Scripts\python.exe -m pytest tests/test_rule_based_agent.py -v

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

- **Báo cáo độc lập:** Hai chế độ phải được đánh giá riêng biệt; chế độ `fixed` không phản ánh tính công bằng về thời gian thực thi. Ở chế độ `time`, số mô phỏng hoặc độ sâu của chế độ `fixed` không được dùng làm điều kiện dừng bổ sung. Tác tử Rule-based đưa ra phản xạ nhanh được phép trả về ngay khi có quyết định, không cần chờ hết 2 giây.
- **Hỗ trợ API:** Lớp `SearchBudget` (`config/search_budget.py`) lưu trữ các trường `mode`, `simulations`, `depth`, `time_limit_s`. Truyền `budget` xuống `BaseAgent` khi khởi tạo subclass; không thay đổi chữ ký phương thức chung.
- **Cơ chế dừng hợp tác (Cooperative Stopping):** Tác tử phải chủ động dừng tìm kiếm trước deadline để kịp hoàn tất và gửi kết quả. `self.time_exhausted()` chỉ trả về đúng khi đã tới deadline, không tự trừ biên an toàn. Mốc dừng sớm khởi đầu đề xuất là 1,9 giây trên hạn 2,0 giây (chừa khoảng 100 ms); thuật toán tự quản lý, công khai và kiểm chứng biên này trên máy chạy. Đây không phải bảo đảm tuyệt đối tránh timeout; hạn cứng của runner vẫn là 2,0 giây, không được cộng thêm 100 ms.
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

Cấu trúc thuật toán Rule-based cần phân định ranh giới trách nhiệm rõ ràng giữa các tầng logic:

| Thành phần | Yêu cầu tổ chức và kỹ thuật |
| :--- | :--- |
| **Nhận diện chiến thuật** | Phân định chính xác nước tạo thành 5 quân thắng ngay và nước đối phương đe dọa thắng ngay ở lượt tiếp theo theo đúng luật Game Engine. |
| **Nhận diện mẫu hình (Pattern)** | Nhận diện các hình thái: chuỗi liên tục, chuỗi có khoảng trống, đầu mở (open), đầu bị chặn và vị trí biên bàn cờ. |
| **Thứ tự ưu tiên và lượng giá** | Thiết lập bảng phân cấp độ ưu tiên rõ ràng giữa: tấn công (tạo đe dọa), phòng thủ (ngăn chặn đe dọa) và kiểm soát vị trí chiến lược. |
| **Quyết định nước đi** | Xác định tập ứng viên cho từng tầng luật, cơ chế phá hòa (tie-breaking) khi nhiều nước bằng điểm và nước đi dự phòng khi không có pattern nổi bật. |
| **Giải thích quyết định** | Cung cấp thông tin truy vết nguyên nhân (luật nào được kích hoạt) phục vụ phân tích trong notebook, giữ nguyên kiểu trả về `(row, col)` của API. |

> **Lưu ý thiết kế:** Mỗi thành phần cần có ca kiểm thử độc lập; không chỉ thử nghiệm trên tác tử hoàn chỉnh cuối cùng. Tuyệt đối không đưa các chức năng phụ như vẽ biểu đồ, đọc file notebook hay vận hành giải đấu vào phương thức lựa chọn nước đi.

---

## 5. Khung lập trình toàn bộ notebook

File `notebooks/01_rule_based.ipynb` được tổ chức tuần tự theo 10 phần dưới đây. Mỗi phần cần có cell Markdown mô tả mục tiêu, cell thực thi mã nguồn và phần phân tích, nhận xét kết quả:

1. **Câu hỏi nghiên cứu và phạm vi:** Việc bổ sung nhận diện pattern có thực sự nâng cao chất lượng nước đi so với việc chỉ dùng luật thắng ngay / chặn thua kết hợp điểm vị trí cơ bản hay không?
2. **Môi trường và cấu hình:** Ghi nhận trọng số điểm, bảng thứ tự ưu tiên, bán kính ứng viên, cơ chế phá hòa và `seed` ngẫu nhiên (nếu có).
3. **Bộ vị trí chiến thuật:** Mô tả chi tiết các tình huống thử nghiệm: trạng thái bàn cờ, bên đến lượt, danh sách các nước đi chấp nhận được và giải thích lý do.
4. **Kiểm chứng luật thắng và chặn:** Quan sát phản ứng của tác tử ở cả 4 hướng (ngang, dọc, 2 đường chéo), các vị trí sát biên bàn cờ, chuỗi dài hơn 5 quân và tình huống đồng thời xuất hiện cả cơ hội thắng lẫn đe dọa thua.
5. **Định nghĩa pattern và độ ưu tiên:** Trình bày định nghĩa cụ thể của từng pattern, điểm số gán cho pattern và lý do sắp xếp thứ tự ưu tiên trên các ví dụ đối chứng.
6. **Giải thích quyết định nước đi:** Trực quan hóa luật được kích hoạt và lý do lựa chọn nước đi tại các vị trí cờ từ đơn giản đến phức tạp có xung đột luật.
7. **Đánh giá tác dụng thành phần (Ablation):** So sánh tác tử đầy đủ với biến thể tắt nhận diện pattern (chỉ giữ luật thắng/chặn), giữ nguyên toàn bộ các thành phần còn lại.
8. **Thi đấu thử nghiệm:** Chạy đối đầu với `RandomAgent` theo từng cặp ván đổi bên đi trước/sau; ghi nhận kết quả $W/D/L$, điểm số và thời gian thực thi.
9. **Phân tích các trường hợp thất bại:** Phân tích các thế cờ bị đối phương tạo đe dọa kép (double threat), nhận diện nhầm pattern hoặc các giới hạn cố hữu của hệ thống luật.
10. **Tổng kết và hướng dẫn tái lập:** Đưa ra câu trả lời dựa trên số liệu thực tế, nêu rõ các hạn chế kỹ thuật, cấu hình đề xuất bàn giao và thứ tự chạy lại notebook.

### Nguyên tắc thực nghiệm trong Notebook
- Tập trung toàn bộ cấu hình ở phần đầu notebook; ghi lại cấu hình cụ thể đi kèm mỗi thí nghiệm khi có sự thay đổi.
- Sử dụng chung bộ vị trí bàn cờ và phương pháp ghi nhận kết quả để đảm bảo tính công bằng khi so sánh các biến thể. Luôn cố định `seed` nếu có yếu tố ngẫu nhiên.
- Thực thi thuật toán trực tiếp từ module `agents/rule_based_agent.py`; notebook chỉ đóng vai trò tổ chức thí nghiệm và trình bày kết quả.
- Mỗi tình huống thử nghiệm cần thể hiện rõ: trạng thái bàn cờ, bên đến lượt, kỳ vọng trước khi chạy và quan sát thực tế sau khi chạy.
- Luôn chạy tuần tự từ một kernel mới (Restart Kernel & Run All); tránh sử dụng các biến còn sót lại từ các cell đã chạy trước đó.
- Không thay thế bộ unit test bằng việc chỉ quan sát bàn cờ trong notebook.
- Chưa có số liệu thì ghi nhận là chưa thực hiện. Tuyệt đối không điền bảng dữ liệu giả định, không đưa ra kết luận vội vã chỉ dựa trên một ván thắng đơn lẻ.
- Kết quả thử nghiệm cục bộ không thay thế giải đấu chính thức (tối thiểu 100 ván/cặp, đổi bên, nhiều seed). Tuyệt đối không dùng các ván đánh giá cuối cùng để điều chỉnh tham số.

---

## 6. Giai đoạn 0 — Khởi tạo môi trường và xác nhận nền tảng

- Đọc kỹ `README.md` gốc của repository, cài đặt môi trường ảo và dependencies từ `requirements.txt`.
- Chạy toàn bộ test nền tảng và kiểm tra hoạt động của `notebooks/00_game_engine_test.ipynb`.
- Mở `notebooks/01_rule_based.ipynb`, xác nhận Jupyter đang sử dụng đúng kernel `.venv` và các ví dụ tích hợp mẫu chạy thành công.
- Kiểm tra trạng thái Git (`git status`), commit hoặc lưu trữ (stash) các công việc đang dở trước khi chuyển nhánh.
- Giai đoạn 0 không tạo commit rỗng hay PR riêng. Các nhánh của từng giai đoạn tiếp theo được tạo tuần tự từ `main` sau khi PR của giai đoạn trước đó đã được merge hoàn tất.
- Nếu nhánh tương ứng đã tồn tại sẵn, chuyển sang nhánh đó thay vì tạo mới. Kiểm tra kỹ lịch sử Git trước khi đổi tên nhánh.

---

## 7. Giai đoạn 1 — Hoàn thiện luật thắng ngay và chặn thua

### Mục tiêu và nhiệm vụ
1. Cài đặt cơ chế kiểm tra nước đi thử nghiệm và nhận diện nước thắng theo 4 hướng mà không làm thay đổi mảng `board` gốc.
2. Thiết lập mức độ ưu tiên tuyệt đối cho nước thắng ngay của bên đang đi trước khi xét đến việc phòng thủ; tìm kiếm đe dọa của đối thủ trên toàn bộ ô trống để tránh bỏ sót do giới hạn tập ứng viên.
3. Quy định cơ chế chọn nước dự phòng ở các vị trí chưa có đe dọa trực tiếp để phiên bản tối thiểu luôn trả về nước đi hợp lệ; xác định rõ đây là bản baseline chiến thuật.
4. Hoàn thiện nội dung các phần 1–4 trong notebook; ghi nhận định nghĩa tập nước đi chấp nhận được khi có nhiều lựa chọn tương đương.

### Kiểm thử và tiêu chí nghiệm thu
- Kiểm tra chính xác nước thắng ngay và nước chặn thua ở cả 4 hướng (ngang, dọc, 2 đường chéo) cho cả hai phía X và O, các trường hợp sát biên và chuỗi nối dài hơn 5 quân.
- Kiểm tra tình huống đồng thời xuất hiện cả nước thắng của mình và đe dọa của đối phương: tác tử phải ưu tiên đánh nước thắng ngay để kết thúc ván cờ.
- Kiểm tra các trường hợp biên: bàn cờ trống, bàn cờ chỉ còn một ô trống duy nhất, tính hợp lệ của đầu ra và tính bất biến của mảng bàn cờ đầu vào.

> **Mốc hoàn thành (Milestone):** Phiên bản luật chiến thuật hoàn thành trọn vẹn ván cờ, bộ test thắng/chặn vượt qua 100% và chưa tuyên bố đã bao quát toàn bộ pattern nâng cao.

### Hướng dẫn thiết kế ca kiểm thử
1. Tạo fixture kiểm thử có mô tả rõ ràng: trạng thái bàn cờ, bên đến lượt, mối đe dọa và danh sách nước đi hợp lệ chấp nhận được; sử dụng engine thực hiện chuỗi nước hợp lệ để dựng trạng thái bàn cờ.
2. Dùng cùng một nhóm test để bao phủ 4 hướng (ngang, dọc, 2 chéo), cho cả hai phía X và O, các thế cờ sát mép biên và trường hợp nối dài chuỗi hơn 5 quân.
3. Chuẩn bị ca kiểm thử đặc biệt có đồng thời cơ hội thắng của mình và đe dọa của đối phương nhằm xác nhận thứ tự ưu tiên chiến thuật hoạt động chính xác.

**Tiêu chí đạt chuẩn trước khi yêu cầu Review:**
- [ ] Chọn chính xác nước thắng khi có cơ hội kết thúc ván cờ ngay; khi không có nước thắng và đối thủ có đe dọa duy nhất, chọn đúng nước chặn thua.
- [ ] Nước thắng ngay luôn được ưu tiên hơn nước chặn đối thủ trên toàn bộ các ca kiểm thử tương ứng.
- [ ] Bàn cờ trống hoặc chỉ còn một ô đều trả về nước đi hợp lệ; mảng bàn cờ đầu vào hoàn toàn không bị biến đổi.

**Bằng chứng bắt buộc trong PR:** Lệnh kiểm thử đã chạy, số lượng test đạt/thất bại, danh sách fixture đã thử nghiệm và các hạn chế kỹ thuật đã biết.

### Quy trình nhánh, commit và đẩy mã nguồn
- **Tên nhánh:** `feature/rule-based-tactics` *(tạo từ main sau khi nền tảng chung sẵn sàng)*.

<details>
<summary>Lệnh thao tác Git (PowerShell)</summary>

```powershell
git status
git switch main
git pull --ff-only origin main
git switch -c feature/rule-based-tactics

# Thực hiện code và kiểm thử trước khi commit
.\.venv\Scripts\python.exe -m pytest tests -v

git add -- agents/rule_based_agent.py notebooks/01_rule_based.ipynb tests/test_rule_based_agent.py docs/Tan-Phat.md
git diff --cached --stat
git diff --cached
git commit -m "feat(rule-based): bổ sung luật thắng ngay và chặn thua"
git push -u origin feature/rule-based-tactics
```

</details>

### Tiêu đề và mô tả PR Giai đoạn 1
- **Tiêu đề PR:** `feat(rule-based): bổ sung tác tử luật với chiến thuật thắng và chặn`
- **Mô tả nội dung mẫu:**
  > Bổ sung phiên bản Rule-based tối thiểu với khả năng nhận diện nước thắng ngay và ngăn chặn đối thủ thắng ngay lập tức. Notebook trình bày chi tiết bảng thứ tự ưu tiên và các tình huống kiểm chứng thực tế. Các luật nhận diện pattern nâng cao sẽ được triển khai ở giai đoạn tiếp theo.

---

## 8. Giai đoạn 2 — Hoàn thiện Pattern và thứ tự ưu tiên

### Mục tiêu và nhiệm vụ
1. Định nghĩa rõ ràng các hình thái cờ: bốn quân mở (open-four), ba quân mở (open-three), chuỗi có khoảng trống ở giữa và chuỗi bị chặn một đầu; không chỉ dựa đơn thuần vào tổng số lượng quân.
2. Xác định bảng thứ tự ưu tiên giữa: tạo đe dọa tấn công, vô hiệu hóa đe dọa phòng thủ và chiếm các điểm vị trí chiến lược; lập bảng quy tắc xử lý khi nhiều luật cùng kích hoạt đồng thời.
3. Thống nhất thang điểm trọng số và cơ chế phá hòa (tie-breaking); sử dụng `seed` riêng nếu có yếu tố chọn ngẫu nhiên. Lọc ứng viên chỉ áp dụng cho các tầng luật đã công khai phạm vi.
4. Bổ sung cơ chế truy vết nguyên nhân chọn nước để phục vụ giải thích trực quan trong notebook, đảm bảo giữ nguyên kiểu dữ liệu trả về của `get_action`.
5. Hoàn thiện các phần 5–6 trong notebook, ghi nhận rõ ràng các tình huống đối phương tạo đe dọa kép không thể phòng thủ hoàn toàn.

### Kiểm thử và tiêu chí nghiệm thu
- Kiểm tra nhận diện pattern trong các trường hợp: có đầu mở, không có đầu mở, sát biên bàn cờ, chuỗi bị ngắt quãng bởi quân đối thủ và các cấu hình tương tự nhưng không hợp lệ.
- Kiểm tra tính đúng đắn khi phát sinh xung đột giữa tấn công và phòng thủ theo đúng bảng ưu tiên đã công bố; đảo góc nhìn X/O để phát hiện các sai lệch thiên vị.
- Tình huống đối phương có hai nước thắng độc lập (đe dọa kép) cần được phân tích như một giới hạn tự nhiên của luật; không ép buộc tác tử phải chặn được cả hai trong một lượt đi.
- Kiểm tra tính ổn định hoặc khả năng tái lập kết quả khi phá hòa theo đúng quy tắc đã công bố.

> **Mốc hoàn thành (Milestone):** Toàn bộ các pattern và bảng quy tắc giải quyết xung đột luật có bài test độc lập vượt qua 100%; notebook giải thích thuyết phục quyết định nước đi và các giới hạn kỹ thuật.

### Hướng dẫn thiết kế ca kiểm thử
1. Với mỗi loại pattern, xây dựng một ví dụ khớp mẫu chuẩn xác và một ví dụ tương tự nhưng không khớp mẫu; phân biệt rõ quân đối thủ, ô trống và mép biên bàn cờ.
2. Kiểm tra độc lập từng luật đơn lẻ trước, sau đó kiểm tra các cặp luật kích hoạt đồng thời để xác nhận thứ tự ưu tiên.
3. Đối chiếu kết quả khi đảo ngược góc nhìn người chơi giữa X và O; kiểm tra các tình huống nhiều nước bằng điểm dựa trên tập hành động cho phép hoặc seed đã chốt.
4. Ca kiểm thử thông tin giải thích phải phản ánh chính xác quyết định thực tế; không tự ý suy diễn tên luật chỉ dựa vào tọa độ nước đi được chọn.

**Tiêu chí đạt chuẩn trước khi yêu cầu Review:**
- [ ] Bộ fixture nhận diện chính xác các pattern chuẩn và không nhận nhầm các ví dụ đối chứng giả định.
- [ ] Quyết định khi xung đột luật tuân thủ đúng bảng ưu tiên; kết quả đối xứng giữa hai phía X và O không bị thiên vị ngoài thiết kế.
- [ ] Tình huống đe dọa kép không thể ngăn chặn được ghi nhận đúng như một hạn chế kỹ thuật; không yêu cầu giải quyết bài toán bất khả thi.
- [ ] Cơ chế phá hòa hoạt động ổn định hoặc tái lập theo seed, đầu ra luôn tuân thủ nghiêm ngặt hợp đồng của `BaseAgent`.

**Bằng chứng bắt buộc trong PR:** Lệnh kiểm thử đã chạy, số lượng test đạt/thất bại, danh sách fixture đã thử nghiệm và các hạn chế kỹ thuật đã biết.

### Quy trình nhánh, commit và đẩy mã nguồn
- **Tên nhánh:** `feature/rule-based-patterns` *(chỉ tạo sau khi PR Giai đoạn 1 đã được merge vào main)*.

<details>
<summary>Lệnh thao tác Git (PowerShell)</summary>

```powershell
git status
git switch main
git pull --ff-only origin main
git switch -c feature/rule-based-patterns

# Thực hiện code và kiểm thử trước khi commit
.\.venv\Scripts\python.exe -m pytest tests -v

git add -- agents/rule_based_agent.py notebooks/01_rule_based.ipynb tests/test_rule_based_agent.py docs/Tan-Phat.md
git diff --cached --stat
git diff --cached
git commit -m "feat(rule-based): bổ sung pattern và thứ tự ưu tiên chiến thuật"
git push -u origin feature/rule-based-patterns
```

</details>

### Tiêu đề và mô tả PR Giai đoạn 2
- **Tiêu đề PR:** `feat(rule-based): hoàn thiện nhận diện pattern và ưu tiên Rule-based`
- **Mô tả nội dung mẫu:**
  > Bổ sung cơ chế nhận diện pattern nâng cao, điểm đánh giá vị trí và quy tắc giải quyết xung đột luật. Notebook giải thích chi tiết các quyết định nước đi thông qua các thế cờ chiến thuật cụ thể. Bộ test bao phủ các vị trí biên bàn cờ, góc nhìn hai phía và các dạng đe dọa đa dạng.

---

## 9. Giai đoạn 3 — Đánh giá tác dụng Pattern và bàn giao

### Mục tiêu và nhiệm vụ
1. Thiết lập hai cấu hình phục vụ thực nghiệm đối chứng: biến thể đầy đủ (Full Pattern) và biến thể tắt pattern (No Pattern); giữ nguyên luật thắng/chặn, danh sách seed, tập vị trí và cơ chế phá hòa.
2. Đo đạc tỷ lệ giải quyết tình huống chiến thuật thành công, thời gian thực thi và kết quả thi đấu thử nghiệm; không chọn lọc riêng các ván thắng để kết luận.
3. Tổ chức thi đấu thử nghiệm với `RandomAgent` theo từng cặp ván đổi bên đi trước/sau; ghi nhận đầy đủ thông số cấu hình và số lượng ván đấu thực tế.
4. Hoàn thiện nội dung các phần 7–10 trong notebook; phân tích các mẫu lỗi thường gặp và đánh giá tác động thực tế của từng tầng luật.

### Kiểm thử và tiêu chí nghiệm thu
- Chạy lại toàn bộ test tự động và notebook từ một kernel mới hoàn toàn.
- Đảm bảo việc bật/tắt pattern chỉ thay đổi duy nhất thành phần cần khảo sát; không vô tình tắt bỏ cả luật thắng ngay và chặn thua.
- Đảm bảo số lượng ván đấu, danh sách seed và các số liệu thống kê khớp hoàn toàn với dữ liệu log; mọi trường hợp phát sinh lỗi đều phải được ghi nhận rõ ràng.

> **Mốc hoàn thành (Milestone):** Có số liệu đối chứng thực nghiệm rõ ràng về tác dụng của pattern, kết quả thi đấu thử nghiệm khách quan; không suy diễn quá mức về sức mạnh của tác tử khi chưa thi đấu chính thức.

### Hướng dẫn thiết kế ca kiểm thử
1. Kiểm tra cấu hình bật/tắt pattern trên các thế cờ mà sự hiện diện của pattern thực sự làm thay đổi quyết định chọn nước; đảm bảo luật thắng/chặn vẫn hoạt động chính xác ở cả hai chế độ.
2. Trong `tests/test_rule_based_agent.py`, kiểm tra cấu trúc log của tác tử và gọi hàm tính chỉ số chung (`evaluation/metrics.py`) trên tập dữ liệu nhỏ đã biết trước kết quả. Không tự ý viết lại logic tính toán $W/D/L$ hay $score$ trong agent hoặc notebook.
3. Trong notebook, sử dụng các cặp ván đổi lượt với cùng danh sách seed; không loại bỏ các ván thua hay ván lỗi ra khỏi báo cáo tổng hợp.

**Tiêu chí đạt chuẩn trước khi yêu cầu Review:**
- [ ] Biến thể tắt pattern chỉ loại bỏ đúng thành phần pattern được khảo sát; toàn bộ test thắng/chặn từ Giai đoạn 1 vẫn vượt qua thành công.
- [ ] Số liệu thống kê khớp chính xác với số ván, danh sách seed và cấu hình thử nghiệm; notebook có thể chạy lại trơn tru từ đầu.
- [ ] Không yêu cầu tác tử bản đầy đủ phải luôn thắng mọi ván hoặc vượt trội ở mọi tình huống; nếu kết quả thực nghiệm chưa cải thiện, vẫn báo cáo và phân tích trung thực nguyên nhân.

**Bằng chứng bắt buộc trong PR:** Lệnh kiểm thử đã chạy, số lượng test đạt/thất bại, danh sách fixture đã thử nghiệm và các hạn chế kỹ thuật đã biết.

### Quy trình nhánh, commit và đẩy mã nguồn
- **Tên nhánh:** `feature/rule-based-evaluation` *(chỉ tạo sau khi PR Giai đoạn 2 đã được merge vào main)*.

<details>
<summary>Lệnh thao tác Git (PowerShell)</summary>

```powershell
git status
git switch main
git pull --ff-only origin main
git switch -c feature/rule-based-evaluation

# Thực hiện code và kiểm thử trước khi commit
.\.venv\Scripts\python.exe -m pytest tests -v

git add -- agents/rule_based_agent.py notebooks/01_rule_based.ipynb tests/test_rule_based_agent.py docs/Tan-Phat.md
git diff --cached --stat
git diff --cached
git commit -m "test(rule-based): bổ sung đánh giá tác dụng pattern và phân tích lỗi"
git push -u origin feature/rule-based-evaluation
```

</details>

### Tiêu đề và mô tả PR Giai đoạn 3
- **Tiêu đề PR:** `test(rule-based): bổ sung đánh giá thành phần và kết quả Rule-based`
- **Mô tả nội dung mẫu:**
  > Bổ sung thực nghiệm đối chứng giữa tác tử đầy đủ và biến thể tắt pattern, kết hợp thi đấu thử nghiệm có cân bằng quyền đi trước/sau. Notebook lưu trữ chi tiết cấu hình, số liệu thực nghiệm và phân tích sâu các tình huống thất bại để làm rõ giá trị thực tế của từng tầng luật.

---

## 10. Quy trình tạo PR và xử lý phản hồi Review

### Thiết lập thông tin bắt buộc trên GitHub
- **Reviewer:** @10tothepowerof6 (Nhóm trưởng). Chọn trực tiếp trong mục Reviewers trên giao diện PR khi đã sẵn sàng review; chỉ nhắc tên trong mô tả sẽ không tạo thông báo chính thức.
- **Assignee:** Chưa gán (sẽ cập nhật tài khoản người phụ trách sau khi được cấp quyền truy cập repository). Người phụ trách có trách nhiệm theo dõi và hoàn thiện PR; reviewer chịu trách nhiệm kiểm tra và đánh giá thay đổi.
- **Milestone:**
  - Task 1–2 / PR tương ứng: [Báo cáo tiến độ nửa dự án](https://github.com/10tothepowerof6/AdvancedAI-Repo/milestone/2), hạn 09/11/2026.
  - Task 3: [milestone tổng](https://github.com/10tothepowerof6/AdvancedAI-Repo/milestone/1).
- **Liên kết Issue:**
  - Giai đoạn 1: [Task 1 (Rule-based)](https://github.com/10tothepowerof6/AdvancedAI-Repo/issues/7)
  - Giai đoạn 2: [Task 2 (Rule-based)](https://github.com/10tothepowerof6/AdvancedAI-Repo/issues/8)
  - Giai đoạn 3: [Task 3 (Rule-based)](https://github.com/10tothepowerof6/AdvancedAI-Repo/issues/9)
  *(Không tạo issue cho Giai đoạn 0).*
- **Đóng issue:** Trong phần mô tả của PR hoàn thành task, sử dụng cú pháp `Closes #<số_issue>` (ví dụ: `Closes #7`). PR nháp hoặc PR chỉ hoàn thành một phần thì chỉ gắn liên kết tham chiếu thông thường, không dùng từ khóa `Closes`.
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
4. Nếu phát sinh xung đột (conflict): Mở từng file bị xung đột, kiểm tra kỹ lưỡng nội dung của cả hai bên, tiến hành giải quyết, chạy lại bộ test và hoàn tất commit hợp nhất: `chore(rule-based): đồng bộ thay đổi từ main`. Tuyệt đối không tự ý xóa bỏ phần code của thành viên khác.

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
