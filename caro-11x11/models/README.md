# Checkpoint CNN Policy Caro 11×11

Checkpoint chính thức `policy.pt` và báo cáo `report.json` được theo dõi trong `models/caro-policy-v1/`. Checkpoint lưu trọng số, phiên bản kiến trúc, epoch được chọn và SHA-256 của manifest dữ liệu. Báo cáo lưu cấu hình, loss và độ chính xác theo epoch, kết quả test, thời gian chạy và SHA-256 của checkpoint.

## Tái tạo checkpoint

Chạy từ thư mục `caro-11x11`. Nếu đã có bộ `caro11-v1` từ Task 1, bắt đầu từ lệnh huấn luyện. Dùng `--output` với thư mục mới khi chạy lại để giữ bản hiện có.

```powershell
.\.venv\Scripts\python.exe -m training.generate_data --games 1000 --seed 42 --opening-plies 4
.\.venv\Scripts\python.exe -m training.train --output models/caro-policy-v1-retrain --device auto --epochs 20 --batch-size 64 --lr 0.001 --patience 5 --seed 42
```

`--device auto` dùng CUDA khi khả dụng, hoặc CPU. Kết quả có thể khác giữa các phiên bản thư viện hoặc thiết bị. Mỗi lượt chọn checkpoint có validation loss thấp nhất rồi đánh giá trên tập test giữ riêng. Notebook Neural Network hiển thị đường cong từ báo cáo đã sinh.

## Kiểm tra bản đã huấn luyện

SHA-256 của checkpoint chính thức là `4f7022148043a5aea35031831abcec28268d8667eddb8affeb7f165a68dd4181`. Với checkpoint tái tạo, đối chiếu với `checkpoint.sha256` trong báo cáo cùng lượt chạy.

```powershell
Get-FileHash models/caro-policy-v1/policy.pt -Algorithm SHA256
```

Module `training.model` cung cấp `load_policy` để nạp checkpoint ở chế độ đánh giá. Mô hình nhận hai kênh quân mình và quân đối thủ, mỗi kênh 11×11, rồi trả logit cho 121 ô theo thứ tự hàng trước, cột sau.
