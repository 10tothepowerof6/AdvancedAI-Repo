# AdvancedAI-Repo — Hướng dẫn setup Caro AI 11×11

Repository chứa bài thực hành và đồ án môn Trí tuệ nhân tạo nâng cao.
Dự án Caro nằm trong `caro-11x11/`. Các lệnh sau dùng PowerShell trên Windows.

## 1. Chuẩn bị công cụ

Cài Git, Python **3.11.9 bản 64-bit** và VS Code.
Trong VS Code, cài extension Python và Jupyter của Microsoft.

```powershell
git --version
py -3.11 --version
```

Python cần hiển thị 3.11.9. Nếu không tìm thấy py, kiểm tra Python Launcher
hoặc dùng đường dẫn đầy đủ đến Python đã cài.

## 2. Kéo repository về máy

Mở terminal tại thư mục dùng để lưu mã nguồn:

```powershell
git clone https://github.com/10tothepowerof6/AdvancedAI-Repo.git
cd AdvancedAI-Repo
git status
cd caro-11x11
```

Nếu đã clone, không clone chồng lên thư mục cũ. Sau khi lưu công việc đang làm
và kiểm tra git status, cập nhật main:

```powershell
git switch main
git pull --ff-only origin main
```

Nếu Git yêu cầu lưu thay đổi hoặc báo phân kỳ, kiểm tra trước khi tiếp tục.
Không dùng reset để xóa công việc chỉ nhằm chạy được pull.

## 3. Tạo môi trường và cài thư viện

Từ đây chạy lệnh tại `AdvancedAI-Repo/caro-11x11`:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Mỗi máy tự tạo .venv, không sao chép từ máy khác và không đưa lên GitHub.
Requirements khóa phiên bản các gói; không tự nâng phiên bản khi làm thuật toán.
Có thể kích hoạt bằng `.\.venv\Scripts\Activate.ps1`, nhưng không bắt buộc:
các lệnh trong tài liệu đã gọi trực tiếp interpreter của .venv.
Nếu PowerShell chặn Activate.ps1, tiếp tục dùng đường dẫn trực tiếp này.

## 4. Kiểm tra môi trường

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import sys, numpy, torch; print('Python:', sys.version); print('NumPy:', numpy.__version__); print('PyTorch:', torch.__version__); print('CUDA khả dụng:', torch.cuda.is_available())"
```

pip check cần báo `No broken requirements found.`
Bộ requirements hiện chứa gói Windows và PyTorch CUDA 12.6; chưa xác nhận dùng
nguyên file trên Linux/macOS. Nếu CUDA trả về False, kiểm tra khả năng chạy CPU
và ghi rõ thiết bị thực tế. Nếu cài lỗi, lưu thông báo và trao đổi với nhóm trưởng
trước khi thay bộ dependency.

## 5. Mở notebook

Trong VS Code, mở caro-11x11, chọn `Python: Select Interpreter` →
`.venv/Scripts/python.exe`. Với notebook, chọn cùng môi trường ở `Select Kernel`.

Hoặc chạy Jupyter Lab:

```powershell
.\.venv\Scripts\python.exe -m jupyter lab --notebook-dir=.
```

Mở notebooks/00_game_engine_test.ipynb và chạy toàn bộ cell từ kernel mới.
Các notebook 01–05 là khung thí nghiệm; Random mẫu chỉ kiểm tra tích hợp.

## 6. Chạy kiểm thử

```powershell
.\.venv\Scripts\python.exe -m pytest tests -v
```

Các test phải pass trước khi bắt đầu thuật toán. Nếu chưa thấy tests/notebooks,
kiểm tra branch và phiên bản bàn giao đã được đẩy lên GitHub.
Không ghi nhận test thành công khi chưa chạy thực tế.

Luật chung: bàn 11×11; X đi trước; 0 trống, 1 X, -1 O.
Từ 5 quân liên tiếp trở lên là thắng, kể cả bị chặn hai đầu.
Bàn đầy không có người thắng thì hòa.

## 7. Đọc phân công và tạo branch

- [Phi Hùng — MCTS](caro-11x11/docs/Phi-Hung.md)
- [Ngọc Phúc — Minimax và Alpha-Beta](caro-11x11/docs/Ngoc-Phuc.md)
- [Tấn Phát — Rule-based](caro-11x11/docs/Tan-Phat.md)

Tài liệu gồm phạm vi file, API, giai đoạn thực hiện, milestone commit/push,
hướng dẫn PR và checklist bàn giao. Đường dẫn trong từng tài liệu tính từ
caro-11x11, trừ khi được ghi rõ khác.
