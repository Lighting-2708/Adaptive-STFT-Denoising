# Adaptive-STFT-Denoising 🎧

Dự án môn học Xử lý Tín hiệu Số. Xây dựng và Tối ưu hóa Hệ thống Khử nhiễu Âm thanh STFT Thích nghi, kết hợp đánh giá đối sánh với Lọc Cổ điển (IIR Butterworth) và Trí tuệ Nhân tạo (Meta Demucs).

## 📊 Về dự án
Hệ thống được thiết kế để giải quyết bài toán khử nhiễu môi trường động. Dự án bao gồm 3 thành phần chính:
1. **Adaptive STFT:** Hệ thống lõi tự xây dựng áp dụng bộ lọc Wiener, thuật toán Decision-Directed và kỹ thuật Mask Smoothing (MATLAB).
2. **IIR Filter (Lower-bound):** Bộ lọc thông thấp Butterworth bậc 4, cắt tại tần số 3000 Hz (MATLAB).
3. **Meta Demucs DNS64 (Upper-bound):** Mô hình AI Deep Learning (Python/PyTorch).

## 📂 Cấu trúc mã nguồn
- `/src_matlab`: Mã nguồn cốt lõi xử lý STFT và Lọc IIR.
- `/src_python`: Mã nguồn triển khai mô hình AI Demucs.
- `/audio_samples`: Các tệp `.wav` trước và sau khi xử lý. 
- `/docs`: Báo cáo chi tiết định dạng PDF (được biên soạn bằng LaTeX).

## 🎧 Nghe thử kết quả (Audio Samples)
*Bạn có thể tải các file trong thư mục `/audio_samples` về để nghe thử sự khác biệt giữa các hệ thống.*

## ⚙️ Yêu cầu môi trường
* **MATLAB:** Cài đặt Signal Processing Toolbox.
* **Python:** `pip install torch torchaudio denoiser soundfile scipy`

---
*Dự án thực hiện bởi Nguyễn Hữu Quang - Kỹ thuật Máy tính.*
