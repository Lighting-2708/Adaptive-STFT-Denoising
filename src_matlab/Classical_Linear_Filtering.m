%% =========================================================
%  BỘ LỌC ÂM THANH CỔ ĐIỂN (CLASSICAL IIR LOW-PASS FILTER)
%% =========================================================
clear; clc; close all;

%% 1. Cài đặt File
inputFile = 'voice_with_noisy.wav';
outputFile = 'denoised_iir.wav'; % Lưu tên khác để phân biệt với file của STFT

if ~exist(inputFile, 'file')
    error('Không tìm thấy file đầu vào: %s', inputFile);
end

% Đọc file và chuyển về dạng Mono
[x, fs] = audioread(inputFile);
x = mean(x, 2); 

%% 2. Thiết kế Bộ lọc Butterworth Thông thấp (Low-pass)
fc = 3400;  % Tần số cắt (Cut-off frequency) ở 3400 Hz
order = 4;  % Bậc của bộ lọc (Bậc 4 là phổ biến, cắt khá dốc)

% Chuẩn hóa tần số cắt theo tần số Nyquist (fs/2)
Wn = fc / (fs / 2); 

% Lấy hệ số của phương trình sai phân (b: tử số, a: mẫu số)
[b, a] = butter(order, Wn, 'low'); 

%% 3. Thực hiện lọc tín hiệu
% Hàm 'filter' sẽ quét qua tín hiệu x. Nó áp dụng đúng công thức 
% phương trình sai phân H(z) mà bạn đã viết trong báo cáo LaTeX.
x_filtered = filter(b, a, x);

% Chuẩn hóa biên độ (Normalize) để tránh bị rè (clipping)
x_filtered = x_filtered / (max(abs(x_filtered)) + eps);

%% 4. Lưu kết quả và Báo cáo
audiowrite(outputFile, x_filtered, fs);

fprintf('\n--- BÁO CÁO LỌC IIR ---\n');
fprintf('Đã lọc xong bằng IIR Butterworth (Bậc %d, Tần số cắt: %d Hz).\n', order, fc);
fprintf('Đã lưu file: %s\n', outputFile);

% Phát thử âm thanh để cảm nhận sự "mù quáng" của IIR (bị nghẹt mũi do mất âm cao)
fprintf('Đang phát âm thanh sau khi lọc IIR...\n');
playerIIR = audioplayer(x_filtered, fs);
play(playerIIR);