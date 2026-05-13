%% =========================================================
%  TRỰC QUAN HÓA SO SÁNH TỔNG HỢP 4 TÍN HIỆU
%  1. Noisy (Gốc)
%  2. Lọc IIR (Cổ điển)
%  3. Lọc STFT Thích nghi / Wiener (Hệ thống đề tài)
%  4. Lọc AI (Meta Demucs)
%% =========================================================
clear; clc; close all;

%% 1. Khai báo thông số phân tích (Giữ nguyên cấu trúc STFT)
params.frameLength = 1024;          
params.hopLength = 256;             
params.nfft = 1024;
params.window = hamming(params.frameLength, 'periodic');

% CẬP NHẬT TÊN FILE: Đảm bảo các tên file này khớp với máy của bạn
file_noisy = 'voice_with_noisy.wav';              % File gốc có nhiễu (hoặc 'noise.wav')
file_iir   = 'iir_denoised.wav';     % File sinh ra từ code IIR
file_stft  = 'denoised.wav';         % File sinh ra từ hệ thống STFT của bạn
file_ai    = 'ai_denoised.wav';   % File sinh ra từ code Python (Meta Demucs)

if ~exist(file_noisy, 'file') || ~exist(file_iir, 'file') || ...
   ~exist(file_stft, 'file')  || ~exist(file_ai, 'file')
    error('Thiếu file âm thanh! Hãy kiểm tra lại tên file trong thư mục hiện tại.');
end

%% 2. Đọc, chuẩn hóa và đồng bộ dữ liệu
[x_noisy, fs] = audioread(file_noisy);
[x_iir, ~]    = audioread(file_iir);
[x_stft, ~]   = audioread(file_stft);
[x_ai, ~]     = audioread(file_ai);

% Chuyển tất cả về Mono (1 kênh)
x_noisy = mean(x_noisy, 2);
x_iir   = mean(x_iir, 2);
x_stft  = mean(x_stft, 2);
x_ai    = mean(x_ai, 2);

% Chuẩn hóa biên độ (Normalize) để dễ so sánh đồ thị
x_noisy = x_noisy / (max(abs(x_noisy)) + eps);
x_iir   = x_iir / (max(abs(x_iir)) + eps);
x_stft  = x_stft / (max(abs(x_stft)) + eps);
x_ai    = x_ai / (max(abs(x_ai)) + eps);

% Lấy độ dài nhỏ nhất để đồ thị khớp nhau hoàn toàn trên trục thời gian
min_len = min([length(x_noisy), length(x_iir), length(x_stft), length(x_ai)]);
x_noisy = x_noisy(1:min_len);
x_iir   = x_iir(1:min_len);
x_stft  = x_stft(1:min_len);
x_ai    = x_ai(1:min_len);

t = (0:min_len - 1) / fs; % Tạo trục thời gian chung

%% ================== VẼ CÁC ĐỒ THỊ CHUYÊN SÂU ====================

% --- Hình 1: Waveform Comparison (4x1) ---
% Định dạng cửa sổ to hơn để hiển thị 4 hàng không bị ép
figure('Name', 'Miền Thời Gian (Waveform)', 'Color', 'w', 'Position', [100, 100, 800, 700]);

subplot(4, 1, 1);
plot(t, x_noisy, 'k'); grid on;
title('1. Tín hiệu gốc (Noisy)'); ylabel('Amp');

subplot(4, 1, 2);
plot(t, x_iir, 'g'); grid on;
title('2. Sau lọc IIR (Cắt thông thấp 3kHz)'); ylabel('Amp');

subplot(4, 1, 3);
plot(t, x_stft, 'b'); grid on;
title('3. Sau lọc STFT Thích nghi (Wiener & Mask Smoothing)'); ylabel('Amp');

subplot(4, 1, 4);
plot(t, x_ai, 'r'); grid on;
title('4. Sau lọc Trí tuệ Nhân tạo (Meta Demucs)'); xlabel('Time (s)'); ylabel('Amp');

% --- Hình 2: Spectrogram Comparison (Lưới 2x2) ---
figure('Name', 'Phổ đồ thời gian - tần số (Spectrogram)', 'Color', 'w', 'Position', [150, 150, 900, 650]);
noverlap = params.frameLength - params.hopLength;

subplot(2, 2, 1);
spectrogram(x_noisy, params.window, noverlap, params.nfft, fs, 'yaxis');
title('1. Spectrogram - Gốc (Noisy)');

subplot(2, 2, 2);
spectrogram(x_iir, params.window, noverlap, params.nfft, fs, 'yaxis');
title('2. Spectrogram - Lọc IIR');

subplot(2, 2, 3);
spectrogram(x_stft, params.window, noverlap, params.nfft, fs, 'yaxis');
title('3. Spectrogram - STFT Thích nghi');

subplot(2, 2, 4);
spectrogram(x_ai, params.window, noverlap, params.nfft, fs, 'yaxis');
title('4. Spectrogram - AI (Demucs)');

% --- Hình 3: PSD Comparison (Gộp 4 đường) ---
[pxx_noisy, f_psd] = pwelch(x_noisy, params.window, noverlap, params.nfft, fs);
[pxx_iir, ~]       = pwelch(x_iir, params.window, noverlap, params.nfft, fs);
[pxx_stft, ~]      = pwelch(x_stft, params.window, noverlap, params.nfft, fs);
[pxx_ai, ~]        = pwelch(x_ai, params.window, noverlap, params.nfft, fs);

pxx_noisy_dB = 10 * log10(pxx_noisy + eps);
pxx_iir_dB   = 10 * log10(pxx_iir + eps);
pxx_stft_dB  = 10 * log10(pxx_stft + eps);
pxx_ai_dB    = 10 * log10(pxx_ai + eps);

figure('Name', 'Miền Tần Số (PSD Analysis)', 'Color', 'w', 'Position', [200, 200, 850, 550]);
plot(f_psd, pxx_noisy_dB, 'k', 'LineWidth', 1.5); hold on;  % Đen: Gốc
plot(f_psd, pxx_iir_dB, 'g', 'LineWidth', 1.2);             % Xanh lá: IIR
plot(f_psd, pxx_stft_dB, 'b', 'LineWidth', 1.2);            % Xanh biển: STFT
plot(f_psd, pxx_ai_dB, 'r', 'LineWidth', 1.2);              % Đỏ: AI
grid on;

title('So sánh Mật độ Phổ Công suất (PSD) của 4 Phương pháp');
xlabel('Tần số (Hz)');
ylabel('Công suất / Tần số (dB/Hz)');
legend('Gốc (Noisy)', 'Lọc IIR', 'Lọc STFT Thích nghi', 'AI (Demucs)', 'Location', 'northeast');
xlim([0 fs/2]); 

% Đánh dấu các mốc tần số quan trọng trong báo cáo
xline(300, '--k', 'Label', '300 Hz', 'LabelOrientation', 'horizontal');
xline(3400, '--k', 'Label', '3.4 kHz', 'LabelOrientation', 'horizontal');
xline(8000, '--r', 'Label', '8 kHz (Giới hạn AI)', 'LabelOrientation', 'horizontal', 'Color', 'r');
hold off;