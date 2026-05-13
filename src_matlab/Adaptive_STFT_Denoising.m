    %  PRACTICAL STFT SPEECH DENOISING SYSTEM (STANDALONE VERSION)
    %  - Hybrid noise estimation
    %  - Adaptive Wiener-style masking
    %  - Time/frequency mask smoothing
    %  - PSD Analysis
clear;
clc;
close all;
%% ================== 1. File settings ====================
inputFile = fullfile(pwd, 'voice_with_noisy.wav');
outputFile = fullfile(pwd, 'denoised.wav');
if ~exist(inputFile, 'file')
    error('Input file not found: %s', inputFile);
end
%% ================== 2. System parameters ================
params.frameLength = 1024;          % 512 to 2048
params.hopLength = 256;             % 25% overlap step for smoother tracking
params.nfft = 1024;
params.window = hamming(params.frameLength, 'periodic');
% Hybrid noise estimation
params.initialNoiseRatio = 0.08;    % portion of total duration
params.minNoiseDuration = 0.10;     % seconds
params.maxNoiseDuration = 0.50;     % seconds
params.lowEnergyPercent = 10;       % use low-energy frames as extra noise cues
params.noiseBlendInitial = 0.60;    % weight for initial noise estimate
params.noiseBlendLowEnergy = 0.40;  % weight for low-energy frame estimate
% Adaptive Wiener filtering with decision-directed SNR estimation
params.decisionDirected = 0.98;
params.gainFloor = 0.2;
params.oversubtraction = 1.6;
params.speechPresenceThreshold = 2.0;
params.noiseUpdateWhenNoise = 0.80;
params.noiseUpdateWhenSpeech = 0.5;
% Smoothing reduces musical noise
params.smoothFreqBins = 5;
params.smoothTimeFrames = 7;
params.blendOriginal = 0.01;
%% ================== 3. Read and normalize ===============
[x, fs] = audioread(inputFile);
x = mean(x, 2); % Convert to mono
x = x / (max(abs(x)) + eps);
audioDuration = numel(x) / fs;
%% ================== 4. STFT =============================
[S, f, t] = stft(x, fs, ...
    'Window', params.window, ...
    'OverlapLength', params.frameLength - params.hopLength, ...
    'FFTLength', params.nfft);
S_mag = abs(S);
S_pow = S_mag .^ 2;
numFrames = size(S_pow, 2);
%% ================== 5. Hybrid noise estimation ==========
[initialNoisePsd, noiseFrames, noiseDuration] = estimate_initial_noise_psd(S_pow, x, fs, params);
lowEnergyNoisePsd = estimate_low_energy_noise_psd(S_pow, params);
noisePsd0 = params.noiseBlendInitial * initialNoisePsd + ...
            params.noiseBlendLowEnergy * lowEnergyNoisePsd;
%% ================== 6. Adaptive denoising ===============
[gainMask, trackedNoisePsd] = build_adaptive_gain(S_pow, noisePsd0, params);
gainMask = smooth_gain_mask(gainMask, params);
S_denoised = gainMask .* S;
%% ================== 7. ISTFT ============================
x_denoised = istft(S_denoised, fs, ...
    'Window', params.window, ...
    'OverlapLength', params.frameLength - params.hopLength, ...
    'FFTLength', params.nfft);
x_denoised = real(x_denoised);
x_denoised = align_signal_length(x_denoised, length(x));
% Blend with original to reduce robotic sound slightly
x_denoised = (1 - params.blendOriginal) * x_denoised + params.blendOriginal * x;
x_denoised = x_denoised / (max(abs(x_denoised)) + eps);
%% ================== 8. Save output ======================
audiowrite(outputFile, x_denoised, fs);
%% ================== 9. Metrics (No-Reference) ==========
% Tính toán lượng tín hiệu bị triệt tiêu chung
residual = x - x_denoised;
outputToResidualDb = 10 * log10(sum(x_denoised .^ 2) / (sum(residual .^ 2) + eps));
fprintf('\n--- BÁO CÁO KẾT QUẢ ĐẦU RA ---\n');
fprintf('Saved denoised file: %s\n', outputFile);
fprintf('Audio duration: %.2f s\n', audioDuration);
fprintf('Adaptive initial noise duration: %.2f s\n', noiseDuration);
fprintf('Initial noise frames: %d / %d\n', noiseFrames, numFrames);
fprintf('Low-energy frames used: %d / %d\n', max(1, round(params.lowEnergyPercent * numFrames / 100)), numFrames);
fprintf('Output-to-residual ratio (Mức độ nén tín hiệu): %.2f dB\n', outputToResidualDb);
%% ================== 10. Playback ========================
fprintf('\nPlaying denoised signal...\n');
playerDenoised = audioplayer(x_denoised, fs);
assignin('base', 'playerDenoised', playerDenoised);
play(playerDenoised);
fprintf('To stop playback, run: stop(playerDenoised)\n');
%% ================== 11. Recommendations =================
fprintf('\nRecommended tuning:\n');
fprintf('- Increase oversubtraction if broadband noise is still strong.\n');
fprintf('- Increase gainFloor slightly if speech sounds metallic or broken.\n');
fprintf('- Increase lowEnergyPercent if the recording has long silent/noisy regions.\n');
fprintf('- Reduce noiseUpdateWhenSpeech if the noise changes quickly over time.\n');
%% ================== HELPER FUNCTIONS ====================
function [noisePsd, noiseFrames, noiseDuration] = estimate_initial_noise_psd(S_pow, x, fs, params)
    totalDuration = numel(x) / fs;
    noiseDuration = min(max(params.minNoiseDuration, params.initialNoiseRatio * totalDuration), params.maxNoiseDuration);
    noiseFrames = floor(noiseDuration * fs / params.hopLength);
    noiseFrames = max(1, min(size(S_pow, 2), noiseFrames));
    noisePsd = mean(S_pow(:, 1:noiseFrames), 2);
end
function noisePsd = estimate_low_energy_noise_psd(S_pow, params)
    frameEnergy = mean(S_pow, 1);
    numFramesToUse = max(1, round(params.lowEnergyPercent * numel(frameEnergy) / 100));
    [~, sortedIdx] = sort(frameEnergy, 'ascend');
    selectedIdx = sortedIdx(1:numFramesToUse);
    noisePsd = mean(S_pow(:, selectedIdx), 2);
end
function [gainMask, trackedNoisePsd] = build_adaptive_gain(S_pow, noisePsd0, params)
    numFreq = size(S_pow, 1);
    numFrames = size(S_pow, 2);
    
    gainMask = zeros(numFreq, numFrames);
    trackedNoisePsd = zeros(numFreq, numFrames);
    
    prevGain = ones(numFreq, 1);
    prevPosterior = S_pow(:, 1) ./ max(noisePsd0, eps);
    trackedNoisePsd(:, 1) = noisePsd0;
    
    for frameIdx = 1:numFrames
        currentNoisePsd = trackedNoisePsd(:, max(frameIdx - 1, 1));
        posterior = S_pow(:, frameIdx) ./ max(params.oversubtraction * currentNoisePsd, eps);
        
        if frameIdx == 1
            priori = max(posterior - 1, 0);
        else
            priori = params.decisionDirected * (prevGain .^ 2) .* prevPosterior + ...
                     (1 - params.decisionDirected) * max(posterior - 1, 0);
        end
        priori = max(priori, 0);
        
        gain = priori ./ (1 + priori);
        gain = max(gain, params.gainFloor);
        
        speechPresence = posterior > params.speechPresenceThreshold;
        updateLambda = params.noiseUpdateWhenNoise * ones(numFreq, 1);
        updateLambda(speechPresence) = params.noiseUpdateWhenSpeech;
        
        if frameIdx < numFrames
            trackedNoisePsd(:, frameIdx + 1) = ...
                updateLambda .* currentNoisePsd + (1 - updateLambda) .* min(S_pow(:, frameIdx), params.oversubtraction * currentNoisePsd);
        end
        
        gainMask(:, frameIdx) = gain;
        prevGain = gain;
        prevPosterior = posterior;
    end
end
function smoothedMask = smooth_gain_mask(gainMask, params)
    kernel = ones(params.smoothFreqBins, params.smoothTimeFrames);
    kernel = kernel / sum(kernel(:));
    smoothedMask = conv2(gainMask, kernel, 'same');
    smoothedMask = min(max(smoothedMask, params.gainFloor), 1);
end
function y = align_signal_length(y, targetLength)
    y = y(:);
    if length(y) > targetLength
        y = y(1:targetLength);
    elseif length(y) < targetLength
        y(end + 1:targetLength, 1) = 0;
    end
end