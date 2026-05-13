import argparse
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal


def read_mono_audio(path: Path) -> tuple[np.ndarray, int]:
    if not path.is_file():
        raise FileNotFoundError(f'Không tìm thấy file "{path}".')

    x, fs = sf.read(path, always_2d=True)
    x = x.mean(axis=1).astype(np.float32)
    peak = np.max(np.abs(x))
    if peak > 0:
        x = x / peak
    return x, fs


def ai_pretrained_demucs(x: np.ndarray, fs: int, device: str = "cpu") -> np.ndarray:
    try:
        import torch
        from denoiser import pretrained
        from denoiser.dsp import convert_audio
    except ImportError as exc:
        raise RuntimeError("Thiếu thư viện. Hãy chạy: pip install torch torchaudio denoiser") from exc

    print("  -> Đang tải mô hình DNS64 của Meta...")
    model = pretrained.dns64().to(device)
    model.eval()

    wav = torch.from_numpy(x).unsqueeze(0).to(device)
    model_fs = 16000
    wav = convert_audio(wav, fs, model_fs, model.chin)

    print("  -> AI đang bóc tách nhiễu...")
    with torch.no_grad():
        denoised = model(wav[None])[0]

    y = denoised.squeeze().cpu().numpy()

    if fs != model_fs and len(y) > 0:
        gcd = math.gcd(fs, model_fs)
        y = signal.resample_poly(y, fs // gcd, model_fs // gcd)

    if len(y) > len(x):
        y = y[:len(x)]
    elif len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))

    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak

    return y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="cut.wav", help="Đường dẫn file wav đầu vào.")
    args = parser.parse_args()

    input_path = Path(args.input)
    # Thư mục lưu kết quả
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    # Đã đổi tên file đầu ra tại đây
    output_path = out_dir / "ai_denoised.wav"

    print(f"[*] Đang đọc file âm thanh: {input_path} ...")
    x, fs = read_mono_audio(input_path)

    print("[*] Bắt đầu gọi Trí tuệ nhân tạo (AI)...")
    try:
        x_ai = ai_pretrained_demucs(x, fs)
    except Exception as exc:
        print(f"\n[!] Lỗi khi chạy AI: {exc}\n")
        return

    print(f"[*] Đang lưu file âm thanh đã lọc nhiễu tại: {output_path} ...")
    sf.write(output_path, x_ai, fs)

    print("[+] Hoàn tất!")


if __name__ == "__main__":
    main()