"""
Configuration constants matching the paper's methodology exactly.
"""
import torch
DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                     else "cuda" if torch.cuda.is_available()
                     else "cpu")


# Audio Processing (matching paper Table 1)
SAMPLE_RATE = 16000          # 16 kHz sampling rate
STFT_WIN_MS = 25             # 25ms Hann window
STFT_HOP_MS = 10             # 10ms hop length
STFT_N_FFT = 512             # FFT size -> 257 freq bins
POWER_LAW_P = 0.3            # Power-law compression exponent

# Video Processing (matching paper)
TARGET_FPS = 25              # Convert all videos to 25 FPS
FACE_IMG_SZ = 160            # Face crop size for FaceNet
CHUNK_DURATION = 3.0         # 3-second chunks

# Spectrogram dimensions (calculated)
# With 16kHz, 10ms hop, 3 seconds -> 298 time frames
# With 512 FFT -> 257 frequency bins
SPEC_FREQ_BINS = 257         # (512 // 2) + 1
SPEC_TIME_FRAMES = 298       # int(3.0 * SAMPLE_RATE / (STFT_HOP_MS * SAMPLE_RATE / 1000))

AUDIO_EMB_SHAPE = (SPEC_FREQ_BINS, SPEC_TIME_FRAMES, 2)  # (freq_bins, time_frames, 2 for magnitude/phase


# Noise mixing (from paper methodology)
NOISE_GAIN = 0.3             # AudioSet noise gain factor

# Device selection



# Paths (you can modify these)
DEFAULT_NOISE_ROOT = "~/data/musan"  # Adjust to your MUSAN path