"""
Refactored Audio Processing Utilities
Clean separation between training and inference modes
"""

import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple

import soundfile as sf
import torch
import torchaudio

from avspeech.utils.noise_mixer import NoiseMixer
from avspeech.utils.structs import AudioChunk, SampleT

# Constants (move to config.py or constants.py)
SAMPLE_RATE = 16000
STFT_N_FFT = 512
STFT_WIN_LENGTH = 400  # 25ms at 16kHz
STFT_HOP_LENGTH = 160  # 10ms at 16kHz
POWER_LAW_P = 0.3  # Power-law compression exponent
FRAMES_PER_CHUNK = 298  # STFT frames per 3-second chunk
SAMPLES_PER_CHUNK = 48000  # 3 seconds at 16kHz

# Duration boundaries for training
CHUNK_DURATIONS = {
    1: 48000,  # 3 seconds
    2: 96000,  # 6 seconds
    3: 144000,  # 9 seconds
}


# ─────────────────────── Audio Extraction and Processing ─────────────────────────
def extract_audio(video_path: Path, mode: str = "training") -> torch.Tensor:
    """
    Extract audio from video with mode-specific processing.

    Args:
        video_path: Path to video file
        mode: "training" (trim to chunks), "inference" (pad to chunks), or "raw" (no processing)

    Returns:
        Normalized audio tensor
    """
    if mode not in ["training", "inference", "raw"]:
        raise ValueError(f"Invalid mode: {mode}. Use 'training', 'inference', or 'raw'")

    # Extract raw audio from video
    raw_audio = _extract_raw_audio(video_path)

    # Apply mode-specific processing
    if mode == "training":
        audio = _process_for_training(raw_audio)
    elif mode == "inference":
        audio = _process_for_inference(raw_audio)
    else:  # raw
        audio = raw_audio

    # Normalize to [-1, 1]
    return audio


def compute_stft_features(audio: torch.Tensor) -> torch.Tensor:
    """
    Compute STFT features with power-law compression.

    Args:
        audio: Audio tensor

    Returns:
        STFT features of shape [257, T_frames, 2] (real, imag)
    """
    # Pad the tail for last STFT frame
    audio = _pad_for_stft(audio)

    # Compute STFT
    stft_complex = _compute_stft(audio)

    # Apply power-law compression
    return _apply_power_law_compression(stft_complex)


def chunk_stft_features(
    features: torch.Tensor, frames_per_chunk: int = FRAMES_PER_CHUNK
) -> List[torch.Tensor]:
    """
    Slice STFT features into fixed-size chunks.

    Args:
        features: STFT tensor of shape [257, T, 2]
        frames_per_chunk: Frames per chunk (default 298 for 3 seconds)

    Returns:
        List of chunks, each of shape [257, frames_per_chunk, 2]
    """
    total_frames = features.shape[1]
    num_complete_chunks = total_frames // frames_per_chunk

    chunks = []
    for i in range(num_complete_chunks):
        start = i * frames_per_chunk
        end = start + frames_per_chunk
        chunk = features[:, start:end, :].clone()
        chunks.append(chunk)

    # if len(chunks) > 0:
    #     remaining_frames = total_frames % frames_per_chunk
    #     if remaining_frames > 0:
    #         print(f"  Dropped {remaining_frames} frames at the end")

    return chunks


# ─────────────────────── Core Audio Extraction ─────────────────────────


# def _extract_raw_audio(video_path: Path) -> torch.Tensor:
#     """
#     Extract raw audio from video file using ffmpeg.

#     Returns audio as float32 tensor at SAMPLE_RATE Hz.
#     """
#     # Load audio waveform and original sample rate
#     try:
#         waveform, orig_sr = torchaudio.load(
#                 str(video_path),
#                 backend="soundfile",  # Explicitly use ffmpeg backend
#                 channels_first=True
#             )


#         # waveform shape: [channels, num_frames], dtype=float32 by default

#         # Resample to 16 kHz if needed (TorchAudio will use efficient sinc resampler)
#         if orig_sr != SAMPLE_RATE:
#             waveform = resample(waveform, orig_sr, SAMPLE_RATE)

#         # Downmix to mono by selecting the left channel (channel 0)
#         if waveform.shape[0] > 1:
#             waveform = waveform[0:1, :]  # keep only first channel

#         # (Optionally, you can return waveform[0] to get a 1D tensor of samples)
#         return waveform

#     except Exception as e:
#         print(f"Torchaudio dont like mp4s bro (Error loading audio: {e})")
#         raise e


def _extract_raw_audio(video_path: Path) -> torch.Tensor:
    """
    Extract raw audio from video file using ffmpeg.

    Returns audio as float32 tensor at SAMPLE_RATE Hz.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # print(f"  Extracting audio from: {video_path}")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        # Extract mono audio at target sample rate
        cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "1",  # Mono
            "-acodec",
            "pcm_s16le",
            tmp.name,
        ]

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg failed: {e.stderr}")

        # Load the extracted audio
        try:
            wav_data, sr = sf.read(tmp.name, dtype="float32")

            if sr != SAMPLE_RATE:
                raise ValueError(f"Sample rate mismatch: {sr} != {SAMPLE_RATE}")

            if len(wav_data) == 0:
                raise RuntimeError("Extracted audio is empty")

            # print(
            #     f"  Extracted {len(wav_data)} samples ({len(wav_data) / SAMPLE_RATE:.1f}s)"
            # )

            return torch.from_numpy(wav_data).float()

        finally:
            Path(tmp.name).unlink(missing_ok=True)


# ─────────────────────── Mode-Specific Processing ─────────────────────────


def _process_for_training(audio: torch.Tensor) -> torch.Tensor:
    """
    Process audio for training: trim to chunk boundaries.

    Training expects complete 3-second chunks (48000 samples each).
    Trims to 3s, 6s, or 9s based on duration.
    """
    num_samples = len(audio)
    min_samples = CHUNK_DURATIONS[1]  # 3 seconds minimum

    if num_samples < min_samples:
        raise ValueError(
            f"Audio too short for training: {num_samples} < {min_samples} samples"
        )

    # Determine how many complete chunks we can use
    if num_samples >= CHUNK_DURATIONS[3]:  # >= 9 seconds
        audio = audio[: CHUNK_DURATIONS[3]]

    elif num_samples >= CHUNK_DURATIONS[2]:  # >= 6 seconds
        audio = audio[: CHUNK_DURATIONS[2]]

    else:  # >= 3 seconds
        audio = audio[: CHUNK_DURATIONS[1]]

    return audio


def _process_for_inference(audio: torch.Tensor) -> torch.Tensor:
    """
    Process audio for inference: pad to chunk boundaries.

    Inference can handle any length but needs padding to chunk boundaries.
    """
    num_samples = len(audio)

    # Calculate padding needed
    remainder = num_samples % SAMPLES_PER_CHUNK
    if remainder != 0:
        pad_size = SAMPLES_PER_CHUNK - remainder
        audio = torch.nn.functional.pad(audio, (0, pad_size), mode="constant", value=0)

        num_chunks = len(audio) // SAMPLES_PER_CHUNK
        print(
            f"  Padded to {num_chunks} chunks ({len(audio)} samples, "
            f"added {pad_size} zeros)"
        )

    return audio


# ─────────────────────── STFT Processing ─────────────────────────


def _pad_for_stft(audio: torch.Tensor) -> torch.Tensor:
    """Pad audio tail for STFT to include last frame."""
    pad_tail = STFT_N_FFT - STFT_WIN_LENGTH
    if pad_tail > 0:
        audio = torch.nn.functional.pad(audio, (0, pad_tail))
    return audio


def _compute_stft(audio: torch.Tensor) -> torch.Tensor:
    """Compute complex STFT."""
    window = torch.hann_window(STFT_WIN_LENGTH, device=audio.device)

    return torch.stft(
        audio,
        n_fft=STFT_N_FFT,
        hop_length=STFT_HOP_LENGTH,
        win_length=STFT_WIN_LENGTH,
        window=window,
        return_complex=True,
        center=False,
        pad_mode="reflect",
    )


def _apply_power_law_compression(stft_complex: torch.Tensor) -> torch.Tensor:
    """
    Apply power-law compression to STFT.

    Returns tensor of shape [257, T, 2] with compressed real and imaginary parts.
    """
    real = stft_complex.real
    imag = stft_complex.imag

    # Apply compression while preserving sign
    real_compressed = torch.sign(real) * real.abs().pow(POWER_LAW_P)
    imag_compressed = torch.sign(imag) * imag.abs().pow(POWER_LAW_P)

    return torch.stack([real_compressed, imag_compressed], dim=-1)


# ─────────────────────── ISTFT Processing ─────────────────────────


# ─────────────────────── Utility Functions ─────────────────────────


def normalize_audio(audio: torch.Tensor) -> torch.Tensor:
    """Normalize audio to [-1, 1] range."""
    max_val = torch.max(torch.abs(audio))
    if max_val == 0:
        raise ValueError("Audio is silent (max amplitude is 0)")
    return audio / max_val


# ─────────────────────── High-Level Pipeline Functions ─────────────────────────


def add_noise_to_audio(
    waveform: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    noise_mixer = NoiseMixer(noise_root=Path("data/musan"))
    s1 = noise_mixer.mix_1s_noise(waveform)
    s2c = noise_mixer.mix_2s_clean(waveform)
    s2n = noise_mixer.mix_paper_2s_noise(waveform)
    return s1, s2c, s2n


def process_audio_with_noise(
    video_path: Path, noise_type: SampleT = SampleT.S1_NOISE
) -> List[torch.Tensor]:
    # Extract and process audio for inference
    audio = extract_audio(video_path, mode="inference")
    noised = add_noise_to_audio(audio)[
        0
        if noise_type == SampleT.S1_NOISE
        else 1
        if noise_type == SampleT.S2_CLEAN
        else 2
    ]

    # Compute STFT features
    # clean_stft = compute_stft_features(audio)
    noiseds_stft = compute_stft_features(noised)

    # Chunk for model input
    # clean_embeddings = chunk_stft_features(clean_stft)
    noised_embeddings = chunk_stft_features(noiseds_stft)
    return noised_embeddings


def save_audio(audio: torch.Tensor, video_path: Path, suffix: str = "mixed") -> None:
    debug_dir = Path("./debug")
    debug_dir.mkdir(exist_ok=True)

    # Use absolute path to avoid working directory issues
    path = debug_dir / f"{video_path.stem}_{suffix}.wav"

    # Ensure tensor is on CPU and detached from computation graph
    audio_cpu = audio.detach().cpu()

    # For torchaudio.save, tensor should be shape (channels, samples)
    # Our audio is mono (1D), so we need to add a channel dimension
    if audio_cpu.dim() == 1:
        audio_cpu = audio_cpu.unsqueeze(
            0
        )  # Add channel dimension: (samples,) -> (1, samples)

    print(f"Saving debug audio with shape: {audio_cpu.shape}")
    torchaudio.save(str(path), audio_cpu, SAMPLE_RATE)
    print(f"  Debug audio saved to: {path}")


def process_audio_for_training(
    video_path: Path, noise_mixer: NoiseMixer
) -> List[AudioChunk]:
    """
    Complete training pipeline: extract → trim → add noise → compute STFT → chunk.

    Args:
        video_path: Path to video file
        noise_mixer: NoiseMixer instance for adding noise to audio

    Returns:
        List of STFT chunks, each of shape [257, 298, 2]
    """
    # Extract and process audio for training
    clean_audio = extract_audio(video_path, mode="training")
    clean_stft = compute_stft_features(clean_audio)
    clean_embedding = chunk_stft_features(clean_stft)

    # Mix original clip's audio with noise - (1S+Noise, 2S_Clean, 2S+Noise)
    mixed_audio = noise_mixer.mix_with_selected_set_type(clean_audio)

    save_debug_audio = False
    if save_debug_audio:
        # Ensure debug directory exists
        save_audio(mixed_audio, video_path)

    mixed_stft = compute_stft_features(mixed_audio)
    mixed_embedding = chunk_stft_features(mixed_stft)

    # return Pairs of (clean_embedding, mixed_embedding)
    return [
        AudioChunk(clean_emb, mixed_emb)
        for clean_emb, mixed_emb in zip(clean_embedding, mixed_embedding)
    ]


def process_audio_for_inference(video_path: Path) -> List[torch.Tensor]:
    """
    Complete inference pipeline: extract → pad → compute STFT → chunk.

    Args:
        video_path: Path to video file

    Returns:
        Tuple of (list of STFT chunks, original audio length in samples)
    """
    # Extract and process audio for inference
    audio = extract_audio(video_path, mode="inference")
    pre_norm = audio.abs().max()
    audio = normalize_audio(audio)
    post_norm = audio.abs().max()
    print(
        f" process_audio_for_inference - normalized: pre={pre_norm:.4f}, post={post_norm:.4f} (max amplitude)"
    )

    # Compute STFT features
    features_stft = compute_stft_features(audio)

    # Chunk for model input
    embeddings = chunk_stft_features(features_stft)

    return embeddings


# ─────────────────────── Backward Compatibility ─────────────────────────


def extract_audio_from_clip(mp4_path: Path, trim: bool = True) -> torch.Tensor:
    """Legacy function for backward compatibility."""
    mode = "training" if trim else "inference"
    return extract_audio(mp4_path, mode=mode)


def stft_once(audio: torch.Tensor) -> torch.Tensor:
    """Legacy function for backward compatibility."""
    return compute_stft_features(audio)


def slice_stft_features(
    features: torch.Tensor, frames_per_chunk: int = 298
) -> List[torch.Tensor]:
    """Legacy function for backward compatibility."""
    return chunk_stft_features(features, frames_per_chunk)
