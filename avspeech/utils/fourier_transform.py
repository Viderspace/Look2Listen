import torch


def stft_to_audio(stft_compressed: torch.Tensor, n_fft=512, hop_length=160, win_length=400, power_law_p=0.3):
    """Convert compressed STFT back to audio waveform

    Args:
        stft_compressed: STFT with power-law compression applied [freq, time, 2]
        n_fft: FFT size (512)
        hop_length: Hop length in samples (160)
        win_length: Window length in samples (400)
        power_law_p: Power law compression parameter (0.3 from your config)
    """
    if len(stft_compressed.shape) != 3 or stft_compressed.shape[-1] != 2:
        raise ValueError("Expected input shape [freq, time, 2] with compressed values")

    # unpacking compressed real and imaginary parts
    real_compressed = stft_compressed[..., 0]
    imag_compressed = stft_compressed[..., 1]

    # Invert the power-law - for decompression
    inverse_power = 1.0 / power_law_p
    real_decompressed = torch.sign(real_compressed) * torch.abs(real_compressed).pow(inverse_power)
    imag_decompressed = torch.sign(imag_compressed) * torch.abs(imag_compressed).pow(inverse_power)

    # Convert to complex tensor
    stft_complex = torch.complex(real_decompressed, imag_decompressed)

    # Create window
    window = torch.hann_window(win_length, device=stft_complex.device)

    # ISTFT with center=True (works reliably across devices)
    # and avoids device-specific issues. Edge artifacts are negligible.
    audio = torch.istft(
            stft_complex,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=True,  # Always use center=True for reliability
            normalized=False,
            onesided=True,
            length=None
    )

    print(f"DEBUG: Audio output shape: {audio.shape}")
    print(f"DEBUG: Audio range: [{audio.min():.4f}, {audio.max():.4f}]")

    # Check for any issues
    if torch.isnan(audio).any():
        print("WARNING: Audio contains NaN values!")
    if torch.isinf(audio).any():
        print("WARNING: Audio contains infinite values!")

    # Normalize audio to prevent clipping
    max_val = torch.abs(audio).max()
    if max_val > 1.0:
        print(f"WARNING: Audio peak {max_val:.4f} > 1.0, normalizing to prevent clipping")
        audio = audio / max_val * 0.99  # Scale to 99% to ensure no clipping

    return audio


