from typing import Tuple, List

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys
from tqdm import tqdm
import soundfile as sf

from avspeech.model.av_model import AudioVisualModel


class LookingToListenInference:

    def __init__(self, learned_weights_file : Path, device=None, verbose=False):
        self.device = set_device(device)
        self.model = AudioVisualModel().to(device)
        self.verbose = verbose
        checkpoint = torch.load(learned_weights_file, map_location=device)  # Load trained weights

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.print_verbose(f"✓ Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
        self.print_verbose(f"✓ Training loss: {checkpoint.get('loss', 'unknown'):.4f}")
        
    def print_verbose(self, message):
        """Print debug messages if verbose mode is enabled"""
        if self.verbose:
            print(message)


    def debug_audio_parameters(self, embeddings_folder):
        """Debug function to check STFT parameters match training"""

        self.print_verbose("🔍 Checking your preprocessing parameters...")

        # Load one chunk to inspect
        embeddings_path = Path(embeddings_folder)
        first_chunk = next(embeddings_path.iterdir())
        audio_path = first_chunk / 'audio' / 'audio_embs.pt'

        if audio_path.exists():
            audio_stft = torch.load(audio_path, map_location='cpu')
            self.print_verbose(f"STFT shape: {audio_stft.shape}")
            self.print_verbose(f"STFT dtype: {audio_stft.dtype}")

            # Based on paper: STFT with Hann window 25ms, hop 10ms, FFT 512
            # At 16kHz: 25ms = 400 samples, 10ms = 160 samples
            expected_freq_bins = 257  # = 257 (matches your shape!)
            expected_time_frames = 298  # ≈ 298 for 3s

            self.print_verbose(f"Expected shape for 3s audio: [{expected_freq_bins}, {expected_time_frames}, 2]")
            self.print_verbose(f"Your shape: {list(audio_stft.shape)}")

            if list(audio_stft.shape) == [257, 298, 2]:
                self.print_verbose("✅ STFT shape matches paper specifications!")
                return {
                        'n_fft'      : 512,
                        'hop_length' : 160,
                        'win_length' : 400,
                        'power_law_p': 0.3  # Add power law parameter from config
                }
            else:
                self.print_verbose("⚠️  STFT shape doesn't match expected - may need parameter adjustment")

        return None

    def stft_to_audio(self, stft_compressed : torch.Tensor, n_fft=512, hop_length=160, win_length=400, power_law_p=0.3):
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

        # self.print_verbose(f"DEBUG: Audio output shape: {audio.shape}")
        # self.print_verbose(f"DEBUG: Audio range: [{audio.min():.4f}, {audio.max():.4f}]")

        # Check for any issues
        if torch.isnan(audio).any():
            self.print_verbose("WARNING: Audio contains NaN values!")
        if torch.isinf(audio).any():
            self.print_verbose("WARNING: Audio contains infinite values!")

        # Normalize audio to prevent clipping
        max_val = torch.abs(audio).max()
        if max_val > 1.0:
            self.print_verbose(f"WARNING: Audio peak {max_val:.4f} > 1.0, normalizing to prevent clipping")
            audio = audio / max_val * 0.99  # Scale to 99% to ensure no clipping

        return audio

        
        

    def process_video(self, audio_embeddings : List[torch.Tensor], face_embeddings : List[torch.Tensor])\
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process entire video by processing all chunks"""

        """todo - validate input shapes and types"""

        # Debug STFT parameters first
        stft_params = {
                        'n_fft'      : 512,
                        'hop_length' : 160,
                        'win_length' : 400,
                        'power_law_p': 0.3  # Add power law parameter from config
                }

        # Process each chunk
        denoised_audio_chunks = []
        for audio_emb, face_emb in zip(audio_embeddings, face_embeddings):
            enhanced_stft, mask = self.enhance_chunk(audio_emb, face_emb)
            denoised_audio_chunks.append(enhanced_stft)

        # Concatenate all STFTs along time dimension [257, 298*5, 2]
        full_enhanced_stft = torch.cat(denoised_audio_chunks, dim=1)
        full_original_stft = torch.cat(audio_embeddings, dim=1)
        full_video_frames = torch.cat(face_embeddings, dim=1)

        self.print_verbose(f" full_enhanced_stft shape: {full_enhanced_stft.shape}")
        self.print_verbose(f" full_original_stft shape: {full_original_stft.shape}")
        self.print_verbose(f" full_video_frames shape: {full_video_frames.shape}")

        # Single ISTFT on concatenated spectrograms
        full_enhanced = self.stft_to_audio(full_enhanced_stft, **stft_params)
        full_original = self.stft_to_audio(full_original_stft, **stft_params)

        self.print_verbose(f"Video duration: {len(full_video_frames)/25:.1f} seconds")
        self.print_verbose(f"Enhanced audio duration: {len(full_enhanced)/16000:.1f} seconds")

        # Save results
        # self.save_results(full_enhanced, full_original, output_dir, Path(embeddings_folder).name)

        return full_enhanced, full_original, full_video_frames

    def enhance_chunk(self, audio_stft, face_embeddings):
        """Run inference on a single chunk"""
        # Purpose: This method processes one 3-second chunk of audio + face data through your trained model.

        with torch.no_grad():

            audio_stft = audio_stft.to(self.device)
            face_embeddings = face_embeddings.to(self.device)

            # Run model - outputs mask for target speaker
            mask = self.model(audio_stft, face_embeddings)

            # Apply mask to enhance target speaker
            enhanced_stft = audio_stft * mask

            return enhanced_stft.squeeze(0), mask.squeeze(0)



    def save_results(self, enhanced_audio, original_audio, output_dir, video_name):
        """Save enhanced audio and create visualization"""

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # Ensure audio is 1D and on CPU
        if len(enhanced_audio.shape) > 1:
            enhanced_audio = enhanced_audio.squeeze()
        if len(original_audio.shape) > 1:
            original_audio = original_audio.squeeze()

        # Move to CPU and convert to numpy
        enhanced_audio = enhanced_audio.cpu().numpy()
        original_audio = original_audio.cpu().numpy()

        # Save audio files using soundfile
        sf.write(
                output_dir / f"{video_name}_original.wav",
                original_audio,
                16000
        )

        sf.write(
                output_dir / f"{video_name}_enhanced.wav",
                enhanced_audio,
                16000
        )

        # Create visualization
        create_visualization(
                torch.from_numpy(enhanced_audio),
                torch.from_numpy(original_audio),
                output_dir,
                video_name
        )

        self.print_verbose(f"✓ Results saved to {output_dir}")
        self.print_verbose(f"   • {video_name}_original.wav ({len(original_audio) / 16000:.1f}s)")
        self.print_verbose(f"   • {video_name}_enhanced.wav ({len(enhanced_audio) / 16000:.1f}s)")
        self.print_verbose(f"   • {video_name}_comparison.png")


def set_device(device=None):
    """Auto-detect the best device for inference"""
    if device is None:
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'

    print(f"Inference Engine Using device: {device}")
    return device


def load_embeddings_from_folder(embeddings_folder : Path)->Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Load all preprocessed embeddings from your folder structure"""

    embeddings_path = Path(embeddings_folder)

    # Find all chunk folders (sorted by number)
    chunk_folders = sorted([f for f in embeddings_path.iterdir()
                            if f.is_dir() and f.name.split('_')[-1].isdigit()],
                           key=lambda x: int(x.name.split('_')[-1]))

    audio_chunks = []
    face_chunks = []

    print(f"Found {len(chunk_folders)} chunks to process")

    for chunk_folder in chunk_folders:
        # Load audio embeddings (STFT)
        audio_path = chunk_folder / 'audio' / 'audio_embs.pt'
        face_path = chunk_folder / 'face' / 'face_embs.pt'

        if audio_path.exists() and face_path.exists():
            audio_emb = torch.load(audio_path, map_location='cpu')
            face_emb = torch.load(face_path, map_location='cpu')

            # Ensure correct shapes
            print(f"Chunk {chunk_folder.name}: Audio {audio_emb.shape}, Face {face_emb.shape}")

            audio_chunks.append(audio_emb)
            face_chunks.append(face_emb)
        else:
            print(f"⚠️  Missing embeddings in {chunk_folder.name}")

    return audio_chunks, face_chunks


def create_visualization(enhanced_audio, original_audio, output_dir, video_name):
    """Create before/after spectrogram comparison"""

    fig, axes = plt.subplots(2, 1, figsize=(15, 8))

    # Create window to match your preprocessing
    window = torch.hann_window(400)  # 25ms window at 16kHz

    # Original mixed audio spectrogram
    original_stft = torch.stft(
            original_audio,
            n_fft=512,
            hop_length=160,
            win_length=400,
            window=window,
            return_complex=True,
            center=False  # Match your preprocessing
    )
    original_mag = torch.log(torch.abs(original_stft) + 1e-8)

    # Enhanced audio spectrogram
    enhanced_stft = torch.stft(
            enhanced_audio,
            n_fft=512,
            hop_length=160,
            win_length=400,
            window=window,
            return_complex=True,
            center=False  # Match your preprocessing
    )
    enhanced_mag = torch.log(torch.abs(enhanced_stft) + 1e-8)

    # Plot spectrograms
    im1 = axes[0].imshow(original_mag.numpy(), aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title('Original Mixed Audio (Speaker + Noise/Interference)')
    axes[0].set_ylabel('Frequency Bin')

    im2 = axes[1].imshow(enhanced_mag.numpy(), aspect='auto', origin='lower', cmap='viridis')
    axes[1].set_title('Enhanced Audio (Target Speaker Isolated)')
    axes[1].set_ylabel('Frequency Bin')
    axes[1].set_xlabel('Time Frame')

    # Add colorbars
    plt.colorbar(im1, ax=axes[0], label='Log Magnitude')
    plt.colorbar(im2, ax=axes[1], label='Log Magnitude')

    plt.tight_layout()
    plt.savefig(output_dir / f"{video_name}_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
