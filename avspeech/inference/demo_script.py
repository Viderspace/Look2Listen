from typing import Tuple, List
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import soundfile as sf

from avspeech.model.av_model import AudioVisualModel


class LookingToListenInference:

    def __init__(self, learned_weights_file: Path, device=None, verbose=False):
        self.device = self.set_device(device)
        self.model = AudioVisualModel().to(self.device)
        self.verbose = verbose
        checkpoint = torch.load(learned_weights_file, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def set_device(self, device=None):
        """Auto-detect the best device for inference"""
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        print(f"Using device: {device}")
        return device

    def print_verbose(self, message):
        if self.verbose:
            print(message)

    def stft_to_audio(self, stft_compressed: torch.Tensor, n_fft=512, hop_length=160,
                      win_length=400, power_law_p=0.3):
        """Convert compressed STFT back to audio waveform"""
        if len(stft_compressed.shape) != 3 or stft_compressed.shape[-1] != 2:
            raise ValueError("Expected input shape [freq, time, 2]")

        # Unpack compressed real and imaginary parts
        real_compressed = stft_compressed[..., 0]
        imag_compressed = stft_compressed[..., 1]

        # Invert power-law compression
        inverse_power = 1.0 / power_law_p
        real_decompressed = torch.sign(real_compressed) * torch.abs(real_compressed).pow(inverse_power)
        imag_decompressed = torch.sign(imag_compressed) * torch.abs(imag_compressed).pow(inverse_power)

        # Convert to complex tensor
        stft_complex = torch.complex(real_decompressed, imag_decompressed)

        # Create window and perform ISTFT
        window = torch.hann_window(win_length, device=stft_complex.device)
        audio = torch.istft(
                stft_complex,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                center=True,
                normalized=False,
                onesided=True,
                length=None
        )

        # Normalize if needed
        max_val = torch.abs(audio).max()
        if max_val > 1.0:
            self.print_verbose(f"Normalizing audio (peak: {max_val:.4f})")
            audio = audio / max_val * 0.99

        return audio

    def enhance_chunk(self, audio_stft, face_embeddings):
        """Run inference on a single chunk"""
        with torch.no_grad():
            audio_stft = audio_stft.to(self.device)
            face_embeddings = face_embeddings.to(self.device)

            # Run model - outputs mask for target speaker
            mask = self.model(audio_stft, face_embeddings)

            # Apply mask to enhance target speaker
            enhanced_stft = audio_stft * mask

            return enhanced_stft.squeeze(0), mask.squeeze(0)

    def process_video(self, embeddings_folder: Path, output_dir: Path):
        """Process entire video from embeddings folder"""
        print(f"\n📂 Loading embeddings from: {embeddings_folder}")

        # Load embeddings
        audio_chunks, face_chunks = self.load_embeddings(embeddings_folder)

        if not audio_chunks:
            print("❌ No valid chunks found!")
            print("   Make sure your folder contains subfolders with:")
            print("   - audio/mixture_embs.pt (or audio_embs.pt)")
            print("   - face/face_embs.pt")
            return None, None

        print(f"✓ Loaded {len(audio_chunks)} chunks")

        # STFT parameters
        stft_params = {
                'n_fft'      : 512,
                'hop_length' : 160,
                'win_length' : 400,
                'power_law_p': 0.3
        }

        # Process each chunk
        print("\n🔊 Enhancing audio...")
        denoised_chunks = []
        for i, (audio_emb, face_emb) in enumerate(zip(audio_chunks, face_chunks)):
            enhanced_stft, _ = self.enhance_chunk(audio_emb, face_emb)
            denoised_chunks.append(enhanced_stft)
            print(f"  Chunk {i + 1}/{len(audio_chunks)} processed")

        # Concatenate all STFTs
        full_enhanced_stft = torch.cat(denoised_chunks, dim=1)
        full_original_stft = torch.cat(audio_chunks, dim=1)

        print(f"\n🎵 Converting to audio...")
        print(f"  Enhanced STFT shape: {full_enhanced_stft.shape}")
        print(f"  Original STFT shape: {full_original_stft.shape}")

        # Convert to audio
        enhanced_audio = self.stft_to_audio(full_enhanced_stft, **stft_params)
        original_audio = self.stft_to_audio(full_original_stft, **stft_params)

        # Save results
        self.save_results(enhanced_audio, original_audio, output_dir, embeddings_folder.name)

        return enhanced_audio, original_audio

    def load_embeddings(self, embeddings_folder: Path):
        """Load all preprocessed embeddings from folder"""
        # Get all subdirectories (your structure has video_id_chunk_id folders)
        chunk_folders = sorted([f for f in embeddings_folder.iterdir() if f.is_dir()])

        audio_chunks = []
        face_chunks = []

        for chunk_folder in chunk_folders:
            # Try different audio file names - your structure has mixture_embs.pt
            audio_path_mixture = chunk_folder / 'audio' / 'mixture_embs.pt'
            audio_path_clean = chunk_folder / 'audio' / 'clean_embs.pt'
            audio_path_default = chunk_folder / 'audio' / 'audio_embs.pt'
            face_path = chunk_folder / 'face' / 'face_embs.pt'

            # Check which audio file exists and use mixture (the mixed audio to enhance)
            audio_path = None
            if audio_path_mixture.exists():
                audio_path = audio_path_mixture
                print(f"  Using mixture audio from {chunk_folder.name}")
            elif audio_path_default.exists():
                audio_path = audio_path_default
                print(f"  Using default audio from {chunk_folder.name}")

            if audio_path and audio_path.exists() and face_path.exists():
                audio_emb = torch.load(audio_path, map_location='cpu')
                face_emb = torch.load(face_path, map_location='cpu')

                # Fix dimensions for both audio and face
                # Audio should be [batch, freq, time, 2]
                if audio_emb.dim() == 3:
                    audio_emb = audio_emb.unsqueeze(0)  # Add batch dimension

                # Face embeddings: [75, 512] -> [batch, 512, 75]
                if face_emb.dim() == 2:
                    # face_emb is [75, 512], need [1, 512, 75]
                    face_emb = face_emb.transpose(0, 1).unsqueeze(0)  # [512, 75] -> [1, 512, 75]
                elif face_emb.dim() == 3 and face_emb.shape[1] == 75:
                    # If it's [1, 75, 512], transpose to [1, 512, 75]
                    face_emb = face_emb.transpose(1, 2)

                print(f"    Audio shape: {audio_emb.shape}, Face shape: {face_emb.shape}")

                audio_chunks.append(audio_emb)
                face_chunks.append(face_emb)
            else:
                if not face_path.exists():
                    print(f"  ⚠️ Missing face embeddings in {chunk_folder.name}")
                if audio_path is None or not audio_path.exists():
                    print(f"  ⚠️ Missing audio embeddings in {chunk_folder.name}")

        return audio_chunks, face_chunks

    def save_results(self, enhanced_audio, original_audio, output_dir: Path, video_name: str):
        """Save enhanced audio and create visualization"""
        output_dir.mkdir(exist_ok=True, parents=True)

        # Ensure audio is 1D and on CPU
        enhanced_audio = enhanced_audio.squeeze().cpu().numpy()
        original_audio = original_audio.squeeze().cpu().numpy()

        # Save audio files
        original_path = output_dir / f"{video_name}_original.wav"
        enhanced_path = output_dir / f"{video_name}_enhanced.wav"

        sf.write(original_path, original_audio, 16000)
        sf.write(enhanced_path, enhanced_audio, 16000)

        print(f"\n✅ Results saved:")
        print(f"  📁 {output_dir}")
        print(f"  🔊 {original_path.name} ({len(original_audio) / 16000:.1f}s)")
        print(f"  🎯 {enhanced_path.name} ({len(enhanced_audio) / 16000:.1f}s)")

        # Create visualization
        self.create_visualization(enhanced_audio, original_audio, output_dir, video_name)
        print(f"  📊 {video_name}_comparison.png")

    def create_visualization(self, enhanced_audio, original_audio, output_dir: Path, video_name: str):
        """Create before/after spectrogram comparison"""
        fig, axes = plt.subplots(2, 1, figsize=(15, 8))

        # Convert back to tensor for STFT
        enhanced_audio = torch.from_numpy(enhanced_audio)
        original_audio = torch.from_numpy(original_audio)

        window = torch.hann_window(400)

        # Compute spectrograms
        original_stft = torch.stft(
                original_audio, n_fft=512, hop_length=160,
                win_length=400, window=window,
                return_complex=True, center=False
        )
        original_mag = torch.log(torch.abs(original_stft) + 1e-8)

        enhanced_stft = torch.stft(
                enhanced_audio, n_fft=512, hop_length=160,
                win_length=400, window=window,
                return_complex=True, center=False
        )
        enhanced_mag = torch.log(torch.abs(enhanced_stft) + 1e-8)

        # Plot
        im1 = axes[0].imshow(original_mag.numpy(), aspect='auto', origin='lower', cmap='viridis')
        axes[0].set_title('Original Mixed Audio')
        axes[0].set_ylabel('Frequency Bin')

        im2 = axes[1].imshow(enhanced_mag.numpy(), aspect='auto', origin='lower', cmap='viridis')
        axes[1].set_title('Enhanced Audio (Target Speaker)')
        axes[1].set_ylabel('Frequency Bin')
        axes[1].set_xlabel('Time Frame')

        plt.colorbar(im1, ax=axes[0], label='Log Magnitude')
        plt.colorbar(im2, ax=axes[1], label='Log Magnitude')

        plt.tight_layout()
        plt.savefig(output_dir / f"{video_name}_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()


def main():
    """
    ==========================================
    QUICK DEMO - MODIFY THESE PATHS
    ==========================================
    """

    # ⚡ HARDCODED PATHS - CHANGE THESE TO YOUR PATHS
    CHECKPOINT_PATH = "/Users/jonatanvider/Desktop/Look2Listen_Stuff/checkpoints/checkpoint_epoch_32.pt"
    EMBEDDINGS_FOLDER = "/Users/jonatanvider/Desktop/Look2Listen_Stuff/small_test_datasets/validation/2s_clean"
    OUTPUT_DIR = "/Users/jonatanvider/Desktop/Look2Listen_Stuff/small_test_datasets/validation/2s_clean/inference"

    # Example paths (replace with your actual paths):
    # CHECKPOINT_PATH = "/home/user/models/ltl_epoch_50.pth"
    # EMBEDDINGS_FOLDER = "/home/user/data/preprocessed/sample_video_001"
    # OUTPUT_DIR = "/home/user/results/enhanced_audio"

    # ==========================================

    # Convert to Path objects
    checkpoint_path = Path(CHECKPOINT_PATH)
    embeddings_folder = Path(EMBEDDINGS_FOLDER)
    output_dir = Path(OUTPUT_DIR)

    # Validate paths
    print("🔍 Checking paths...")
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    if not embeddings_folder.exists():
        print(f"❌ Embeddings folder not found: {embeddings_folder}")
        return

    print(f"✓ Checkpoint: {checkpoint_path}")
    print(f"✓ Embeddings: {embeddings_folder}")
    print(f"✓ Output dir: {output_dir}")

    # Initialize model
    print("\n🚀 Initializing model...")
    inference = LookingToListenInference(
            learned_weights_file=checkpoint_path,
            device=None,  # Auto-detect GPU/CPU
            verbose=True  # Show debug messages
    )

    # Process video
    print("\n🎬 Processing video...")
    result = inference.process_video(embeddings_folder, output_dir)

    if result[0] is not None:
        print("\n✨ Demo complete! Check the output folder for:")
        print("  - Original audio (mixed/noisy)")
        print("  - Enhanced audio (target speaker isolated)")
        print("  - Spectrogram comparison visualization")
    else:
        print("\n❌ Processing failed. Please check your embeddings structure.")


if __name__ == "__main__":
    main()