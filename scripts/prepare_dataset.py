#!/usr/bin/env python3
"""
Prepare AV-Speech training dataset from video clips.

Processes raw video clips into training-ready chunks with:
- Audio STFT features with noise augmentation
- Face embeddings (75 frames x 512 dims)
- 3-second aligned chunks for training
"""

import argparse
import time
from pathlib import Path
from typing import Dict

from tqdm import tqdm

from avspeech.preprocessing.clips_loader import DataLoader
from avspeech.preprocessing.pipeline import init_models, process_single_clip
from avspeech.utils.FaceManager import FaceManager
from avspeech.utils.noise_mixer import NoiseMixer
from avspeech.utils.structs import SampleT


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare AV-Speech training dataset from video clips",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Get the project root directory (parent of scripts/)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Required arguments
    parser.add_argument(
        "clips_dir", type=Path, help="Directory containing AVSpeech video clips"
    )
    parser.add_argument(
        "output_dir", type=Path, help="Output directory for processed dataset"
    )

    # Optional arguments with defaults relative to project root
    parser.add_argument(
        "--metadata",
        type=Path,
        default=project_root / "data" / "metadata.jsonl",
        help="Path to AVSpeech metadata file",
    )
    parser.add_argument(
        "--noise-dir",
        type=Path,
        default=project_root / "data" / "musan",
        help="Path to MUSAN noise dataset",
    )
    parser.add_argument(
        "--max-clips", type=int, default=0, help="Maximum clips to process (0=all)"
    )
    parser.add_argument(
        "--augmentation",
        type=SampleT,
        choices=list(SampleT),
        default=SampleT.S2_CLEAN,
        help="Type of noise augmentation",
    )

    return parser.parse_args()


def validate_environment(args) -> None:
    """Validate that all required files and directories exist."""
    errors = []

    if not args.clips_dir.exists():
        errors.append(f"Clips directory not found: {args.clips_dir}")

    if not args.metadata.exists():
        errors.append(f"Metadata file not found: {args.metadata}")

    if not args.noise_dir.exists():
        errors.append(f"Noise directory not found: {args.noise_dir}")
    else:
        noise_files = list(args.noise_dir.rglob("*.wav"))
        if not noise_files:
            errors.append(f"No .wav files found in: {args.noise_dir}")

    if errors:
        print("Setup errors found:")
        for error in errors:
            print(f"  ❌ {error}")
        print("\nPlease run: python scripts/setup_data.py")
        raise FileNotFoundError("Missing required data files")

    print("✓ Environment validated")


def sample_already_processed(clip_id: str, output_dir: Path) -> bool:
    """Check if a clip has already been processed and saved in that output folder."""
    clip_output_dir = output_dir / f"_{clip_id}_0"
    return clip_output_dir.exists()


def process_dataset(
    data_loader: DataLoader,
    output_dir: Path,
    face_manager: FaceManager,
    noise_mixer: NoiseMixer,
) -> Dict[str, int]:
    """Process all clips in the dataset."""

    stats = {"processed": 0, "failed": 0, "chunks": 0, "start_time": time.time()}

    progress_bar = tqdm(data_loader, desc="Processing")
    chunks_limit = 100_000_000  # Safety limit to avoid excessive processing

    for clip_data in progress_bar:
        try:
            # check if the folder for this clip already exists
            if sample_already_processed(clip_data.unique_clip_id, output_dir):
                print(f"Skipping already processed clip: {clip_data.unique_clip_id}")
                continue

            chunks_created = process_single_clip(
                face_manager=face_manager,
                clip_data=clip_data,
                noise_mixer=noise_mixer,
                output_dir=output_dir,
            )

            if chunks_created > 0:
                stats["processed"] += 1
                stats["chunks"] += chunks_created
            else:
                stats["failed"] += 1

            elapsed = time.time() - stats["start_time"]
            total_clips = stats["processed"] + stats["failed"]
            clips_per_min = (total_clips / elapsed * 60) if elapsed > 0 else 0
            avg_chunks = (
                stats["chunks"] / stats["processed"] if stats["processed"] > 0 else 0
            )

            progress_bar.set_postfix(
                {
                    "✓": stats["processed"],
                    "✗": stats["failed"],
                    "chunks": f"{stats['chunks']}/{chunks_limit}",
                    "avg": f"{avg_chunks:.1f}",
                    "rate": f"{clips_per_min:.1f}/min",
                }
            )

            if stats["chunks"] >= chunks_limit:
                print("\nChunks limit reached. Stopping processing.")
                break

        except KeyboardInterrupt:
            print("\nProcessing interrupted")
            break
        except Exception as e:
            print(
                f"\n process_dataset() - Error processing {clip_data.unique_clip_id}: {e}"
            )
            stats["failed"] += 1
            continue

    stats["elapsed"] = time.time() - stats["start_time"]
    return stats


def print_summary(stats: Dict[str, int]) -> None:
    """Print processing summary."""
    print("\n" + "=" * 50)
    print("PROCESSING COMPLETE")
    print("=" * 50)
    print(f"Processed: {stats['processed']} clips")
    print(f"Failed:    {stats['failed']} clips")
    print(f"Chunks:    {stats['chunks']} (3-second segments)")
    print(f"Time:      {stats['elapsed'] / 60:.1f} minutes")

    if stats["processed"] > 0:
        print(f"\nAverage chunks per clip: {stats['chunks'] / stats['processed']:.1f}")
        print(f"Total training duration: {stats['chunks'] * 3 / 60:.1f} minutes")


def main():
    # Parse arguments
    args = parse_arguments()

    # Validate environment
    validate_environment(args)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize components
    print("Initializing models...")
    data_loader = DataLoader(
        clips_dir=args.clips_dir,
        metadata_path=args.metadata,
        max_clips=args.max_clips,
    )
    face_manager = FaceManager()

    new_speech_root = Path(
        "/Users/jonatanvider/Documents/LookingToListenProject/av-speech-enhancement/scripts/data/speech_library"
    )
    new_noise_root = Path(
        "/Users/jonatanvider/Documents/LookingToListenProject/av-speech-enhancement/data/noise"
    )

    noise_mixer = NoiseMixer.from_audio_dirs(
        speech_root=new_speech_root,
        noise_root=new_noise_root,
        set_type=SampleT.S2_CLEAN,
    )

    # Process dataset
    print(f"Processing {len(data_loader)} clips...")
    stats = process_dataset(data_loader, args.output_dir, face_manager, noise_mixer)

    # Print summary
    print_summary(stats)


if __name__ == "__main__":
    main()


# Example usage:
# python prepare_dataset.py /Users/jonatanvider/Downloads/AVSpeech/clips/al-aq /Users/jonatanvider/Desktop/Look2Listen_Stuff/NEW_DB
#

# /Users/jonatanvider/Downloads/AVSpeech/clips/xau-xba
# /Users/jonatanvider/Desktop/Look2Listen_Stuff/Main_Dataset/S2N/S2N_45K


# demi clips folder - /Users/jonatanvider/Downloads/AVSpeech/clips/used/first_60k_aa2ai/xab
# demi output folder - /Users/jonatanvider/Desktop/Look2Listen_Stuff/tests_clips_junk