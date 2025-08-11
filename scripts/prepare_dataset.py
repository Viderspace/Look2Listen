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

from avspeech.preprocessing.pipeline import process_single_clip, init_models
from avspeech.preprocessing.clips_loader import DataLoader
from avspeech.utils.face_embedder import FaceEmbedder
from avspeech.utils.noise_mixer import NoiseMixer
from avspeech.utils.structs import SampleT
from avspeech.utils.constants import SAMPLE_RATE


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
            description="Prepare AV-Speech training dataset from video clips",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Get the project root directory (parent of scripts/)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Required arguments
    parser.add_argument(
            "clips_dir",
            type=Path,
            help="Directory containing AVSpeech video clips"
    )
    parser.add_argument(
            "output_dir",
            type=Path,
            help="Output directory for processed dataset"
    )

    # Optional arguments with defaults relative to project root
    parser.add_argument(
            "--metadata",
            type=Path,
            default=project_root / "data" / "metadata.jsonl",
            help="Path to AVSpeech metadata file"
    )
    parser.add_argument(
            "--noise-dir",
            type=Path,
            default=project_root / "data" / "musan",
            help="Path to MUSAN noise dataset"
    )
    parser.add_argument(
            "--max-clips",
            type=int,
            default=0,
            help="Maximum clips to process (0=all)"
    )
    parser.add_argument(
            "--augmentation",
            type=SampleT,
            choices=list(SampleT),
            default=SampleT.S1_NOISE,
            help="Type of noise augmentation"
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

    print(f"✓ Environment validated")

def sample_already_processed(clip_id: str, output_dir: Path) -> bool:
    """Check if a clip has already been processed and saved in that output folder."""
    clip_output_dir = output_dir / f"_{clip_id}_0"
    return  clip_output_dir.exists()

def process_dataset(
        data_loader: DataLoader,
        output_dir: Path,
        face_embedder: FaceEmbedder,
        noise_mixer: NoiseMixer
) -> Dict[str, int]:
    """Process all clips in the dataset."""

    stats = {
            "processed" : 0,
            "failed"    : 0,
            "chunks"    : 0,
            "start_time": time.time()
    }

    for clip_data in tqdm(data_loader, desc="Processing"):
        try:
            # check if the folder for this clip already exists
            if sample_already_processed(clip_data.unique_clip_id, output_dir):
                print(f"Skipping already processed clip: {clip_data.unique_clip_id}")
                continue


            chunks_created = process_single_clip(
                    face_embedder=face_embedder,
                    clip_data=clip_data,
                    noise_mixer=noise_mixer,
                    output_dir=output_dir,
            )

            if chunks_created > 0:
                stats["processed"] += 1
                stats["chunks"] += chunks_created
            else:
                stats["failed"] += 1

        except KeyboardInterrupt:
            print("\nProcessing interrupted")
            break
        except Exception as e:
            print(f"\nError processing {clip_data.unique_clip_id}: {e}")
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

    if stats['processed'] > 0:
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
    face_embedder = init_models()
    noise_mixer = NoiseMixer(
            noise_root=args.noise_dir,
            set_type=args.augmentation,
            sample_rate=SAMPLE_RATE
    )

    # Process dataset
    print(f"Processing {len(data_loader)} clips...")
    stats = process_dataset(data_loader, args.output_dir, face_embedder, noise_mixer)

    # Print summary
    print_summary(stats)


if __name__ == "__main__":
    main()


# Example usage:
# python scripts/prepare_dataset.py --max-clips 2 --augmentation 2s_clean /Users/jonatanvider/Downloads/AVSpeech/clips/al-aq/al-an /Users/jonatanvider/Downloads/AVSpeech/New_2s_Datasets/2s_clean_al_an