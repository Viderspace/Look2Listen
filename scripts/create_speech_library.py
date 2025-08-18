#!/usr/bin/env python3
import os
from pathlib import Path
import subprocess
from typing import Optional
from avspeech.utils.constants import SAMPLE_RATE
from tqdm import tqdm
import argparse

SPEECH_LIB_ROOT = Path("data/speech_library")
THUMB_DIR_NAMES = {".@__thumb", "@eaDir"}  # common thumbnail/metadata dirs


def _skip_dir(p: Path) -> bool:
    return any(part.startswith(".") or part in THUMB_DIR_NAMES for part in p.parts)

def extract_audio_from_video(video_path: Path, output_dir: Path, sample_rate: int = 16000)-> bool:
    """Extract mono audio from video file using ffmpeg"""
    # Create output filename (same name, .wav extension)
    output_path = output_dir / f"{video_path.stem}.wav"

    # Skip if already processed
    if output_path.exists():
        return True

    # FFmpeg command for mono 16kHz extraction
    cmd = [
            'ffmpeg',
            '-i', str(video_path),
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # 16-bit PCM
            '-ar', str(sample_rate),  # Sample rate
            '-ac', '1',  # Mono
            '-y',  # Overwrite
            str(output_path),
            '-loglevel', 'error'  # Quiet output
    ]

    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed: {video_path.name} - {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Extract audio from AVSpeech videos")
    parser.add_argument("video_dir", type=Path, help="Directory containing video files")
    parser.add_argument("--output_dir_name", type=str, default=None, help="Output directory custom name (default = 'data/speech_library/input_folder_name')")
    args = parser.parse_args()

    # If specific output folder name is given, use it, otherwise use the video directory name as subdir
    output_subdir = args.output_dir_name if args.output_dir_name else args.video_dir.stem
    output_path = SPEECH_LIB_ROOT / output_subdir

    output_path.mkdir(parents=True, exist_ok=True)


    # Find all video files recursively args.video_dir.rglob("*.mp4")
    video_files = sorted(f for f in args.video_dir.rglob("*.mp4") if f.is_file() and not _skip_dir(f.parent))
    print(f"Found {len(video_files)} videos to process under {args.video_dir}")

    # Process each video
    success = 0
    for video_path in tqdm(video_files, desc="Extracting audio"):
        if extract_audio_from_video(video_path, output_path, SAMPLE_RATE):
            success += 1

    print(f"\nComplete! Extracted {success}/{len(video_files)} audio files")
    print(f"Audio files saved to: {output_path}")


if __name__ == "__main__":
    main()