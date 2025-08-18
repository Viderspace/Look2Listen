"""
This script runs the full pipeline of processing a video clip to do noise isolation

Ingredients:
- Video file (MP4)
- Inference model for audio denoising (.pt weights)
- Optional face hint (x, y) coordinates for better face detection - if not provided, ask the user to provide a hint
- Output directory to save the processed video

The process includes:
1. Opening a video file and extract the frames in the original frame rate and resolution (cap cv)
2. Launch a ClipProcessor instance
3. provide a face hint to the ClipProcessor instance
4. applying the inference model, via ClipProcessor, receiving the enhanced and original audio waveforms
5. trimming the tails of the audio waveforms (padded for inference with 0 values) to the original video length
6. using the enhanced audio and the original video frames, combining them into a video container
7. using the original audio and the original video frames, combining them into a video container
8. export the 2 video containers as mp4's to the output directory
9. (Optional, later) analyze and visualize the result metrics and export them as well
"""

import os
import sys
from pathlib import Path
from argparse import ArgumentParser
from typing import Tuple, Optional
import torch
import torchaudio
import cv2
import numpy as np

# Enable MPS fallback for unsupported operations (like istft on Apple Silicon)
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

from avspeech.utils.ClipProcessor import ClipProcessor


def get_video_info(video_path: Path) -> dict:
    """Get basic video information for trimming purposes."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0

    cap.release()
    return {
        'fps': fps,
        'frame_count': frame_count,
        'duration': duration
    }


def trim_audio_to_video_length(audio_waveform: torch.Tensor, video_duration: float, sample_rate: int = 16000) -> torch.Tensor:
    """Trim audio waveform to match original video duration."""
    target_length = int(video_duration * sample_rate)
    if audio_waveform.size(-1) > target_length:
        return audio_waveform[..., :target_length]
    return audio_waveform


def save_audio_video_combined(video_path: Path, audio_waveform: torch.Tensor, output_path: Path,
                              sample_rate: int = 16000):
    """Combine video frames with new audio and save as MP4."""
    import ffmpeg

    # Save audio as temporary WAV file
    temp_audio_path = output_path.with_suffix('.temp.wav')
    torchaudio.save(str(temp_audio_path), audio_waveform.unsqueeze(0), sample_rate)

    try:
        # Create ffmpeg inputs
        video_input = ffmpeg.input(str(video_path))
        audio_input = ffmpeg.input(str(temp_audio_path))

        # Method 1: Using separate map calls with shortest flag
        stream = ffmpeg.output(
                video_input['v'],  # Video stream from first input
                audio_input['a'],  # Audio stream from second input
                str(output_path),
                vcodec='copy',  # Copy video without re-encoding
                acodec='aac',  # Re-encode audio as AAC
                audio_bitrate='320k',
                shortest=None  # Match shortest stream
        )

        stream.overwrite_output().run(capture_stdout=True, capture_stderr=True)
        print(f"Successfully saved: {output_path}")

    except ffmpeg.Error as e:
        print("FFmpeg error:", e.stderr.decode())
        print("FFmpeg stdout:", e.stdout.decode())
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise
    finally:
        # Clean up temporary audio file
        if temp_audio_path.exists():
            temp_audio_path.unlink()
def interactive_face_hint_selection(clip_processor: ClipProcessor) -> Tuple[float, float]:
    """Interactively get face hint from user."""
    print(f"\nFace hint selection for: {clip_processor.video_path.name}")
    print("You need to provide coordinates where the target speaker's face is located.")
    print("\nCoordinate System Guide:")
    print("  • x=0.0 is LEFT edge,   x=1.0 is RIGHT edge")
    print("  • y=0.0 is TOP edge,    y=1.0 is BOTTOM edge")
    print("  • Center of frame is:   x=0.5, y=0.5")
    print("\nExamples:")
    print("  • Face in center:       0.5,0.5")
    print("  • Face in upper-right:  0.8,0.3")
    print("  • Face in lower-left:   0.2,0.7")

    while True:
        try:
            hint_input = input("\nEnter face hint as 'x,y' (e.g., '0.5,0.3'): ").strip()

            if not hint_input:
                print("Please enter coordinates in format 'x,y'")
                continue

            if ',' not in hint_input:
                print("Please use comma-separated format: 'x,y'")
                continue

            parts = hint_input.split(',')
            if len(parts) != 2:
                print("Please enter exactly two coordinates: 'x,y'")
                continue

            x = float(parts[0].strip())
            y = float(parts[1].strip())

            if not (0.0 <= x <= 1.0) or not (0.0 <= y <= 1.0):
                print("Both coordinates must be between 0.0 and 1.0. Please try again.")
                continue

            print(f"Face hint set: x={x} (0.0=left, 1.0=right), y={y} (0.0=top, 1.0=bottom)")
            return (x, y)

        except ValueError:
            print("Invalid input. Please enter numeric values in format 'x,y' (e.g., '0.5,0.3')")
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            sys.exit(1)


def interactive_input():
    """Get all required inputs interactively from user."""
    print("=== Interactive Video Enhancement Setup ===")

    # Get video path
    while True:
        video_input = input("\nEnter path to video file (.mp4): ").strip().strip('"\'')
        if not video_input:
            print("Video path cannot be empty.")
            continue

        video_path = Path(video_input)
        if not video_path.exists():
            print(f"Video file not found: {video_path}")
            continue

        if not video_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
            print("Please provide a valid video file (.mp4, .avi, .mov, .mkv)")
            continue

        break

    # Get model checkpoint path
    while True:
        checkpoint_input = input("\nEnter path to model checkpoint file (.pt): ").strip().strip('"\'')
        if not checkpoint_input:
            print("Checkpoint path cannot be empty.")
            continue

        checkpoint_path = Path(checkpoint_input)
        if not checkpoint_path.exists():
            print(f"Checkpoint file not found: {checkpoint_path}")
            continue

        if not checkpoint_path.suffix.lower() == '.pt':
            print("Please provide a valid PyTorch checkpoint file (.pt)")
            continue

        break

    # Get output directory
    while True:
        output_input = input("\nEnter output directory: ").strip().strip('"\'')
        if not output_input:
            print("Output directory cannot be empty.")
            continue

        output_dir = Path(output_input)

        # Create directory if it doesn't exist
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            break
        except Exception as e:
            print(f"Cannot create output directory: {e}")
            continue

    return video_path, checkpoint_path, output_dir


def main():
    parser = ArgumentParser(description="Video Audio Enhancement Pipeline")
    parser.add_argument("--video_path", type=str, help="Path to input video file")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint (.pt file)")
    parser.add_argument("--output_dir", type=str, help="Output directory for processed videos")
    parser.add_argument("--face_hint", type=str, help="Face hint coordinates as 'x,y' (e.g., '0.5,0.3')")

    args = parser.parse_args()

    # Get inputs either from arguments or interactively
    if args.video_path and args.checkpoint and args.output_dir:
        video_path = Path(args.video_path)
        checkpoint_path = Path(args.checkpoint)
        output_dir = Path(args.output_dir)

        # Validate paths
        if not video_path.exists():
            print(f"Error: Video file not found: {video_path}")
            sys.exit(1)
        if not checkpoint_path.exists():
            print(f"Error: Checkpoint file not found: {checkpoint_path}")
            sys.exit(1)

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        video_path, checkpoint_path, output_dir = interactive_input()

    # Parse face hint if provided
    face_hint = None
    if args.face_hint:
        try:
            x, y = map(float, args.face_hint.split(','))
            if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                face_hint = (x, y)
            else:
                print("Warning: Face hint coordinates must be between 0.0 and 1.0. Will ask interactively.")
        except ValueError:
            print("Warning: Invalid face hint format. Expected 'x,y'. Will ask interactively.")

    print(f"\n=== Processing Video: {video_path.name} ===")

    try:
        # Step 1 & 2: Initialize ClipProcessor (automatically extracts frames and audio)
        print("1. Loading video and extracting frames/audio...")
        from avspeech.utils.structs import SampleT
        clip_processor = ClipProcessor(video_path)

        if not clip_processor.is_video_loaded():
            print("Error: Failed to load video. Please check the file format and integrity.")
            sys.exit(1)

        print(f"   ✓ Extracted {len(clip_processor.video_frames)} video frames")
        print(f"   ✓ Extracted {len(clip_processor.audio_embeddings)} audio chunks")

        # Step 3: Set face hint
        print("\n2. Setting face hint...")
        if face_hint:
            clip_processor.set_face_hint(face_hint)
            print(f"   ✓ Using provided face hint: {face_hint}")
        else:
            face_hint = interactive_face_hint_selection(clip_processor)
            clip_processor.set_face_hint(face_hint)
            print(f"   ✓ Face hint set: {face_hint}")

        # Verify video is ready
        if not clip_processor.is_video_ready():
            print("Error: Video processing failed. Cannot proceed with inference.")
            sys.exit(1)

        print(f"   ✓ Generated {len(clip_processor.face_embeddings)} face embeddings")

        # Step 4: Apply inference model
        print("\n3. Applying inference model...")
        enhanced_audio, original_audio = clip_processor.apply_inference(checkpoint_path)
        print("   ✓ Audio enhancement completed")

        # Step 5: Trim audio to original video length
        print("\n4. Trimming audio to original video length...")
        video_info = get_video_info(video_path)
        enhanced_audio_trimmed = trim_audio_to_video_length(enhanced_audio, video_info['duration'])
        original_audio_trimmed = trim_audio_to_video_length(original_audio, video_info['duration'])
        print(f"   ✓ Trimmed audio to {video_info['duration']:.2f} seconds")

        # Steps 6-8: Combine and export videos
        print("\n5. Exporting processed videos...")

        base_name = video_path.stem
        epoch_name = checkpoint_path.stem.split('_')[-1]  # Get the last part of the checkpoint name
        enhanced_output = output_dir / f"{base_name}_enhanced_{epoch_name}.mp4"
        original_output = output_dir / f"{base_name}_original.mp4"

        # Save enhanced version
        print("   Saving enhanced video...")
        save_audio_video_combined(video_path, enhanced_audio_trimmed, enhanced_output)

        # Save original version (for comparison)
        print("   Saving original comparison video...")
        save_audio_video_combined(video_path, original_audio_trimmed, original_output)

        print("\n🎉 Processing completed successfully!")
        print(f"📁 Output files saved to: {output_dir}")
        print(f"   • Enhanced: {enhanced_output.name}")
        print(f"   • Original: {original_output.name}")

    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()