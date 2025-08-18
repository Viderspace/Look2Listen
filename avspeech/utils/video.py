from pathlib import Path
from typing import List, Optional, Set, Tuple

import cv2
import numpy as np
import torch

from avspeech.utils.face_embedder import FaceEmbedder

# Constants
TARGET_FPS = 25
FRAMES_PER_CHUNK = 75  # 3 seconds at 25fps
MAX_TRAINING_FRAMES = 225  # 9 seconds max for training


def as_rgb(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if len(frame.shape) == 3 else frame


def extract_frames(
    video_path: Path, target_fps: int = TARGET_FPS, mode: str = "training"
) -> List[np.ndarray]:
    """
    Extract frames from video with mode-specific processing.

    Args:
        video_path: Path to video file
        target_fps: Target frame rate for extraction
        mode: "training" (trim to chunks), "inference" (pad to chunks), or "raw" (no processing)

    Returns:
        List of frames as numpy arrays
    """
    if mode not in ["training", "inference", "raw"]:
        raise ValueError(f"Invalid mode: {mode}. Use 'training', 'inference', or 'raw'")

    # Open video and get metadata
    cap, video_info = _open_and_analyze_video(video_path)
    if cap is None:
        return []

    try:
        # Calculate which frames to grab (handles FPS resampling)
        frame_indices = _calculate_resample_indices(
            video_info["total_frames"],
            video_info["fps"],
            target_fps,
            # TODO - instead of none, force a max frames limit (say 15-20 minutes of video)
            #  in a new app components file 'rules.py'
            max_frames=MAX_TRAINING_FRAMES if mode == "training" else None,
        )

        # Extract the frames
        frames = _read_frames_from_video(cap, frame_indices)

        # Apply mode-specific post-processing
        if mode == "training":
            frames = _process_for_training(frames)
        elif mode == "inference":
            frames = _process_for_inference(frames)
        # mode == "raw" returns frames as-is

        return frames

    finally:
        cap.release()


def _open_and_analyze_video(
    video_path: Path,
) -> Tuple[Optional[cv2.VideoCapture], dict]:
    """
    Open video file and extract metadata.

    Returns:
        Tuple of (VideoCapture object, metadata dict) or (None, {}) if failed
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Warning: Cannot open video: {video_path}")
        return None, {}

    video_info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    video_info["duration"] = (
        video_info["total_frames"] / video_info["fps"] if video_info["fps"] > 0 else 0
    )

    # print(
    #     f"  Video: {video_info['total_frames']} frames @ {video_info['fps']:.1f} FPS "
    #     f"({video_info['duration']:.1f}s)"
    # )

    return cap, video_info


def _calculate_resample_indices(
    total_frames: int,
    original_fps: float,
    target_fps: float,
    max_frames: Optional[int] = None,
) -> Optional[Set[int]]:
    """
    Calculate which frame indices to extract for FPS resampling.

    Args:
        total_frames: Total frames in video
        original_fps: Original video FPS
        target_fps: Desired output FPS
        max_frames: Optional maximum number of frames to extract

    Returns:
        Set of frame indices to extract, or None to extract all frames
    """
    if original_fps <= 0:
        return None

    # If FPS already matches, no resampling needed
    if abs(original_fps - target_fps) < 0.1:  # Small tolerance for float comparison
        if max_frames and total_frames > max_frames:
            # Even if FPS matches, we might need to limit frames
            return set(range(max_frames))
        return None  # Extract all frames

    # Calculate how many frames we need at target FPS
    video_duration = total_frames / original_fps
    desired_frames = int(round(video_duration * target_fps))

    # Apply max frames limit if specified
    if max_frames:
        desired_frames = min(desired_frames, max_frames)

    # Calculate evenly-spaced frame indices
    if desired_frames >= total_frames:
        # If we need more frames than available, take all
        return None

    indices = np.linspace(0, total_frames - 1, desired_frames).astype(int)

    # print(
    #     f"  Resampling: {total_frames} frames @ {original_fps:.1f} FPS "
    #     f"→ {desired_frames} frames @ {target_fps:.1f} FPS"
    # )

    return set(indices.tolist())


def _read_frames_from_video(
    cap: cv2.VideoCapture, frame_indices: Optional[Set[int]]
) -> List[np.ndarray]:
    """
    Read frames from video, optionally filtering by indices.

    Args:
        cap: OpenCV VideoCapture object
        frame_indices: Set of frame indices to extract, or None for all frames

    Returns:
        List of frames as numpy arrays
    """
    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Check if we should keep this frame
        if frame_indices is None or frame_idx in frame_indices:
            frames.append(frame)

        frame_idx += 1

    # print(f"  Extracted {len(frames)} frames")
    return frames


def _process_for_training(frames: List[np.ndarray]) -> List[np.ndarray]:
    """
    Process frames for training: trim to chunk boundaries.

    Training expects complete 3-second chunks (75 frames each).
    """
    if not frames:
        return []

    # First apply max frame limit
    if len(frames) > MAX_TRAINING_FRAMES:
        frames = frames[:MAX_TRAINING_FRAMES]

    # Trim to complete chunks only
    complete_chunks = len(frames) // FRAMES_PER_CHUNK

    if complete_chunks == 0:
        return []

    frames_to_keep = complete_chunks * FRAMES_PER_CHUNK
    if frames_to_keep < len(frames):
        frames = frames[:frames_to_keep]

    return frames


def _process_for_inference(frames: List[np.ndarray]) -> List[np.ndarray]:
    """
    Process frames for inference: pad to chunk boundaries.

    Inference can handle any length but needs padding to chunk boundaries.
    """
    if not frames:
        return []

    remainder = len(frames) % FRAMES_PER_CHUNK
    if remainder != 0:
        pad_size = FRAMES_PER_CHUNK - remainder
        padding = [np.zeros_like(frames[0])] * pad_size
        frames.extend(padding)
        print(
            f"  Padded with {pad_size} frames to reach chunk boundary "
            f"({len(frames)} total)"
        )

    return frames


# Convenience functions for clarity
def extract_frames_for_training(video_path: Path) -> List[np.ndarray]:
    """Extract and prepare frames for model training."""
    return extract_frames(video_path, mode="training")


def extract_frames_for_inference(video_path: Path) -> List[np.ndarray]:
    """Extract and prepare frames for model inference."""
    return extract_frames(video_path, mode="inference")


# Backward compatibility wrapper
def extract_frames_legacy(
    mp4_path: Path, target_fps: int = TARGET_FPS, trim: bool = True
) -> List[np.ndarray]:
    """Legacy function signature for backward compatibility."""
    mode = "training" if trim else "inference"
    return extract_frames(mp4_path, target_fps=target_fps, mode=mode)


"""
Main pre processing calls
"""


def process_video_for_inference(
    mp4_path: Path,
    face_embedder: FaceEmbedder,
    user_hint: [Tuple[float, float]] = (
        0.5,
        0.5,
    ),  # Default hint. center is enough if only 1 face
) -> List[torch.Tensor]:
    # Extract video frames
    video_frames = extract_frames_for_inference(mp4_path)
    if not video_frames:
        raise ValueError(f"No video frames extracted for {mp4_path.name}")

    # Core processing: always needed
    face_crops = face_embedder.crop_faces(video_frames, user_hint[0], user_hint[1])
    face_embeddings_chunks = face_embedder.compute_embeddings(face_crops)
    print(
        f"  Extracted {len(face_embeddings_chunks)} face embeddings from {len(video_frames)} frames"
    )
    return face_embeddings_chunks


def process_frames_for_inference(
    frames: List[np.ndarray], hint_pos: Tuple[float, float]
) -> List[torch.Tensor]:
    """
    Process a list of frames for inference using the provided hint position.

    Args:
        frames: List of video frames as numpy arrays
        hint_pos: Tuple of (hint_x, hint_y) for face cropping

    Returns:
        List of face embeddings as torch tensors
    """
    if not frames:
        raise ValueError("No frames provided for inference")

    face_embedder = FaceEmbedder()  # Initialize your face embedder here
    face_crops = face_embedder.crop_faces(frames, hint_pos[0], hint_pos[1])
    return face_embedder.compute_embeddings(face_crops)


def process_video_for_training(
    mp4_path: Path, face_embedder: FaceEmbedder, face_hint_pos: Tuple[float, float]
) -> List[torch.Tensor]:
    """
    Complete video/face pipeline with optimized debug processing.
    Debug images are only created when save_debug=True.
    """
    # all valid video paths should have a face hint (metadata)
    hint_x, hint_y = face_hint_pos

    # Extract video frames
    video_frames = extract_frames_for_training(mp4_path)
    if not video_frames:
        raise ValueError(f"No video frames extracted for {mp4_path.name}")

    # Core processing: always needed
    face_crops = face_embedder.crop_faces(video_frames, hint_x, hint_y)
    return face_embedder.compute_embeddings(face_crops)
