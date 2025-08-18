"""
Clean app flow for face detection and processing.
Minimal coordination of existing functions.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from components.video_handler import extract_segment

from avspeech.utils.face_embedder import FaceEmbedder
from avspeech.utils.video import as_rgb, extract_frames_for_inference


@dataclass
class AnalysisResult:
    """Result from analyze step - everything needed for next steps."""

    preview_image: np.ndarray  # Annotated best frame
    frames: List[np.ndarray]  # All extracted frames
    all_detections: List[Any]  # Detection results per frame
    best_frame_idx: int  # Which frame had most faces
    best_frame_boxes: List[Dict]  # Bounding boxes for face selection
    segment_path: str  # Path to extracted segment


# ======================== STEP 1: ANALYZE ========================


def analyze_video_segment(
    video_path: str, start_time: float, end_time: float, face_embedder: FaceEmbedder
) -> AnalysisResult:
    """
    Step 1: Analyze video segment.
    Extract frames, detect faces, create preview.

    Args:
        video_path: Path to video file
        start_time: Start time in seconds
        end_time: End time in seconds
        face_embedder: FaceEmbedder instance

    Returns:
        AnalysisResult with all data needed for next steps
    """
    # Extract segment
    segment_path = extract_segment(Path(video_path), start_time, end_time)

    # Extract frames (with padding for inference)
    frames = extract_frames_for_inference(Path(segment_path))

    # Detect faces in all frames (ONCE)
    all_detections = [face_embedder.detect_frame(frame) for frame in frames]

    # Find best frame (most faces)
    best_frame_idx = 0
    max_faces = 0
    for i, detections in enumerate(all_detections):
        num_faces = len(detections)
        if num_faces > max_faces:
            max_faces = num_faces
            best_frame_idx = i

    # Create annotated preview image
    preview_image = frames[best_frame_idx].copy()
    # convert to rgb
    preview_image = as_rgb(preview_image)

    # ** Best frame is the first one with most faces detected **
    best_detections = all_detections[best_frame_idx]
    print(f"Best frame has {len(best_detections)} faces (index {best_frame_idx})")
    for detection in best_detections:
        preview_image = face_embedder.draw_detection(preview_image, detection)

    # Extract bounding boxes for face selection UI
    # TODO - Replace this below (redundant with FaceEmbedder) by introducing FaceDetection class to App scripts
    best_frame_boxes = []
    for i, detection in enumerate(best_detections):
        best_frame_boxes.append(
            {
                "face_id": i,
                "x": detection.x1,
                "y": detection.y1,
                "width": detection.width(),
                "height": detection.height(),
                "confidence": detection.confidence,
            }
        )

    return AnalysisResult(
        preview_image=preview_image,
        frames=frames,
        all_detections=all_detections,
        best_frame_idx=best_frame_idx,
        best_frame_boxes=best_frame_boxes,
        segment_path=segment_path,
    )


# ======================== STEP 2: PROCESS WITH SELECTION ========================


def process_with_face_selection(
    analysis_result: AnalysisResult,
    face_embedder: FaceEmbedder,
    selected_face_id: Optional[int] = None,
) -> Tuple[List[torch.Tensor], Tuple[float, float]]:
    """
    Step 2: Generate embeddings using selected face.

    Args:
        analysis_result: Result from analyze step
        face_embedder: FaceEmbedder instance
        selected_face_id: Which face to track (None = auto-select first/best)

    Returns:
        Tuple of (face_embedding_chunks, hint_coordinates)
    """
    # Determine hint coordinates from selection
    if selected_face_id is None:
        # Auto-select: use first face or center if no faces
        if analysis_result.best_frame_boxes:
            selected_box = analysis_result.best_frame_boxes[0]
        else:
            # No faces - use center
            h, w = analysis_result.frames[0].shape[:2]
            selected_box = {"x": w // 2, "y": h // 2, "width": 0, "height": 0}
    else:
        # User selected a specific face
        selected_box = analysis_result.best_frame_boxes[selected_face_id]

    # Convert to relative hint coordinates (0-1)
    h, w = analysis_result.frames[0].shape[:2]
    hint_x = (selected_box["x"] + selected_box["width"] / 2) / w
    hint_y = (selected_box["y"] + selected_box["height"] / 2) / h

    # Now crop faces using the hint
    face_crops = []
    for frame, detections in zip(
        analysis_result.frames, analysis_result.all_detections
    ):
        if len(detections) > 1:
            # Find nearest face to the hint position
            best_detection = face_embedder.find_nearest_face(detections, hint_x, hint_y)
            if best_detection:
                crop = face_embedder._crop_face_with_padding(frame, best_detection)
            else:
                crop = np.zeros((160, 160, 3), dtype=np.uint8)
        else:
            # No faces in this frame
            crop = np.zeros((160, 160, 3), dtype=np.uint8)

        face_crops.append(crop)

    # Generate embeddings
    face_embedding_chunks = face_embedder.compute_embeddings(face_crops)

    return face_embedding_chunks, (hint_x, hint_y)


# ======================== STEP 3: RUN INFERENCE ========================


def run_inference(
    audio_chunks, face_embeddings, checkpoint_path: str, device: str = "cpu"
):
    """
    Complete inference pipeline using the loaded model.

    Args:
        audio_chunks: List of STFT tensors from process_audio_for_inference
        face_embeddings: List of face embedding tensors
        checkpoint_path: Path to model checkpoint file
        device: Device to run inference on

    Returns:
        Tuple of (enhanced_audio_tensor, original_audio_tensor)
    """
    import torch

    import avspeech.utils.fourier_transform as fourier
    from avspeech.model.av_model import AudioVisualModel

    device = torch.device(device)

    # Load model
    model = AudioVisualModel().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    inference_results = []

    # Process each chunk
    with torch.no_grad():
        for audio_chunk, face_chunk in zip(audio_chunks, face_embeddings):
            audio_chunk = audio_chunk.to(device)
            face_chunk = face_chunk.to(device)

            # Add batch dimension if missing
            if len(audio_chunk.shape) == 3:
                audio_chunk = audio_chunk.unsqueeze(0)
            if len(face_chunk.shape) == 2:
                face_chunk = face_chunk.unsqueeze(0)

            # Run model - outputs mask for target speaker
            mask = model(audio_chunk, face_chunk)

            # Apply mask to enhance target speaker
            enhanced_chunk = audio_chunk * mask
            inference_results.append(enhanced_chunk.squeeze(0))

    # Concatenate all STFTs along time dimension
    full_enhanced_stft = torch.cat(inference_results, dim=1)
    full_original_stft = torch.cat(audio_chunks, dim=1)

    # Convert STFT back to audio
    enhanced_audio = fourier.stft_to_audio(full_enhanced_stft)
    original_audio = fourier.stft_to_audio(full_original_stft)

    return enhanced_audio, original_audio


def run_inference_with_clip_processor(
    segment_path: str, face_hint: Tuple[float, float], checkpoint_path: str
):
    """
    Alternative implementation using ClipProcessor for complete pipeline.

    Args:
        segment_path: Path to video segment
        face_hint: (x, y) coordinates for face detection hint
        checkpoint_path: Path to model checkpoint

    Returns:
        Tuple of (enhanced_audio_tensor, original_audio_tensor)
    """
    from avspeech.utils.ClipProcessor import ClipProcessor

    # Create ClipProcessor and run complete pipeline
    clip_processor = ClipProcessor(Path(segment_path))

    if not clip_processor.is_video_loaded():
        raise ValueError("Failed to load video segment")

    # Set face hint and generate embeddings
    clip_processor.set_face_hint(face_hint)

    if not clip_processor.is_video_ready():
        raise ValueError("Video processing failed")

    # Run inference
    enhanced_audio, original_audio = clip_processor.apply_inference(
        Path(checkpoint_path)
    )

    return enhanced_audio, original_audio


# ======================== CONVENIENCE FUNCTIONS ========================


def get_status_message(analysis_result: AnalysisResult) -> str:
    num_faces = len(analysis_result.best_frame_boxes)
    frames_with_faces = sum(1 for d in analysis_result.all_detections if len(d) > 0)
    total_frames = len(analysis_result.frames)

    if num_faces == 0:
        return f"⚠️ No faces detected in {total_frames} frames"
    elif num_faces == 1:
        conf = analysis_result.best_frame_boxes[0]["confidence"]
        return (
            f"✅ 1 face detected (confidence: {conf:.2%})\n"
            f"Found faces in {frames_with_faces}/{total_frames} frames"
        )
    else:
        return (
            f"👥 {num_faces} faces detected\n"
            f"Found faces in {frames_with_faces}/{total_frames} frames\n"
            f"Click on a face to select"
        )


def auto_proceed_if_single_face(analysis_result: AnalysisResult) -> bool:
    """Only One face in the video -> No need to select it"""
    return len(analysis_result.best_frame_boxes) == 1
