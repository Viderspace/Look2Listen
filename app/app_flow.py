"""
Clean app flow for face detection and processing.
Minimal coordination of existing functions.
"""

from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from pathlib import Path

from avspeech.utils.face_embedder import FaceEmbedder
from avspeech.utils.video import extract_frames_for_inference, as_rgb
from components.video_handler import extract_segment


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
        video_path: str,
        start_time: float,
        end_time: float,
        face_embedder: FaceEmbedder
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
    #convert to rgb
    preview_image = as_rgb(preview_image)

    # ** Best frame is the first one with most faces detected **
    best_detections = all_detections[best_frame_idx]
    print(f"Best frame has {len(best_detections)} faces (index {best_frame_idx})")
    for detection in best_detections:
        preview_image = face_embedder.draw_detection(preview_image, detection)

    # Extract bounding boxes for face selection UI
    #TODO - Replace this below (redundant with FaceEmbedder) by introducing FaceDetection class to App scripts
    best_frame_boxes = []
    for i, detection in enumerate(best_detections):
        best_frame_boxes.append({
                'face_id'   : i,
                'x'         : detection.x1,
                'y'         : detection.y1,
                'width'     : detection.width(),
                'height'    : detection.height(),
                'confidence': detection.confidence
        })

    return AnalysisResult(
            preview_image=preview_image,
            frames=frames,
            all_detections=all_detections,
            best_frame_idx=best_frame_idx,
            best_frame_boxes=best_frame_boxes,
            segment_path=segment_path
    )


# ======================== STEP 2: PROCESS WITH SELECTION ========================

def process_with_face_selection(
        analysis_result: AnalysisResult,
        face_embedder: FaceEmbedder,
        selected_face_id: Optional[int] = None
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
            selected_box = {'x': w // 2, 'y': h // 2, 'width': 0, 'height': 0}
    else:
        # User selected a specific face
        selected_box = analysis_result.best_frame_boxes[selected_face_id]

    # Convert to relative hint coordinates (0-1)
    h, w = analysis_result.frames[0].shape[:2]
    hint_x = (selected_box['x'] + selected_box['width'] / 2) / w
    hint_y = (selected_box['y'] + selected_box['height'] / 2) / h

    # Now crop faces using the hint
    face_crops = []
    for frame, detections in zip(analysis_result.frames, analysis_result.all_detections):

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


def run_inference(audio_chunks, face_embeddings, sample_rate: int):
    """
    Returns a single enhanced mono waveform (float32, [-1,1]) and sample_rate.
    Implement your model’s forward pass here.
    """
    # TODO: Replace this mock with your actual model inference
    # Example shape assumptions:
    # - audio_chunks: List[np.ndarray] of shape [T_chunk] at sample_rate
    # - face_embeddings: List[np.ndarray] aligned per-chunk, or a single embedding
    enhanced = np.concatenate(audio_chunks, axis=0).astype(np.float32)
    return enhanced, sample_rate



# ======================== CONVENIENCE FUNCTIONS ========================

def get_status_message(analysis_result: AnalysisResult) -> str:
    num_faces = len(analysis_result.best_frame_boxes)
    frames_with_faces = sum(1 for d in analysis_result.all_detections
                            if len(d) > 0)
    total_frames = len(analysis_result.frames)

    if num_faces == 0:
        return f"⚠️ No faces detected in {total_frames} frames"
    elif num_faces == 1:
        conf = analysis_result.best_frame_boxes[0]['confidence']
        return (f"✅ 1 face detected (confidence: {conf:.2%})\n"
                f"Found faces in {frames_with_faces}/{total_frames} frames")
    else:
        return (f"👥 {num_faces} faces detected\n"
                f"Found faces in {frames_with_faces}/{total_frames} frames\n"
                f"Click on a face to select")


def auto_proceed_if_single_face(analysis_result: AnalysisResult) -> bool:
    """Only One face in the video -> No need to select it"""
    return len(analysis_result.best_frame_boxes) == 1