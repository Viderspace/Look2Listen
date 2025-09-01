#!/usr/bin/env python3
"""
face_embedder.py
================

Clean FaceEmbedder class that encapsulates face detection and embedding generation.
Optimized to separate detection logic from debug image creation for performance.
"""

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from avspeech.utils.constants import FACE_IMG_SZ
from avspeech.utils.face_detection_toolbox import nms_faces
from avspeech.utils.structs import FaceDetection

confidence_threshold = 0.7  # Minimum detection confidence. Not sure it affects anything


class FaceDetector:
    """
    Encapsulates face detection and embedding generation.
    Optimized for performance: debug images created only when needed.

    Uses:
    - MediaPipe for face detection
    - InceptionResnetV1 (facenet_pytorch) for 512-D face embeddings on MPS
    - Paper's approach: missing faces → zero embeddings (not dummy crops)
    """

    def __init__(self):
        """Initialize face detector and encoder models."""
        logging.info("🔍 Initializing MediaPipe face detector...")
        self._init_detector()

        self.frames_per_chunk = 75

        self.undetected_frame_count = 0

    def _init_detector(self):
        """Initialize MediaPipe face detection."""
        import mediapipe as mp

        mp_face_detection = mp.solutions.face_detection
        self.detector = mp_face_detection.FaceDetection(
                model_selection=1,  # 0 for short-range (2m), 1 for full-range (5m)
                min_detection_confidence=confidence_threshold,
        )

    def crop_faces(
            self, frames: List[np.ndarray], hint_x: float, hint_y: float
    ) -> List[np.ndarray]:
        face_crops = []
        successful_detections = 0

        for frame in frames:
            # MediaPipe expects RGB
            rgb_frame = (
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if frame.shape[-1] == 3
                    else frame
            )

            # Detect faces
            detections = self.detect_frame(rgb_frame)
            face_crop = None
            if detections:
                # Find best face using hint
                best_detection = self.find_nearest_face(detections, hint_x, hint_y)
                if best_detection is not None:
                    face_crop = self._crop_face_with_padding(frame, best_detection)
                    successful_detections += 1

            # Use dummy crop if no face found (ORIGINAL WORKING LOGIC)
            if face_crop is None:
                self.undetected_frame_count += 1
                face_crop = np.zeros((FACE_IMG_SZ, FACE_IMG_SZ, 3), dtype=np.uint8)

            face_crops.append(face_crop)  # Always valid crop (real or dummy)

        sample = face_crops[0] if len(face_crops) > 0 else None
        if sample is not None:
            print(f"sample image input (for embedding): {sample.shape}, dtype: {sample.dtype}, min: {sample.min()}, max: {sample.max()}")

        return face_crops


    def detect_frame(self, frame: np.ndarray) -> List[FaceDetection]:
        raw_results = self.detector.process(frame).detections
        if not raw_results:
            return []
        all_detections = [FaceDetection(detection, frame) for detection in raw_results]
        without_overlaps = nms_faces(all_detections)
        return without_overlaps


    # TODO - Migrate this method into a function in toolbox
    def find_nearest_face(
            self, detections: List[FaceDetection], hint_x: float, hint_y: float
    ) -> Optional[FaceDetection]:
        """Find the face detection closest to the hint coordinates."""
        best_detection = None
        min_distance = float("inf")

        # frame_height, frame_width = frame_shape[:2]

        for detection in detections:
            # Calculate distance to hint in ** UV SPACE **
            relative_squared_distance = detection.get_squared_distance(hint_x, hint_y)

            if relative_squared_distance < min_distance:
                min_distance = relative_squared_distance
                best_detection = detection

        return best_detection

    def _crop_face_with_padding(
            self, frame: np.ndarray, detection: FaceDetection, padding_factor=0.15
    ) -> np.ndarray:
        """Crop face with padding for better context."""
        frame_height, frame_width = frame.shape[:2]

        # Convert to absolute coordinates
        x1, y1, x2, y2 = detection.x1y1x2y2()

        # Add padding
        width, height = x2 - x1, y2 - y1
        pad_x = int(width * padding_factor)
        pad_y = int(height * padding_factor)

        x1_pad = max(0, x1 - pad_x)
        y1_pad = max(0, y1 - pad_y)
        x2_pad = min(frame_width, x2 + pad_x)
        y2_pad = min(frame_height, y2 + pad_y)

        # Crop and resize
        cropped = frame[y1_pad:y2_pad, x1_pad:x2_pad]
        return cv2.resize(cropped, (FACE_IMG_SZ, FACE_IMG_SZ))

    def draw_detection(
            self,
            frame: np.ndarray,
            detection: FaceDetection,
            color: Tuple[int, int, int] = (0, 255, 0),
    ) -> np.ndarray:
        """Draw detection bounding box on frame for debugging."""
        # Getting pre-calculated absolute coordinates (FaceDetection class)
        x1, y1, x2, y2 = detection.x1y1x2y2()
        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Add confidence score
        cv2.putText(
                frame,
                f"{detection.confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.0,
                color,
                3,
        )

        return frame

