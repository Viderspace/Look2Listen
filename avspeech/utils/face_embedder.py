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


class FaceEmbedder:
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

        logging.info("🧠 Initializing 512-D face encoder on MPS (facenet_pytorch)…")
        self._init_encoder()

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

    def _init_encoder(self):
        from facenet_pytorch import InceptionResnetV1

        """Initialize face encoder on MPS."""
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS not available - this implementation requires Apple Silicon"
            )

        self.device = torch.device("mps")
        face_model = InceptionResnetV1(pretrained="vggface2", classify=False).to(
            self.device
        )
        self.encoder = TorchEmbedder(face_model, self.device)

    def print_missing_frame_count(self):
        print(
            f"FaceEmbedder missed {self.undetected_frame_count} frames total (confidence = {confidence_threshold})"
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

        return face_crops

    def detect_and_crop_faces(
        self, video_frames: List[np.ndarray], hint_x: float, hint_y: float
    ) -> Tuple[List[np.ndarray], List[Optional[object]]]:
        """
        Detect faces in video frames and crop them to 160x160.
        Returns crops and raw detection objects (for optional debug image creation).

        Args:
            video_frames: List of video frames (H, W, 3) in RGB format
            hint_x: Face hint x coordinate (0-1)
            hint_y: Face hint y coordinate (0-1)

        Returns:
            Tuple of (face_crops, detections)
            - face_crops: List with valid crops (160x160x3) - dummy crops for missing faces
            - detections: List with MediaPipe detection objects or None for missing faces
        """
        face_crops = []
        detections = []
        successful_detections = 0

        for frame_idx, frame in enumerate(video_frames):
            # MediaPipe expects RGB
            rgb_frame = (
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if frame.shape[-1] == 3
                else frame
            )

            # Detect faces
            results = self.detect_frame(rgb_frame)

            face_crop = None
            best_detection = None

            if results:
                # Find best face using hint
                best_detection = self.find_nearest_face(results, hint_x, hint_y)
                if best_detection is not None:
                    face_crop = self._crop_face_with_padding(frame, best_detection)
                    successful_detections += 1

            # Use dummy crop if no face found (ORIGINAL WORKING LOGIC)
            if face_crop is None:
                self.undetected_frame_count += 1
                face_crop = np.zeros((FACE_IMG_SZ, FACE_IMG_SZ, 3), dtype=np.uint8)

            face_crops.append(face_crop)  # Always valid crop (real or dummy)
            detections.append(best_detection)  # None or MediaPipe detection object

        return face_crops, detections

    def detect_frame(self, frame: np.ndarray) -> List[FaceDetection]:
        raw_results = self.detector.process(frame).detections
        if not raw_results:
            return []
        all_detections = [FaceDetection(detection, frame) for detection in raw_results]
        without_overlaps = nms_faces(all_detections)
        return without_overlaps

    def compute_embeddings(self, face_crops: List[np.ndarray]) -> List[torch.Tensor]:
        """
        Compute face embeddings for crops and split into chunks.
        Implements zero embedding logic: dummy crops (all zeros) → zero embeddings.

        Args:
            face_crops: List of face crops (160x160x3) - includes dummy crops for missing faces

        Returns:
            List of tensors, one per chunk, each shaped (frames_per_chunk, 512)
        """
        if not face_crops:
            return []

        # Process each frame: dummy crop → zero embedding, valid crop → neural network
        embeddings = []
        valid_crops_batch = []
        valid_indices = []

        # Separate valid crops from dummy crops (all zeros)
        for i, crop in enumerate(face_crops):
            if np.all(crop == 0):  # Dummy crop (all zeros) → zero embedding
                embeddings.append(torch.zeros(512, dtype=torch.float32))
            else:
                # Collect valid crops for batch processing
                valid_crops_batch.append(crop)
                valid_indices.append(i)
                embeddings.append(None)  # Placeholder

        # Batch process valid crops through neural network
        if valid_crops_batch:
            valid_embs_np = self.encoder.get(np.stack(valid_crops_batch, axis=0))
            valid_embs = torch.from_numpy(valid_embs_np).float()

            # Insert computed embeddings back into the correct positions
            for batch_idx, original_idx in enumerate(valid_indices):
                embeddings[original_idx] = valid_embs[batch_idx]

        # Convert list to tensor
        all_embeddings = torch.stack(embeddings)  # (N, 512)

        # Split into chunks
        chunks = []
        for i in range(0, len(all_embeddings), self.frames_per_chunk):
            chunk_embs = all_embeddings[i : i + self.frames_per_chunk]

            # Pad if needed to maintain consistent chunk size
            if len(chunk_embs) < self.frames_per_chunk:
                padding_size = self.frames_per_chunk - len(chunk_embs)
                padding = torch.zeros(padding_size, 512, dtype=torch.float32)
                chunk_embs = torch.cat([chunk_embs, padding], dim=0)

            chunks.append(chunk_embs.clone())

        return chunks

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

    def _crop_face(self, frame: np.ndarray, detection) -> np.ndarray:
        """Crop face from frame using detection bounding box."""
        frame_height, frame_width = frame.shape[:2]
        bbox = detection.location_data.relative_bounding_box

        # Convert relative coordinates to absolute
        x1 = max(0, int(bbox.xmin * frame_width))
        y1 = max(0, int(bbox.ymin * frame_height))
        x2 = min(frame_width, int((bbox.xmin + bbox.width) * frame_width))
        y2 = min(frame_height, int((bbox.ymin + bbox.height) * frame_height))

        # Crop and resize to target size
        cropped = frame[y1:y2, x1:x2]
        if cropped.size > 0:
            resized = cv2.resize(cropped, (FACE_IMG_SZ, FACE_IMG_SZ))
            return resized
        else:
            # Return dummy if crop failed
            return np.zeros((FACE_IMG_SZ, FACE_IMG_SZ, 3), dtype=np.uint8)

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


class TorchEmbedder:
    """
    Wrapper for PyTorch face embedding model to provide .get(batch_np) API.
    """

    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model.eval()
        self.device = device

    def get(self, batch_np):
        """
        Process batch of face images and return embeddings.

        Args:
            batch_np: Numpy array of shape (N, H, W, 3) in uint8 format

        Returns:
            Numpy array of shape (N, 512) with face embeddings
        """
        # Convert to torch tensor
        if isinstance(batch_np, torch.Tensor):
            x = batch_np
        else:
            x = torch.from_numpy(batch_np)

        # Convert NHWC to NCHW and normalize
        if x.ndim == 4 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
        x = x.float()
        if x.max() > 1.5:
            x = x / 255.0

        # Resize to 160x160 if needed (facenet_pytorch expects this)
        if x.shape[-2:] != (160, 160):
            x = F.interpolate(x, size=(160, 160), mode="bilinear", align_corners=False)

        # Normalize to [-1, 1] (facenet_pytorch preprocessing)
        x = (x - 0.5) / 0.5

        # Compute embeddings
        with torch.no_grad():
            embeddings = self.model(x.to(self.device)).cpu().numpy()

        return embeddings
