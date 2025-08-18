"""

Named tuple definitions for consistent chunk data structures across audio and video pipelines.
Provides type safety and self-documenting code for pipeline outputs.
"""
from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple, Tuple, List, Optional

import numpy as np
import torch



class SampleT(Enum):
    S1_NOISE = "1s_noise"
    S2_CLEAN = "2s_clean"
    S2_NOISE = "2s_noise"



class AudioChunk(NamedTuple):
    """
    Audio data for a single 3-second chunk.

    Attributes:
        clean: Clean audio STFT embeddings [257, 298, 2]
        mixed: Mixed audio STFT embeddings [257, 298, 2]
    """
    clean: torch.Tensor
    mixed: torch.Tensor



class Sample(NamedTuple):
    """
    Sample data for a single audio-visual sample.

    Attributes:
        mixture: Mixture audio embeddings [257, 298, 2]
        clean: Clean audio embeddings [257, 298, 2]
        face: Face embeddings [75, 512]
        sample_id: Unique identifier for the sample
    """
    face: torch.Tensor
    mix: torch.Tensor
    clean: torch.Tensor





class FaceDetection:
    """Small adapter for a face detection data (bounding box, confidence, etc.)"""
    x1: int
    y1: int
    x2: int
    y2: int
    u : float
    v : float
    confidence: float

    def __init__(self, detection, frame):
        frame_height, frame_width = frame.shape[:2]
        bbox = detection.location_data.relative_bounding_box

        # Convert to absolute coordinates
        self.u = bbox.xmin + bbox.width / 2
        self.v = bbox.ymin + bbox.height / 2

        self.x1 = int(bbox.xmin * frame_width)
        self.y1 = int(bbox.ymin * frame_height)
        self.x2 = int((bbox.xmin + bbox.width) * frame_width)
        self.y2 = int((bbox.ymin + bbox.height) * frame_height)  # FIXED: was frame_width
        self.confidence = float(detection.score[0]) if detection.score else 0.0

    def x1y1x2y2(self):
        """Get coordinates in x1, y1, x2, y2 format"""
        return self.x1, self.y1, self.x2, self.y2

    def width(self) -> int:
        """Get width of the bounding box"""
        return self.x2 - self.x1

    def height(self) -> int:
        """Get height of the bounding box"""
        return self.y2 - self.y1

    def center(self)-> tuple[int, int]:
        """Get center coordinates of the bounding box"""
        return self.x1 + self.width() // 2, self.y1 + self.height() // 2

    def get_squared_distance(self, x: float, y : float) -> float:
        """Calculate squared distance to a point in uv space."""
        return (self.u - x) ** 2 + (self.v - y) ** 2


    def uv(self) -> Tuple[float, float]:
        """Get relative center coordinates (0-1) of the bounding box in the frame"""
        return self.u, self.v

