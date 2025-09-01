from typing import List, Optional

import numpy as np
import torch

from avspeech.utils.FaceDetector import FaceDetector
from avspeech.utils.FaceEmbedderV2 import FaceEmbedderV2


class FaceManager:


    def __init__(self):
        self.detector = FaceDetector()
        self.embedder = FaceEmbedderV2()



    def crop_frames(self, frames, hint_x, hint_y):
        return self.detector.crop_faces(frames, hint_x, hint_y)

    def compute_embeddings(self,
                           face_crops: List[np.ndarray],
                           drop_tail: bool = False,
                           pad_tail: bool = False,
                           max_3_chunks: bool = False) -> Optional[List[torch.Tensor]]:
        return self.embedder.compute_embeddings(face_crops, drop_tail, pad_tail, max_3_chunks)