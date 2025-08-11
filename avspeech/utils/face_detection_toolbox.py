import numpy as np

from typing import List
import numpy as np
from avspeech.utils.structs import FaceDetection

def iou(a: FaceDetection, b: FaceDetection) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    iw = max(0, x2 - x1)
    ih = max(0, y2 - y1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
    area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
    return inter / (area_a + area_b - inter)


def nms_faces(dets: List[FaceDetection], iou_thresh: float = 0.45) -> List[FaceDetection]:
    """Keep highest-score boxes and remove overlaps above iou_thresh."""
    if not dets:
        return []

    order = np.argsort([d.confidence for d in dets])[::-1]
    keep: List[FaceDetection] = []

    while order.size > 0:
        i = int(order[0])
        keep.append(dets[i])
        if order.size == 1:
            break
        rest = order[1:]
        rest_keep = []
        for j in rest:
            if iou(dets[i], dets[int(j)]) <= iou_thresh:
                rest_keep.append(j)
        order = np.array(rest_keep, dtype=int)

    return keep

