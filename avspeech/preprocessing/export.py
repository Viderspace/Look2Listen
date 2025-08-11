
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import json
import logging

import numpy as np
from avspeech.utils.structs import Sample
import torch
from PIL import Image, ImageDraw, ImageFont
from avspeech.utils.constants import SPEC_FREQ_BINS, CHUNK_DURATION, TARGET_FPS
from avspeech.preprocessing.clips_loader import ClipData

logger = logging.getLogger(__name__)



def save_single_sample(folder_path : Path, sample: Sample, overwrite: bool = True) -> None:
    """
    Save a single chunk of audio and face embeddings to disk.

    Args:
        path (Path): Directory where the chunk will be saved.
        chunk (Sample): The chunk data containing audio and face embeddings.
        overwrite (bool): Whether to overwrite existing files.
    """
    folder_path = _mkdir(folder_path)

    # Save audio embeddings (clean and mixed))
    audio_sub_dir = _mkdir(folder_path/ "audio")
    _torch_save(sample.clean, audio_sub_dir /  "clean_embs.pt", overwrite)
    _torch_save(sample.mix, audio_sub_dir / "mixture_embs.pt", overwrite)

    # Save face embeddings
    face_sub_dir = _mkdir(folder_path / "face")
    _torch_save(sample.face, face_sub_dir / "face_embs.pt", overwrite)


def save_processed_clip(embeddings : List[Sample],
                         clip_id : str,
                         output_path : Path,
                         overwrite: bool = True) -> int:
    """
    Persist the whole bundle to disk under
    <out_base>/<clip_id>/{audio,face,debug}/…

    After it succeeds the method is silent; on error it raises.
    """
    save_count = 0
    for idx, sample in enumerate(embeddings):
        try:
            folder_path = output_path / f"_{clip_id}_{idx}" # e.g. _XAAb_03_0
            save_single_sample(folder_path, sample, overwrite)
            save_count += 1

        except Exception as e:
            logging.error(f"❌ Error creating chunk {idx}: {e}")
            continue

    logger.info(f"✅ saved <{clip_id}> to {output_path / f'_{clip_id}'} with {save_count} chunks")
    return save_count




# ───────────────────────── helper utilities ──────────────────────────
def _mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _torch_save(obj: torch.Tensor, dst: Path, overwrite: bool) -> None:
    if dst.exists() and not overwrite:
        raise FileExistsError(dst)
    torch.save(obj, dst)


def _json_dump(obj: Dict, dst: Path, overwrite: bool) -> None:
    if dst.exists() and not overwrite:
        raise FileExistsError(dst)
    with open(dst, "w") as f:
        json.dump(obj, f, indent=2)