

from __future__ import annotations

import math
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def save_debug_collage(outdir: Path, face_crops : List[np.ndarray]) -> None:
    """
    Create and save a collage of all face crops for this chunk.

    """
    if not face_crops:
        return

    # Convert numpy arrays to PIL Images, skip None values (missing faces)
    pil_crops = []
    crop_indices = []  # Track original indices for labeling

    for idx, crop in enumerate(face_crops):
        if crop is None:
            print(f"Skipping face crop {idx}: Missing face (None)")
            continue

        if not isinstance(crop, np.ndarray):
            print(f"⚠️ Skipping face crop {idx}: Not a numpy array ({type(crop)})")
            continue

        if crop.dtype != np.uint8:
            try:
                crop = crop.astype(np.uint8)
            except Exception as e:
                print(f"⚠️ Skipping face crop {idx}: Cannot cast to uint8 ({e})")
                continue

        if crop.ndim != 3 or crop.shape[2] != 3:
            print(f"⚠️ Skipping face crop {idx}: Unexpected shape {crop.shape}")
            continue

        # Convert BGR to RGB (face crops come from OpenCV processing)
        crop_rgb = crop[:, :, ::-1]  # BGR → RGB

        try:
            pil_crops.append(Image.fromarray(crop_rgb))
            crop_indices.append(idx)
        except Exception as e:
            print(f"⚠️ Skipping face crop {idx}: PIL conversion failed ({e})")

    if not pil_crops:
        print(f"⚠️ No valid face crops to create collage for {outdir.name}")
        return

    # Calculate grid dimensions
    n = len(pil_crops)
    tile_w, tile_h = pil_crops[0].size
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    # Create collage
    collage = Image.new("RGB", (cols * tile_w, rows * tile_h), (0, 0, 0))
    draw = ImageDraw.Draw(collage)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    # Paste each crop with original frame index label
    for i, (img, original_idx) in enumerate(zip(pil_crops, crop_indices)):
        r, c = divmod(i, cols)
        x = c * tile_w
        y = r * tile_h
        collage.paste(img, (x, y))

        # Add original frame number label
        label = f"{original_idx:02d}"
        if font:
            # Black outline for readability
            for ox, oy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                draw.text((x + 4 + ox, y + 4 + oy), label, fill=(0, 0, 0), font=font)
            draw.text((x + 4, y + 4), label, fill=(255, 255, 255), font=font)
        else:
            draw.text((x + 4, y + 4), label, fill=(255, 255, 255))

    # Save collage in chunk's debug directory
    outdir.mkdir(parents=True, exist_ok=True)
    collage_path = outdir / "face_crops_collage.png"
    collage.save(collage_path)
    print(f"  💾 Saved face crops collage: {collage_path.name} ({len(pil_crops)}/{len(face_crops)} faces)")

