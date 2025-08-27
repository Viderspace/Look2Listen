# avspeech/training/metrics_viz.py
from __future__ import annotations
from typing import List, Optional
import csv
import math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


def _read_csv(csv_path: Path):
    epochs: List[int] = []
    train: List[float] = []
    val: List[float] = []
    val_s2n: List[Optional[float]] = []
    val_2sc: List[Optional[float]] = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                e = int(row["epoch"]) ; epochs.append(e)
                train.append(float(row["train_loss"]))
                val.append(float(row["val_loss"]))
                v1 = row.get("val_s2_noise", "").strip()
                v2 = row.get("val_2s_clean", "").strip()
                val_s2n.append(float(v1) if v1 else None)
                val_2sc.append(float(v2) if v2 else None)
            except Exception:
                # skip malformed rows
                continue
    return epochs, train, val, val_s2n, val_2sc


def update_metrics_plot(csv_path, out_png):
    """Read metrics CSV and save a PNG plot with epoch-wise trends.
    Shows three lines when available: train_loss, val_s2_noise, val_2s_clean.
    """
    csv_path = Path(csv_path)
    out_png = Path(out_png)
    if not csv_path.exists():
        return

    epochs, train, val, val_s2n, val_2sc = _read_csv(csv_path)
    if not epochs:
        return

    plt.figure(figsize=(8, 5), dpi=120)
    # Always plot train and overall val
    plt.plot(epochs, train, label="train_loss")
    plt.plot(epochs, val, label="val_loss")

    # Optional specialized metrics if available
    if any(v is not None for v in val_s2n):
        xs = [e for e, v in zip(epochs, val_s2n) if v is not None]
        ys = [v for v in val_s2n if v is not None]
        plt.plot(xs, ys, label="val_s2_noise")
    if any(v is not None for v in val_2sc):
        xs = [e for e, v in zip(epochs, val_2sc) if v is not None]
        ys = [v for v in val_2sc if v is not None]
        plt.plot(xs, ys, label="val_2s_clean")

    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training metrics")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
