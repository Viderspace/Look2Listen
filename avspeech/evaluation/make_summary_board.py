#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
from matplotlib.patches import Rectangle
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from avspeech.utils.structs import SampleT as T

# ----- Labels & Benchmarks -----
TITLE: Dict[T, str] = {T.S1_NOISE: "1S+Noise", T.S2_CLEAN: "2S Clean", T.S2_NOISE: "2S+Noise"}
PAPER_BENCH: Dict[T, float] = {T.S1_NOISE: 16.0, T.S2_CLEAN: 10.3, T.S2_NOISE: 10.6}

# ===================== Helpers =====================

def pretty_ckpt(name: str) -> str:
    s = str(name)
    s = s.replace("checkpoint_", "")
    s = s.replace("epoch_", "e")
    s = s.replace("_pre_", "pre")
    return s

def _norm(s: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()

def pick(colnames, needle, exclude=None) -> str:
    """Robust column picker with optional exclusion (avoids 'Mixture SDR')."""
    want = _norm(needle)
    excl = _norm(exclude) if exclude else None
    # exact match first
    for c in colnames:
        if _norm(c) == want:
            return c
    # contains match with exclusion
    for c in colnames:
        nc = _norm(c)
        if excl and excl in nc:
            continue
        if want in nc:
            return c
    raise KeyError(f"Column containing '{needle}' not found (exclude={exclude}). Got: {list(colnames)}")

def load_task(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    col_sdri_bss = pick(df.columns, "SDRi (BSS mean)")
    col_sdr_bss  = pick(df.columns, "SDR (BSS mean)", exclude="mixture")
    col_pesq     = pick(df.columns, "PESQ (mean)")
    col_stoi     = pick(df.columns, "STOI (mean)")
    out = df[["Checkpoint", col_sdri_bss, col_sdr_bss, col_pesq, col_stoi]].rename(
        columns={
            col_sdri_bss: "SDRi_BSS",
            col_sdr_bss:  "SDR_BSS",
            col_pesq:     "PESQ",
            col_stoi:     "STOI",
        }
    )
    out["Checkpoint"] = out["Checkpoint"].astype(str)
    out.sort_values("Checkpoint", inplace=True)
    return out

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Make one-page summary board (clean, modular)")
    ap.add_argument("--out", default="metrics_viz/summary_board.png")
    ap.add_argument("--csv_1s",  default="metrics_viz/1s_noise/metrics_summary_1s_noise.csv")
    ap.add_argument("--csv_2sc", default="metrics_viz/2s_clean/metrics_summary_2s_clean.csv")
    ap.add_argument("--csv_2sn", default="metrics_viz/2s_noise/metrics_summary_2s_noise.csv")
    ap.add_argument("--orientation", choices=["landscape", "portrait"], default="portrait",
                    help="A4 page orientation for print")
    ap.add_argument("--png_dpi", type=int, default=300, help="PNG DPI (300 is print quality)")
    ap.add_argument("--tick_chars", type=int, default=3, help="Characters kept in checkpoint tick labels")
    ap.add_argument("--margins_mm", type=float, default=12.0, help="Print-safe margins in millimeters")
    return ap.parse_args()

# ===================== Figure Setup =====================

def make_figure(orientation: str, margins_mm: float) -> Tuple[plt.Figure, Tuple[float, float], GridSpec]:
    """Create an exact A4 canvas with constrained layout and print-safe margins; return fig, size, and GridSpec."""
    A4_PORTRAIT  = (8.27, 11.69)  # inches
    A4_LANDSCAPE = (11.69, 8.27)  # inches
    fig_size = A4_LANDSCAPE if orientation == "landscape" else A4_PORTRAIT

    fig = plt.figure(figsize=fig_size, layout="constrained")

    # Portable spacing
    try:
        le = fig.get_layout_engine()  # >=3.4
        if hasattr(le, "set"):
            le.set(w_pad=0.6, h_pad=0.6, wspace=0.25, hspace=0.35)
        else:
            raise AttributeError
    except Exception:
        plt.subplots_adjust(wspace=0.25, hspace=0.35)

    # Print margins
    W, H = fig_size
    inch = 1 / 25.4
    mm = margins_mm
    ml = mr = mt = mb = mm * inch
    fig.subplots_adjust(left=ml / W, right=1 - mr / W, bottom=mb / H, top=1 - mt / H)

    # 6-row grid: top charts, spacer, heatmap, spacer, table, footer
    gs = GridSpec(6, 3, figure=fig, height_ratios=[0.75, 0.18, 0.55, 0.16, 1.20, 0.10])
    return fig, fig_size, gs

# ===================== Plot Sections =====================


def plot_top_row(fig: plt.Figure, gs: GridSpec, tasks, tick_chars: int = 3, highlight_peak: bool = True) -> None:
    """Top 3 mini-charts: SDRi (BSS) vs paper benchmark, one per task, with optional best-epoch HORIZONTAL line."""
    for i, (task, df) in enumerate(tasks):
        ax = fig.add_subplot(gs[0, i])
        xs = list(range(len(df)))
        ys = df["SDRi_BSS"].to_numpy(dtype=float)

        # Base line
        ax.plot(xs, ys, marker="o", linewidth=1)
        ax.set_xticks(xs)
        ax.set_xticklabels([pretty_ckpt(x)[:tick_chars] for x in df["Checkpoint"]],
                           rotation=45, ha="right", fontsize=5)
        ax.tick_params(axis="y", labelsize=9)
        ax.set_title(f"{TITLE[task]} · SDR Improvement", fontsize=8, pad=8)
        if i == 0:
            ax.set_ylabel("dB", fontsize=9)

        # Benchmark line (horizontal)
        bench = PAPER_BENCH.get(task)
        if bench is not None:
            ax.axhline(bench, linestyle="--", color="red")

        # Find best epoch/value for this task
        best_ckpt_lbl, y_best = None, None
        if highlight_peak and np.any(~np.isnan(ys)):
            finite_idx = np.where(~np.isnan(ys))[0]
            j_best = finite_idx[np.argmax(ys[finite_idx])]
            y_best = float(ys[j_best])
            best_ckpt_lbl = pretty_ckpt(df["Checkpoint"].iloc[j_best])[:tick_chars]

        # Include both benchmark and best in limits so neither line clips
        finite_vals = [float(v) for v in ys if not np.isnan(v)]
        if bench is not None:
            finite_vals.append(float(bench))
        if y_best is not None:
            finite_vals.append(float(y_best))
        if finite_vals:
            lo, hi = min(finite_vals), max(finite_vals)
            pad = 0.2 * (hi - lo) if hi > lo else 1.0
            ax.set_ylim(lo - pad, hi + pad)

        # Annotate benchmark (clamped safely)
        if bench is not None and finite_vals:
            y0, y1 = ax.get_ylim()
            by = max(min(bench + 0.85, y1 - 0.05*(y1 - y0)), y0 + 0.05*(y1 - y0))
            ax.annotate(f"{bench:.1f}dB benchmark",
                        xy=(0.0, by), xycoords=("axes fraction", "data"),
                        xytext=(6, 0), textcoords="offset points",
                        va="center", ha="left", color="red", fontsize=8)

        # --- Best performance HORIZONTAL line + right-edge label ---
        if y_best is not None:
            ax.axhline(y_best, linestyle="--", color="black", linewidth=1.2, alpha=0.9)
            # Nudge label if it's very close to the benchmark to reduce overlap
            v_nudge = 10 if (bench is not None and abs(y_best - bench) < 0.4) else 0
            ax.annotate(f"{y_best:.1f}dB {best_ckpt_lbl}",
                        xy=(0.0, y_best), xycoords=("axes fraction", "data"),
                        xytext=(3.0, v_nudge+ + 5), textcoords="offset points",
                        ha="left", va="center", fontsize=6, color="black")
def plot_heatmap(
    fig: plt.Figure,
    gs: GridSpec,
    tasks: List[Tuple[T, pd.DataFrame]],
    tick_chars: int = 3,
    highlight_best: bool = True,
    weights: Tuple[float, float, float] | None = None,   # per-task weights if you want
):
    """Middle heatmap: delta to paper (SDRi BSS). Optionally highlight best column."""
    ax = fig.add_subplot(gs[2, :])

    # assume same number/order of checkpoints across tasks (sorted earlier)
    d1 = tasks[0][1]; d2 = tasks[1][1]; d3 = tasks[2][1]
    M = np.vstack([
        (d1["SDRi_BSS"] - PAPER_BENCH[T.S1_NOISE]).to_numpy(),
        (d2["SDRi_BSS"] - PAPER_BENCH[T.S2_CLEAN]).to_numpy(),
        (d3["SDRi_BSS"] - PAPER_BENCH[T.S2_NOISE]).to_numpy(),
    ])

    im = ax.imshow(M, aspect="auto")
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels([TITLE[T.S1_NOISE], TITLE[T.S2_CLEAN], TITLE[T.S2_NOISE]], fontsize=9)
    ax.set_xticks(list(range(len(d1))))
    ax.set_xticklabels([pretty_ckpt(x)[:tick_chars] for x in d1["Checkpoint"]],
                       rotation=28, ha="right", fontsize=9)
    ax.set_title("Δ to paper (SDRi BSS in dB): negative = below", fontsize=9, pad=6)

    # annotate cells
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:+.1f}",
                    ha="center", va="center",
                    fontsize=9,
                    color=("white" if M[i, j] < -4 else "black"))

    # colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Δ dB", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    # ---- Highlight best column (overall across the 3 tasks) ----
    if highlight_best and M.size:
        # column-wise score = weighted mean over rows (tasks)
        if weights is None:
            w = np.ones(M.shape[0], dtype=float)
        else:
            w = np.asarray(weights, dtype=float)
            if w.shape != (M.shape[0],):
                raise ValueError(f"weights must have length {M.shape[0]} (got {w.shape})")
        # normalize weights (ignore NaNs)
        w = w / w.sum()

        # nan-aware weighted mean across rows
        scores = []
        for j in range(M.shape[1]):
            col = M[:, j]
            mask = ~np.isnan(col)
            if not np.any(mask):
                scores.append(-np.inf)
            else:
                ww = w[mask] / w[mask].sum()
                scores.append(float(np.sum(col[mask] * ww)))
        best_j = int(np.argmax(scores))

        # draw a rectangle frame around the best column
        # Rectangle coords in imshow space: (x0, y0) with width=1, height=#rows
        rect = Rectangle((best_j - 0.5, -0.5),
                         width=1.0, height=M.shape[0],
                         fill=False, linewidth=2.0, edgecolor="yellow", linestyle="--")
        ax.add_patch(rect)

        # # add a small star marker below the x tick (purely decorative)
        # ax.annotate("★ best overall",
        #             xy=(best_j, M.shape[0] - 0.5), xycoords=("data", "data"),
        #             xytext=(0, 12), textcoords="offset points",
        #             ha="center", va="bottom", fontsize=9, color="tab:black")

def plot_table(fig: plt.Figure, gs: GridSpec, tasks: List[Tuple[T, pd.DataFrame]],
               tick_chars: int = 3) -> None:
    """Lower table: SDR (BSS mean), PESQ, STOI."""
    ax = fig.add_subplot(gs[4, :])
    ax.axis("off")
    rows = []
    for task, df in tasks:
        for _, r in df.iterrows():
            rows.append([
                TITLE[task], pretty_ckpt(r["Checkpoint"])[:tick_chars],
                f"{r['SDR_BSS']:.2f}", f"{r['PESQ']:.2f}", f"{r['STOI']:.3f}"
            ])
    col_labels = ["Task", "Checkpoint", "SDR (BSS mean)", "PESQ (mean)", "STOI (mean)"]
    tbl = ax.table(cellText=rows, colLabels=col_labels, loc="center",
                   cellLoc="center", colLoc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(4)
    tbl.scale(1, 1.15)

def plot_footer(fig: plt.Figure, gs: GridSpec) -> None:
    """Footer notes (dedicated row avoids overlaps)."""
    ax = fig.add_subplot(gs[5, :])
    ax.axis("off")
    footer = ("Note: top row shows SDRi (BSS). Heatmap = delta vs paper SDRi. Table = supporting metrics.\n"
              "Benchmarks: 1s+noise = 16.0 dB, 2s clean = 10.3 dB, 2s+noise = 10.6 dB.")
    ax.text(0.0, 0.5, footer, fontsize=9, va="center", ha="left", transform=ax.transAxes)

def set_title(fig: plt.Figure) -> None:
    fig.suptitle("Model Checkpoints · AVSpeech Synthetic Tasks (Apples-to-Apples, BSS Eval)",
                 fontsize=9, y=0.98)

# ===================== Orchestration =====================

def build_board(d1: pd.DataFrame, d2: pd.DataFrame, d3: pd.DataFrame,
                orientation: str, margins_mm: float, tick_chars: int) -> Tuple[plt.Figure, Tuple[float, float]]:
    """Create full A4 board and return (figure, size)."""
    fig, fig_size, gs = make_figure(orientation=orientation, margins_mm=margins_mm)
    tasks = [(T.S1_NOISE, d1), (T.S2_CLEAN, d2), (T.S2_NOISE, d3)]

    plot_top_row(fig, gs, tasks, tick_chars=tick_chars)
    # spacer rows are already in GridSpec (1 and 3)
    plot_heatmap(fig, gs, tasks, tick_chars=tick_chars, highlight_best=True)
    plot_table(fig, gs, tasks, tick_chars=tick_chars)
    plot_footer(fig, gs)
    set_title(fig)
    return fig, fig_size

# ===================== Entry =====================

def main():
    args = parse_args()

    d1 = load_task(Path(args.csv_1s))
    d2 = load_task(Path(args.csv_2sc))
    d3 = load_task(Path(args.csv_2sn))

    fig, _ = build_board(d1, d2, d3,
                         orientation=args.orientation,
                         margins_mm=args.margins_mm,
                         tick_chars=args.tick_chars)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.png_dpi)           # PNG @ 300 DPI (default)
    fig.savefig(out_path.with_suffix(".pdf"))         # vector PDF, exact A4 page size
    print(f"Saved {out_path}")

if __name__ == "__main__":
    main()