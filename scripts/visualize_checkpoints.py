#!/usr/bin/env python3
"""
visualize_checkpoints.py

Evaluate 4 (or more) model checkpoints (.pt files) on a folder of test samples,
compute average metrics (SDR, SDR Improvement, SI-SDR, PESQ, STOI),
and generate per-metric plots with an external benchmark line.

- Checkpoints directory: contains .pt files (sorted alphabetically)
- Test clips directory: contains subfolders; each has:
    audio/clean_embs.pt
    audio/mixture_embs.pt
    face/face_embs.pt

Outputs:
- metrics_viz/metrics_summary.csv
- metrics_viz/<METRIC>.png for each metric
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
import matplotlib
matplotlib.use("TkAgg")  # TkAgg is fine on macOS; plots are saved (no GUI needed)
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import the audio-visual model and metrics module
from avspeech.model.av_model import AudioVisualModel
import avspeech.evaluation.metrics as metrics


def stft_to_audio(stft_compressed: torch.Tensor,
                  n_fft: int = 512,
                  hop_length: int = 160,
                  win_length: int = 400,
                  power_law_p: float = 0.3) -> torch.Tensor:
    """
    Convert compressed STFT back to a time-domain audio waveform.
    Expects last dim=2 for real/imag, shape [freq, time, 2] or [B, freq, time, 2].
    """
    if stft_compressed.dim() == 4:
        # Assume [B, F, T, 2] -> process batch elem-wise then cat
        audios = []
        for b in range(stft_compressed.size(0)):
            audios.append(stft_to_audio(stft_compressed[b], n_fft, hop_length, win_length, power_law_p))
        return torch.stack(audios, dim=0)

    if stft_compressed.dim() != 3 or stft_compressed.shape[-1] != 2:
        raise ValueError("Expected STFT shape [freq, time, 2] or [B, freq, time, 2].")

    real_c = stft_compressed[..., 0]
    imag_c = stft_compressed[..., 1]

    # Invert power-law compression
    inv_p = 1.0 / power_law_p
    real = torch.sign(real_c) * torch.abs(real_c).pow(inv_p)
    imag = torch.sign(imag_c) * torch.abs(imag_c).pow(inv_p)

    stft_complex = torch.complex(real, imag)
    window = torch.hann_window(win_length, device=stft_complex.device)

    audio = torch.istft(
        stft_complex,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        normalized=False,
        onesided=True,
        return_complex=False,
    )
    return audio


def load_model_from_checkpoint(ckpt_file: Path, device: str) -> AudioVisualModel:
    """Load model from a .pt checkpoint file (.get('model_state_dict') fallback)."""
    model = AudioVisualModel().to(device)
    checkpoint = torch.load(ckpt_file, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.eval()
    return model


def evaluate_sample(model: AudioVisualModel, sample_dir: Path, device: str) -> Dict[str, float]:
    """Run inference on one test sample directory and compute metrics."""

    audio_dir = sample_dir / "audio"
    face_dir = sample_dir / "face"
    clean = torch.load(audio_dir / "clean_embs.pt", map_location="cpu").unsqueeze(0).to(device)
    # Support either mixture_embs.pt or audio_embs.pt (naming differences)
    mix_path = audio_dir / "mixture_embs.pt"
    if not mix_path.exists():
        alt = audio_dir / "audio_embs.pt"
        if alt.exists():
            mix_path = alt
        else:
            raise FileNotFoundError(f"Missing mixture STFT: {audio_dir}/mixture_embs.pt (or audio_embs.pt)")
    mixture = torch.load(mix_path, map_location="cpu").unsqueeze(0).to(device)
    face = torch.load(face_dir / "face_embs.pt", map_location="cpu").unsqueeze(0).to(device)

    with torch.no_grad():
        mask = model(mixture, face)          # [B, F, T, 2] mask
        separated = mixture * mask           # apply mask to mixture STFT


    # Convert to time-domain (CPU to avoid MPS op gaps)
    def safe_audio(x):
        x_cpu = x.detach().to("cpu")
        try:
            return stft_to_audio(x_cpu)
        except Exception as e:
            raise RuntimeError(f"ISTFT failed on CPU: {e}")
    separated_audio = safe_audio(separated.squeeze(0))
    clean_audio = safe_audio(clean.squeeze(0))
    mixture_audio = safe_audio(mixture.squeeze(0))

    # Compute metrics via your metrics.py (expect CPU tensors)

    return {
        "sdr": metrics.sdr(separated_audio, clean_audio),
        "sdr_improvement": metrics.sdr_improvement(separated_audio, clean_audio, mixture_audio),
        "si_sdr": metrics.si_sdr(separated_audio, clean_audio),
        "pesq": metrics.pesq(separated_audio, clean_audio),
        "stoi": metrics.stoi(separated_audio, clean_audio),
    }



def evaluate_model(model: AudioVisualModel, samples: List[Path], device: str) -> List[Dict[str, float]]:
    out: List[Dict[str, float]] = []
    failed = 0
    for s in tqdm(samples, desc="Evaluating samples"):
        try:
            out.append(evaluate_sample(model, s, device))
        except Exception as e:
            failed += 1
            print(f"[WARN] Skipping {s}: {type(e).__name__}: {e}")
    print(f"Finished: {len(out)} ok, {failed} failed")
    return out


def average_metrics(rows: List[Dict[str, float]]) -> Dict[str, Optional[float]]:
    if not rows:
        return {}
    keys = rows[0].keys()
    avgs: Dict[str, Optional[float]] = {}
    for k in keys:
        vals = [r[k] for r in rows if r.get(k) is not None]
        avgs[k] = sum(vals) / len(vals) if vals else None
    return avgs


def line_plot_metric(metric_key: str,
                     values: List[Optional[float]],
                     labels: List[str],
                     benchmark_value: Optional[float],
                     out_dir: Path) -> None:
    """Plot metric as a simple line+markers with a horizontal benchmark line."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = list(range(len(values)))
    y = [v if v is not None else float("nan") for v in values]

    ax.plot(x, y, marker="o", linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")

    title = metric_key.replace("_", " ").upper() if metric_key != "si_sdr" else "SI-SDR"
    if metric_key == "sdr_improvement":
        title = "SDR Improvement"

    ax.set_title(f"Average {title} per Checkpoint")
    ax.set_ylabel(title)
    ax.grid(True, axis="y", alpha=0.3)

    if benchmark_value is not None:
        ax.axhline(benchmark_value, linestyle="--", color="red", label="Benchmark")
        ax.legend(loc="best")

    # reasonable y-limits
    finite_vals = [v for v in y if v == v]  # filter NaN
    if finite_vals:
        lo, hi = min(finite_vals), max(finite_vals)
        pad = 0.1 * (hi - lo) if hi > lo else 1.0
        if metric_key in ("pesq", "stoi"):
            # known/common bounds
            if metric_key == "pesq":
                ax.set_ylim(1.0, 4.5)
            else:
                ax.set_ylim(0.0, 1.0)
        else:
            ax.set_ylim(lo - pad, hi + pad)

    fname = metric_key.upper() if metric_key != "sdr_improvement" else "SDR_improvement"
    fig.tight_layout()
    fig.savefig(out_dir / f"{fname}.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize checkpoints vs benchmark")
    parser.add_argument("checkpoints_dir", type=str, help="Folder with .pt checkpoint files")
    parser.add_argument("test_clips_dir", type=str, help="Folder with pre-processed test sample subfolders")
    parser.add_argument("--output", type=str, default="metrics_viz", help="Output folder for CSV/plots")
    parser.add_argument("--benchmark_json", type=str, default="", help="Optional JSON file with benchmark values")
    parser.add_argument("--inspect", action="store_true", help="Print a few discovered sample dirs and exit")
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "mps", "cpu"], default="auto", help="Force device (default auto)")
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoints_dir)
    test_dir = Path(args.test_clips_dir)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Checkpoints folder not found: {ckpt_dir}")
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test clips folder not found: {test_dir}")

    # Gather .pt files (alphabetical)
    ckpt_files = sorted([p for p in ckpt_dir.iterdir() if p.is_file() and p.suffix == ".pt"], key=lambda p: p.name)
    if len(ckpt_files) == 0:
        print("[WARN] No .pt files found in checkpoints_dir.")
    if len(ckpt_files) != 4:
        print(f"[WARN] Expected 4 checkpoints; found {len(ckpt_files)}. Proceeding with what we found.")

    # Device selection (prefer CUDA, then MPS, then CPU)
    if args.device != "auto":
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device} (model forward); ISTFT + metrics on CPU)")
    # Load benchmark values (defaults from paper Table 5: Mandarin enhancement)
    benchmark = {
        "sdr": 6.1,
        "sdr_improvement": 6.1,   # use same reference unless you provide a per-task value
        "si_sdr": 6.1,            # same note as above
        "pesq": 2.5,
        "stoi": 0.71,
    }
    if args.benchmark_json:
        try:
            with open(args.benchmark_json, "r") as f:
                user_bench = json.load(f)
            benchmark.update({k: user_bench.get(k, v) for k, v in benchmark.items()})
        except Exception as e:
            print(f"[WARN] Failed to read benchmark_json: {e}. Using defaults.")


    # Collect test sample subdirectories (recursively) that contain required files
    def has_required_files(d: Path) -> bool:
        audio_dir = d / "audio"
        face_dir = d / "face"
        return (
            audio_dir.is_dir() and face_dir.is_dir() and
            ( (audio_dir / "mixture_embs.pt").exists() or (audio_dir / "audio_embs.pt").exists() ) and
            (audio_dir / "clean_embs.pt").exists() and
            (face_dir / "face_embs.pt").exists()
        )

    sample_dirs = [d for d in test_dir.rglob("*") if d.is_dir() and has_required_files(d)]
    sample_dirs = sorted(sample_dirs, key=lambda p: p.name)
    if not sample_dirs:
        raise RuntimeError(f"No valid test samples found under: {test_dir}\n"
                           f"Expected per-sample folder with:\n"
                           f"  audio/clean_embs.pt\n"
                           f"  audio/mixture_embs.pt (or audio/audio_embs.pt)\n"
                           f"  face/face_embs.pt")
    print(f"Discovered {len(sample_dirs)} valid samples (recursive search).")
    if args.inspect:
        for d in sample_dirs[:5]:
            audio_dir = d / "audio"
            face_dir = d / "face"
            print(f"  - {d}")
            print(f"      clean_embs.pt: {(audio_dir / 'clean_embs.pt').exists()}")
            print(f"      mixture_embs.pt: {(audio_dir / 'mixture_embs.pt').exists()}")
            print(f"      audio_embs.pt (alt): {(audio_dir / 'audio_embs.pt').exists()}")
            print(f"      face_embs.pt: {(face_dir / 'face_embs.pt').exists()}")
        return


    # Evaluate all checkpoints
    summaries: List[tuple[str, Dict[str, Optional[float]]]] = []
    for ckpt_file in ckpt_files:
        model_name = ckpt_file.stem
        print(f"\nEvaluating checkpoint: {model_name}")
        model = load_model_from_checkpoint(ckpt_file, device)
        results = evaluate_model(model, sample_dirs, device)
        avg = average_metrics(results)
        summaries.append((model_name, avg))
        avg_str = ", ".join(f"{k}={v:.3f}" for k, v in avg.items() if v is not None)
        print(f"  Averages: {avg_str}")

    # Write summary CSV
    csv_path = out_dir / "metrics_summary.csv"
    with open(csv_path, "w") as f:
        f.write("Checkpoint,SDR,SDR Improvement,SI-SDR,PESQ,STOI\n")
        for name, avg in summaries:
            f.write(f"{name},"
                    f"{(avg.get('sdr') or float('nan')):.4f},"
                    f"{(avg.get('sdr_improvement') or float('nan')):.4f},"
                    f"{(avg.get('si_sdr') or float('nan')):.4f},"
                    f"{(avg.get('pesq') or float('nan')):.4f},"
                    f"{(avg.get('stoi') or float('nan')):.4f}\n")
    print(f"\nSaved CSV: {csv_path}")

    # Generate plots per metric

    if summaries:
        summaries.sort(key=lambda t: t[0])  # sort by name for visual consistency
        labels = [name for name, _ in summaries]
        for metric in ["sdr", "sdr_improvement", "si_sdr", "pesq", "stoi"]:
            values = [avg.get(metric) for _, avg in summaries]
            if all(v is None or (isinstance(v, float) and (v != v)) for v in values):  # all NaN/None
                print(f"[WARN] Skipping plot for {metric}: all values are NaN/None.")
                continue
            line_plot_metric(metric, values, labels, benchmark.get(metric), out_dir)
        print(f"Saved plots to: {out_dir}")
    else:
        print("No summaries to plot; nothing saved.")

if __name__ == "__main__":
    main()


    # 1s_noise test samples = /Users/jonatanvider/Desktop/Look2Listen_Stuff/small_test_datasets/validation/1s_noise
    # 2s_clean test samples = /Users/jonatanvider/Desktop/Look2Listen_Stuff/small_test_datasets/validation/2s_clean
    # 2s_noise test samples = /Users/jonatanvider/Desktop/Look2Listen_Stuff/small_test_datasets/validation/2s_noise

    #checkpoints_dir = "/Users/jonatanvider/Desktop/Look2Listen_Stuff/checkpoints"
    #                   /Users/jonatanvider/Desktop/Look2Listen_Stuff/checkpoints