#!/usr/bin/env python3
"""
visualize_checkpoints_tasks.py

Evaluate model checkpoints (.pt) on THREE tasks (1s_noise, 2s_clean, 2s_noise),
compute averages (mean & median) for SDR, SDRi, SI-SDR, PESQ, STOI, and *BSS Eval* SDR/SDRi,
and generate per-metric plots with task-appropriate benchmark lines from the paper.

Inputs:
- checkpoints_dir: folder with .pt checkpoint files (sorted alphabetically)
- test_1s_noise_dir: folder with 3s samples (audio/clean_embs.pt, audio/mixture_embs.pt OR audio/audio_embs.pt, face/face_embs.pt)
- test_2s_clean_dir: same structure
- test_2s_noise_dir: same structure

Outputs:
- metrics_viz/<task>/metrics_summary_<task>.csv
- metrics_viz/<task>/<METRIC>.png
- metrics_viz/<task>/<checkpoint>_per_sample.csv
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Enable CPU fallback for missing MPS ops
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# Project-specific imports
from avspeech.model.av_model import AudioVisualModel
import metrics
import metrics_bss


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
    model = AudioVisualModel().to(device)
    checkpoint = torch.load(ckpt_file, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.eval()
    return model


def _find_samples(root: Path, limit : int = 500) -> List[Path]:
    def has_required(d: Path) -> bool:
        a = d / "audio"
        f = d / "face"
        return (
            a.is_dir() and f.is_dir() and
            ((a / "mixture_embs.pt").exists() or (a / "audio_embs.pt").exists()) and
            (a / "clean_embs.pt").exists() and
            (f / "face_embs.pt").exists()
        )

    samples = []
    for i, d in enumerate(root.rglob("*")):
        if i >= limit:
            break
        if d.is_dir() and has_required(d):
            samples.append(d)

    return sorted(samples, key=lambda p: p.name)


def _safe_audio_cpu(x: torch.Tensor) -> torch.Tensor:
    x_cpu = x.detach().to("cpu")
    return stft_to_audio(x_cpu)


def evaluate_sample(model: AudioVisualModel, sample_dir: Path, device: str) -> Dict[str, float]:
    """Compute all metrics (including BSS Eval) for a single 3s sample."""
    a = sample_dir / "audio"
    f = sample_dir / "face"
    clean = torch.load(a / "clean_embs.pt", map_location="cpu").unsqueeze(0).to(device)
    mix_path = a / "mixture_embs.pt"
    if not mix_path.exists():
        alt = a / "audio_embs.pt"
        if alt.exists():
            mix_path = alt
        else:
            raise FileNotFoundError(f"Missing mixture STFT: {a}/mixture_embs.pt (or audio_embs.pt)")
    mixture = torch.load(mix_path, map_location="cpu").unsqueeze(0).to(device)
    face = torch.load(f / "face_embs.pt", map_location="cpu").unsqueeze(0).to(device)

    with torch.no_grad():
        mask = model(mixture, face)
        separated = mixture * mask

    # Reconstruct on CPU
    sep_audio = _safe_audio_cpu(separated.squeeze(0))
    clean_audio = _safe_audio_cpu(clean.squeeze(0))
    mix_audio = _safe_audio_cpu(mixture.squeeze(0))

    # Compute metrics
    out = {
        "sample_id": sample_dir.name,
        # your original metrics (may be SI-like)
        "sdr": float(metrics.sdr(sep_audio, clean_audio)),
        "sdr_improvement": float(metrics.sdr_improvement(sep_audio, clean_audio, mix_audio)),
        "si_sdr": float(metrics.si_sdr(sep_audio, clean_audio)),
        "pesq": float(metrics.pesq(sep_audio, clean_audio)),
        "stoi": float(metrics.stoi(sep_audio, clean_audio)),
    }
    # BSS Eval (apples-to-apples with paper)
    try:
        out["sdr_bss"] = float(metrics_bss.sdr_bss(sep_audio, clean_audio))
        out["sdr_improvement_bss"] = float(metrics_bss.sdri_bss(sep_audio, clean_audio, mix_audio))
        out["mixture_sdr_bss"] = float(metrics_bss.sdr_bss(mix_audio, clean_audio))
    except Exception as e:
        # If mir_eval/museval not installed, we at least keep the other metrics
        out["sdr_bss"] = None
        out["sdr_improvement_bss"] = None
        out["mixture_sdr_bss"] = None
        print(f"[WARN] BSS Eval unavailable for {sample_dir}: {e}")

    return out


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


def agg_stats(rows: List[Dict[str, float]]) -> Dict[str, Optional[float]]:
    if not rows:
        return {}
    keys = set().union(*(r.keys() for r in rows)) - {"sample_id"}
    out: Dict[str, Optional[float]] = {}
    for k in keys:
        vals = [r[k] for r in rows if r.get(k) is not None]
        if not vals:
            out[k] = None
            out[f"{k}_median"] = None
            continue
        vals = [float(v) for v in vals]
        vals.sort()
        out[k] = sum(vals) / len(vals)
        mid = len(vals) // 2
        out[f"{k}_median"] = vals[mid] if len(vals) % 2 == 1 else 0.5 * (vals[mid - 1] + vals[mid])
    return out


def plot_metric(metric_key: str,
                values: List[Optional[float]],
                labels: List[str],
                benchmark_value: Optional[float],
                out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = list(range(len(values)))
    y = [v if v is not None else float("nan") for v in values]
    ax.plot(x, y, marker="o", linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")

    pretty = metric_key.replace("_", " ").upper().replace("BSS", "BSS")
    if metric_key == "sdr_improvement_bss":
        pretty = "SDR Improvement (BSS Eval)"
    elif metric_key == "sdr_bss":
        pretty = "SDR (BSS Eval)"
    elif metric_key == "si_sdr":
        pretty = "SI-SDR"

    ax.set_title(f"Average {pretty}")
    ax.set_ylabel(pretty)
    ax.grid(True, axis="y", alpha=0.3)

    if benchmark_value is not None:
        ax.axhline(benchmark_value, linestyle="--", color="red", label="Paper benchmark")
        ax.legend(loc="best")

    # y-limits (include benchmark if present so the line is visible)
    finite = [v for v in y if v == v]  # drop NaNs
    if benchmark_value is not None:
        finite.append(float(benchmark_value))
    if finite:
        lo, hi = min(finite), max(finite)
        pad = 0.1 * (hi - lo) if hi > lo else 1.0
        if metric_key in ("pesq", "stoi"):
            low, high = (1.0, 4.5) if metric_key == "pesq" else (0.0, 1.0)
            ax.set_ylim(low, high)
        else:
            ax.set_ylim(lo - pad, hi + pad)

    # ... then compute y-limits including the benchmark so it never clips:
    finite = [v for v in y if v == v]
    if benchmark_value is not None:
        finite.append(float(benchmark_value))
    if finite:
        lo, hi = min(finite), max(finite)
        pad = 0.1 * (hi - lo) if hi > lo else 1.0
        if metric_key in ("pesq", "stoi"):
            ax.set_ylim((1.0, 4.5) if metric_key == "pesq" else (0.0, 1.0))
        else:
            ax.set_ylim(lo - pad, hi + pad)

    # annotate the benchmark value at the right margin
    if benchmark_value is not None:
        ax.annotate(
            f"{benchmark_value:.1f} dB",
            xy=(1.0, benchmark_value),
            xycoords=("axes fraction", "data"),  # x in axes coords, y in data coords
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            color="red",
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_csv(path, summaries):
    import csv

    header = [
        "Checkpoint",
        "Mixture SDR (BSS mean)", "Mixture SDR (BSS median)",
        "SDR (BSS mean)", "SDR (BSS median)",
        "SDRi (BSS mean)", "SDRi (BSS median)",
        "SDR (mean)", "SDR (median)",
        "SDRi (mean)", "SDRi (median)",
        "SI-SDR (mean)", "SI-SDR (median)",
        "PESQ (mean)", "PESQ (median)",
        "STOI (mean)", "STOI (median)",
    ]

    def fmt(x):
        try:
            return f"{float(x):.4f}"
        except Exception:
            return "nan"

    with open(path, "w", newline="") as f:
        w = csv.writer(f)  # handles quoting safely
        w.writerow(header)
        for name, avg in summaries:
            row = [
                name,
                fmt(avg.get("mixture_sdr_bss")),
                fmt(avg.get("mixture_sdr_bss_median")),
                fmt(avg.get("sdr_bss")),
                fmt(avg.get("sdr_bss_median")),
                fmt(avg.get("sdr_improvement_bss")),
                fmt(avg.get("sdr_improvement_bss_median")),
                fmt(avg.get("sdr")),
                fmt(avg.get("sdr_median")),
                fmt(avg.get("sdr_improvement")),
                fmt(avg.get("sdr_improvement_median")),
                fmt(avg.get("si_sdr")),
                fmt(avg.get("si_sdr_median")),
                fmt(avg.get("pesq")),
                fmt(avg.get("pesq_median")),
                fmt(avg.get("stoi")),
                fmt(avg.get("stoi_median")),
            ]
            w.writerow(row)
def run_task(task_name: str,
             test_dir: Path,
             ckpt_files: List[Path],
             device: str,
             out_root: Path,
             benchmarks: Dict[str, Dict[str, Optional[float]]]) -> None:
    print(f"\n=== Task: {task_name} ===")
    samples = _find_samples(test_dir, limit=500)  # limit to 100 samples for speed
    print(f"Discovered {len(samples)} samples under {test_dir}")
    out_dir = out_root / task_name
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: List[Tuple[str, Dict[str, Optional[float]]]] = []
    for ckpt in ckpt_files:
        name = ckpt.stem
        print(f"Evaluating checkpoint: {name}")
        model = load_model_from_checkpoint(ckpt, device)
        res = evaluate_model(model, samples, device)
        avg = agg_stats(res)

        bench = benchmarks.get(task_name, {}).get("sdr_improvement_bss")
        if bench is not None and avg.get("sdr_improvement_bss") is not None:
            avg["sdri_bss_delta_to_paper"] = avg["sdr_improvement_bss"] - float(bench)

        summaries.append((name, avg))

        # per-checkpoint per-sample CSV
        import csv
        cols = ["sample_id","mixture_sdr_bss","sdr_bss","sdr_improvement_bss","sdr","sdr_improvement","si_sdr","pesq","stoi"]
        with open(out_dir / f"{name}_per_sample.csv", "w", newline="") as fcsv:
            w = csv.DictWriter(fcsv, fieldnames=cols)
            w.writeheader()
            for r in res:
                w.writerow({k: r.get(k) for k in cols})

    # Write summary CSV
    sum_csv = out_dir / f"metrics_summary_{task_name}.csv"
    write_csv(sum_csv, summaries)
    print(f"Wrote summary CSV: {sum_csv}")

    # Plots (means)
    summaries.sort(key=lambda t: t[0])
    labels = [n for n, _ in summaries]
    for metric in ["sdr_improvement_bss", "sdr_bss", "sdr", "sdr_improvement", "si_sdr", "pesq", "stoi"]:
        vals = [avg.get(metric) for _, avg in summaries]
        bench = benchmarks.get(task_name, {}).get(metric)
        plot_metric(metric, vals, labels, bench, out_dir / f"{metric.upper()}.png")
    print(f"Saved plots to: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Visualize checkpoints on 3 tasks with BSS Eval metrics")
    parser.add_argument("checkpoints_dir", type=str, help="Folder with .pt checkpoint files")
    parser.add_argument("test_1s_noise_dir", type=str, help="Folder with 1s+noise samples")
    parser.add_argument("test_2s_clean_dir", type=str, help="Folder with 2s clean samples")
    parser.add_argument("test_2s_noise_dir", type=str, help="Folder with 2s+noise samples")
    parser.add_argument("--output", type=str, default="metrics_viz", help="Root output folder")
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--bench_json", type=str, default="", help="Optional JSON with benchmark lines per task/metric")
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoints_dir)
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Checkpoints folder not found: {ckpt_dir}")
    ckpt_files = sorted([p for p in ckpt_dir.iterdir() if p.is_file() and p.suffix == ".pt"], key=lambda p: p.name)
    if len(ckpt_files) == 0:
        print("[WARN] No .pt files found.")
    elif len(ckpt_files) != 4:
        print(f"[WARN] Expected 4 checkpoints, found {len(ckpt_files)}. Proceeding.")

    # device
    if args.device != "auto":
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device} (model forward); ISTFT+metrics on CPU")

    # Output root
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    # Default benchmarks from the paper (AVSpeech synthetic tasks)
    # These are SDR improvement (BSS Eval) lines per task.
    default_benchmarks = {
        "1s_noise": {
            "sdr_improvement_bss": 16.0,  # Table 3: 1S+Noise, SDRi (BSS Eval)
            # other metrics aren't reported by paper for AVSpeech synthetic tasks -> keep None
        },
        "2s_clean": {
            "sdr_improvement_bss": 10.3,  # Table 3: 2S clean, SDRi (BSS Eval)
        },
        "2s_noise": {
            "sdr_improvement_bss": 10.6,  # Table 3: 2S+Noise, SDRi (BSS Eval)
        },
    }

    # Optional override via JSON
    benchmarks = default_benchmarks
    if args.bench_json:
        try:
            with open(args.bench_json, "r") as f:
                user_bench = json.load(f)
            # deep-merge
            for task, d in user_bench.items():
                benchmarks.setdefault(task, {}).update(d)
            print(f"Loaded benchmark overrides from {args.bench_json}")
        except Exception as e:
            print(f"[WARN] Could not load bench_json: {e}. Using defaults.")

    # Run all tasks
    run_task("1s_noise", Path(args.test_1s_noise_dir), ckpt_files, device, out_root, benchmarks)
    run_task("2s_clean", Path(args.test_2s_clean_dir), ckpt_files, device, out_root, benchmarks)
    run_task("2s_noise", Path(args.test_2s_noise_dir), ckpt_files, device, out_root, benchmarks)


if __name__ == "__main__":
    main()

    """
    
    QUICK 50 SAMPLES PER TYPE TEST SETS
    # 1s_noise test samples = /Users/jonatanvider/Desktop/Look2Listen_Stuff/small_test_datasets/validation/1s_noise
    # 2s_clean test samples = /Users/jonatanvider/Desktop/Look2Listen_Stuff/small_test_datasets/validation/2s_clean
    # 2s_noise test samples = /Users/jonatanvider/Desktop/Look2Listen_Stuff/small_test_datasets/validation/2s_noise
    
    BEEFY 2000+ PER TYPE TEST SETS
    /Users/jonatanvider/Downloads/AVSpeech/clips/xar/1s_noise
    /Users/jonatanvider/Downloads/AVSpeech/clips/xar/2s_clean
    /Users/jonatanvider/Downloads/AVSpeech/clips/xar/2s_noise

    #checkpoints_dir = "/Users/jonatanvider/Desktop/Look2Listen_Stuff/checkpoints"
    #                   /Users/jonatanvider/Desktop/Look2Listen_Stuff/checkpoints

    
python visualize_checkpoints_tasks.py /Users/jonatanvider/Desktop/Look2Listen_Stuff/checkpoints /Users/jonatanvider/Desktop/Look2Listen_Stuff/small_test_datasets/validation/1s_noise Users/jonatanvider/Desktop/Look2Listen_Stuff/small_test_datasets/validation/2s_clean Users/jonatanvider/Desktop/Look2Listen_Stuff/small_test_datasets/validation/2s_noise
  --output metrics_viz \
  --device auto             # or: cpu / mps / cuda
  # --bench_json paper_bench.json  # optional override"""
