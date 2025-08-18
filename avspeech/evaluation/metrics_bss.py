"""
metrics_bss.py
Upgrade of metrics to include BSS Eval SDR and SDR Improvement compatible with the paper.
Keeps simple torch/np interfaces and tolerates torch.Tensors or numpy arrays.

Dependencies:
- Prefer: mir_eval  (pip install mir-eval)
- Fallback: museval  (pip install museval)

Note:
A "sample" in your pipeline is a 3-second segment already, matching the paper's evaluation window.
So global BSS Eval SDR on the 3-second sample is equivalent to the paper's 3s window aggregation.
"""

from typing import Tuple, Optional, Union
import torch
import numpy as np

# Prefer mir_eval; fallback to museval if not available
_BACKEND = None
try:
    import mir_eval.separation as mirsep  # type: ignore
    _BACKEND = "mir_eval"
except Exception:
    try:
        import museval  # type: ignore
        _BACKEND = "museval"
    except Exception:
        _BACKEND = None

def _to_numpy_1d(x: Union[np.ndarray,torch.Tensor]) -> np.ndarray:
    """Ensure 1D numpy array (mono). Trim to shortest length if needed externally."""
    try:
        import torch  # local import to avoid hard dep
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    except Exception:
        pass
    x = np.asarray(x).squeeze()
    # If multi-channel, take mean as mono (or you can raise an error if not expected).
    if x.ndim > 1:
        x = x.mean(axis=-1)
    return x.astype(np.float64, copy=False)

def _align_lengths(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = min(len(a), len(b))
    return a[:n], b[:n]

def sdr_bss(est: Union[np.ndarray,torch.Tensor],
            ref: Union[np.ndarray,torch.Tensor]) -> float:
    """
    BSS Eval SDR for single-source case, in dB.
    Uses mir_eval if available; else museval; else raises.
    """
    if _BACKEND is None:
        raise ImportError("Need mir-eval or museval installed. Try: pip install mir-eval  (or: pip install museval)")

    est = _to_numpy_1d(est)
    ref = _to_numpy_1d(ref)
    ref, est = _align_lengths(ref, est)

    if _BACKEND == "mir_eval":
        # shapes: (nsrc, nsamples)
        res = mirsep.bss_eval_sources(ref[None, :], est[None, :])
        # mir_eval returns (sdr, sir, sar, perm)
        if len(res) == 4:
            sdr, sir, sar, perm = res
        elif len(res) == 5:  # just-in-case variant
            sdr, isr, sir, sar, perm = res
        else:
            raise RuntimeError(f"Unexpected bss_eval_sources return of length {len(res)}")
        return float(sdr[0])
    else:
        # museval expects (nsrc, nsamples) as well
        import museval
        sdr, isr, sir, sar = museval.metrics.bss_eval_sources_framewise(ref[None, :], est[None, :])
        # Take the mean over frames (our segments are already 3s; this is robust if framewise is returned)
        s = np.nanmean(sdr[0]) if np.ndim(sdr[0]) > 0 else sdr[0]
        return float(s)

def sdri_bss(est: Union[np.ndarray,torch.Tensor],
             ref: Union[np.ndarray,torch.Tensor],
             mix: Union[np.ndarray,torch.Tensor]) -> float:
    """
    SDR Improvement using BSS Eval: SDR(est,ref) - SDR(mix,ref).
    """
    return sdr_bss(est, ref) - sdr_bss(mix, ref)
