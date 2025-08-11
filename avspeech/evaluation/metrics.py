# metrics.py
import torch
import numpy as np
from typing import Optional, Union
from pystoi import stoi as compute_stoi
from pesq import pesq as compute_pesq


def sdr(estimated: torch.Tensor, target: torch.Tensor) -> float:
    """Signal-to-Distortion Ratio (SDR)"""
    min_len = min(len(estimated), len(target))
    estimated = estimated[:min_len].flatten()
    target = target[:min_len].flatten()

    alpha = torch.dot(estimated, target) / (torch.dot(target, target) + 1e-8)
    s_target = alpha * target
    e_noise = estimated - s_target

    sdr_value = 10 * torch.log10(
            torch.sum(s_target ** 2) / (torch.sum(e_noise ** 2) + 1e-8)
    )
    return sdr_value.item()


def sdr_improvement(estimated: torch.Tensor, target: torch.Tensor, mixture: torch.Tensor) -> float:
    """SDR Improvement (SDRi) - main metric from paper"""
    sdr_est = sdr(estimated, target)
    sdr_mix = sdr(mixture, target)
    return sdr_est - sdr_mix


def si_sdr(estimated: torch.Tensor, target: torch.Tensor) -> float:
    """Scale-Invariant SDR"""
    min_len = min(len(estimated), len(target))
    estimated = estimated[:min_len].flatten()
    target = target[:min_len].flatten()

    # Zero-mean
    estimated = estimated - torch.mean(estimated)
    target = target - torch.mean(target)

    alpha = torch.dot(target, estimated) / (torch.dot(target, target) + 1e-8)
    s_target = alpha * target
    e_noise = estimated - s_target

    si_sdr_value = 10 * torch.log10(
            torch.sum(s_target ** 2) / (torch.sum(e_noise ** 2) + 1e-8)
    )
    return si_sdr_value.item()


def pesq(estimated: Union[torch.Tensor, np.ndarray],
         target: Union[torch.Tensor, np.ndarray],
         sample_rate: int = 16000) -> Optional[float]:
    """PESQ - requires external library"""
    est_np = estimated.cpu().numpy() if isinstance(estimated, torch.Tensor) else estimated
    tgt_np = target.cpu().numpy() if isinstance(target, torch.Tensor) else target
    return compute_pesq(sample_rate, tgt_np, est_np, 'wb')



def stoi(estimated: Union[torch.Tensor, np.ndarray],
         target: Union[torch.Tensor, np.ndarray],
         sample_rate: int = 16000) -> Optional[float]:
    est_np = estimated.cpu().numpy() if isinstance(estimated, torch.Tensor) else estimated
    tgt_np = target.cpu().numpy() if isinstance(target, torch.Tensor) else target
    return compute_stoi(tgt_np, est_np, sample_rate, extended=False)
