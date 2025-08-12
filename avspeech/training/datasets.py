import os
import tarfile
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
from torch.utils.data import Dataset
from avspeech.utils.structs import SampleT

"""
Note - look at 'CombinedAVDataset' (bottom of this file)

its the main dataset class for multiple datasets combined (e.g., 1s_noise, 2s_clean, 2s_noise).
"""


# Base class for AVSpeech datasets
class AVSpeechDatasetBase(Dataset):
    """Base dataset class for AVSpeech datasets."""

    def __init__(self, root_dir: Path):
        self.root = Path(root_dir)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.root}")

        # Get all sample directories
        self.sample_dirs: List[Path] = [p for p in self.root.iterdir() if p.is_dir()]
        self.sample_dirs.sort() # For reproducibility



    def __len__(self) -> int:
        return len(self.sample_dirs)

    def get_type(self) -> SampleT:
        raise NotImplementedError


    def _load_tensor(self, sample_dir: Path, name: str) -> torch.Tensor:
        path = sample_dir / f"{name}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing tensor: {path}")
        return torch.load(path, map_location="cpu")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample_dir = self.sample_dirs[idx]

        # Load only what we need
        mixture_audio = torch.load(sample_dir / 'audio' / 'mixture_embs.pt')
        clean_audio = torch.load(sample_dir / 'audio' / 'clean_embs.pt')
        face_embs = torch.load(sample_dir / 'face' / 'face_embs.pt')

        return {
                'mixture'  : mixture_audio,  # [257, 298, 2]
                'clean'    : clean_audio,  # [257, 298, 2]
                'face'     : face_embs,  # [75, 512]
                'sample_id': sample_dir.name,
                "mix_type" : self.get_type(),  # Use SampleT enum for type
        }


# === Concrete datasets (renamed as requested) ===
class OneSpeakerNoiseDataset(AVSpeechDatasetBase):

    def get_type(self) -> SampleT: return SampleT.S1_NOISE


class TwoSpeakerCleanDataset(AVSpeechDatasetBase):

    def get_type(self) -> SampleT: return SampleT.S2_CLEAN


class TwoSpeakerNoiseDataset(AVSpeechDatasetBase):

    def get_type(self) -> SampleT: return SampleT.S2_NOISE


class CombinedAVDataset:
    """Combine multiple datasets; handles (SampleT, idx) indexing."""

    def __init__(self,
                 s1_noise_ds: Optional[OneSpeakerNoiseDataset] = None,
                 s2_clean_ds: Optional[TwoSpeakerCleanDataset] = None,
                 s2_noise_ds: Optional[TwoSpeakerNoiseDataset] = None):
        self.datasets = {}
        if s1_noise_ds:
            self.datasets[SampleT.S1_NOISE] = s1_noise_ds
        if s2_clean_ds:
            self.datasets[SampleT.S2_CLEAN] = s2_clean_ds
        if s2_noise_ds:
            self.datasets[SampleT.S2_NOISE] = s2_noise_ds

    def __len__(self) -> int:
        return sum(len(d) for d in self.datasets.values())

    def __getitem__(self, idx):
        """
        Args:
            idx: Either (SampleT, int) tuple or will be called by our custom sampler
        """
        if isinstance(idx, tuple):
            sample_type, sample_idx = idx
            if sample_type in self.datasets:
                return self.datasets[sample_type][sample_idx]
        return None