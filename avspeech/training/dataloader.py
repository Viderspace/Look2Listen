# dataloader.py
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Optional, Iterator, List, Any
from pathlib import Path
import torch
from avspeech.utils.structs import SampleT
from avspeech.training.datasets import CombinedAVDataset, OneSpeakerNoiseDataset, TwoSpeakerCleanDataset, \
    TwoSpeakerNoiseDataset
from avspeech.training.sampler import SampleMixer


class MixedDataLoader:

    def __init__(
            self,
            train_paths: Dict[SampleT, Path],
            val_paths: Dict[SampleT, Path],
            probabilities: Dict[SampleT, float],
            batch_size: int = 32,
            num_workers: int = 2,
            seed: int = 42
    ):
        """
        Initialize mixed data loader with combined training dataset and separate validation datasets.
        :param train_paths: Paths to training datasets for each sample type.
        """
        # Create datasets
        self.train_dataset = self._create_combined_dataset(train_paths)
        self.val_datasets = self._create_separate_datasets(val_paths)

        # Get sizes for sampler
        train_sizes = self._get_dataset_sizes(train_paths)

        # Create sampler
        self.sampler = SampleMixer(train_sizes, probabilities, batch_size, seed)

        # Store config
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _create_combined_dataset(self, paths: Dict[SampleT, Path]) -> CombinedAVDataset:
        datasets = {}
        if SampleT.S1_NOISE in paths:
            datasets['s1_noise_ds'] = OneSpeakerNoiseDataset(paths[SampleT.S1_NOISE])
        if SampleT.S2_CLEAN in paths:
            datasets['s2_clean_ds'] = TwoSpeakerCleanDataset(paths[SampleT.S2_CLEAN])
        if SampleT.S2_NOISE in paths:
            datasets['s2_noise_ds'] = TwoSpeakerNoiseDataset(paths[SampleT.S2_NOISE])
        return CombinedAVDataset(**datasets)

    def _create_separate_datasets(self, paths: Dict[SampleT, Path]) -> Dict[SampleT, Dataset]:
        # Keep validation datasets separate for per-type evaluation
        val_datasets = {}
        if SampleT.S1_NOISE in paths:
            val_datasets[SampleT.S1_NOISE] = OneSpeakerNoiseDataset(paths[SampleT.S1_NOISE])
        if SampleT.S2_CLEAN in paths:
            val_datasets[SampleT.S2_CLEAN] = TwoSpeakerCleanDataset(paths[SampleT.S2_CLEAN])
        if SampleT.S2_NOISE in paths:
            val_datasets[SampleT.S2_NOISE] = TwoSpeakerNoiseDataset(paths[SampleT.S2_NOISE])
        return val_datasets

    def _get_dataset_sizes(self, paths: Dict[SampleT, Path]) -> Dict[SampleT, int]:
        sizes = {}
        for st, path in paths.items():
            sample_dirs = [d for d in path.iterdir() if d.is_dir()]
            sizes[st] = len(sample_dirs)
        return sizes

    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        mixture = torch.stack([item['mixture'] for item in batch])
        clean = torch.stack([item['clean'] for item in batch])
        face = torch.stack([item['face'] for item in batch])

        return {
                'mixture'  : mixture,
                'clean'    : clean,
                'face'     : face,
                'sample_id': [item['sample_id'] for item in batch],
                'mix_type' : [item['mix_type'] for item in batch]
        }

    def get_train_loader(self):
        # Custom iterator to handle (SampleT, idx) tuples from sampler
        class BatchIterator:

            def __init__(self, dataset: CombinedAVDataset, sampler: SampleMixer, collate_fn):
                self.dataset = dataset
                self.sampler = sampler
                self.collate_fn = collate_fn

            def __iter__(self) -> Iterator[Dict[str, Any]]:
                for indices in self.sampler:
                    batch = [self.dataset[idx] for idx in indices]
                    yield self.collate_fn(batch)

            def __len__(self) -> int:
                return len(self.sampler)

        return BatchIterator(self.train_dataset, self.sampler, self._collate_fn)



    def get_val_loader(self, sample_type: SampleT) -> Optional[DataLoader]:
        if sample_type not in self.val_datasets:
            return None

        return DataLoader(
                self.val_datasets[sample_type],
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=self._collate_fn
        )