# sampler.py
import numpy as np
"""
SampleMixer: A batch sampler that maintains exact dataset ratios while maximizing data usage.

Main Algorithm:
For a batch_size=32 with ratios (45%, 45%, 10%) (of sample types: 2s_clean, 2s_noise, 1s_noise), 
we need non-integer samples per batch:

- 2S Clean: 32 × 0.45 = 14.4 samples
- 2S+Noise: 32 × 0.45 = 14.4 samples  
- 1S+Noise: 32 × 0.10 = 3.2 samples

Per-batch mixing formula:
1. Base allocation: floor(batch_size × ratio) for each type → (14, 14, 3) = 31 samples
2. Remainder slot: Use fractional parts as probabilities (0.4, 0.4, 0.2) for the last slot
3. Drift correction: Last 10% of batches allocate based on accumulated deficit

Key methods:
- _calculate_batch_allocation(): Pre-computes base counts and remainder probabilities
- _get_batch_composition(): Implements the per-batch mixing logic with drift correction
- _calculate_epoch_size(): Finds bottleneck dataset to maximize usage (2S Clean: 33,757 samples)

Result: Exact ratios over epoch while keeping each batch within ±1 sample of target.
"""

class SampleMixer:

    def __init__(self, dataset_sizes, probabilities, batch_size=32, seed=42):
        self.dataset_sizes = dataset_sizes
        self.probabilities = probabilities
        self.batch_size = batch_size
        self.rng = np.random.RandomState(seed)
        self.sample_types = list(dataset_sizes.keys())

        # Pre-calculate per-batch allocation
        self._calculate_batch_allocation()

        # Initialize epoch counter
        self.current_epoch = 0
        self.seed = seed

        # Calculate epoch parameters
        self._calculate_epoch_size()

        # Initialize indices
        self._prepare_epoch()
        self._prepare_indices()

    def _calculate_batch_allocation(self):
        ideal = {st: self.batch_size * self.probabilities[st]
                 for st in self.sample_types}

        self.base_counts = {st: int(ideal[st]) for st in self.sample_types}
        self.remainders = {st: ideal[st] - self.base_counts[st]
                           for st in self.sample_types}
        self.slots_to_fill = self.batch_size - sum(self.base_counts.values())

        # Pre-calculate remainder probabilities
        if self.slots_to_fill > 0:
            remainder_values = [self.remainders[st] for st in self.sample_types]
            self.remainder_probs = np.array(remainder_values) / sum(remainder_values)

    def _calculate_epoch_size(self):
        # Find bottleneck - dataset that limits our epoch size
        max_possible_samples = float('inf')

        for st in self.sample_types:
            if self.probabilities[st] > 0:
                # How many total samples if we use ALL of this dataset?
                possible = self.dataset_sizes[st] / self.probabilities[st]
                max_possible_samples = min(max_possible_samples, possible)

        # Round down to nearest batch_size multiple
        total_samples = int(max_possible_samples)
        self.batches_per_epoch = total_samples // self.batch_size
        self.epoch_size = self.batches_per_epoch * self.batch_size

        return self.epoch_size

    def _prepare_epoch(self):
        # Track actual usage vs target
        total_samples = self._calculate_epoch_size()
        self.target_counts = {st: int(total_samples * self.probabilities[st])
                              for st in self.sample_types}
        self.actual_counts = {st: 0 for st in self.sample_types}

    def _prepare_indices(self):
        # Shuffle and prepare indices for each dataset
        self.available_indices = {}
        self.index_positions = {}  # Track position in each dataset

        for st in self.sample_types:
            # Calculate how many samples we'll use from this dataset
            samples_needed = int(self.epoch_size * self.probabilities[st])
            samples_needed = min(samples_needed, self.dataset_sizes[st])

            # Create shuffled indices
            indices = np.arange(self.dataset_sizes[st])
            self.rng.shuffle(indices)

            # Store only what we need
            self.available_indices[st] = indices[:samples_needed]
            self.index_positions[st] = 0

    def _get_batch_composition(self, batch_idx, total_batches):
        batch_counts = self.base_counts.copy()

        if self.slots_to_fill > 0:
            # For last ~10% of batches, correct the drift
            if batch_idx > total_batches * 0.9:
                # Calculate deficit for each type
                deficits = {st: self.target_counts[st] - self.actual_counts[st]
                            for st in self.sample_types}
                # Allocate remaining slots to highest deficits
                for _ in range(self.slots_to_fill):
                    max_deficit_type = max(deficits, key=deficits.get)
                    batch_counts[max_deficit_type] += 1
                    deficits[max_deficit_type] -= 1
            else:
                # Random allocation for most batches
                extras = self.rng.choice(self.sample_types,
                                         size=self.slots_to_fill,
                                         p=self.remainder_probs)
                for st in extras:
                    batch_counts[st] += 1

        # Update counters
        for st, count in batch_counts.items():
            self.actual_counts[st] += count

        return batch_counts

    def _get_batch_indices(self, batch_composition):
        """
        Given composition like {S1_NOISE: 3, S2_CLEAN: 14, S2_NOISE: 14},
        return list of (SampleT, idx) tuples
        """
        batch_indices = []

        for st, count in batch_composition.items():
            # Get next 'count' indices for this sample type
            pos = self.index_positions[st]
            indices = self.available_indices[st][pos:pos + count]

            # Add as (SampleT, idx) tuples
            for idx in indices:
                batch_indices.append((st, int(idx)))

            # Update position
            self.index_positions[st] += count

        # Shuffle to mix types within batch
        self.rng.shuffle(batch_indices)

        return batch_indices

    def reset(self):
        """Call at start of new epoch"""
        # Optionally increment seed for different shuffling each epoch
        self.rng = np.random.RandomState(self.seed + self.current_epoch)
        self.current_epoch += 1

        # Reset counters and prepare for new epoch
        self._prepare_epoch()

        # Re-shuffle and reset indices
        self._prepare_indices()

    def __iter__(self):
        # Auto-reset for new epoch
        self.reset()

        for batch_idx in range(self.batches_per_epoch):
            composition = self._get_batch_composition(batch_idx, self.batches_per_epoch)
            indices = self._get_batch_indices(composition)
            yield indices

    def __len__(self):
        return self.batches_per_epoch