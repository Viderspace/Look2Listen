"""
Pre-capped, drop-last sampler that precomputes an epoch plan.
- Pre-cap only (never exhaust a type)
- Drop-last only (yield full batches)
- No backfills / oversampling
- Seeded (planning + shuffles)
- Plan preview via get_plan() and __str__
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, List, Mapping, Sequence, Iterator, Iterable, Optional, Hashable, Tuple
import math
import random

# Types
SampleT = Hashable
BatchIndex = Tuple[SampleT, int]          # (type, local_index)
Batch = List[BatchIndex]
BatchComposition = Dict[SampleT, int]
Quotas = Dict[SampleT, int]


@dataclass(frozen=True)
class PlanSummary:
    """A snapshot of the planned epoch: sizes, quotas, and per-batch mix."""
    batches: int
    epoch_total: int
    quotas: Quotas
    per_batch: List[BatchComposition]
    realized_ratios: Dict[SampleT, float]
    batch_size: int


class SampleMixer(Iterable[Batch]):
    """Pre-capped, drop-last batch sampler enforcing target ratios per epoch."""

    # ---- Public API ----
    def __init__(
        self,
        dataset_sizes: Dict[SampleT, int],
        probabilities: Dict[SampleT, float],
        batch_size: int = 32,
        seed: int = 42,
    ) -> None:
        """Store sizes/ratios and basic config. No heavy work here."""
        self.dataset_sizes: Dict[SampleT, int] = {t: int(n) for t, n in dict(dataset_sizes).items()}
        self.probabilities: Dict[SampleT, float] = dict(probabilities)
        self.batch_size: int = int(batch_size)
        self.seed: Optional[int] = int(seed) if seed is not None else None

        self._check_inputs()
        # Normalize to sum exactly 1.0
        tot = sum(self.probabilities.values())
        self.probabilities = {t: p / tot for t, p in self.probabilities.items()}

        # Planning state
        self._planned: bool = False
        self._batches: int = 0
        self._epoch_total: int = 0
        self._quotas: Quotas = {}
        self._per_batch: List[BatchComposition] = []
        # Per-type pools of local indices 0..size-1 (shuffled in plan_epoch)
        self._pools: Dict[SampleT, List[int]] = {}
        # Type iteration order (stable across epoch given seed)
        self._type_order: List[SampleT] = list(self.dataset_sizes.keys())



        self.plan_epoch(seed)  # Automatically build the plan on init

    def plan_epoch(self, seed: Optional[int] = None) -> None:
        """Build the plan (E, B, quotas, per-batch mix) and shuffle per-type pools."""
        rng = random.Random(self.seed if seed is None else seed)

        capacities = self._capacities()
        epoch_total, batches = self._calc_epoch_size(capacities)
        quotas = self._calc_quotas(epoch_total)

        # Deterministic type order for all tie-breaks
        self._type_order = list(self.probabilities.keys())
        rng.shuffle(self._type_order)

        # Build global type sequence of length E by maximum-deficit scheduling
        # assigned[t] counts how many of type t we already placed
        assigned: Dict[SampleT, int] = {t: 0 for t in self._type_order}
        # Use Fractions for exact deficits: deficit_t(s) = s * p[t] - assigned[t]
        probs_frac: Dict[SampleT, Fraction] = {
            t: Fraction(self.probabilities[t]).limit_denominator(1_000_000) for t in self._type_order
        }
        type_sequence: List[SampleT] = []
        for s in range(1, epoch_total + 1):
            # compute best type by largest deficit; break ties by shuffled _type_order
            best_t: Optional[SampleT] = None
            best_num: Optional[Fraction] = None
            for t in self._type_order:
                if assigned[t] >= quotas[t]:
                    continue
                # deficit numerator = s * p - assigned
                deficit = probs_frac[t] * s - assigned[t]
                if best_num is None or deficit > best_num:
                    best_t = t
                    best_num = deficit
            # best_t must exist because sum assigned < E and sum quotas = E
            assert best_t is not None
            assigned[best_t] += 1
            type_sequence.append(best_t)

        # Chunk into batches and count per-batch composition
        per_batch: List[BatchComposition] = []
        for k in range(batches):
            start = k * self.batch_size
            stop = start + self.batch_size
            comp: BatchComposition = {t: 0 for t in self._type_order}
            for t in type_sequence[start:stop]:
                comp[t] += 1
            # remove zeros for cleaner plan
            comp = {t: n for t, n in comp.items() if n}
            assert sum(comp.values()) == self.batch_size
            per_batch.append(comp)

        # Shuffle per-type pools of local indices deterministically
        self._pools = {t: list(range(capacities[t])) for t in capacities}
        for t in self._pools:
            rng.shuffle(self._pools[t])

        # Save plan
        self._epoch_total = epoch_total
        self._batches = batches
        self._quotas = quotas
        self._per_batch = per_batch

        # Final checks
        self._check_plan(capacities, epoch_total, batches, quotas, per_batch)
        self._planned = True

    def get_plan(self) -> PlanSummary:
        """Return the current epoch plan. Call after plan_epoch()."""
        if not self._planned:
            raise RuntimeError("get_plan() called before plan_epoch().")
        realized = {t: (self._quotas[t] / self._epoch_total) for t in self._quotas}
        return PlanSummary(
            batches=self._batches,
            epoch_total=self._epoch_total,
            quotas=dict(self._quotas),
            per_batch=[dict(bc) for bc in self._per_batch],
            realized_ratios=realized,
            batch_size=self.batch_size,
        )

    def __iter__(self) -> Iterator[Batch]:
        """Yield exactly B full batches, as planned. No hidden resets."""
        if not self._planned:
            raise RuntimeError("__iter__ called before plan_epoch().")
        ptr: Dict[SampleT, int] = {t: 0 for t in self._pools}
        for comp in self._per_batch:
            batch: Batch = []
            for t, n in comp.items():
                start = ptr[t]
                end = start + n
                pool = self._pools[t]
                # map local indices to (type, local_index)
                batch.extend((t, idx) for idx in pool[start:end])
                ptr[t] = end
            assert len(batch) == self.batch_size
            yield batch

    def __len__(self) -> int:
        """Number of planned batches (B)."""
        if not self._planned:
            raise RuntimeError("__len__ called before plan_epoch().")
        return self._batches

    def __str__(self) -> str:
        """Multi-line summary of the current plan or a short note if not planned yet."""
        if not self._planned:
            return "SampleMixer(plan: not built)"
        lines: List[str] = []
        lines.append("SampleMixer(plan)")
        lines.append(f"  batch_size: {self.batch_size}")
        lines.append(f"  batches (B): {self._batches}")
        lines.append(f"  epoch_total (E): {self._epoch_total}")
        # Order quotas by descending count then name for readability
        for_quota = sorted(self._quotas.items(), key=lambda kv: (-kv[1], str(kv[0])))
        lines.append("  quotas:")
        for i, (t, q) in enumerate(for_quota):
            if i > 10:
                break  # Limit
            ratio = q / self._epoch_total if self._epoch_total else 0.0
            lines.append(f"    - {t}: {q} ({ratio:.3f})")
        lines.append("  batches (type:count):")
        for i, comp in enumerate(self._per_batch, 1):
            parts = [f"{t}:{n}" for t, n in sorted(comp.items(), key=lambda kv: (str(kv[0])))]
            lines.append(f"    [{i:>3}] " + ", ".join(parts))
        return "\n".join(lines)

    # ---- Internal helpers ----
    def _check_inputs(self) -> None:
        """Validate sizes, ratios, and batch_size."""
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if not self.dataset_sizes:
            raise ValueError("dataset_sizes must not be empty")
        if not self.probabilities:
            raise ValueError("probabilities must not be empty")
        keys_sizes = set(self.dataset_sizes.keys())
        keys_probs = set(self.probabilities.keys())
        if keys_sizes != keys_probs:
            missing_sizes = keys_probs - keys_sizes
            missing_probs = keys_sizes - keys_probs
            parts = []
            if missing_sizes:
                parts.append("missing dataset_sizes for: [" + ", ".join(sorted(map(str, missing_sizes))) + "]")
            if missing_probs:
                parts.append("missing probabilities for: [" + ", ".join(sorted(map(str, missing_probs))) + "]")
            raise ValueError("Key mismatch: " + "; ".join(parts))
        for t, n in self.dataset_sizes.items():
            if int(n) <= 0:
                raise ValueError(f"dataset_sizes[{t!r}] must be > 0 (got {n})")
        for t, p in self.probabilities.items():
            if not (p > 0.0):
                raise ValueError(f"probabilities[{t!r}] must be > 0 (got {p})")

    def _capacities(self) -> Dict[SampleT, int]:
        """Return available counts per type from dataset_sizes."""
        return {t: int(n) for t, n in self.dataset_sizes.items()}

    def _calc_epoch_size(self, capacities: Mapping[SampleT, int]) -> Tuple[int, int]:
        """Compute (E, B) using pre-cap, then snap E to full batches: E = B * batch_size."""
        # min over types of cap[t] / p[t]
        candidates: List[float] = []
        for t, cap in capacities.items():
            p = self.probabilities[t]
            candidates.append(cap / p)
        e_star = math.floor(min(candidates)) if candidates else 0
        epoch_total = (e_star // self.batch_size) * self.batch_size
        if epoch_total <= 0:
            raise RuntimeError("Not enough data to form a single full batch at requested ratios and batch_size.")
        batches = epoch_total // self.batch_size
        return epoch_total, batches

    def _calc_quotas(self, epoch_total: int) -> Quotas:
        """Integer per-type totals for this epoch that sum to epoch_total (largest remainder)."""
        # Ideal totals (Fractions for exactness)
        ideals: Dict[SampleT, Fraction] = {
            t: Fraction(epoch_total) * Fraction(self.probabilities[t]).limit_denominator(1_000_000)
            for t in self.probabilities
        }
        base: Dict[SampleT, int] = {t: int(math.floor(float(v))) for t, v in ideals.items()}
        remainder = epoch_total - sum(base.values())
        if remainder < 0:
            raise AssertionError("Negative remainder in quotas")
        # Largest fractional parts first; break ties by higher probability then name
        def frac(v: Fraction) -> float:
            f = float(v)
            return f - math.floor(f)
        order = sorted(self.probabilities.keys(), key=lambda t: (frac(ideals[t]), self.probabilities[t], str(t)), reverse=True)
        quotas: Quotas = dict(base)
        for t in order:
            if remainder <= 0:
                break
            quotas[t] += 1
            remainder -= 1
        if sum(quotas.values()) != epoch_total:
            raise AssertionError("Quotas do not sum to epoch_total")
        # Safety: ensure quotas do not exceed capacities (pre-cap should guarantee)
        for t, q in quotas.items():
            if q > self.dataset_sizes[t]:
                raise AssertionError(f"Quota for {t!r} exceeds capacity {self.dataset_sizes[t]} (got {q})")
        return quotas

    def _make_batch_plan(self, quotas: Quotas, batches: int) -> List[BatchComposition]:
        """(Unused in current implementation — kept for API symmetry)"""
        # We build per-batch plan via the global type_sequence and chunking in plan_epoch.
        # This function remains as a placeholder for compatibility with earlier stubs.
        per_batch: List[BatchComposition] = []
        return per_batch

    def _check_plan(
        self,
        capacities: Mapping[SampleT, int],
        epoch_total: int,
        batches: int,
        quotas: Quotas,
        per_batch: Sequence[BatchComposition],
    ) -> None:
        """Light invariants before sampling (sums match, quotas <= capacities, etc.)."""
        assert epoch_total == batches * self.batch_size
        assert sum(quotas.values()) == epoch_total
        for t, cap in capacities.items():
            assert quotas.get(t, 0) <= cap, f"quota[{t}]={quotas.get(t,0)} > cap={cap}"
        for i, comp in enumerate(per_batch):
            s = sum(comp.values())
            assert s == self.batch_size, f"batch {i} sums to {s}, expected {self.batch_size}"
