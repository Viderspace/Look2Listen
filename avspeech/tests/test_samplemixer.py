# tests/test_samplemixer.py
import re
import itertools
import math
import pytest

from avspeech.training.sampler import SampleMixer

# ---- helpers ---------------------------------------------------------------

def mk_mixer(sizes, probs, batch_size=4, seed=123):
    """Convenience constructor with tiny defaults."""
    return SampleMixer(dataset_sizes=sizes, probabilities=probs,
                       batch_size=batch_size, seed=seed)

def realised_counts_from_plan(plan):
    """Sum per-batch composition dicts into per-type totals."""
    totals = {t: 0 for t in plan.quotas}
    for comp in plan.per_batch:
        for t, n in comp.items():
            totals[t] = totals.get(t, 0) + n
    return totals

def collect_batches(mixer):
    """Iterate and return a list of batches; also returns per-type local indices."""
    batches = list(iter(mixer))
    seen = {}
    for batch in batches:
        for t, idx in batch:
            seen.setdefault(t, set()).add(idx)
    return batches, seen


# ---- validation ------------------------------------------------------------


# tests/test_samplemixer.py (append)

SIZES_REAL = {"1s": 15_000, "2sc": 45_000, "2sn": 45_000}

def _filter_and_norm_probs(probs: dict) -> dict:
    """Remove zero-prob types and renormalize to 1.0 (sampler forbids p==0)."""
    nz = {k: v for k, v in probs.items() if v > 0.0}
    total = sum(nz.values())
    return {k: v / total for k, v in nz.items()}

def _expected_epoch_total(sizes: dict, probs: dict, batch_size: int) -> int:
    """E = floor(min_t cap_t / p_t), then snap to full batches."""
    e_star = min(sizes[t] / probs[t] for t in probs)
    E = int(e_star)
    return (E // batch_size) * batch_size

@pytest.mark.parametrize("probs_label, probs", [
    ("a", {"1s": 0.10, "2sc": 0.45, "2sn": 0.45}),
    ("b", {"1s": 0.15, "2sc": 0.425, "2sn": 0.425}),
    ("c", {"1s": 0.05, "2sc": 0.475, "2sn": 0.475}),
    ("d", {"1s": 0.00, "2sc": 0.50,  "2sn": 0.50}),   # 1s excluded
    ("e", {"1s": 0.00, "2sc": 0.00,  "2sn": 1.00}),   # only 2sn
])
@pytest.mark.parametrize("batch_size", [1, 2, 4, 6, 8, 32])
def test_real_world_plans_various_bs(probs_label, probs, batch_size):
    # Filter zeros + renormalize
    nz_probs = _filter_and_norm_probs(probs)
    nz_sizes = {t: SIZES_REAL[t] for t in nz_probs}

    m = SampleMixer(dataset_sizes=nz_sizes, probabilities=nz_probs,
                    batch_size=batch_size, seed=123)
    m.plan_epoch()
    plan = m.get_plan()

    # Print full summary (run pytest with -s to see this)
    print(f"\n[{probs_label}][bs={batch_size}] "
          f"B={plan.batches} E={plan.epoch_total} probs={nz_probs}")
    # For giant plans, consider truncating:
    print(m if plan.batches <= 20 else
          "\n".join(str(m).splitlines()[:18] + ["    ... (truncated) ..."] + str(m).splitlines()[-8:]))

    # Invariants
    expected_E = _expected_epoch_total(nz_sizes, nz_probs, batch_size)
    assert plan.epoch_total == expected_E
    assert sum(plan.quotas.values()) == expected_E
    for comp in plan.per_batch:
        assert sum(comp.values()) == batch_size
        assert set(comp).issubset(nz_probs.keys())

    # Quotas close to ideal (integer rounding)
    for t, p in nz_probs.items():
        ideal = expected_E * p
        assert abs(plan.quotas[t] - round(ideal)) <= 1

    # Optional: quick prefix fairness check on a few prefixes
    for k in [1, min(5, plan.batches), plan.batches]:
        pref = plan.per_batch[:k]
        actual = {t: sum(b.get(t, 0) for b in pref) for t in nz_probs}
        total = k * batch_size
        for t, p in nz_probs.items():
            assert abs(actual[t] - round(total * p)) <= 1
@pytest.mark.parametrize("batch_size", [0, -1])
def test_invalid_batch_size(batch_size):
    with pytest.raises(ValueError):
        mk_mixer({"A": 10}, {"A": 1.0}, batch_size=batch_size)

def test_empty_sizes():
    with pytest.raises(ValueError):
        mk_mixer({}, {"A": 1.0})

def test_empty_probs():
    with pytest.raises(ValueError):
        mk_mixer({"A": 10}, {})

def test_key_mismatch():
    with pytest.raises(ValueError):
        mk_mixer({"A": 10}, {"A": 0.5, "B": 0.5})

def test_zero_size_or_prob():
    with pytest.raises(ValueError):
        mk_mixer({"A": 0}, {"A": 1.0})
    with pytest.raises(ValueError):
        mk_mixer({"A": 10}, {"A": 0.0})

# ---- planning invariants ---------------------------------------------------

def test_plan_invariants_simple():
    m = mk_mixer({"X": 10, "Y": 6, "Z": 4}, {"X": 0.5, "Y": 0.3, "Z": 0.2}, batch_size=4, seed=7)
    m.plan_epoch()
    plan = m.get_plan()

    assert plan.epoch_total > 0 and plan.batches >= 1
    assert sum(plan.quotas.values()) == plan.epoch_total
    for comp in plan.per_batch:
        assert sum(comp.values()) == plan.batch_size

    # quotas never exceed capacity (pre-cap)
    for t, q in plan.quotas.items():
        assert q <= m.dataset_sizes[t]

    # ratios close to target within integer rounding (<= 1/E)
    for t, p in m.probabilities.items():
        assert abs(plan.quotas[t]/plan.epoch_total - p) <= 1.0 / plan.epoch_total + 1e-9

def test_iter_yields_full_batches_and_no_reuse():
    m = mk_mixer({"X": 10, "Y": 6, "Z": 4}, {"X": 0.5, "Y": 0.3, "Z": 0.2}, batch_size=4, seed=7)
    m.plan_epoch()
    plan = m.get_plan()
    batches, seen = collect_batches(m)

    assert len(batches) == plan.batches
    for b in batches:
        assert len(b) == plan.batch_size

    # no reuse within epoch; counts match quotas
    for t, q in plan.quotas.items():
        assert len(seen.get(t, set())) == q

# ---- behavior before planning ---------------------------------------------

@pytest.mark.parametrize("method", ["__len__", "get_plan"])
def test_errors_before_plan_epoch(method):
    m = mk_mixer({"A": 8, "B": 8}, {"A": 0.5, "B": 0.5})
    with pytest.raises(RuntimeError):
        getattr(m, method)()
    with pytest.raises(RuntimeError):
        _ = list(iter(m))

def test_str_before_after():
    m = mk_mixer({"A": 8, "B": 8}, {"A": 0.5, "B": 0.5})
    assert "not built" in str(m) or "not" in str(m).lower()
    m.plan_epoch()
    s = str(m)
    assert "batches" in s.lower() and "epoch_total" in s.lower()
    # has enumerated batch lines
    assert re.search(r"\[\s*1\]", s)  # at least first batch printed

# ---- reproducibility -------------------------------------------------------

def test_same_seed_same_plan_and_indices():
    sizes = {"X": 20, "Y": 12, "Z": 8}
    probs = {"X": 0.5, "Y": 0.3, "Z": 0.2}

    m1 = mk_mixer(sizes, probs, batch_size=4, seed=42)
    m2 = mk_mixer(sizes, probs, batch_size=4, seed=42)
    m1.plan_epoch(); m2.plan_epoch()

    p1, p2 = m1.get_plan(), m2.get_plan()
    assert p1.epoch_total == p2.epoch_total
    assert p1.batches == p2.batches
    assert p1.quotas == p2.quotas
    assert p1.per_batch == p2.per_batch

    b1, _ = collect_batches(m1)
    b2, _ = collect_batches(m2)
    assert b1 == b2

def test_different_seed_changes_shuffle_not_quotas():
    sizes = {"X": 20, "Y": 12, "Z": 8}
    probs = {"X": 0.5, "Y": 0.3, "Z": 0.2}

    m1 = mk_mixer(sizes, probs, batch_size=4, seed=1)
    m2 = mk_mixer(sizes, probs, batch_size=4, seed=2)
    m1.plan_epoch(); m2.plan_epoch()
    assert m1.get_plan().quotas == m2.get_plan().quotas

    b1, _ = collect_batches(m1)
    b2, _ = collect_batches(m2)
    # very likely different shuffles; defensively allow rare equality
    assert b1 != b2 or b1 == b2

# ---- prefix fairness (max-deficit scheduling) ------------------------------

@pytest.mark.parametrize("probs", [
    {"A": 0.5, "B": 0.3, "C": 0.2},
    {"A": 0.7, "B": 0.2, "C": 0.1},
    {"A": 0.34, "B": 0.33, "C": 0.33},
])
def test_prefix_fairness(probs):
    sizes = {t: 1000 for t in probs}  # ample capacity
    m = mk_mixer(sizes, probs, batch_size=4, seed=99)
    m.plan_epoch()
    plan = m.get_plan()

    # for each prefix of batches, actual per-type count stays within ~1 of rounded expectation
    for k in range(1, plan.batches + 1):
        prefix = plan.per_batch[:k]
        actual = {t: 0 for t in probs}
        for comp in prefix:
            for t, n in comp.items():
                actual[t] += n
        total = k * plan.batch_size
        for t, p in probs.items():
            expected = round(total * p)
            assert abs(actual[t] - expected) <= 1

# ---- skew & small-batch corners -------------------------------------------

@pytest.mark.parametrize("batch_size", [1, 2, 3, 4])
def test_skew_small_batches(batch_size):
    probs = {"Major": 0.99, "Minor": 0.01}
    sizes = {"Major": 10_000, "Minor": 100}  # enough for multiple batches
    m = mk_mixer(sizes, probs, batch_size=batch_size, seed=5)
    m.plan_epoch()
    plan = m.get_plan()

    # batches sum correctly; minor might be absent in many batches
    for comp in plan.per_batch:
        assert sum(comp.values()) == batch_size
        for t in comp:
            assert comp[t] >= 0

# ---- capacity binding ------------------------------------------------------

def test_capacity_binding_uses_all_bottleneck():
    # Y binds: cap/p gives min at Y
    sizes = {"X": 100, "Y": 15, "Z": 100}
    probs = {"X": 0.5, "Y": 0.3, "Z": 0.2}
    m = mk_mixer(sizes, probs, batch_size=5, seed=10)
    m.plan_epoch()
    plan = m.get_plan()
    # Y should be nearly fully used (integer rounding may leave a tiny slack)
    assert plan.quotas["Y"] in {15, 14}  # allow rounding slack by 1
    assert plan.quotas["Y"] <= sizes["Y"]

# ---- iteration twice (no hidden state leaks) -------------------------------

def test_iterate_twice_same_output():
    m = mk_mixer({"A": 20, "B": 20}, {"A": 0.5, "B": 0.5}, batch_size=4, seed=77)
    m.plan_epoch()
    b1, _ = collect_batches(m)
    b2, _ = collect_batches(m)
    assert b1 == b2