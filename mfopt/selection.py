"""Test subset selection for multi-fidelity evaluation.

Setting: every test has been run once at the cheap level and once at the
expensive level (a calibration run). We want a small subset to run at the
expensive level in future runs, while the cheap level keeps running on all
tests because it is free. The subset should
  (a) cover every failure category seen at the expensive level, and
  (b) let us predict the expensive result from the cheap result on the tests
      we no longer run. For that the subset also holds a small random sample
      per cheap outcome (pass/fail), used to estimate P(expensive pass | cheap
      result) and from it the expensive accuracy on the full set.
"""

import random
from collections import Counter
from dataclasses import dataclass, field


@dataclass(frozen=True)
class TestRecord:
    __test__ = False  # not a pytest test class

    id: str
    cost: float
    cheap_pass: bool
    exp_pass: bool
    elements: frozenset = field(default_factory=frozenset)


def universe(records):
    out = set()
    for r in records:
        out |= r.elements
    return out


def stratified_sample(records, per_stratum, seed=0, exclude=()):
    """Random sample of up to per_stratum tests among cheap passes and among
    cheap failures. The cheap result is known for every test in every run, so
    this sample can be used to estimate P(expensive pass | cheap result)
    without the bias of the coverage tests, which were picked for failing."""
    rng = random.Random(seed)
    excluded = set(exclude)
    chosen = []
    for cheap_value in (True, False):
        pool = sorted(r.id for r in records if r.cheap_pass == cheap_value and r.id not in excluded)
        chosen += rng.sample(pool, min(per_stratum, len(pool)))
    return chosen


def greedy_set_cover(records):
    """Classic greedy set cover: repeatedly take the test that covers the most
    still-uncovered failure categories (ties: lower cost, then lower id)."""
    uncovered = universe(records)
    chosen = []
    remaining = sorted(records, key=lambda r: r.id)
    while uncovered:
        # max() keeps the first of equal keys, so ties go to the lower id
        best = max(remaining, key=lambda r: (len(r.elements & uncovered), -r.cost))
        gain = best.elements & uncovered
        if not gain:
            break
        chosen.append(best.id)
        uncovered -= gain
        remaining.remove(best)
    return chosen


def cost_aware_scores(records):
    """Failure detection value per unit cost. value = sum over the test's
    failure categories of 1 / (number of tests with that category), so rare
    categories count more."""
    freq = Counter(e for r in records for e in r.elements)
    return {r.id: sum(1.0 / freq[e] for e in r.elements) / max(r.cost, 1e-9) for r in records}


def cost_aware_ranking(records):
    """Rank tests by detection value per cost once, then take them in that
    order until every failure category is covered. Tests that add nothing
    new when their turn comes are skipped."""
    scores = cost_aware_scores(records)
    uncovered = universe(records)
    chosen = []
    for r in sorted(records, key=lambda r: (-scores[r.id], r.cost, r.id)):
        if not uncovered:
            break
        if r.elements & uncovered:
            chosen.append(r.id)
            uncovered -= r.elements
    return chosen


def select(method, records, per_stratum=10, seed=0):
    """Full selection: coverage tests from `method` plus a stratified sample
    for prediction. Returns (all ids, sample ids)."""
    cover = method(records)
    sample = stratified_sample(records, per_stratum, seed, exclude=cover)
    return cover + sample, sample


def random_selection(records, size, seed):
    rng = random.Random(seed)
    ids = [r.id for r in records]
    rng.shuffle(ids)
    return ids[:size]


def random_order(records, seed):
    return random_selection(records, len(records), seed)


def full_order(records, method_ids):
    """Extend a selected subset into a full ordering (remaining tests by
    cost-aware score), used for cost vs coverage curves."""
    scores = cost_aware_scores(records)
    rest = sorted((r for r in records if r.id not in set(method_ids)),
                  key=lambda r: (-scores[r.id], r.cost, r.id))
    return list(method_ids) + [r.id for r in rest]


# ---- evaluation of a subset -----------------------------------------------

def coverage(selected_ids, records):
    """Share of failure categories in `records` that the subset also detects."""
    target = universe(records)
    if not target:
        return 1.0
    ids = set(selected_ids)
    found = universe([r for r in records if r.id in ids])
    return len(found & target) / len(target)


def cost_saved(selected_ids, records):
    total = sum(r.cost for r in records)
    ids = set(selected_ids)
    used = sum(r.cost for r in records if r.id in ids)
    return 1.0 - used / total


def estimate_expensive_accuracy(selected_ids, records, sample_ids=None):
    """Estimate expensive-level accuracy on all tests.

    Tests in the subset were run at the expensive level, so their result is
    known. For every other test we predict P(expensive pass | cheap result),
    estimated on the sample with add-one smoothing. `sample_ids` defaults to
    the whole subset, which is right for a random subset."""
    ids = set(selected_ids)
    sample = set(sample_ids) if sample_ids is not None else ids
    sub = [r for r in records if r.id in sample]

    def rate(cheap_value):
        group = [r for r in sub if r.cheap_pass == cheap_value]
        return (sum(r.exp_pass for r in group) + 1) / (len(group) + 2)

    p = {True: rate(True), False: rate(False)}
    total = sum(r.exp_pass if r.id in ids else p[r.cheap_pass] for r in records)
    return total / len(records)


def estimation_error(selected_ids, records, sample_ids=None):
    actual = sum(r.exp_pass for r in records) / len(records)
    return abs(estimate_expensive_accuracy(selected_ids, records, sample_ids) - actual)


def coverage_curve(order, records):
    """(cost used in %, coverage in %) after each test in `order`."""
    by_id = {r.id: r for r in records}
    total = sum(r.cost for r in records)
    target = universe(records)
    found, used, curve = set(), 0.0, [(0.0, 0.0 if target else 100.0)]
    for test_id in order:
        r = by_id[test_id]
        used += r.cost
        found |= r.elements & target
        curve.append((100 * used / total, 100 * len(found) / len(target) if target else 100.0))
    return curve
