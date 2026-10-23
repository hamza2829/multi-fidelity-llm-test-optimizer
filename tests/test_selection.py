import pytest

from mfopt.selection import (TestRecord, cost_aware_ranking, cost_aware_scores, cost_saved,
                             coverage, coverage_curve, estimate_expensive_accuracy,
                             greedy_set_cover, random_selection, select, stratified_sample,
                             universe)


def rec(i, cost, elems=(), cheap=True, exp=None):
    exp = not elems if exp is None else exp
    return TestRecord(id=f"t{i}", cost=cost, cheap_pass=cheap, exp_pass=exp,
                      elements=frozenset(elems))


@pytest.fixture
def records():
    return [
        rec(1, 1.0, {"a", "b", "c"}),
        rec(2, 1.0, {"a"}),
        rec(3, 0.5, {"d"}),
        rec(4, 5.0, {"d", "e"}),
        rec(5, 0.2, {"e"}),
        rec(6, 1.0),
        rec(7, 1.0, cheap=False),
    ]


def test_greedy_set_cover_covers_universe_with_few_tests(records):
    ids = greedy_set_cover(records)
    assert coverage(ids, records) == 1.0
    # t1 covers a, b, c; then one test of size 2 (t4) covers d and e
    assert ids == ["t1", "t4"]


def test_cost_aware_ranking_covers_universe_and_prefers_cheap(records):
    ids = cost_aware_ranking(records)
    assert coverage(ids, records) == 1.0
    assert "t4" not in ids  # d and e are covered by the cheaper t3 and t5
    assert cost_saved(ids, records) > cost_saved(greedy_set_cover(records), records)


def test_cost_aware_scores_weight_rare_categories(records):
    scores = cost_aware_scores(records)
    # t2 covers only "a", which t1 also covers: half value per cost of t1's "a" part
    assert scores["t2"] == pytest.approx(0.5)
    assert scores["t6"] == 0.0


def test_empty_universe():
    recs = [rec(1, 1.0), rec(2, 1.0)]
    assert universe(recs) == set()
    assert greedy_set_cover(recs) == []
    assert cost_aware_ranking(recs) == []
    assert coverage([], recs) == 1.0


def test_stratified_sample_respects_strata_and_exclusion(records):
    sample = stratified_sample(records, per_stratum=2, seed=1, exclude=["t1"])
    by_id = {r.id: r for r in records}
    assert "t1" not in sample
    assert sum(by_id[i].cheap_pass for i in sample) == 2
    assert sum(not by_id[i].cheap_pass for i in sample) == 1  # only t7 fails cheap
    assert stratified_sample(records, 2, seed=1) == stratified_sample(records, 2, seed=1)


def test_select_returns_cover_plus_sample(records):
    ids, sample = select(greedy_set_cover, records, per_stratum=1, seed=0)
    assert ids[:2] == ["t1", "t4"]
    assert set(sample) <= set(ids) and len(sample) == 2


def test_random_selection_is_seeded(records):
    assert random_selection(records, 3, 5) == random_selection(records, 3, 5)
    assert len(set(random_selection(records, 3, 5))) == 3


def test_estimate_uses_known_results_and_conditional_rates():
    recs = [rec(i, 1.0, cheap=True, exp=True) for i in range(4)]
    recs += [rec(i, 1.0, {"x"}, cheap=False, exp=False) for i in range(4, 8)]
    # whole set selected: estimate equals the true accuracy
    assert estimate_expensive_accuracy([r.id for r in recs], recs) == pytest.approx(0.5)
    # nothing selected: smoothed rates are 0.5 for both strata
    assert estimate_expensive_accuracy([], recs) == pytest.approx(0.5)
    # one test per stratum as sample: P(pass|cheap pass)=2/3, P(pass|cheap fail)=1/3
    est = estimate_expensive_accuracy(["t0", "t4"], recs)
    assert est == pytest.approx((1 + 0 + 3 * 2 / 3 + 3 * 1 / 3) / 8)


def test_coverage_curve_is_monotonic_and_ends_at_full(records):
    curve = coverage_curve([r.id for r in records], records)
    costs = [c for c, _ in curve]
    covs = [v for _, v in curve]
    assert costs == sorted(costs) and covs == sorted(covs)
    assert curve[-1] == (pytest.approx(100.0), pytest.approx(100.0))
