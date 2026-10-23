"""Full pipeline: evaluate all levels, compare them, select test subsets, report.

Usage: python -m mfopt.run [--out docs] [--random-seeds 20] [--eval-seeds 5]
"""

import argparse
import json
import statistics
import time
from pathlib import Path

from .assistant import MockModelAssistant, build_assistants
from .metrics import agreement, evaluate, failure_elements, summarize
from .selection import (TestRecord, cost_aware_ranking, cost_saved, coverage, coverage_curve,
                        estimation_error, full_order, greedy_set_cover, random_order,
                        random_selection, select)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TESTSET = ROOT / "data" / "testset.json"
LEVELS = ["low", "medium", "high"]


def load_tests(path=DEFAULT_TESTSET):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def make_records(tests, cheap_results, exp_results, cost_key):
    return [TestRecord(id=t["id"], cost=t["cost"][cost_key],
                       cheap_pass=cheap_results[t["id"]]["passed"],
                       exp_pass=exp_results[t["id"]]["passed"],
                       elements=failure_elements(exp_results[t["id"]]))
            for t in tests]


def subset_metrics(ids, records, eval_runs, sample_ids=None):
    """Metrics of one subset on the calibration run and on fresh runs."""
    out = {
        "size": len(ids),
        "cost_saved": cost_saved(ids, records),
        "coverage_calibration": coverage(ids, records),
        "estimation_error_calibration": estimation_error(ids, records, sample_ids),
    }
    if eval_runs:
        out["coverage_fresh"] = statistics.mean(coverage(ids, run) for run in eval_runs)
        out["estimation_error_fresh"] = statistics.mean(
            estimation_error(ids, run, sample_ids) for run in eval_runs)
    return out


def aggregate(values):
    return {"mean": statistics.mean(values), "std": statistics.pstdev(values),
            "min": min(values), "max": max(values)}


def curve_on_grid(order, runs, grid):
    """Mean coverage over runs at each cost percentage in grid."""
    curves = [coverage_curve(order, run) for run in runs]
    return [statistics.mean(max(c for cost, c in curve if cost <= x + 1e-9) for curve in curves)
            for x in grid]


def run_pipeline(testset=DEFAULT_TESTSET, out_dir=ROOT / "docs", seed=0, eval_seeds=5,
                 random_seeds=20, per_stratum=10, enable_llm=None, make_charts=True):
    started = time.time()
    tests = load_tests(testset)
    assistants = build_assistants(seed=seed, enable_llm=enable_llm)

    results, notes = {}, []
    results["low"] = evaluate(assistants["low"], tests)
    results["medium"] = evaluate(assistants["medium"], tests)
    high = assistants["high"]
    if high.is_available_for([t["utterance"] for t in tests]):
        results["high"] = evaluate(high, tests)
    else:
        notes.append("High fidelity (LLM) skipped: disabled and not fully cached. "
                     "Medium is used as the expensive level.")
    levels = [lv for lv in LEVELS if lv in results]
    summaries = {lv: summarize(results[lv], tests) for lv in levels}

    agreements = {}
    for i, a in enumerate(levels):
        for b in levels[i + 1:]:
            agreements[f"{a}_vs_{b}"] = agreement(results[a], results[b], tests)

    cheap, expensive = "low", levels[-1]
    cost_key = f"{expensive}_s"
    records = make_records(tests, results[cheap], results[expensive], cost_key)

    # Fresh runs of the expensive level with new seeds: does the subset chosen
    # on the calibration run still find the failures of a new run?
    eval_runs = []
    if expensive == "medium":
        for s in range(seed + 1, seed + 1 + eval_seeds):
            fresh = evaluate(MockModelAssistant(seed=s, error_rate=assistants["medium"].error_rate),
                             tests)
            eval_runs.append(make_records(tests, results[cheap], fresh, cost_key))
    else:
        notes.append("Expensive level is the cached LLM, which is deterministic, "
                     "so no fresh-run evaluation is done.")

    methods = {
        "greedy_set_cover": greedy_set_cover,
        "cost_aware_ranking": cost_aware_ranking,
    }
    selection, cover_ids = {}, {}
    for name, method in methods.items():
        # The coverage part is deterministic; the prediction sample is random,
        # so method metrics are averaged over the same number of seeds as the
        # random baseline.
        runs = []
        for s in range(random_seeds):
            ids_s, sample_s = select(method, records, per_stratum, seed + s)
            runs.append(subset_metrics(ids_s, records, eval_runs, sample_s))
        ids, sample = select(method, records, per_stratum, seed)
        cover_ids[name] = [i for i in ids if i not in set(sample)]
        rand = [subset_metrics(random_selection(records, len(ids), s), records, eval_runs)
                for s in range(random_seeds)]
        selection[name] = {
            "ids": ids,
            "coverage_ids": cover_ids[name],
            "sample_ids": sample,
            "method": {k: statistics.mean(r[k] for r in runs) for k in runs[0]},
            "method_spread": {k: aggregate([r[k] for r in runs]) for k in runs[0] if k != "size"},
            "random_same_size": {k: aggregate([r[k] for r in rand]) for k in rand[0] if k != "size"},
        }

    grid = list(range(0, 101, 2))
    curve_runs = eval_runs or [records]
    curves = {"cost_pct": grid}
    for name, ids in cover_ids.items():
        curves[name] = curve_on_grid(full_order(records, ids), curve_runs, grid)
    random_curves = [curve_on_grid(random_order(records, s), curve_runs, grid)
                     for s in range(random_seeds)]
    curves["random_mean"] = [statistics.mean(col) for col in zip(*random_curves)]
    curves["random_min"] = [min(col) for col in zip(*random_curves)]
    curves["random_max"] = [max(col) for col in zip(*random_curves)]

    tests_by_id = {t["id"]: t for t in tests}
    missed_examples = []
    for tid in agreements[f"{cheap}_vs_{expensive}"]["tests_failing_only_at_expensive"]:
        missed_examples.append({
            "id": tid,
            "utterance": tests_by_id[tid]["utterance"],
            "tags": tests_by_id[tid]["tags"],
            "expected": tests_by_id[tid]["expected"],
            "cheap_output": results[cheap][tid]["raw"],
            "expensive_output": results[expensive][tid]["raw"],
            "expensive_error": results[expensive][tid]["error_type"],
        })

    report = {
        "config": {"seed": seed, "eval_seeds": eval_seeds, "random_seeds": random_seeds,
                   "per_stratum": per_stratum, "n_tests": len(tests),
                   "medium_error_rate": assistants["medium"].error_rate,
                   "cheap_level": cheap, "expensive_level": expensive, "cost_unit": cost_key},
        "notes": notes,
        "levels": levels,
        "summaries": summaries,
        "total_cost": {lv: sum(t["cost"][f"{lv}_s"] for t in tests) for lv in LEVELS},
        "total_high_tokens": sum(t["cost"]["high_tokens"] for t in tests),
        "agreements": agreements,
        "fresh_runs_failure_categories": [len(set().union(*(r.elements for r in run)))
                                          for run in eval_runs],
        "calibration_failure_categories": len(set().union(*(r.elements for r in records))),
        "selection": selection,
        "cheap_only_estimate_error": abs(summaries[cheap]["exact_match"]
                                         - summaries[expensive]["exact_match"]),
        "curves": curves,
        "cheap_misses": missed_examples,
        "runtime_s": round(time.time() - started, 2),
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n",
                                          encoding="utf-8")
    from .report import write_markdown
    write_markdown(report, tests, results, out_dir / "results.md")
    if make_charts:
        from .charts import make_all
        make_all(report, tests, results, out_dir)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--testset", default=str(DEFAULT_TESTSET))
    parser.add_argument("--out", default=str(ROOT / "docs"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seeds", type=int, default=5)
    parser.add_argument("--random-seeds", type=int, default=20)
    parser.add_argument("--per-stratum", type=int, default=10)
    parser.add_argument("--no-charts", action="store_true")
    args = parser.parse_args()
    report = run_pipeline(args.testset, args.out, args.seed, args.eval_seeds, args.random_seeds,
                          args.per_stratum, make_charts=not args.no_charts)
    for note in report["notes"]:
        print("note:", note)
    for lv in report["levels"]:
        s = report["summaries"][lv]
        print(f"{lv:>6}: exact match {s['exact_match']:.1%}  intent acc {s['intent_accuracy']:.1%}"
              f"  slot F1 {s['slot_f1']:.3f}  failures {s['failures']}")
    for name, sel in report["selection"].items():
        m = sel["method"]
        print(f"{name}: {m['size']} tests, cost saved {m['cost_saved']:.1%}, "
              f"coverage calib {m['coverage_calibration']:.1%}"
              + (f", fresh {m['coverage_fresh']:.1%}" if "coverage_fresh" in m else ""))
    print(f"wrote {args.out}/results.md and results.json in {report['runtime_s']} s")


if __name__ == "__main__":
    main()
