"""Scoring of single predictions and aggregate metrics."""

import json
import re
from collections import Counter, defaultdict

from .intents import ACTIONS, ERROR_TYPES, normalize_slots

FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$")


def parse_output(raw):
    """Parse raw model output into {"action", "slots"}, or None if invalid.

    Accepts plain JSON or JSON inside a markdown code fence, which LLMs
    sometimes return.
    """
    if not isinstance(raw, str):
        return None
    text = FENCE_RE.sub("", raw.strip())
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or data.get("action") not in ACTIONS:
        return None
    return {"action": data["action"], "slots": normalize_slots(data.get("slots", {}))}


def score(expected, raw):
    """Compare one prediction with the expected intent.

    Returns a dict with: passed, action_ok, slot tp/fp/fn, the list of error
    types and the primary error type (first in ERROR_TYPES order, or None).
    """
    exp_slots = normalize_slots(expected["slots"])
    pred = parse_output(raw)
    errors = []
    if pred is None:
        errors.append("invalid_output")
        pred_slots, action_ok = {}, False
    else:
        pred_slots = pred["slots"]
        action_ok = pred["action"] == expected["action"]
        if not action_ok:
            errors.append("wrong_action")
    tp = sum(1 for k, v in pred_slots.items() if exp_slots.get(k) == v)
    fp = len(pred_slots) - tp
    fn = len(exp_slots) - tp
    if pred is not None:
        if any(k not in pred_slots for k in exp_slots):
            errors.append("missing_slot")
        if any(k in exp_slots and exp_slots[k] != v for k, v in pred_slots.items()):
            errors.append("wrong_slot_value")
        if any(k not in exp_slots for k in pred_slots):
            errors.append("extra_slot")
    primary = next((e for e in ERROR_TYPES if e in errors), None)
    return {"passed": not errors, "action_ok": action_ok, "tp": tp, "fp": fp, "fn": fn,
            "errors": errors, "error_type": primary,
            "predicted_action": pred["action"] if pred else None}


def evaluate(assistant, tests):
    """Run one assistant on all tests. Returns {test_id: result dict}."""
    results = {}
    for test in tests:
        raw = assistant.predict(test["utterance"])
        result = score(test["expected"], raw)
        result["raw"] = raw
        result["category"] = test["tags"]["category"]
        result["tags"] = test["tags"]
        results[test["id"]] = result
    return results


FAILURE_TAGS = ("category", "difficulty", "language", "noise")


def failure_elements(result):
    """The failure categories a failing result covers.

    A failure category is a (tag=value, error type) pair, for example
    ("language=de", "missing_slot"). A failing test covers one pair per tag
    and error type it shows, so one test can cover several categories."""
    return frozenset((f"{tag}={result['tags'][tag]}", e)
                     for tag in FAILURE_TAGS for e in result["errors"])


def slot_prf(results):
    tp = sum(r["tp"] for r in results)
    fp = sum(r["fp"] for r in results)
    fn = sum(r["fn"] for r in results)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def summarize(results, tests):
    """Aggregate metrics for one level."""
    rows = [results[t["id"]] for t in tests]
    n = len(rows)
    precision, recall, f1 = slot_prf(rows)
    by_tag = {}
    for tag in ("category", "difficulty", "language", "noise"):
        groups = defaultdict(list)
        for t in tests:
            groups[t["tags"][tag]].append(results[t["id"]]["passed"])
        by_tag[tag] = {k: {"n": len(v), "accuracy": sum(v) / len(v)} for k, v in sorted(groups.items())}
    return {
        "n": n,
        "exact_match": sum(r["passed"] for r in rows) / n,
        "intent_accuracy": sum(r["action_ok"] for r in rows) / n,
        "slot_precision": precision,
        "slot_recall": recall,
        "slot_f1": f1,
        "failures": sum(not r["passed"] for r in rows),
        "error_types": dict(Counter(r["error_type"] for r in rows if not r["passed"])),
        "failure_categories": sorted({"/".join(e) for r in rows for e in failure_elements(r)}),
        "by_tag": by_tag,
    }


def cohen_kappa(a, b):
    """Cohen's kappa for two equal-length boolean lists."""
    n = len(a)
    if n == 0:
        return 0.0
    observed = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    if expected == 1:
        return 1.0
    return (observed - expected) / (1 - expected)


def agreement(results_a, results_b, tests):
    """Pass/fail agreement between two levels, a = cheaper level."""
    a = [results_a[t["id"]]["passed"] for t in tests]
    b = [results_b[t["id"]]["passed"] for t in tests]
    same_output = sum(
        parse_output(results_a[t["id"]]["raw"]) == parse_output(results_b[t["id"]]["raw"])
        for t in tests)
    cells = Counter(zip(a, b))
    missed = [t["id"] for t in tests
              if results_a[t["id"]]["passed"] and not results_b[t["id"]]["passed"]]
    elems_a = set().union(*(failure_elements(results_a[t["id"]]) for t in tests))
    elems_b = set().union(*(failure_elements(results_b[t["id"]]) for t in tests))
    return {
        "pass_fail_agreement": sum(x == y for x, y in zip(a, b)) / len(tests),
        "kappa": cohen_kappa(a, b),
        "same_output": same_output / len(tests),
        "both_pass": cells[(True, True)],
        "cheap_pass_expensive_fail": cells[(True, False)],
        "cheap_fail_expensive_pass": cells[(False, True)],
        "both_fail": cells[(False, False)],
        "tests_failing_only_at_expensive": missed,
        "failure_categories_only_at_expensive": sorted("/".join(e) for e in elems_b - elems_a),
        "failure_categories_only_at_cheap": sorted("/".join(e) for e in elems_a - elems_b),
    }
