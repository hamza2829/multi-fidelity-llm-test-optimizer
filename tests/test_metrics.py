import json

import pytest

from mfopt.metrics import agreement, cohen_kappa, failure_elements, parse_output, score, slot_prf

EXPECTED = {"action": "set_temperature", "slots": {"temperature": "21", "zone": "driver"}}


def out(action, slots):
    return json.dumps({"action": action, "slots": slots})


def test_parse_output_plain_fenced_and_invalid():
    assert parse_output(out("navigate", {"destination": "home"}))["action"] == "navigate"
    fenced = "```json\n" + out("navigate", {"destination": "home"}) + "\n```"
    assert parse_output(fenced)["slots"] == {"destination": "home"}
    assert parse_output('{"action": "navigate", "slots": {"destination": "ho') is None
    assert parse_output(out("fly_to_moon", {})) is None
    assert parse_output(None) is None


def test_score_exact_match_and_number_normalization():
    r = score(EXPECTED, out("set_temperature", {"temperature": 21, "zone": "Driver"}))
    assert r["passed"] and r["tp"] == 2 and r["fp"] == 0 and r["fn"] == 0
    assert r["error_type"] is None


@pytest.mark.parametrize("raw, primary, errors", [
    ("not json", "invalid_output", ["invalid_output"]),
    (out("set_fan_speed", {"temperature": "21", "zone": "driver"}), "wrong_action", ["wrong_action"]),
    (out("set_temperature", {"temperature": "21"}), "missing_slot", ["missing_slot"]),
    (out("set_temperature", {"temperature": "22", "zone": "driver"}), "wrong_slot_value",
     ["wrong_slot_value"]),
    (out("set_temperature", {"temperature": "21", "zone": "driver", "level": "3"}), "extra_slot",
     ["extra_slot"]),
    (out("set_fan_speed", {"level": "3"}), "wrong_action",
     ["wrong_action", "missing_slot", "extra_slot"]),
])
def test_score_error_types(raw, primary, errors):
    r = score(EXPECTED, raw)
    assert not r["passed"]
    assert r["error_type"] == primary
    assert r["errors"] == errors


def test_slot_prf_micro_average():
    rows = [score(EXPECTED, out("set_temperature", {"temperature": "21", "zone": "driver"})),
            score(EXPECTED, out("set_temperature", {"temperature": "22"}))]
    # tp = 2, fp = 1, fn = 2
    p, r, f1 = slot_prf(rows)
    assert p == pytest.approx(2 / 3)
    assert r == pytest.approx(2 / 4)
    assert f1 == pytest.approx(2 * (2 / 3) * 0.5 / (2 / 3 + 0.5))


def test_cohen_kappa():
    assert cohen_kappa([True, False, True, False], [True, False, True, False]) == pytest.approx(1.0)
    assert cohen_kappa([True, False, True, False], [False, True, False, True]) == pytest.approx(-1.0)
    assert cohen_kappa([True, True], [True, True]) == pytest.approx(1.0)


def test_failure_elements_one_per_tag_and_error():
    r = score(EXPECTED, out("set_fan_speed", {"level": "3"}))
    r["tags"] = {"category": "climate", "difficulty": "easy", "language": "de", "noise": "clean"}
    elems = failure_elements(r)
    assert ("language=de", "missing_slot") in elems
    assert len(elems) == 4 * 3
    ok = score(EXPECTED, out("set_temperature", {"temperature": "21", "zone": "driver"}))
    ok["tags"] = r["tags"]
    assert failure_elements(ok) == frozenset()


def test_agreement_counts_cheap_misses():
    tags = {"category": "climate", "difficulty": "easy", "language": "en", "noise": "clean"}
    tests = [{"id": f"t{i}", "tags": tags} for i in range(3)]

    def res(passed):
        raw = out("set_temperature", {"temperature": "21", "zone": "driver"} if passed else {})
        r = score(EXPECTED, raw)
        r["tags"] = tags
        r["raw"] = raw
        return r

    cheap = {"t0": res(True), "t1": res(True), "t2": res(False)}
    expensive = {"t0": res(True), "t1": res(False), "t2": res(False)}
    a = agreement(cheap, expensive, tests)
    assert a["both_pass"] == 1 and a["both_fail"] == 1
    assert a["cheap_pass_expensive_fail"] == 1
    assert a["tests_failing_only_at_expensive"] == ["t1"]
    assert a["pass_fail_agreement"] == pytest.approx(2 / 3)
