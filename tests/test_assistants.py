import json

import pytest

from mfopt.assistant import (HighFidelityUnavailable, LLMAssistant, MockModelAssistant,
                             RuleBasedAssistant)
from mfopt.metrics import parse_output


def test_rule_based_canonical_commands():
    a = RuleBasedAssistant()
    assert parse_output(a.predict("set temperature to 21")) == {
        "action": "set_temperature", "slots": {"temperature": "21"}}
    assert parse_output(a.predict("navigiere zur nächsten Ladestation"))["slots"] == {
        "destination": "charging_station"}
    assert parse_output(a.predict("tell me a joke"))["action"] == "unsupported"


def test_mock_model_handles_typos_without_noise():
    a = MockModelAssistant(error_rate=0.0)
    assert parse_output(a.predict("set volumme to 9")) == {
        "action": "set_volume", "slots": {"level": "9"}}


def test_mock_model_is_deterministic_per_seed():
    texts = [f"set the temperature to {t} degrees" for t in range(16, 29)] * 3
    a1, a2 = MockModelAssistant(seed=3, error_rate=0.5), MockModelAssistant(seed=3, error_rate=0.5)
    assert [a1.predict(t) for t in texts] == [a2.predict(t) for t in texts]
    b = MockModelAssistant(seed=4, error_rate=0.5)
    assert [a1.predict(t) for t in texts] != [b.predict(t) for t in texts]


def test_mock_model_error_rate_controls_noise():
    texts = [f"set the temperature to {t} degrees" for t in range(16, 29)]
    clean = MockModelAssistant(error_rate=0.0)
    noisy = MockModelAssistant(error_rate=0.6)
    diff = sum(clean.predict(t) != noisy.predict(t) for t in texts)
    assert all(parse_output(clean.predict(t))["action"] == "set_temperature" for t in texts)
    assert diff > 0


def test_llm_disabled_without_cache_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    llm = LLMAssistant(enabled=True, cache_path=tmp_path / "cache.json")
    assert not llm.enabled
    assert not llm.is_available_for(["call Anna"])
    with pytest.raises(HighFidelityUnavailable):
        llm.predict("call Anna")


def test_llm_reads_from_cache(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    path = tmp_path / "cache.json"
    llm = LLMAssistant(cache_path=path)
    answer = json.dumps({"action": "call_contact", "slots": {"contact": "anna"}})
    path.write_text(json.dumps({llm._key("call Anna"): {"output": answer}}))
    cached = LLMAssistant(cache_path=path)
    assert cached.is_available_for(["call Anna"])
    assert cached.predict("call Anna") == answer
