import json

from mfopt.run import load_tests, run_pipeline


def test_testset_shape():
    tests = load_tests()
    assert len(tests) == 150
    assert len({t["id"] for t in tests}) == 150
    assert {t["tags"]["language"] for t in tests} == {"en", "de"}
    assert {t["tags"]["noise"] for t in tests} == {"clean", "typo"}
    for t in tests:
        assert set(t["cost"]) == {"low_s", "medium_s", "high_s", "high_tokens"}
        assert t["cost"]["low_s"] < t["cost"]["medium_s"] < t["cost"]["high_s"]


def test_full_pipeline_runs_without_api(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("MFOPT_ENABLE_LLM", raising=False)
    report = run_pipeline(out_dir=tmp_path, eval_seeds=2, random_seeds=3)

    assert report["levels"][:2] == ["low", "medium"]
    for name in ("results.md", "results.json", "accuracy_by_fidelity.png",
                 "cost_vs_coverage.png", "failure_heatmap.png"):
        assert (tmp_path / name).stat().st_size > 0
    saved = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert saved["config"]["n_tests"] == 150

    for sel in report["selection"].values():
        m = sel["method"]
        assert m["coverage_calibration"] == 1.0
        assert 0 < m["cost_saved"] < 1
        assert 0 <= m["coverage_fresh"] <= 1
    low = report["summaries"]["low"]["exact_match"]
    medium = report["summaries"]["medium"]["exact_match"]
    assert 0 < low < medium < 1
