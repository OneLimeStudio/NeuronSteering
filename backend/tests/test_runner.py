"""Runner durability: atomic manifest, resume, fingerprint checks, synthetic rejection."""

import json
import os

import pytest

os.environ.setdefault("MOCK_MODE", "1")

from fastapi.testclient import TestClient

from backend.app import app
from backend.runner import (
    IncompatibleCacheError,
    aggregate_sweep,
    default_config,
    init_run,
    load_manifest,
    read_jsonl,
    reject_synthetic,
    run_sweep_phase,
    set_state,
)


def _client():
    return TestClient(app)


def test_init_manifest_states_and_counts(tmp_path):
    cfg = default_config(prompts=[{"id": "p1", "category": "everyday", "text": "Say hi."}],
                         doses=[0.0, 0.8], max_tokens=20)
    run_dir, m = init_run(str(tmp_path), cfg, ["sweep"], run_id="smoke-test-1")
    m = load_manifest(run_dir)
    assert m["state"] == "created"
    assert m["phases_status"] == {"sweep": "pending"}
    assert m["counts"]["successful"] == 0
    set_state(run_dir, "preflight_passed")
    assert load_manifest(run_dir)["state"] == "preflight_passed"


def test_smoke_run_mock_records_but_live_aggregation_refused(tmp_path):
    cfg = default_config(prompts=[{"id": "p1", "category": "everyday", "text": "Say hi."}],
                         doses=[0.0, 0.8], max_tokens=20, backend="mock")
    run_dir, _ = init_run(str(tmp_path), cfg, ["sweep"], run_id="smoke-test-2")
    counts = run_sweep_phase(_client(), run_dir, cfg)
    assert counts["successful"] == 2 and counts["failed"] == 0
    recs, corrupt = read_jsonl(os.path.join(run_dir, "raw_generations.jsonl"))
    assert len(recs) == 2 and corrupt == 0
    # mock rows are plumbing only: live aggregation must refuse them
    with pytest.raises(IncompatibleCacheError):
        aggregate_sweep(run_dir, only_live=True)
    with pytest.raises(IncompatibleCacheError):
        reject_synthetic([{"mock": True, "label": "pre-recorded", "record_id": "x"}])
    agg = aggregate_sweep(run_dir, only_live=False)
    assert agg["n_usable"] == 2
    assert {r["dose"] for r in agg["per_dose_keyword"]} == {0.0, 0.8}


def test_resume_reuses_matching_records(tmp_path):
    cfg = default_config(prompts=[{"id": "p1", "category": "everyday", "text": "Say hi."}],
                         doses=[0.0, 0.8], max_tokens=20, backend="mock")
    run_dir, _ = init_run(str(tmp_path), cfg, ["sweep"], run_id="smoke-test-3")
    first = run_sweep_phase(_client(), run_dir, cfg)
    assert first["model_calls"] == 1
    second = run_sweep_phase(_client(), run_dir, cfg)
    assert second["reused"] == 2 and second["model_calls"] == 0
    recs, _ = read_jsonl(os.path.join(run_dir, "raw_generations.jsonl"))
    assert len(recs) == 2  # no duplicates appended


def test_config_change_refuses_reuse(tmp_path):
    cfg = default_config(prompts=[{"id": "p1", "category": "everyday", "text": "Say hi."}],
                         doses=[0.0, 0.8], max_tokens=20, backend="mock")
    run_dir, _ = init_run(str(tmp_path), cfg, ["sweep"], run_id="smoke-test-4")
    run_sweep_phase(_client(), run_dir, cfg)
    cfg2 = dict(cfg, seed=99)
    with pytest.raises(IncompatibleCacheError):
        run_sweep_phase(_client(), run_dir, cfg2)
