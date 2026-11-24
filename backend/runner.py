"""Minimal resumable experiment runner over the existing backend API.

Scope: plumbing for paired sweep evaluations with durable records. It does
NOT implement LLM judging, bootstrap significance families, or the full
100-prompt study — those remain pending (see report). Mock smoke runs verify
plumbing only and are never accepted as scientific evidence.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import tempfile
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from .calibration import calibration_fingerprint
from .stats import bootstrap_mean_ci

RUN_STATES = ("created", "preflight_passed", "running", "interrupted",
              "failed", "completed", "validated")

DOSE_GRID = [-1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0,
             0.25, 0.5, 0.8, 1.0, 1.25, 1.5]

# Plumbing-only smoke prompts (new, documented). NOT evaluation evidence.
SMOKE_PROMPTS = [
    {"id": "smoke-01", "category": "everyday", "text": "What should I pack for a beach trip?"},
    {"id": "smoke-02", "category": "factual", "text": "Explain how rain forms in two sentences."},
]


class IncompatibleCacheError(ValueError):
    pass


def new_run_id(prefix: str = "run") -> str:
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{stamp}-{uuid.uuid4().hex[:8]}"


def sha256_canonical(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def code_fingerprint(files: List[str]) -> Dict[str, str]:
    out = {}
    for f in files:
        try:
            with open(f, "rb") as fh:
                out[f] = hashlib.sha256(fh.read()).hexdigest()
        except Exception as e:
            out[f] = f"unreadable:{e}"
    return out


def config_fingerprint(config: Dict[str, Any]) -> str:
    return sha256_canonical(config)


def atomic_write_json(path: str, obj: Any) -> None:
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except Exception:
            pass
        raise


def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> Tuple[List[Dict[str, Any]], int]:
    records, corrupt = [], 0
    if not os.path.exists(path):
        return records, 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                corrupt += 1
    return records, corrupt


def default_config(prompts: Optional[List[Dict[str, str]]] = None,
                   mask: str = "jesus", doses: Optional[List[float]] = None,
                   max_tokens: int = 60, seed: int = 0,
                   backend: str = "mock") -> Dict[str, Any]:
    from .steering import MODEL_ID, PROFILES

    return {
        "model_id": MODEL_ID,
        "model_revision": None,  # unverified until Colab preflight records it
        "mask": mask,
        "profiles": {k: v["neurons"] for k, v in PROFILES.items()},
        "doses": list(doses) if doses is not None else list(DOSE_GRID),
        "prompts": prompts if prompts is not None else [dict(p) for p in SMOKE_PROMPTS],
        "max_tokens": max_tokens,
        "seed": seed,
        "backend": backend,
        "calibration": calibration_fingerprint(),
    }


def init_run(runs_root: str, config: Dict[str, Any], phases: List[str],
             run_id: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    run_id = run_id or new_run_id("smoke" if config.get("backend") == "mock" else "run")
    run_dir = os.path.join(runs_root, run_id)
    os.makedirs(run_dir, exist_ok=False)
    manifest = {
        "run_id": run_id,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "state": "created",
        "config": config,
        "config_sha": config_fingerprint(config),
        "phases_requested": list(phases),
        "phases_status": {p: "pending" for p in phases},
        "counts": {"expected": 0, "attempted": 0, "successful": 0,
                   "failed": 0, "missing": 0, "reused": 0, "model_calls": 0},
        "timing_s": {},
        "calibration_fallbacks": 0,
        "peak_memory_mb": None,  # instrumented only in Colab (nvidia-smi); None here
        "artifacts": [],
    }
    atomic_write_json(os.path.join(run_dir, "manifest.json"), manifest)
    return run_dir, manifest


def load_manifest(run_dir: str) -> Dict[str, Any]:
    with open(os.path.join(run_dir, "manifest.json"), encoding="utf-8") as fh:
        return json.load(fh)


def set_state(run_dir: str, state: str, phase: Optional[str] = None,
              phase_state: Optional[str] = None) -> Dict[str, Any]:
    assert state in RUN_STATES, f"unknown state {state}"
    m = load_manifest(run_dir)
    m["state"] = state
    if phase is not None and phase_state is not None:
        m["phases_status"][phase] = phase_state
    atomic_write_json(os.path.join(run_dir, "manifest.json"), m)
    return m


def is_live_record(rec: Dict[str, Any]) -> bool:
    return rec.get("mock") is False and rec.get("label") == "live"


def reject_synthetic(records: List[Dict[str, Any]]) -> None:
    """Refuse to accept pre-recorded mock rows as live evidence."""
    for r in records:
        if not is_live_record(r):
            raise IncompatibleCacheError(
                "synthetic/pre-recorded record cannot enter accepted live analysis "
                f"(record_id={r.get('record_id')}, label={r.get('label')}, mock={r.get('mock')})"
            )


class _Client:
    """Accept TestClient-like objects or http base URLs."""

    def __init__(self, client_or_url: Any):
        self._c = client_or_url

    def post(self, path: str, payload: Dict[str, Any], timeout: int = 600) -> Dict[str, Any]:
        if hasattr(self._c, "post"):
            r = self._c.post(path, json=payload)
            if hasattr(r, "raise_for_status"):
                try:
                    r.raise_for_status()
                except Exception as e:
                    body = getattr(r, "text", "")[:500]
                    raise RuntimeError(f"POST {path} failed: {e} :: {body}")
            return r.json() if hasattr(r, "json") else r
        import urllib.request

        base = str(self._c).rstrip("/")
        data = json.dumps(payload).encode()
        req = urllib.request.Request(base + path, data=data,
                                     headers={"Content-Type": "application/json",
                                              "ngrok-skip-browser-warning": "true"},
                                     method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())


def run_sweep_phase(client_or_url: Any, run_dir: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Execute one sweep evaluation per prompt (resumable)."""
    m = load_manifest(run_dir)
    if m["config_sha"] != config_fingerprint(config):
        raise IncompatibleCacheError("config changed since run creation; refusing reuse")
    set_state(run_dir, "running", "sweep", "running")
    client = _Client(client_or_url)
    raw_path = os.path.join(run_dir, "raw_generations.jsonl")
    existing, corrupt = read_jsonl(raw_path)
    done = {(r.get("prompt_id"), r.get("dose")) for r in existing
            if r.get("config_sha") == m["config_sha"]}
    t0 = time.time()
    counts = {"expected": 0, "attempted": 0, "successful": 0, "failed": 0,
              "missing": 0, "reused": len(done), "model_calls": 0}
    t_calls = 0.0
    try:
        for p in config["prompts"]:
            body = {"prompt": p["text"], "mask": config["mask"],
                    "doses": config["doses"], "max_tokens": config["max_tokens"],
                    "seed": config["seed"]}
            # primary neuron passthrough kept server-side; nothing else needed
            pending = [d for d in config["doses"] if (p["id"], d) not in done]
            counts["expected"] += len(config["doses"])
            if not pending:
                continue
            counts["attempted"] += len(pending)
            t1 = time.time()
            try:
                resp = client.post("/api/sweep", body)
                t_calls += time.time() - t1
                counts["model_calls"] += 1
            except Exception as e:
                counts["failed"] += len(pending)
                for d in pending:
                    append_jsonl(raw_path, {"record_id": f"{m['run_id']}:{p['id']}:{d}",
                                            "run_id": m["run_id"], "config_sha": m["config_sha"],
                                            "phase": "sweep", "prompt_id": p["id"],
                                            "prompt_category": p.get("category"),
                                            "prompt": p["text"], "mask": config["mask"],
                                            "dose": d, "status": "failed",
                                            "failure_reason": str(e)[:500],
                                            "mock": None, "label": None, "text": None})
                continue
            by_dose = {r.get("dose"): r for r in resp.get("results", [])}
            for d in pending:
                r = by_dose.get(d)
                if r is None or r.get("text") is None:
                    counts["missing"] += 1
                    append_jsonl(raw_path, {"record_id": f"{m['run_id']}:{p['id']}:{d}",
                                            "run_id": m["run_id"], "config_sha": m["config_sha"],
                                            "phase": "sweep", "prompt_id": p["id"],
                                            "prompt_category": p.get("category"),
                                            "prompt": p["text"], "mask": config["mask"],
                                            "dose": d, "status": "missing",
                                            "failure_reason": "no text for dose",
                                            "mock": resp.get("mock"), "label": resp.get("label"),
                                            "text": None})
                else:
                    counts["successful"] += 1
                    append_jsonl(raw_path, {"record_id": f"{m['run_id']}:{p['id']}:{d}",
                                            "run_id": m["run_id"], "config_sha": m["config_sha"],
                                            "phase": "sweep", "prompt_id": p["id"],
                                            "prompt_category": p.get("category"),
                                            "prompt": p["text"], "mask": config["mask"],
                                            "dose": d, "status": "ok",
                                            "failure_reason": None,
                                            "mock": resp.get("mock"), "label": resp.get("label"),
                                            "text": r.get("text"),
                                            "keyword_score": r.get("keyword_score"),
                                            "kl": r.get("kl"), "similarity": r.get("similarity"),
                                            "ppl": r.get("ppl"), "push": r.get("push"),
                                            "baseline": resp.get("baseline")})
        m = load_manifest(run_dir)
        m["counts"] = counts
        m["timing_s"]["sweep"] = round(time.time() - t0, 2)
        m["timing_s"]["model_calls_s"] = round(t_calls, 2)
        m["corrupt_lines_skipped"] = corrupt
        m["artifacts"] = sorted(set(m.get("artifacts", [])) | {"raw_generations.jsonl"})
        atomic_write_json(os.path.join(run_dir, "manifest.json"), m)
        set_state(run_dir, "running", "sweep",
                  "completed" if counts["failed"] == 0 and counts["missing"] == 0 else "completed_with_missing")
        return counts
    except Exception:
        set_state(run_dir, "interrupted", "sweep", "interrupted")
        raise


def aggregate_sweep(run_dir: str, only_live: bool = True) -> Dict[str, Any]:
    """Aggregate per-dose means with full CI endpoints. Synthetic rows rejected when only_live."""
    m = load_manifest(run_dir)
    records, corrupt = read_jsonl(os.path.join(run_dir, "raw_generations.jsonl"))
    if only_live:
        live = [r for r in records if r.get("status") == "ok" and is_live_record(r)]
        non_live_ok = [r for r in records if r.get("status") == "ok" and not is_live_record(r)]
        if non_live_ok:
            raise IncompatibleCacheError(
                f"{len(non_live_ok)} non-live rows refused for live aggregation "
                "(mock smoke is plumbing only)")
        usable = live
    else:
        usable = [r for r in records if r.get("status") == "ok"]
    per_dose: Dict[str, List[float]] = {}
    for r in usable:
        try:
            per_dose.setdefault(str(r["dose"]), []).append(float(r["keyword_score"]))
        except Exception:
            continue
    table = []
    for d in sorted(per_dose, key=float):
        ci = bootstrap_mean_ci(per_dose[d], unit="prompt")
        table.append({"dose": float(d), **ci})
    agg = {"run_id": m["run_id"], "config_sha": m["config_sha"], "only_live": only_live,
           "n_records": len(records), "n_usable": len(usable), "corrupt_lines": corrupt,
           "per_dose_keyword": table}
    atomic_write_json(os.path.join(run_dir, "aggregates.json"), agg)
    m = load_manifest(run_dir)
    m["artifacts"] = sorted(set(m.get("artifacts", [])) | {"aggregates.json"})
    atomic_write_json(os.path.join(run_dir, "manifest.json"), m)
    return agg
