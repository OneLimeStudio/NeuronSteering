"""Human-annotation workflow (blinded export + label import).

No human labels exist yet. This module prepares the workflow and refuses to
compute agreement without real annotations:

- export: reproducible blinded sample (anonymous IDs, no condition/profile/
  dose/run metadata), fixed sampling rule, seed recorded.
- import: validates label files (known IDs, in-scale integers, no dupes).
- agreement: NOT computed here until real labels arrive; callers must specify
  whether they measure inter-rater or judge-vs-human agreement.
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Dict, List


def blinded_id(record_id: str, salt: str = "lab") -> str:
    return "ann-" + hashlib.sha256(f"{salt}:{record_id}".encode()).hexdigest()[:10]


def export_items(records: List[Dict[str, Any]], n: int, seed: int,
                 strata_key: str = "dose") -> Dict[str, Any]:
    """Reproducible blinded sample across strata (round-robin by sorted key).

    Only exports prompt text + response text. Never exports profile, dose,
    run_id, or record linkage. Returns ``{"items": [...], "spec": {...}}``.
    """
    ok = [r for r in records if r.get("status") == "ok" and r.get("text")]
    strata: Dict[str, List[Dict[str, Any]]] = {}
    for r in ok:
        strata.setdefault(str(r.get(strata_key)), []).append(r)
    for v in strata.values():
        v.sort(key=lambda r: str(r.get("record_id")))
    keys = sorted(strata)
    rng = random.Random(seed)  # reserved for documented tie-breaking; order is deterministic
    _ = rng.random()
    items, i = [], 0
    while len(items) < min(n, len(ok)) and any(strata.values()):
        for k in keys:
            if len(items) >= min(n, len(ok)):
                break
            if strata[k]:
                r = strata[k].pop(0)
                items.append({"item_id": blinded_id(str(r.get("record_id"))),
                              "prompt": r.get("prompt"), "response": r.get("text")})
        i += 1
        if i > len(ok) + 1:
            break
    return {"items": items,
            "spec": {"n_requested": n, "n_exported": len(items), "seed": seed,
                     "strata_key": strata_key, "blinded": True,
                     "withheld": ["profile", "dose", "run_id", "record_id"]}}


def validate_labels(labels: List[Dict[str, Any]], item_ids: List[str],
                    lo: int, hi: int) -> Dict[str, Any]:
    """Validate imported annotations. Returns counts + problems (no agreement)."""
    known = set(item_ids)
    seen, problems = set(), []
    valid = 0
    for row in labels:
        iid = row.get("item_id")
        if iid not in known:
            problems.append(f"unknown item_id {iid!r}")
            continue
        if iid in seen:
            problems.append(f"duplicate item_id {iid!r}")
            continue
        seen.add(iid)
        v = row.get("label")
        if not isinstance(v, int) or not (lo <= v <= hi):
            problems.append(f"item {iid!r}: label must be int in [{lo},{hi}]")
            continue
        valid += 1
    return {"n_items": len(item_ids), "n_labels": len(labels), "n_valid": valid,
            "n_missing": len(item_ids) - len([i for i in item_ids if i in seen]),
            "problems": problems, "agreement": None,
            "note": "agreement pending: specify inter-rater vs judge-vs-human + real labels"}
