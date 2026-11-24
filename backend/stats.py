"""Small, tested statistical helpers with full interval endpoints.

- Bootstrap percentile intervals report ``lo``/``hi`` (never a half-width).
- All helpers report requested vs valid vs missing counts and the seed.
- Pooled rows are NOT treated as independent: callers pass the resampling
  unit explicitly (documented in ``unit``).
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Sequence


def _clean(values: Sequence[float]) -> List[float]:
    out = []
    for v in values:
        try:
            f = float(v)
        except Exception:
            continue
        if math.isfinite(f):
            out.append(f)
    return out


def bootstrap_mean_ci(values: Sequence[float], n_resamples: int = 10000,
                      seed: int = 0, ci: float = 0.95,
                      unit: str = "prompt") -> Dict[str, float | int | str]:
    """Percentile bootstrap CI for the mean with full endpoints."""
    vals = _clean(values)
    n_valid = len(vals)
    n_missing = len(list(values)) - n_valid
    if n_valid == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n_requested": len(list(values)), "n_valid": 0,
                "n_missing": n_missing, "n_resamples": n_resamples,
                "seed": seed, "ci": ci, "unit": unit, "status": "missing"}
    mean = sum(vals) / n_valid
    if n_resamples <= 0:
        return {"mean": mean, "lo": float("nan"), "hi": float("nan"),
                "n_requested": len(list(values)), "n_valid": n_valid,
                "n_missing": n_missing, "n_resamples": 0,
                "seed": seed, "ci": ci, "unit": unit, "status": "no-resample"}
    rng = random.Random(seed)
    alpha = (1.0 - ci) / 2.0
    means = []
    for _ in range(n_resamples):
        s = sum(vals[rng.randrange(n_valid)] for _ in range(n_valid)) / n_valid
        means.append(s)
    means.sort()
    lo_i = min(n_resamples - 1, max(0, int(alpha * n_resamples)))
    hi_i = min(n_resamples - 1, max(0, int((1.0 - alpha) * n_resamples)))
    # use nearest-rank endpoints (documented; not smoothed/interpolated)
    return {"mean": mean, "lo": means[lo_i], "hi": means[hi_i],
            "n_requested": len(list(values)), "n_valid": n_valid,
            "n_missing": n_missing, "n_resamples": n_resamples,
            "seed": seed, "ci": ci, "unit": unit, "status": "ok"}


def paired_signflip_pvalue(a: Sequence[float], b: Sequence[float],
                           n_perm: int = 10000, seed: int = 0) -> Dict[str, float | int | str]:
    """Two-sided paired sign-flip permutation p-value for mean(a-b).

    Uses the +1 smoothing documented in the report; attainable resolution is
    bounded by ``1/(n_perm+1)``. NaN/non-finite pairs are dropped and counted.
    """
    diffs = []
    n_missing = 0
    for x, y in zip(a, b):
        try:
            d = float(x) - float(y)
        except Exception:
            n_missing += 1
            continue
        if math.isfinite(d):
            diffs.append(d)
        else:
            n_missing += 1
    n_valid = len(diffs)
    if n_valid == 0:
        return {"p": float("nan"), "stat": float("nan"), "n_valid": 0,
                "n_missing": n_missing, "n_perm": n_perm, "seed": seed, "status": "missing"}
    obs = abs(sum(diffs) / n_valid)
    rng = random.Random(seed)
    exceed = 0
    for _ in range(n_perm):
        s = sum(d if rng.random() < 0.5 else -d for d in diffs) / n_valid
        if abs(s) >= obs:
            exceed += 1
    p = (exceed + 1) / (n_perm + 1)
    return {"p": p, "stat": sum(diffs) / n_valid, "n_valid": n_valid,
            "n_missing": n_missing, "n_perm": n_perm, "seed": seed, "status": "ok"}


def holm_adjust(pvals: Sequence[float]) -> List[float]:
    """Holm step-down adjusted p-values (NaN stays NaN)."""
    indexed = [(float(p) if p == p else float("nan"), i) for i, p in enumerate(pvals)]
    order = sorted([x for x in indexed if x[0] == x[0]], key=lambda t: t[0])
    m = len(order)
    adj = [float("nan")] * len(pvals)
    running = 0.0
    for rank, (p, i) in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p))
        adj[i] = running
    return adj
