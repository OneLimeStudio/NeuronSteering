"""Calibration provenance and coordinate validation.

The steering math itself lives in :mod:`backend.steering` and is unchanged.
This module adds the provenance that accepted scientific runs require:

- where each layer median came from (stored / cache / downloaded / fallback),
- a strict accessor that refuses the silent ``10.0`` fallback,
- a calibration fingerprint,
- coordinate validation against a loaded model,
- a robust model fingerprint (not a scalar sum).
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, Iterable, List, Tuple

from .steering import MEDIAN, NORMS_URL

FALLBACK_MEDIAN = 10.0


class CalibrationError(ValueError):
    pass


class CoordinateError(ValueError):
    pass


def median_with_source(layer: int) -> Tuple[float, str]:
    """Return ``(value, source)`` for a layer median.

    Sources: ``stored`` (shipped table), ``cache`` (previously downloaded),
    ``downloaded`` (fresh fetch this call), ``fallback`` (unverified 10.0).
    """
    from . import steering as _st

    if layer in MEDIAN:
        return float(MEDIAN[layer]), "stored"
    if layer in _st._NORMS_CACHE:
        return float(_st._NORMS_CACHE[layer]), "cache"
    try:
        import urllib.request

        with urllib.request.urlopen(_st.NORMS_URL, timeout=8) as r:
            payload = json.loads(r.read().decode())
        vals = payload.get("medians", payload.get("norms", payload)) if isinstance(payload, dict) else {}
        if isinstance(vals, dict):
            for k, v in vals.items():
                try:
                    _st._NORMS_CACHE[int(k)] = float(v)
                except Exception:
                    continue
        if layer in _st._NORMS_CACHE:
            return float(_st._NORMS_CACHE[layer]), "downloaded"
    except Exception:
        pass
    return float(FALLBACK_MEDIAN), "fallback"


def get_median_strict(layer: int) -> float:
    """Median for accepted scientific runs: raises on fallback."""
    value, source = median_with_source(layer)
    if source == "fallback":
        raise CalibrationError(
            f"No verified median for layer {layer} (source={NORMS_URL}); "
            "refusing silent fallback 10.0. Remeasure or approve a calibration protocol."
        )
    return value


def calibration_fingerprint() -> Dict[str, Any]:
    """Stable fingerprint of the calibration configuration."""
    canonical = json.dumps(
        {"medians": {str(k): float(v) for k, v in sorted(MEDIAN.items())},
         "url": NORMS_URL,
         "fallback": float(FALLBACK_MEDIAN)},
        sort_keys=True,
    )
    return {
        "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "layers": sorted(MEDIAN.keys()),
        "url": NORMS_URL,
        "fallback": float(FALLBACK_MEDIAN),
    }


def validate_coordinates(layers: Any, coords: Iterable[Tuple[int, int, float]]) -> List[Dict[str, Any]]:
    """Validate every configured coordinate against loaded layers.

    Checks: layer exists, ``mlp.down_proj`` exists with a 2-D weight,
    neuron index in range, dose finite within [-1.5, 1.5], column norm
    finite and nonzero. Records median source; flags fallback as ineligible
    for accepted runs. Raises :class:`CoordinateError` aggregating problems.
    """
    coords = list(coords)
    errors: List[str] = []
    try:
        n_layers = len(layers)
    except Exception:
        raise CoordinateError("layers object has no length")
    out: List[Dict[str, Any]] = []
    for i, (layer, neuron, dose) in enumerate(coords):
        tag = f"coords[{i}]=(L{layer},N{neuron},d={dose})"
        if not isinstance(layer, int) or not (0 <= layer < n_layers):
            errors.append(f"{tag}: layer out of range 0..{n_layers - 1}")
            continue
        try:
            mod = layers[layer].mlp.down_proj
            weight = mod.weight
        except Exception as e:
            errors.append(f"{tag}: no mlp.down_proj weight ({e})")
            continue
        try:
            shape = tuple(weight.shape)
        except Exception:
            errors.append(f"{tag}: weight has no shape")
            continue
        if len(shape) != 2:
            errors.append(f"{tag}: expected 2-D down_proj weight, got {shape}")
            continue
        if not isinstance(neuron, int) or not (0 <= neuron < shape[1]):
            errors.append(f"{tag}: neuron out of range 0..{shape[1] - 1}")
            continue
        if not isinstance(dose, (int, float)) or not math.isfinite(dose) or abs(dose) > 1.5:
            errors.append(f"{tag}: dose must be finite within [-1.5, 1.5]")
            continue
        try:
            w_norm = float(weight[:, neuron].float().norm().item())
        except Exception as e:
            errors.append(f"{tag}: cannot compute column norm ({e})")
            continue
        if not math.isfinite(w_norm) or w_norm <= 0:
            errors.append(f"{tag}: non-finite or zero column norm ({w_norm})")
            continue
        median, source = median_with_source(layer)
        out.append({"layer": layer, "neuron": neuron, "dose": float(dose),
                    "w_norm": w_norm, "median": median, "median_source": source,
                    "push": float(dose) * median / w_norm,
                    "eligible": source != "fallback"})
    if errors:
        raise CoordinateError("; ".join(errors))
    return out


def fingerprint_entries(entries: Iterable[Tuple[str, Tuple[int, ...], str, bytes]]) -> Dict[str, Any]:
    """Robust fingerprint over (name, shape, dtype, sample bytes) entries."""
    h = hashlib.sha256()
    names: List[str] = []
    for name, shape, dtype, sample in entries:
        names.append(name)
        h.update(name.encode("utf-8") + b"\x00")
        h.update(json.dumps(list(shape)).encode() + b"\x00")
        h.update(str(dtype).encode("utf-8") + b"\x00")
        h.update(bytes(sample) + b"\x00")
    return {"sha256": h.hexdigest(), "n_entries": len(names)}


def torch_model_fingerprint(model: Any) -> Dict[str, Any]:
    """Fingerprint a torch model without trusting a scalar sum."""
    try:
        import torch as _t
    except Exception as e:
        raise CalibrationError(f"torch unavailable for fingerprint: {e}")
    entries = []
    n_params = 0
    dtypes, devices = set(), set()
    for name, p in model.named_parameters():
        try:
            shape = tuple(p.shape)
            n_params += int(p.numel())
            dtypes.add(str(p.dtype))
            devices.add(str(p.device))
            flat = p.detach().flatten().cpu()
            k = min(16, flat.numel())
            head = flat[:k].tolist() if k else []
            tail = flat[-k:].tolist() if k else []
            sample = json.dumps([head, tail], sort_keys=True).encode()
        except Exception:
            shape, sample = (), b"unreadable"
        entries.append((name, shape, str(getattr(p, 'dtype', '?')), sample))
    fp = fingerprint_entries(entries)
    fp.update({"n_params": n_params, "dtypes": sorted(dtypes), "devices": sorted(devices),
               "arch": type(model).__name__})
    try:
        import torch as _t2
        fp["torch_version"] = _t2.__version__
    except Exception:
        pass
    return fp
