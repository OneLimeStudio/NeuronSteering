"""Calibration provenance and coordinate validation (no model needed)."""

import pytest

from backend.calibration import (
    calibration_fingerprint,
    get_median_strict,
    median_with_source,
    validate_coordinates,
)
from backend.steering import MEDIAN


class _Col:
    def __init__(self, norm):
        self._norm = norm

    def float(self):
        return self

    def norm(self):
        return self

    def item(self):
        return self._norm


class _Weight:
    def __init__(self, rows, cols, norm=2.0):
        self.shape = (rows, cols)
        self._norm = norm

    def __getitem__(self, key):
        return _Col(self._norm)


class _Mod:
    def __init__(self, weight):
        self.mlp = type("M", (), {"down_proj": type("D", (), {"weight": weight})()})()


def test_stored_median_source():
    layer = next(iter(MEDIAN))
    value, source = median_with_source(layer)
    assert source == "stored"
    assert value == float(MEDIAN[layer])


def test_strict_median_accepts_stored():
    layer = next(iter(MEDIAN))
    assert get_median_strict(layer) == float(MEDIAN[layer])


def test_calibration_fingerprint_stable():
    a = calibration_fingerprint()
    b = calibration_fingerprint()
    assert a["sha256"] == b["sha256"] and len(a["sha256"]) == 64
    assert a["layers"] == sorted(MEDIAN.keys())


def test_validate_coordinates_ok_and_push_math():
    layers = [_Mod(_Weight(8, 100, norm=4.0)) for _ in range(23)]
    layer = next(iter(MEDIAN))
    layers[layer] = _Mod(_Weight(8, 10000, norm=4.0))
    out = validate_coordinates(layers, [(layer, 41, 0.5)])
    assert out[0]["push"] == pytest.approx(0.5 * float(MEDIAN[layer]) / 4.0)
    assert out[0]["median_source"] == "stored"
    assert out[0]["eligible"] is True


def test_validate_coordinates_rejects_bad_layer_neuron_dose():
    layers = [_Mod(_Weight(8, 100)) for _ in range(23)]
    with pytest.raises(Exception):
        validate_coordinates(layers, [(999, 0, 0.5)])
    with pytest.raises(Exception):
        validate_coordinates(layers, [(16, 10**9, 0.5)])
    with pytest.raises(Exception):
        validate_coordinates(layers, [(16, 5, 9.0)])


def test_validate_coordinates_rejects_zero_norm():
    layers = [_Mod(_Weight(8, 100, norm=0.0)) for _ in range(23)]
    with pytest.raises(Exception):
        validate_coordinates(layers, [(16, 5, 0.5)])
