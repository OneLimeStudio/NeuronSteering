"""Statistics helpers report full endpoints and valid denominators."""

from backend.stats import bootstrap_mean_ci, holm_adjust, paired_signflip_pvalue


def test_bootstrap_constant_is_exact():
    r = bootstrap_mean_ci([0.5] * 10, n_resamples=200, seed=0)
    assert r["mean"] == 0.5 and r["lo"] == 0.5 and r["hi"] == 0.5
    assert r["n_valid"] == 10 and r["n_missing"] == 0
    assert r["status"] == "ok"


def test_bootstrap_missing_reported_not_zero():
    r = bootstrap_mean_ci([1.0, float("nan"), 3.0, float("inf")], n_resamples=200, seed=1)
    assert r["n_valid"] == 2 and r["n_missing"] == 2
    assert r["mean"] == 2.0
    assert r["lo"] <= r["mean"] <= r["hi"]


def test_bootstrap_empty_is_missing():
    r = bootstrap_mean_ci([], n_resamples=100, seed=0)
    assert r["status"] == "missing" and r["n_valid"] == 0


def test_paired_identical_gives_p_one():
    r = paired_signflip_pvalue([1, 2, 3], [1, 2, 3], n_perm=200, seed=0)
    assert r["p"] == 1.0 and r["n_valid"] == 3


def test_holm_monotone_and_bounded():
    adj = holm_adjust([0.01, 0.04, 0.5])
    assert len(adj) == 3 and all(0 <= p <= 1 for p in adj)
    assert adj[0] <= adj[1] <= adj[2]
