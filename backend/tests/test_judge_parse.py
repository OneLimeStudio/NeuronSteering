"""Strict judge parser: exact integers only, bounded retries."""

from backend.judge_parse import parse_rating, parse_with_retry


def test_exact_integer_accepted():
    assert parse_rating("3", 1, 5) == (3, "ok")
    assert parse_rating("  4\n", 1, 5) == (4, "ok")
    assert parse_rating("-2", -2, 2) == (-2, "ok")


def test_digit_inside_longer_response_rejected():
    assert parse_rating("I give it a 5!", 1, 5)[1] != "ok"
    assert parse_rating("score: 4", 1, 5)[1] != "ok"
    assert parse_rating("3.0", 1, 5)[1] != "ok"
    assert parse_rating("++3", 1, 5)[1] != "ok"
    assert parse_rating("", 1, 5)[1] != "ok"
    assert parse_rating(None, 1, 5)[1] != "ok"


def test_out_of_range_rejected():
    assert parse_rating("12", 1, 5) == (None, "out-of-range")
    assert parse_rating("0", 1, 5) == (None, "out-of-range")


def test_retry_uses_second_reply_and_counts_attempts():
    r = parse_with_retry(["nope", "4"], 1, 5, max_attempts=2)
    assert r == {"rating": 4, "attempts": 2, "status": "ok", "raw": "4"}


def test_retry_exhaustion_is_missing():
    r = parse_with_retry(["nope", "still no"], 1, 5, max_attempts=2)
    assert r["rating"] is None and r["attempts"] == 2 and r["status"] == "missing"
