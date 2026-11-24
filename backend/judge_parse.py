"""Strict judge-rating parser.

The legacy parser accepted the first permitted digit anywhere in a longer
reply (e.g. ``"I give 5 stars!!!"`` or ``"12"``). This module requires the
whole reply to be exactly one integer in range after stripping whitespace.
Anything else is ``missing`` with a reason, eligible for one bounded retry.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

_STRICT = re.compile(r"^[+-]?\d+$")


def parse_rating(reply: object, lo: int, hi: int) -> Tuple[Optional[int], str]:
    """Parse one judge reply. Returns ``(rating_or_None, status)``."""
    if not isinstance(reply, str):
        return None, "non-string"
    s = reply.strip()
    if not _STRICT.match(s):
        return None, "not-exact-integer"
    try:
        v = int(s, 10)
    except Exception:
        return None, "not-exact-integer"
    if not (lo <= v <= hi):
        return None, "out-of-range"
    return v, "ok"


def parse_with_retry(replies: List[object], lo: int, hi: int,
                     max_attempts: int = 2) -> Dict[str, object]:
    """Consume up to ``max_attempts`` replies; record attempts and status."""
    attempts = 0
    for reply in replies[: max(0, max_attempts)]:
        attempts += 1
        value, status = parse_rating(reply, lo, hi)
        if status == "ok":
            return {"rating": value, "attempts": attempts, "status": "ok",
                    "raw": reply if isinstance(reply, str) else str(reply)}
    return {"rating": None, "attempts": attempts, "status": "missing",
            "raw": (replies[attempts - 1] if attempts else None)}
