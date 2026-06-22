"""Shared time-conversion helpers for ASR adapters.

Different ASR backends return timestamps in different units and shapes:
seconds (float), milliseconds (int), or ``[start, end]`` pairs. These
helpers normalise them to YouDub's internal ``int`` milliseconds.
"""

from __future__ import annotations

from typing import Any


def seconds_to_ms(value: Any, default: int = 0) -> int:
    """Convert a timestamp value to integer milliseconds.

    Heuristic: values > 100 000 are treated as already-milliseconds; values
    > 1000 are treated as seconds; otherwise seconds. Non-numeric values
    fall back to ``default``.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number > 100000:
        return int(round(number))
    return int(round(number * 1000))


def to_ms(seconds: float) -> int:
    """Convert seconds (float) to integer milliseconds."""
    return int(round(float(seconds) * 1000))


def time_value_to_ms(value: Any, default: int = 0) -> int:
    """Convert a timestamp value that may be seconds or milliseconds.

    Values > 1000 are treated as seconds; values <= 1000 are treated as
    seconds too (so ``0.5`` -> 500 ms). Values > 100 000 are treated as
    already-milliseconds. Non-numeric values fall back to ``default``.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number > 100000:
        return int(round(number))
    return to_ms(number / 1000.0) if number > 1000 else to_ms(number)


def timestamp_pair_to_ms(ts: Any) -> tuple[int, int] | None:
    """Convert a timestamp pair (``[start, end]`` or ``{start, end}``) to ms."""
    if isinstance(ts, dict):
        start = ts.get("start_time", ts.get("start"))
        end = ts.get("end_time", ts.get("end"))
        return time_value_to_ms(start), time_value_to_ms(end)
    if isinstance(ts, (list, tuple)) and len(ts) >= 2:
        return int(round(float(ts[0]))), int(round(float(ts[1])))
    return None
