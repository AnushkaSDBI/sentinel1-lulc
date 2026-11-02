"""Calendar-month windows for compositing and STAC queries."""

from __future__ import annotations

from datetime import date, timedelta


def month_windows(start: str, n: int = 12) -> list[tuple[str, date, date]]:
    """n consecutive months from 'YYYY-MM'. Returns (label, first_day, first_day_of_next_month)."""
    try:
        year, month = (int(p) for p in start.split("-"))
        date(year, month, 1)
    except (ValueError, TypeError) as e:
        raise ValueError(f"start must look like 'YYYY-MM', got {start!r}") from e
    out = []
    for i in range(n):
        y = year + (month - 1 + i) // 12
        m = (month - 1 + i) % 12 + 1
        ny = y + m // 12
        nm = m % 12 + 1
        out.append((f"{y:04d}-{m:02d}", date(y, m, 1), date(ny, nm, 1)))
    return out


def stac_interval(first: date, next_first: date) -> str:
    """Inclusive STAC datetime interval covering exactly one month."""
    last = next_first - timedelta(days=1)
    return f"{first:%Y-%m-%d}T00:00:00Z/{last:%Y-%m-%d}T23:59:59Z"
