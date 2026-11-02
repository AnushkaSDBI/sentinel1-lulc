"""Model card: everything the prototype needs to reproduce training-time preprocessing.

Saved next to the weights. The prototype reads it, so normalisation, channel order,
composite recipe and class scheme cannot drift between training and inference.
(Idea from a reference implementation.)
"""

from __future__ import annotations

import json
from pathlib import Path

SCHEMA_VERSION = 1


def build_model_card(
    class_names: list[str],
    channels: list[str],
    norm: dict[str, tuple[float, float]],
    composite: dict,
    window_months: int,
    patch: int,
    ignore_index: int,
    extra: dict | None = None,
) -> dict:
    if set(norm) != set(channels):
        raise ValueError(f"norm keys {sorted(norm)} must match channels {sorted(channels)}")
    if any(std <= 0 for _, std in norm.values()):
        raise ValueError("standard deviations must be positive")
    return {
        "schema_version": SCHEMA_VERSION,
        "class_names": list(class_names),
        "channels": list(channels),
        "norm": {k: {"mean": float(m), "std": float(s)} for k, (m, s) in norm.items()},
        "composite": composite,
        "window_months": int(window_months),
        "patch": int(patch),
        "ignore_index": int(ignore_index),
        **(extra or {}),
    }


def write_model_card(path: Path, card: dict) -> None:
    Path(path).write_text(json.dumps(card, indent=2))


def read_model_card(path: Path) -> dict:
    card = json.loads(Path(path).read_text())
    if card.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported model card version {card.get('schema_version')}")
    return card
