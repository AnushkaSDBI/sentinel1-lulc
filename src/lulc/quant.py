"""8-bit storage of backscatter in dB: dB = code * 0.2 - 35, code 0 = no data.

Halves memory against float16 and the error (+-0.1 dB) is far below speckle.
"""

from __future__ import annotations

import numpy as np

DB_MIN = -35.0
DB_STEP = 0.2
NODATA_CODE = 0


def quantize_db(db: np.ndarray) -> np.ndarray:
    """float dB (NaN = no data) -> uint8 codes 1..255, 0 for no data."""
    db = np.asarray(db, dtype=np.float32)
    codes = np.clip(np.rint((db - DB_MIN) / DB_STEP), 1, 255)
    codes[~np.isfinite(db)] = NODATA_CODE
    return codes.astype(np.uint8)


def dequantize_db(codes: np.ndarray) -> np.ndarray:
    """uint8 codes -> float32 dB, NaN where the code is 0."""
    codes = np.asarray(codes)
    db = codes.astype(np.float32) * DB_STEP + DB_MIN
    db[codes == NODATA_CODE] = np.nan
    return db
