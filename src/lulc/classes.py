"""Class scheme: 6 SAR-separable classes, with mappings from two label products.

Training labels come from Dynamic World (9 classes). WorldCover (11 classes)
is used as an independent validation map, so both are mapped to the same
6 classes here.

Dropped on purpose (mapped to IGNORE_INDEX, excluded from loss and metrics):
  Dynamic World: flooded vegetation (3), snow and ice (8)
  WorldCover:    snow/ice (70), herbaceous wetland (90), mangrove (95), nodata (0)
Reasons: rare in the study regions and weakly mapped (global per-class accuracy
of flooded vegetation is about 53%). Revisit flooded vegetation once the class
shares per region are known; it may matter for the rice fields near Milan and
Valencia. State this choice in the report.
"""

from __future__ import annotations

import numpy as np

IGNORE_INDEX = 255

CLASS_NAMES: tuple[str, ...] = (
    "Tree cover",
    "Grass/shrub",
    "Cropland",
    "Built-up",
    "Bare",
    "Water",
)
N_CLASSES = len(CLASS_NAMES)

CLASS_COLORS: tuple[str, ...] = (
    "#1a7a1a",  # tree cover   dark green
    "#8fd18f",  # grass/shrub  light green
    "#e6c200",  # cropland     yellow
    "#4d4d4d",  # built-up     dark grey
    "#e8e8e8",  # bare         light grey / near-white
    "#1f6fc0",  # water        blue
)

# ESA WorldCover value -> LULC index
WORLDCOVER_TO_LULC: dict[int, int] = {
    10: 0,   # tree cover
    20: 1,   # shrubland
    30: 1,   # grassland
    40: 2,   # cropland
    50: 3,   # built-up
    60: 4,   # bare / sparse vegetation
    80: 5,   # permanent water
    100: 4,  # moss and lichen
}

# Dynamic World label index -> LULC index
DW_CLASS_NAMES: tuple[str, ...] = (
    "water", "trees", "grass", "flooded_vegetation", "crops",
    "shrub_and_scrub", "built", "bare", "snow_and_ice",
)
DYNAMIC_WORLD_TO_LULC: dict[int, int] = {
    0: 5,  # water
    1: 0,  # trees
    2: 1,  # grass
    4: 2,  # crops
    5: 1,  # shrub and scrub
    6: 3,  # built
    7: 4,  # bare
}


# Esri 10 m annual LULC. The community-catalog page lists two value sets (the
# original Impact Observatory values and a remapped 1..9 set) without saying which
# one the Earth Engine asset holds. Detect it from the pixel values, never assume.
ESRI_ORIGINAL_TO_LULC: dict[int, int] = {
    1: 5,    # water
    2: 0,    # trees
    5: 2,    # crops
    7: 3,    # built area
    8: 4,    # bare ground
    11: 1,   # rangeland (merged grass and scrub)
}  # 4 flooded vegetation, 9 snow/ice, 10 clouds -> ignore
ESRI_REMAPPED_TO_LULC: dict[int, int] = {
    1: 5,    # water
    2: 0,    # trees
    4: 2,    # crops
    5: 3,    # built area
    6: 4,    # bare ground
    9: 1,    # rangeland
}  # 3 flooded vegetation, 7 snow/ice, 8 clouds -> ignore


def detect_esri_scheme(values) -> str:
    """'original' or 'remapped', from the set of raw values found in the data.

    Rangeland is 11 in the original scheme and 9 in the remapped one; values 3 and 6
    exist only in the remapped one, 10 and 11 only in the original one.
    """
    v = {int(x) for x in np.unique(np.asarray(values))}
    if v & {10, 11}:
        return "original"
    if v & {3, 6}:
        return "remapped"
    raise ValueError(f"cannot tell the Esri value scheme from values {sorted(v)}")


def make_lut(mapping: dict[int, int]) -> np.ndarray:
    """Lookup table of size 256: source value -> LULC index (IGNORE_INDEX if unmapped)."""
    lut = np.full(256, IGNORE_INDEX, dtype=np.uint8)
    for src, dst in mapping.items():
        lut[src] = dst
    return lut


def _remap(arr: np.ndarray, lut: np.ndarray, what: str) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        if arr.size and (arr.min() < 0 or arr.max() > 255):
            raise ValueError(f"{what} values must be in 0..255")
        arr = arr.astype(np.uint8)
    return lut[arr]


_WC_LUT = make_lut(WORLDCOVER_TO_LULC)
_DW_LUT = make_lut(DYNAMIC_WORLD_TO_LULC)


def remap_worldcover(arr: np.ndarray) -> np.ndarray:
    """Remap a WorldCover array to the 6-class scheme."""
    return _remap(arr, _WC_LUT, "WorldCover")


def remap_dynamic_world(arr: np.ndarray) -> np.ndarray:
    """Remap a Dynamic World label array (0..8, 255 = no data) to the 6-class scheme."""
    return _remap(arr, _DW_LUT, "Dynamic World")


def remap_esri(arr: np.ndarray, scheme: str | None = None) -> np.ndarray:
    """Remap an Esri LULC array to the 6-class scheme (0 = no data -> ignore)."""
    scheme = scheme or detect_esri_scheme(arr[arr > 0])
    if scheme not in ("original", "remapped"):
        raise ValueError(f"unknown Esri scheme {scheme!r}")
    mapping = ESRI_ORIGINAL_TO_LULC if scheme == "original" else ESRI_REMAPPED_TO_LULC
    return _remap(arr, make_lut(mapping), "Esri")
