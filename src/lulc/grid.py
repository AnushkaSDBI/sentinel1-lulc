"""One pixel grid per region, so imagery and labels line up by construction.

Each region gets a square grid in its UTM zone, snapped to multiples of the
pixel size. Both Sentinel-1 and Dynamic World are requested on this exact grid,
so there is no separate resampling step that could shift one against the other.
Large grids are split into tiles because Earth Engine limits the size of one
download request.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from pyproj import Transformer

from .config import Region


def utm_epsg(lat: float, lon: float) -> int:
    """EPSG code of the UTM zone containing the point (WGS84)."""
    if not (-180 <= lon < 180):
        raise ValueError(f"longitude out of range: {lon}")
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


@dataclass(frozen=True)
class Grid:
    """Square grid. (x0, y0) is the top-left corner in CRS units (metres)."""

    epsg: int
    x0: float
    y0: float
    size_px: int
    res: float = 10.0

    @property
    def crs(self) -> str:
        return f"EPSG:{self.epsg}"

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """(xmin, ymin, xmax, ymax)"""
        span = self.size_px * self.res
        return (self.x0, self.y0 - span, self.x0 + span, self.y0)


@dataclass(frozen=True)
class Tile:
    row: int          # first pixel row inside the grid
    col: int          # first pixel column inside the grid
    height: int
    width: int
    x0: float         # top-left corner in CRS units
    y0: float


def grid_for_region(region: Region, res: float = 10.0, size_px: int | None = None) -> Grid:
    """Square UTM grid centred on the region, snapped to multiples of `res`."""
    if size_px is None:
        size_px = int(round(2 * region.half_size_km * 1000 / res))
    epsg = utm_epsg(region.lat, region.lon)
    to_utm = Transformer.from_crs(4326, epsg, always_xy=True)
    e, n = to_utm.transform(region.lon, region.lat)
    half = size_px * res / 2
    x0 = math.floor((e - half) / res) * res
    y0 = math.ceil((n + half) / res) * res
    return Grid(epsg=epsg, x0=x0, y0=y0, size_px=size_px, res=res)


def grid_bounds_lonlat(grid: Grid) -> tuple[float, float, float, float]:
    """Bounding box of the grid in lon/lat (west, south, east, north), for catalog searches."""
    to_ll = Transformer.from_crs(grid.epsg, 4326, always_xy=True)
    xmin, ymin, xmax, ymax = grid.extent
    corners = [to_ll.transform(x, y) for x in (xmin, xmax) for y in (ymin, ymax)]
    lons, lats = zip(*corners)
    return (min(lons), min(lats), max(lons), max(lats))


def split_tiles(grid: Grid, tile_px: int) -> list[Tile]:
    """Cover the grid with tiles of at most tile_px x tile_px, no overlap, no gaps."""
    if tile_px <= 0:
        raise ValueError("tile_px must be positive")
    tiles = []
    for r in range(0, grid.size_px, tile_px):
        for c in range(0, grid.size_px, tile_px):
            tiles.append(
                Tile(
                    row=r,
                    col=c,
                    height=min(tile_px, grid.size_px - r),
                    width=min(tile_px, grid.size_px - c),
                    x0=grid.x0 + c * grid.res,
                    y0=grid.y0 - r * grid.res,
                )
            )
    return tiles


def affine_transform(grid: Grid, tile: Tile) -> dict:
    """Earth Engine grid affine transform for one tile (north-up)."""
    return {
        "scaleX": grid.res,
        "shearX": 0,
        "translateX": tile.x0,
        "shearY": 0,
        "scaleY": -grid.res,
        "translateY": tile.y0,
    }


def assemble(grid: Grid, tiles: list[Tile], arrays: list[np.ndarray]) -> np.ndarray:
    """Place per-tile arrays of shape (bands, h, w) into one (bands, size, size) array."""
    if len(tiles) != len(arrays):
        raise ValueError("tiles and arrays must have the same length")
    bands = arrays[0].shape[0]
    out = np.empty((bands, grid.size_px, grid.size_px), dtype=arrays[0].dtype)
    filled = np.zeros((grid.size_px, grid.size_px), dtype=bool)
    for t, a in zip(tiles, arrays):
        if a.shape != (bands, t.height, t.width):
            raise ValueError(f"tile at ({t.row},{t.col}) has shape {a.shape}, expected {(bands, t.height, t.width)}")
        out[:, t.row:t.row + t.height, t.col:t.col + t.width] = a
        filled[t.row:t.row + t.height, t.col:t.col + t.width] = True
    if not filled.all():
        raise ValueError("tiles do not cover the whole grid")
    return out
