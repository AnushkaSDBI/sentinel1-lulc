"""Interferometric coherence for our regions from ASF HyP3 (Sentinel-1 burst InSAR), as an optional model input (V13).

    python scripts/hyp3_coherence.py plan    --data DATA --regions poltava black_forest   # searches only, costs nothing
    python scripts/hyp3_coherence.py submit  --data DATA --regions ...                    # spends 1 credit per pair
    python scripts/hyp3_coherence.py collect --data DATA --regions ...                    # run again until it says done

Needs a free NASA Earthdata login in ~/.netrc (machine urs.earthdata.nasa.gov) or the EARTHDATA_USERNAME and
EARTHDATA_PASSWORD environment variables. The region's Sentinel-1 files must already be exported (the orbit direction is read from them).

What it does: for every burst over the region, take passes exactly 12 days apart (Sentinel-1A only), ask HyP3 for the
burst coherence (INSAR_ISCE_BURST, 5x1 looks: 40 m resolution on a 20 m pixel spacing, per the HyP3 burst InSAR guide), download it, resample it to a 20 m grid on our region square,
average the pairs per season of the first pass (winter DJF, spring MAM, summer JJA, fall SON), delete the download,
and write <data>/<region>/coherence_hyp3.npz with `coherence` (4, h/2, w/2) float16 and `factor` 2. Safe to rerun: jobs are
found by name, so nothing is paid twice, and finished regions are skipped.

Written from the asf_search 14.1 and hyp3_sdk 7.7 signatures (checked by importing them). No third-party code was copied.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

from lulc.config import load_config  # noqa: E402
from lulc.grid import grid_bounds_lonlat, grid_for_region  # noqa: E402

SEASONS = ("winter", "spring", "summer", "fall")
FACTOR = 2                       # 20 m coherence pixels on our 10 m grid


def season_index(d: str) -> int:
    """'2021-04-15' -> 1 (meteorological seasons: DJF 0, MAM 1, JJA 2, SON 3)."""
    m = int(d[5:7])
    return {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}[m]


def twelve_day_pairs(acq: dict, year: int) -> list[tuple[str, str, str, str]]:
    """{burst: {(date, scene)}} -> [(burst, first date, first scene, second scene)].

    Consecutive passes of one burst exactly 12 days apart, the first pass in `year`. A longer gap decorrelates more and is skipped.
    """
    out = []
    for burst, a in sorted(acq.items()):
        a = sorted(a)
        for (d1, s1), (d2, s2) in zip(a, a[1:]):
            if (datetime.fromisoformat(d2) - datetime.fromisoformat(d1)).days == 12 and d1.startswith(str(year)):
                out.append((burst, d1, s1, s2))
    return out


def seasonal_mean(maps: list[tuple[int, np.ndarray]]) -> np.ndarray:
    """[(season index, 2D coherence with NaN)] -> (4, H, W) mean per season, NaN where a season has no pair."""
    h, w = maps[0][1].shape
    out = np.full((4, h, w), np.nan, np.float32)
    for s in range(4):
        got = [m for k, m in maps if k == s]
        if got:
            with np.errstate(all="ignore"):
                out[s] = np.nanmean(np.stack(got), 0)
    return out


def scene_month(scene: str) -> str:
    """'S1_325416_IW3_20250626T031920_VV_5876-BURST' -> '2025-06-26'"""
    d = scene.split("_")[3]
    return f"{d[:4]}-{d[4:6]}-{d[6:8]}"


def region_direction(data: Path, key: str) -> str:
    """Orbit direction used in the region's Sentinel-1 export, from the meta of its first month file."""
    for f in sorted((data / key).glob("20*-*.npz")):
        with np.load(f) as z:
            d = json.loads(str(z["meta"])).get("direction")
        if d:
            return str(d).upper()
    raise SystemExit(f"{key}: no month file with an orbit direction in {data / key}; export the radar first")


def polygon_wkt(region) -> str:
    w, s, e, n = grid_bounds_lonlat(grid_for_region(region))
    return f"POLYGON(({w} {s},{e} {s},{e} {n},{w} {n},{w} {s}))"


def need(*mods):
    """Stop with the install line instead of a traceback when an optional package is missing."""
    import importlib.util

    missing = [m for m in mods if importlib.util.find_spec(m) is None]
    if missing:
        raise SystemExit(f"missing packages: {missing}. Run:  pip install asf_search hyp3_sdk rasterio")


def find_pairs(region, year: int, direction: str):
    need("asf_search")
    import asf_search as asf

    for attempt in range(4):                 # the search service turns away bursts of requests now and then
        try:
            found = asf.search(dataset=asf.DATASET.SLC_BURST, intersectsWith=polygon_wkt(region), start=f"{year}-01-01",
                               end=f"{year + 1}-01-13", polarization=asf.POLARIZATION.VV, flightDirection=direction)
            break
        except Exception:
            if attempt == 3:
                raise
            time.sleep(30)
    acq = collections.defaultdict(set)
    for r in found:
        p = r.properties
        if p.get("platform") == "SENTINEL-1A" and str(p.get("sceneName", "")).endswith("-BURST"):
            acq[p["burst"]["fullBurstID"]].add((p["startTime"][:10], p["sceneName"]))
    return twelve_day_pairs(acq, year)


def job_name(key: str, year: int) -> str:
    return f"c_{key[:12]}_{year}"            # short on purpose; the HyP3 docs give no name length limit, so stay small


def coherence_on_grid(zip_path: str, region) -> np.ndarray:
    """One product's coherence on the region's 20 m grid (NaN where the burst does not cover the square)."""
    import rasterio
    from rasterio.warp import Resampling, reproject

    g = grid_for_region(region)
    n = g.size_px // FACTOR
    out = np.full((n, n), np.nan, np.float32)
    with zipfile.ZipFile(zip_path) as z:
        inner = next(x for x in z.namelist() if x.endswith("_corr.tif"))
    with rasterio.open(f"zip://{os.path.abspath(zip_path)}!/{inner}") as src:
        reproject(rasterio.band(src, 1), out, src_nodata=src.nodata if src.nodata is not None else 0,
                  dst_transform=rasterio.Affine(g.res * FACTOR, 0, g.x0, 0, -g.res * FACTOR, g.y0), dst_crs=g.crs,
                  dst_nodata=np.nan, resampling=Resampling.bilinear)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["plan", "submit", "collect"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--regions", nargs="+", required=True)
    ap.add_argument("--year", type=int, default=2021)
    ap.add_argument("--config", default=str(ROOT / "configs" / "regions.yaml"))
    ap.add_argument("--raw", default=os.environ.get("COHERENCE_RAW", "coherence_raw"), help="scratch folder for downloads")
    ap.add_argument("--threads", type=int, default=4, help="parallel downloads in collect")
    ap.add_argument("--max-pairs", type=int, default=400, help="refuse to submit more pairs than this in one call")
    a = ap.parse_args(argv)
    need("asf_search", "hyp3_sdk", "rasterio")
    data = Path(a.data)
    cfg_regions = {r.key: r for r in load_config(a.config)[1]}
    regions = [cfg_regions[k] for k in a.regions]
    todo = [r for r in regions if not (data / r.key / "coherence_hyp3.npz").exists()]
    print(f"{len(regions) - len(todo)} region(s) already done, {len(todo)} to do")

    if a.action in ("plan", "submit"):
        plan = {r.key: find_pairs(r, a.year, region_direction(data, r.key)) for r in todo}
        for k, p in plan.items():
            print(f"{k:24s} {len({b for b, *_ in p}):2d} bursts {len(p):4d} pairs")
        total = sum(len(p) for p in plan.values())
        print(f"total {total} pairs. estimate of 1 credit per pair (not stated in the HyP3 guide: check your credits first).")
        if a.action == "plan":
            return 0
        if total > a.max_pairs:
            raise SystemExit(f"{total} pairs is more than --max-pairs {a.max_pairs}; submit fewer regions or raise the limit")
        import hyp3_sdk

        hyp3 = hyp3_sdk.HyP3()
        print("credits left:", hyp3.check_credits())
        for r in todo:
            have = {tuple(j.job_parameters["granules"]) for j in hyp3.find_jobs(name=job_name(r.key, a.year)) if j.job_parameters}
            new = [(s1, s2) for _, _, s1, s2 in plan[r.key] if (s1, s2) not in have]
            print(f"{r.key}: submitting {len(new)} pairs ({len(have)} already at HyP3)")
            for s1, s2 in new:
                hyp3.submit_insar_isce_burst_job(s1, s2, name=job_name(r.key, a.year), looks="5x1")
        return 0

    import hyp3_sdk

    hyp3 = hyp3_sdk.HyP3()
    Path(a.raw).mkdir(parents=True, exist_ok=True)
    for r in todo:
        batch = hyp3.find_jobs(name=job_name(r.key, a.year))
        if not len(batch):
            print(f"{r.key}: nothing submitted yet")
            continue
        if not batch.complete():
            print(f"{r.key}: still running, {dict(collections.Counter(j.status_code for j in batch))}")
            continue
        def one(j, r=r):
            path = os.path.join(a.raw, j.files[0]["filename"])
            if not os.path.exists(path):
                j.download_files(a.raw)
            try:
                return season_index(scene_month(j.job_parameters["granules"][0])), coherence_on_grid(path, r)
            except Exception as e:          # a corrupt or half-downloaded zip: drop this one product, keep the rest;
                print(f"  skipped {os.path.basename(path)}: {type(e).__name__}", flush=True)   # deleted below, so it re-downloads next run
                return None
            finally:
                if os.path.exists(path):
                    os.remove(path)

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(a.threads) as pool:        # the downloads dominate the time, so run several at once
            maps = [m for m in pool.map(one, [j for j in batch if j.succeeded()]) if m is not None]
        if not maps:
            print(f"{r.key}: no job succeeded")
            continue
        coh = seasonal_mean(maps)
        np.savez_compressed(data / r.key / "coherence_hyp3.npz", coherence=coh.astype(np.float16), factor=np.array(FACTOR))
        print(f"{r.key}: done, {len(maps)} of {len(batch)} jobs, covered {np.isfinite(coh).mean():.0%} of the 4 seasonal bands")
    return 0


if __name__ == "__main__":
    sys.exit(main())
