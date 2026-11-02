# Coherence decision (updated 1 Nov 2026, evening)

Decision: coherence from ASF HyP3 is a research variant (V13) on the rural tiles only. It is not in the main model or the prototype.

## Where the data comes from
- Both sources involve ASF (NASA's Alaska Satellite Facility). The difference is the product, not the agency.
- Earth Engine product (our V4): a ready-made seasonal climatology, 92.7 m, Dec 2019 to Nov 2020. Wrong year, coarse. V4 scored 0.564 against 0.620 for V2.
- HyP3 (repo A): coherence computed on demand from Sentinel-1 burst pairs 12 days apart, matching year, 20 m pixel spacing (40 m resolution at 5x1 looks, per the HyP3 burst InSAR guide).
- repo A contains no coherence files, only code with no licence, and his sites are not ours. Nothing can be taken from it; we generate our own with `scripts/hyp3_coherence.py` (written from the asf_search and hyp3_sdk signatures, no code copied).

## What the script produces
Per region: `coherence_hyp3.npz` with 4 seasonal bands (winter, spring, summer, fall), 20 m, from 12-day pairs of the first pass in 2021. Trained as V13 and compared with V2 on the same tiles.

## Why only rural tiles
- Cost: about 60 pairs per 10 km tile (repo A's notes; the HyP3 guide does not state the credit cost). 15 rural tiles is about 900 jobs. The 24 km city squares cost several times more.
- The prototype must run any place and date, and coherence jobs queue for hours, so the prototype stays coherence-free.
- repo A's gain is built vs trees (AUC 0.83, 0.87). Crops vs grass did not replicate (0.63, 0.54). Our weakest class is grass/shrub.

## Order
1. `plan` (free), 2. `submit`, 3. `collect` every few hours, 4. train V13 and V2 on the same tiles with `--only`.
Stop on 5 Nov if the jobs are not finished. Then report V4 as a negative result, V13 as unfinished, and the idea as future work.
