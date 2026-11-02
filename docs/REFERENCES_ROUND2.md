# Peer repos, second pass (1 Nov 2026)

Licences: repo A and repo B have none, so nothing is copied; reference implementation C is MIT.
`repo B-404/sas2pykg` could not be fetched (private or wrong name; it looks like the SAS-to-Python thesis, not this project).

## Adopted now (re-implemented)

| Idea | Source | Where |
|---|---|---|
| Rural land is the target; city-heavy data failed on a Black Forest test | repo B, 31 Oct | `region_balance` sampling (V11, V12): every region counts equally, because 10 big cities supply about 90% of the samples |
| Report a rural headline score next to the all-site score | repo B | `evaluate.py` rural group (already there); put it first in the report |
| Pixels with no radar are filled, flagged, and a month with more than 5% of them is warned about | repo A | `proto.uncovered_share`, warning in the app |
| Prototype highlights what changed since the previous month | repo A | "Highlight pixels that changed" on the Maps tab |
| Both hemispheres, climates chosen on purpose | reference implementation C | 5 southern/steppe/desert train tiles (earlier) and 4 South Africa test tiles |
| Write the decision rule before the result | repo A | see "Decision rules" below |
| Smooth the probabilities, not the hard classes | repo B | `change_eval.probability_smooth`, `evaluate.py --prob-smooth` (measured, not yet the prototype default) |
| Precision, recall, F1, kappa per class | repo B | `evaluate.py` writes `per_class.csv` and kappa in `metrics.json` |
| One final weights file | repo B | `scripts/package_model.py` writes `final_model.pt`; the prototype reads it |

## Decision rules (written before any V5-V12 result)

- A variant replaces V2 only if mean validation mIoU over 3 seeds is higher by at least 0.02 (repo B measured up to 0.02 between identical runs; ours is 0.05 for a single seed gap, so 0.02 on a 3-seed mean is the minimum I will call a gain).
- For V11/V12 the rural validation mIoU (val rural tiles only) must not fall, and the all-site mIoU must not fall by more than 0.02.
- Test regions are read once, after the choice.

## Not adopted

- Their class sets and label rules (7 classes, rangeland into grassland, consensus-only labels): ours are fixed; V10 covers the agreement filter.
- Terrain channels (DEM, slope): not Sentinel-1.
- repo C's from-scratch U-Net on 50 m data: V0 vs V1 already shows ImageNet is better for us.
- Coherence in the main model: see `COHERENCE.md` (research variant V13 on rural tiles).
