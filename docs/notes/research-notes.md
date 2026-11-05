# Research notes

Running record of what was tried, what helped, and what did not. Plain notes, written as the work happened.

## Design decisions
- Sentinel-1 only (VV, VH in dB from Earth Engine). No optical, no terrain channels in the main model.
- Monthly composite: per-pixel median in dB of the dominant orbit direction. No spatial speckle filter (the temporal median is the filter).
- 6 classes: Tree cover, Grass/shrub, Cropland, Built-up, Bare, Water. Flooded vegetation and snow dropped (too rare, ignore index 255).
- Labels: monthly Dynamic World (argmax of mean probabilities). Independent check maps: ESA WorldCover 2021 and Esri 2021, used for validation only.
- Split by region, never by patch. Every validation/test region is a different place from training. Leakage checked (region boxes do not overlap).
- Model: U-Net, ResNet-34 encoder, ImageNet weights (beat scratch: V0 0.466 vs V1 0.597). 3-month input window.
- Report per-class IoU and mIoU, not just accuracy. Select variants on validation only; test read once at the end.

## What helped
- ImageNet encoder over scratch: +0.13 mIoU.
- 3-month window over single month: +0.02.
- Yearly statistic channels (per-pixel min/max/std/range over the year): the main lever. Validation 0.597 -> 0.639, and Grass/shrub IoU 0.151 -> ~0.25. This is the real improvement.
- Region-balanced sampling (every region equal weight): no rural-mIoU gain, small overall gain through steadier Water and Bare. Kept in the chosen model but not oversold.
- 3-month majority smoothing: cuts false month-to-month change by about a third.

## What did not help
- Loss changes (higher class-weight power + Lovasz): hurt (0.609 vs 0.620 baseline).
- Regularisation bundle (dB shift, month dropout, window jitter, label smoothing, heavy weight decay): hurt (0.604).
- Speckle augmentation: neutral (0.633).
- Stacking everything (stats + loss + speckle): worse than stats alone (0.628 vs 0.639). More is not better.
- Confidence cutoff on labels (keep only >=50): slightly lower (0.609 vs 0.620).
- Earth Engine coherence layer (92.7 m, wrong year): 0.564, well below baseline. Wrong data for the question.

## Coherence (research variant, not shipped)
- HyP3 burst InSAR coherence (20 m, matching year, 4 seasonal bands) as extra input: +0.025 validation mIoU on rural tiles, within seed noise.
- Not in the final model or the live prototype: it needs on-demand HyP3 jobs (a NASA login, credits, hours per area), which breaks the any-place/any-date prototype.
- Wired into the prototype as a demo on prepared tiles only, behind a warning. Reported as evidence plus future work.

## Ideas considered and parked (future work)
- Coherence trained with a "present" flag and 30% dropout, so one model works with or without coherence at prediction time, and measuring the coherence-trained model with coherence zeroed. Would let coherence help the radar-only path. Parked: our coherence gain is within noise and retraining is out of scope this late.
- PCA over the 12 monthly values per pixel (mean brightness, seasonal swing) as a cheaper stand-in for the yearly statistics. The statistics already capture this.
- Biome tag per site for reporting and site selection. The per-climate CSV covers the reporting half.

## Known limitations (for the report)
- The monthly maps have no ground truth: labels are annual, and Dynamic World is itself a model.
- Grass/shrub stays weak (~0.25). Radar cannot separate grass, shrub and crops well.
- Weak on arid shrubland (Great Basin 0.08), and on biomes unlike the European-heavy training set (South Africa group 0.29, never trained there).
- No radiometric terrain flattening: hilly areas carry slope-dependent errors.
- Sentinel-1B stopped Dec 2021, so months from 2022 have about half the scenes. The model was trained on 2021.
- Seasonal radar effects (snow, wet soil, harvest) look like change. The false-change noise floor quantifies this.
