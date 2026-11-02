# What we took from the two reference repos

Both repos solve the same SRH brief. Neither has a licence, so nothing is copied.
Every item below is a re-implementation of an idea, and the report should cite them.

- reference implementation A: GEE Sentinel-1 plus monthly Dynamic World, 16 tropical sites. Already used for the monthly-label idea.
- reference implementation B: global site set, WorldCover/Dynamic World agreement labels, U-Net with a 3-month window, extensive run log.

## Adopted

| Idea | Where | Notes |
|---|---|---|
| Incidence-angle mask (they use 31-45 deg) | `CompositeSettings.angle_range`, `export_dataset.py --angle-mask` | Off by default. The EE `angle` band is a coarse interpolated grid (catalog: 20 km pixels), so the mask edge is smooth. The 31-45 range is their choice, not a standard. |
| Mean in linear power, then dB | `CompositeSettings.method='mean_linear'` | Option. Default stays the median of dB so existing exports remain valid. |
| Settings stored per file, no silent mixing | `settings_conflict`, meta `composite` | Our own addition, because two recipes in one dataset would be a silent bug. |
| Independent maps on the same grid | `export_reference.py`, `ee_export.build_reference_image` | WorldCover 2021 plus Esri 10 m LULC 2021 (CC BY 4.0, trained on human-labelled pixels, so independent of Dynamic World). They used Esri as a sanity check too. |
| Patch index instead of copied patches | `patches.py` | Keep a patch if at least 40% has labels and 99% has S1 (their thresholds: at most 60% ignored, at least 99% valid). Last window is moved back to touch the tile edge. |
| Model card saved with the weights | `modelcard.py` | The prototype reads normalisation, channels and composite recipe from it. |
| sqrt median-frequency class weights | `train_math.class_weights` | |
| Warmup + cosine learning rate | `train_math.cosine_warmup_lr` | Same schedule for every variant, so variants are comparable. |
| Measure run-to-run noise, select by validation, not test | training plan | Their same-code reruns moved test mIoU by 0.02-0.06. Plan: 3 seeds for the final variants, report the spread. Choosing the best variant on test is leakage. |

Still to do in the training notebook: EMA of the weights (also cover BatchNorm buffers), Lovasz loss (smp `LovaszLoss` supports `ignore_index`, checked in the smp docs), GPU-side augmentation.

## Not adopted, and why

- **Agreement labels (WorldCover and Dynamic World must agree).** It cleans the labels but removes the hard pixels, so test scores look better than they are. It would also pull WorldCover into training, which breaks it as an independent validation map. We train on Dynamic World with a confidence cutoff and report scores on all valid pixels, on agreement pixels, and against WorldCover and Esri separately.
- **176 global sites.** Too much data for the time left. Our regions are 9 cities (2 out-of-distribution). Revisit after the class shares are known.
- **U-TAE or any new temporal architecture.** Their own notes call it 3-5 h of code with an unknown gain on this data. We use a 3-month window of monthly composites instead.
- **Yearly PCA or yearly statistics as inputs.** In their reruns the PCA variant helped trees and shrub but hurt cropland and wetland on unseen sites (site-specific overfitting). For us it also ties every monthly map to the whole year, which dampens real monthly change. If tried, report it as a variant.
- **Terrain channels (DEM, slope, HAND).** Not Sentinel-1. Allowed only as a clearly labelled extra ablation, never in the main model.
- **Their numbers.** Test mIoU 0.54-0.56 is on agreement-filtered pixels with 8 classes and 176 sites. It is not comparable with ours or with published scores.
- **Their `.env` file with the Earth Engine project ID is committed in the repo.** We keep the ID out of the code (`EE_PROJECT`).

## Checked against documentation

- EE `COPERNICUS/S1_GRD`: thermal noise removal, calibration and terrain correction only, then dB. No speckle filter. The `angle` band is the interpolated incidence angle. (EE catalog)
- Esri LULC asset: `projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS`, CC BY 4.0. The catalog page lists two class-value sets and does not say which the asset holds, so `detect_esri_scheme` checks the pixel values.
- smp: `LovaszLoss(mode, per_image, ignore_index, from_logits)` and `DiceLoss(..., ignore_index, ...)` both support `ignore_index`.

## Not verified

The literature numbers in repo B-404's research notes (U-TAE on PASTIS, EMA gains, label-product accuracies, and so on) are second-hand. Check the original paper before quoting any of them in the report.

## Report notes

- WorldCover and Esri are annual maps. They validate the annual level only. The monthly maps have no independent ground truth: the only monthly reference is Dynamic World, itself a noisy model output. Say this in the report.

## Coherence (proof of concept only)

`scripts/export_coherence.py` exports the global seasonal Sentinel-1 coherence (Kellndorfer et al. 2022,
asset `Earth_Big_Data/GLOBAL_SEASONAL_S1/V2019/COHERENCE`, CC BY 4.0, checked in the EE catalog).
Limits: 92.7 m pixels (we export 100 m, aligned to our grid), four seasonal medians from
Dec 2019 - Nov 2020, so it is a static prior per place and does not match the 2021 labels. It cannot give
monthly coherence. Use for one figure (coherence vs backscatter for built, bare, vegetation) and as a labelled
ablation, not in the main model. Whether coherence counts as "Sentinel-1 only" should be confirmed with the professor.

## Adopted from the peer repos (round 3, 2026-10-29)
Checked by reading their files, not only their READMEs.

| Source | Idea | Where in our code | Status |
| --- | --- | --- | --- |
| repo B-404 | Yearly statistic channels (VV min/max, VH std, VH spread, share of low-VV months) | `yearly.py`, V5 | implemented, not trained |
| repo B-404 | EMA weights, warmup plus cosine schedule | `train.py` | in use |
| repo B-404 | Lovasz loss, class-weight power | V6 | implemented, not trained |
| repo B-404 | dB shift, month dropout, window jitter, label smoothing, weight decay 0.05 | `torch_data.py`, `train.py`, V9 | implemented, not trained |
| repo B-404 | Consensus labels (two label maps must agree) | `apply_agree_filter`, V10 | implemented, not trained |
| repo A | Label agreement filter with WorldCover; dry-region training sites | V10, five new tiles | implemented, not trained |
| repo A | Scores per biome | `climate_groups`, `per_climate.csv` | implemented |
| repo A | Do not use spatial radar filters | compositing | kept (temporal median only) |
| reference implementation C | Speckle augmentation | V7 | implemented, not trained |
| reference implementation C | Warning for areas unlike the training data | `ood.py`, prototype | implemented, needs `ood_reference.json` |
| reference implementation C | Southern-hemisphere and arid tiles | `regions.yaml` | to export |

Not adopted, on purpose: repo B's terrain channels (elevation, slope, HAND: not Sentinel-1 input, would break the brief); PCA of the
yearly vector (calendar-tied, hurt the test score in repo B's own runs); repo A's hybrid yearly-fill labels (large change, uncertain gain);
InSAR coherence (our V4 lost, and repo A's own proof of concept helps only built-up versus trees); 50 m resolution and PC-RTC data
(restarts the pipeline); two-pass noise floor and per-scene standardisation (no time before 9 Nov).
