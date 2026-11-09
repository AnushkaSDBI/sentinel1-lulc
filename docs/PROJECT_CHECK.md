# Project check, 2026-11-06 (3 days to the 9 Nov deadline)

## 1. Requirements against what we have

Status: done = built and tested offline, partly = exists but needs data or work, open = not started.

| Requirement | Status | Where / what is missing |
|---|---|---|
| Sentinel-1 is the only model input | done | Labels, WorldCover, Esri and coherence are never inputs. Coherence is a labelled ablation and needs the professor's OK. |
| Dataset: source, regions, period, classes | done | EE S1_GRD + Dynamic World, 2021-2026 (`runs_multiyear`), 6 classes. |
| Dataset: class distribution | partly | In `notebooks/02_data_preprocessing.ipynb`. Not yet copied into the report document. |
| Calibration, dB, terrain correction | done (by EE) | EE does thermal noise removal, calibration, geometric terrain correction, dB. No radiometric terrain flattening: say so, hilly regions suffer. |
| Speckle filtering | partly | Monthly temporal median is our filter. The slide asks for it explicitly, so run one spatial-filter variant and report the number (repo A: it hurt). |
| Normalisation, patching | done | `patches.py` (train-only stats, index of valid patches). |
| Label matching | done | One grid per region, grid check against EE, overlay QA. Show the QA figure in the report. |
| Augmentation | open | Flips and 90-degree rotations only. Brightness shift optional. |
| Splits by city | done | 16 train, 6 val, 7 test, 2 out-of-distribution, 1 case study. Overlap check in code. |
| Model: architecture, setup, reasoning | done | Notebook `03_train_validate_test.ipynb`, run through V12; border variants V13-V17 in `05_experiments.ipynb`. |
| Test OA, mIoU, per-class IoU and F1, training curves, variant comparison | done | `03_train_validate_test.ipynb` and `05_experiments.ipynb`, with real outputs. 3-seed ensemble shipped. |
| Prototype: lat/lon + start month -> 12 maps | done | `prototype/app.py`, tested on real Heidelberg data. |
| Prototype: monthly compositing, how changes are computed | done | `build_image` reused for inference; `stats.py` for percentages and change in pp. |
| Case study against an independent map | done | Heidelberg 2021, `docs/figures/heidelberg_2021_independent_check.png`. |
| Prototype README, requirements, accounts or API keys | done | README has the prototype steps; EE account is the only account. |
| Dataset delivery (link if large) | open | Link not made yet. |
| Notebooks with executed outputs | done | All five notebooks in `notebooks/`, saved with real outputs. |
| Trained weights | done | `weights/final_model.pt` (3-seed ensemble) and `ood_reference.json`. |
| Screen recording of a full run on an urban area | open | Heidelberg, after the prototype is final. |
| Report: unlike regions | open | Material exists (Great Basin, Iceland, Patagonia, OOD regions); not yet in the report document. |
| Report: SAR class confusion | open | Not yet in the report document. |
| Report: seasonal effects mimicking change | open | Not yet in the report document. |
| Report: classification errors inflating change percentages | open | Not yet in the report document. |

### Differences from the older project doc ("defaults I am proposing")
- Data access: Earth Engine, not Planetary Computer (labels and server-side compositing live in EE).
- Labels: Dynamic World monthly plus WorldCover and Esri annual. The monthly maps still have no independent ground truth, because Dynamic World is itself a model. Say this in the report.
- Swin-Unet dropped. U-Net variants only, because of the 14 days.
- Input: VV, VH, and (optional) 3-month window. The ratio is a fixed function of VV and VH in dB.

### Risks, most serious first
1. Export time and Earth Engine quota are unmeasured. 384 region-months. Check the quota page after the first batch.
2. Prototype is not started and the screen recording depends on it.
3. Dataset size for submission.
4. Hilly regions: no radiometric terrain flattening.
5. Esri class scheme unconfirmed.

### Schedule (updated 2026-11-06)
| Dates | Work | Status |
|---|---|---|
| 26 Oct - 5 Nov | Export, training through V12, border experiments, prototype, Heidelberg case study | done |
| 6-8 Nov | Report content, README links, screen recording, dataset/weights upload | in progress |
| 9 Nov | Submit | pending |

## 2. What we take from the two repos

Evidence: repo A's numbers were recomputed from his result files. repo B's numbers are as reported in his README and not rerun.

| Item | From | Evidence | What it improves for us |
|---|---|---|---|
| Diverse places, including dry and bare rural tiles | repo A exp1, repo B | Strong. repo A: test mIoU 0.21-0.25 -> 0.335, water IoU 0.07-0.10 -> 0.42 | Rare-class IoU, unseen regions |
| Raw radar, no spatial speckle filter | repo A exp1 | Moderate. One seed each, effect much larger than seed noise (test mIoU -0.04) | Avoids a harmful step |
| Label cleaning for training | repo A exp3, repo B | Good. +0.04 mIoU, both seeds, validation and test agree | Crops, low vegetation, flooded vegetation. We adapt it: confidence cutoff, optional WorldCover agreement, Esri as the independent map |
| Seeds, rules written before results, validation-only selection | repo A | Method, confirmed by his own review finding test contamination | Credible comparisons |
| Per-class validation scores | repo A | Method | Single-class test scores on 3 sites swing by up to 0.3 |
| 3-month window | repo B | Moderate. His runs: single month 0.507, three months 0.559 mIoU, one seed | Cropland, wetland, seasonal classes |
| Pretrained vs scratch encoder | both | Conflicting. repo A: ImageNet unstable. repo B: pretrained | Decide with 3 seeds |
| Incidence-angle mask | repo B | Untested | Swath-edge noise |
| Mean-linear composite | repo B | Untested | Possibly less speckle |
| Patch index with validity thresholds | repo B | Engineering | Disk size, no empty patches |
| Model card with the weights | repo B | Engineering | No train/inference drift in the prototype |
| EMA, warmup-cosine schedule, Lovasz loss | repo B notes | Literature numbers unverified. Lovasz `ignore_index` support checked | Smoother validation curve, comparable variants |
| sqrt median-frequency class weights | repo B | Standard | Imbalance |
| Esri 2021 as a second independent map | repo B | Asset exists, CC BY 4.0. Class scheme to confirm | Independent validation |
| uint8 dB storage | repo B | Not adopted yet | Dataset size |
| Rejected: U-Net++, oversampling, copy-paste, yearly context channels, terrain channels, U-TAE | both | repo A: no gain or harm | Saves time |

## 3. Where we can get an edge

Ranked by value for the grade against cost in days.

| # | Idea | Why it stands out | Cost | Risk |
|---|---|---|---|---|
| 1 | **False-change test.** On pixels where WorldCover and Esri agree and the class is stable, measure the change our maps report between months and in total. Then apply a 3-month majority vote and measure again. | It is the project's actual output, and the brief's two critical-discussion points (seasonal effects, errors inflating change). Few will measure it. | 1-2 days | Low |
| 2 | **Out-of-year test.** Export 2024 for the 3-4 test cities and score the 2021-trained model against 2024 Dynamic World. | The prototype will be run on recent months, where Sentinel-1 density differs (one satellite in 2022-2024). This shows whether it holds up. | 0.5 day plus export | Low |
| 3 | **Uncertainty.** Ensemble the 3 seeds we train anyway; show per-pixel confidence and give each change percentage a spread. | Change numbers with error bars are much more honest than point values. | 1 day | Low |
| 4 | **Rigorous evaluation.** Metrics on all pixels, on agreement pixels, and against WorldCover and Esri; bootstrap intervals by city. | Most projects report one number against one label map. | 0.5 day | Low |
| 5 | **Coherence (professor's advice).** Seasonal VV coherence as an ablation variant and one figure of how it separates built, bare and vegetation. | It was recommended. Honest expectation: small gain, because it is 92.7 m, 2020, four seasons. The figure is the value. | 1 day | Medium |
| 6 | **Scene-subsampling augmentation.** Train on composites built from 4-6 random scenes. | Makes the model robust to sparse years, which pairs with #2. | 0.5 day | Low |
| 7 | **Radiometric terrain flattening check.** Planetary Computer RTC on one hilly region. | Turns a limitation into a measurement. | 1.5 days | Medium (coverage unverified) |
| 8 | **True per-month coherence for one city via HyP3.** | Strongest coherence demonstration. | Unknown queue and quota | High. Skip unless time is left. |

Do #1-#4 first. They are cheap and all improve the credibility of the monthly-change output.
