# Refinements added 4 Nov 2026

Every step below is optional. Nothing here changes the shipped model.

| Step | Command (run from the repo root, in the notebook with `%run`) | Decision |
|---|---|---|
| Input ablation table | `scripts/ablate_inputs.py --data data/raw --runs runs_multiyear --variant V12_window3_balanced_stats --splits val --year 2024` | Explains the model; selects nothing. Paste `ablation.md` into the report. |
| Convolutional CRF | `scripts/crf_eval.py --data data/raw --runs runs_multiyear --variant V12_window3_balanced_stats --splits val --year 2024` | Use only if boundary F1 +0.02, mIoU not down more than 0.005, no class down more than 0.05. Then confirm once on test. |
| Orbit direction check | `scripts/check_orbit.py --data data/raw --data data/raw_2025 --data data/raw_2026` | Re-export a region if it shows more than one direction. |
| Yearly Dynamic World label | `scripts/export_yearly_labels.py --project ID --out data/raw --years 2021 2022 2023 2024` | Needed only for `lulc.yearly_labels` (fill holes, agreement scoring). Not used by training. |
| Agreement-pixel score | `scripts/evaluate.py ...` now also prints the row `dw_wc_agree` (Dynamic World scored only where WorldCover agrees; 2021 only). | Free; shows how much error is label noise. |
| Graph head | `scripts/graph_head_eval.py --data data/raw --runs runs_multiyear --variant V12_window3_balanced_stats --train-splits val --eval-splits test test_ood --year 2024 --eval-year 2025 --eval-data data/raw_2025` | Same rule as the CRF. Train on validation regions, judge on test. Experimental. |
| Stable control city | Prototype, sidebar, "Quick demo locations": "Control: Prague" (held out; Munich is a train region, so it would understate false change) | Check the square on satellite view before recording. |

Sharper borders, in order, all in `notebooks/sharper_borders.ipynb` and all judged by the same rule (`sharpness.passes_border_rule`):

| Step | Command | Cost |
|---|---|---|
| Test-time augmentation | `scripts/sharpen_eval.py --data data/raw --runs runs_multiyear --variant V12_window3_balanced_stats --year 2024 --settings tta` | No retrain; inference 8 times slower |
| Convolutional CRF | as in the table above | No retrain |
| Gaussian tile stitching | same as the first row with `--settings gaussian` | No retrain |
| Graph head | as in the table above | Trains a small head |
| Full-resolution branch (V17) | `scripts/train_all.py --variants V17_window3_fullres`, then `sharpen_eval.py ... --settings none --compare-variant V17_window3_fullres` | Full retrain, 3 seeds |

Needs `pip install scikit-image` for the graph head (already in requirements.txt).

CRF idea: Teichmann and Cipolla 2018, "Convolutional CRFs for Semantic Segmentation", re-implemented from the paper's idea. No code copied.
