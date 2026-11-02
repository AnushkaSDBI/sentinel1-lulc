# Experiments

Every variant shares the defaults in `src/lulc/train.py`; `configs/experiments.yaml` lists only the overrides.
Runs are written to `runs/<variant>/seed{0,1,2}/` (weights, model card, summary.json) and evaluated into
`runs/eval_<variant>/` (metrics.json, per_class.csv, per_city.csv, per_climate.csv, by_month.csv, change.csv).
Variants are selected on validation only; the test set is read once at the end.

## Variants and validation mIoU (3 seeds)

| Variant | What it tests | Val mIoU |
|---|---|---|
| V0 scratch (1 month) | encoder from scratch | 0.466 |
| V1 ImageNet (1 month) | pretrained encoder (baseline) | 0.597 |
| V2 window-3 | 3-month input window | 0.620 |
| V3 window-3 + conf cutoff | keep labels with confidence >= 50 | 0.609 |
| V4 window-3 + EE coherence | Earth Engine coherence layer | 0.564 |
| V5 window-3 + yearly stats | per-pixel yearly statistics | 0.639 |
| V6 window-3 + loss | class-weight power 0.75 + Lovasz | 0.609 |
| V7 window-3 + speckle | speckle augmentation | 0.633 |
| V8 window-3 all three | stats + loss + speckle | 0.628 |
| V9 window-3 regularised | dB shift, month dropout, jitter, smoothing, weight decay | 0.604 |
| V11 window-3 balanced | region-balanced sampling | 0.628 |
| **V12 window-3 balanced + stats (chosen)** | balancing + yearly stats | **0.646** |
| V13 window-3 + HyP3 coherence | burst-InSAR coherence input (research) | see docs/COHERENCE.md |

## How to reproduce
```
# train (3 seeds each); V2+ take the encoder from whichever of V0/V1 scored higher
python scripts/train_all.py --data $OUT --out $RUNS --variants V0_scratch_1m V1_imagenet_1m
python scripts/train_all.py --data $OUT --out $RUNS --variants V2_window3 V5_window3_stats V11_window3_balanced V12_window3_balanced_stats
# evaluate and read the rural rows
python scripts/evaluate.py --data $OUT --runs $RUNS --variant V12_window3_balanced_stats
# package the chosen model
python scripts/package_model.py --runs $RUNS --variant V12_window3_balanced_stats --out weights
```

See `docs/research-notes.md` for what each result means and `claude/results-log.md` (in the project) for the full numbers.
