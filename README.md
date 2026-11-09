# Land cover maps from Sentinel-1 radar alone

A U-Net that maps six land cover classes from Sentinel-1 radar only, and a Streamlit prototype that turns a
latitude, longitude and start month into twelve monthly maps with class shares and change in percentage points.
Course project, Deep Learning, MSc Applied Data Science and AI, SRH Hochschule Heidelberg (solo).

## Links

| Item | Link |
|---|---|
| **Dataset** (Sentinel-1 + labels, 67 regions) | https://drive.google.com/file/d/1LddsfSM7CM0RR5r1ygh-hcIta0NG9KMU/view?usp=sharing |
| **Training code** — notebook | https://github.com/AnushkaSDBI/sentinel1-lulc/blob/main/notebooks/03_train_validate_test.ipynb |
| **Training code** — training script | https://github.com/AnushkaSDBI/sentinel1-lulc/blob/main/scripts/train_all.py |
| **Training code** — model, data and metrics modules | https://github.com/AnushkaSDBI/sentinel1-lulc/tree/main/src/lulc |
| **Model weights** (3 seeds) + `ood_reference.json` | https://drive.google.com/drive/folders/1D7t3IX6vt9x8Gu28-bjNOs3YvEnqOdu6?usp=sharing |
| **Prototype code** — app | https://github.com/AnushkaSDBI/sentinel1-lulc/blob/main/prototype/app.py |
| **Prototype code** — notebook | https://github.com/AnushkaSDBI/sentinel1-lulc/blob/main/notebooks/04_app.ipynb |
| Report (Word + PDF) | `report/` |
| Code repository | https://github.com/AnushkaSDBI/sentinel1-lulc |
| Demo video (5 urban/rural areas) | https://drive.google.com/file/d/1Z4eBlwcIl7GYEhNddxRd4zBj3B9PGovz/view?usp=sharing |
| All submission files in one folder | https://drive.google.com/drive/folders/162QX-zKiULvjKzCePR-PTbTka6m43c9g?usp=sharing |

## Results

The shipped model is a 3-seed ensemble trained on 2021-2024 (21 regions) and tested on 39 held-out regions in
2025 and 2026, so the test is held out in space **and** in time. Every number comes from `results/report_numbers.md`.

| Test | Reference map | Pooled mIoU (95% interval) |
|---|---|---|
| 2025, 39 unseen regions | Dynamic World 2025 | 0.540 (0.468-0.615) |
| 2026, 39 unseen regions | Dynamic World 2026 | 0.562 (0.493-0.634) |
| 2021, 9 regions with independent maps | Esri 2021 (independent) | 0.668 (0.524-0.716) |

Per-class IoU (2026): Built-up 0.845, Tree cover 0.599, Water 0.588, Cropland 0.554, Bare 0.416, Grass/shrub 0.367.

![Heidelberg 2021 against independent maps](docs/figures/heidelberg_2021_independent_check.png)

## Repository layout

| Path | Contents |
|---|---|
| `notebooks/` | Five notebooks (01-05), saved with outputs |
| `prototype/` | Streamlit app, its README and requirements |
| `src/lulc/` | Model, data, metrics, prototype helpers |
| `scripts/` | Earth Engine export, training, evaluation |
| `results/` | Every table and number behind the report |
| `weights/` | `final_model.pt` + `ood_reference.json` (download above) |
| `report/` | Report, Word and PDF |
| `docs/` | Figures, references, full pipeline (`docs/PIPELINE.md`) |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e .
pytest -q                                            # 178 tests
```

No account is needed to run the saved notebooks or an offline saved run. A **live** prototype run on a new place
needs a free Earth Engine project in the `EE_PROJECT` environment variable; no ID or token is stored in the repo.

## Run the prototype

```bash
streamlit run prototype/app.py
```

Enter a latitude, longitude and start month to get 12 monthly maps with class shares, month-to-month change,
an independent-map comparison, and a warning for regions unlike the training data. Runs in `saved_runs/` open
offline with "Saved run (offline)". See `prototype/README.md`.

## Reproduce

Each notebook is saved with its outputs and re-runs top to bottom with no data and no GPU: slow steps sit behind
a switch that is off by default, and tables and figures rebuild from `results/`.

| Notebook | What it does |
|---|---|
| `01_data_collection` | Regions, coverage checks, Earth Engine export |
| `02_data_preprocessing` | Preprocessing, class distribution, data exploration |
| `03_train_validate_test` | Training, test runs, every table and figure of the report |
| `04_app` | Prototype walkthrough and how its numbers are computed |
| `05_experiments` | Variants, the input fix, border experiments |

Full pipeline, commands and design decisions: [`docs/PIPELINE.md`](docs/PIPELINE.md).

## Limits

- Monthly maps have no ground truth (labels are annual); stability is measured instead. On land both annual maps
  call stable, the model flips class between months for 1.00% of pixels after smoothing, vs 6.79% for Dynamic World.
- 2025 and 2026 are scored against Dynamic World only; the 2021 Esri comparison is the firmer check.
- Weak classes: Grass/shrub, Bare, Cropland. Hardest regions are arid and cold (Great Basin, Iceland, Patagonia, Sahel).
- Borders are soft; five sharpening methods were tested and none passed the pre-set rule (see `notebooks/05_experiments.ipynb`).

## Licence

MIT, see `LICENSE`. References in `docs/REFERENCES.md`; no GPL code copied.
