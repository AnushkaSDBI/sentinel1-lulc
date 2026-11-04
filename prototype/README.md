# Prototype: monthly land-cover maps from Sentinel-1

Enter a latitude, longitude and start month. The app downloads 12 monthly Sentinel-1 composites from Earth Engine (fewer in the current year: it stops at the last finished month, and needs at least 4),
runs the trained U-Net (seed ensemble), and shows one map per month, the share of each class per month, the month-to-month and
total change in percentage points (with a noise floor and model-spread bands), and a check against WorldCover and Esri.

## Setup
```
pip install -r requirements.txt        # from the repository root (Python 3.11+)
```
For a GPU install the CUDA build of PyTorch first (pytorch.org). The CPU works: a 5.12 km area takes a few minutes.

**Weights.** Copy the winning run from training into `weights/`, one sub-folder per seed:
```
weights/seed0/best.pt  weights/seed0/model_card.json
weights/seed1/...      weights/seed2/...
```
(the folders `runs/<chosen variant>/seed*` of the training run). Also copy `runs/eval_<variant>/ood_reference.json` into `weights/`: without it the
app shows no area warning. Models trained with the yearly-statistic channels (V5, V8) are detected from the model card. Or set `LULC_WEIGHTS=/path/to/folder`.

**Earth Engine (the only account needed).** A Google account with Earth Engine access and a Cloud project registered for it
(earthengine.google.com, non-commercial use is free). Log in once: `earthengine authenticate`. Enter the project ID in the sidebar,
or set `EE_PROJECT`.

## Run
```
streamlit run prototype/app.py
```
Open http://localhost:8501, choose the place and the start month, press Run. Downloaded months are cached in `.cache_lulc/`,
so a second run of the same place is fast.

Offline test without Earth Engine: choose "Local export" and give the export folder (`data/raw`) and a region key.

## What the interface offers
- Maps: class layer over a satellite photo or street map, opacity slider, month slider, click a pixel to see its class in every month.
- Place: type latitude and longitude, or open "Pick the place on a map" in the sidebar and click.
- Class shares: stacked area chart plus per-class lines with the spread between seeds.
- Change: month-to-month table with noise-floor flags, and a map of pixels whose class differs between the first and the last month.
- Check against independent maps (2021): the model, the Dynamic World ceiling and a majority-class baseline against WorldCover, Esri and
  the pixels where both agree, with the share of the area those pixels cover, and a confusion matrix against Esri.
- Extra packages for the map and charts: `folium`, `streamlit-folium`, `plotly` (in `requirements.txt`). Basemap tiles need internet.

## How the change is computed
Class shares use only pixels valid in all months of the run. Change = difference of shares in percentage points, month to month and first to last.
Optional smoothing replaces each month's class by the majority over three consecutive months.
The noise floor (total class churn per month on land that did not change) comes from our false-change test on 9 test regions and is
set in `src/lulc/proto.py` (`NOISE_FLOOR`). Update it if you retrain.

## Limits
The monthly maps have no ground truth (labels are annual and come from another model). Grass/shrub is weak. Hilly areas have no
radiometric terrain correction. After December 2021 there is one Sentinel-1 satellite, so fewer scenes per month than in training.
The reference check uses 2021 maps only.
