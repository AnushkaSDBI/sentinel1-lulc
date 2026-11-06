# Course requirements: Sentinel-1 LULC project

Source: `Documentation/requirements_info.txt` and `lecture_notes.txt` (slides 9-10 of the course introduction).
The task part matches the text Anu pasted. The report part was not pasted by Anu: check it against the slides.
Deadline: 9 Nov 2026. Solo project.

## Task
Collect data and train a deep learning model for Land Use and Land Cover (LULC) semantic segmentation that uses Sentinel-1 data as its only input.

## Prototype
- Input: latitude/longitude of the location and a start month.
- Processing: run the trained model to produce 12 monthly LULC maps.
- Calculation: for each month, the percentage of the area covered by each class. Also the month-to-month change and the total change from the first to the last month, both in percentage points.
- Visualization: LULC map of the selected location over 12 months.

## Submit
1. The dataset: a download link if too big.
2. The notebook(s): must display the execution results and visualizations of training performance.
3. The prototype: source code, a README (setup, dependencies, required accounts or API keys) and a requirements file.
4. The trained model weights.
5. A short screen recording of a full run for at least one urban area of your choice.

## Report must cover
- **Dataset and preprocessing:** data source, regions, time period, classes and class distribution. All preprocessing: calibration, dB conversion, speckle filtering, terrain correction; normalization and patching; how labels were matched to the imagery; augmentation and data splits.
- **Model:** architecture and training setup, with the reasoning behind the design choices.
- **Experimental results:** overall accuracy (OA), mean IoU (mIoU) and per-class IoU/F1 on the test set; training curves; a comparison of model variants if more than one was tested.
- **Prototype:** system architecture; data retrieval and monthly compositing; how class percentages and changes are calculated; a case study of the chosen location, sanity-checked against an independent LULC map not used for training.
- **Critical discussion:** limitations of the model and the prototype, including performance on regions unlike the training data, SAR-specific confusion between classes, seasonal effects that can look like real land-use change, and how classification errors inflate the calculated change percentages.

## Lecture notes (extra points from the course)
- Address data imbalance, and make sure accuracy is high across all classes.
- Use sparse categorical cross-entropy (in PyTorch: cross-entropy with integer labels, `ignore_index` for unlabelled pixels).
- Use early stopping to avoid overfitting.

## Status against the list (2026-10-29)
| Item | Status |
|---|---|
| Dataset | Exported to Drive (32 regions, 2021). Download link not made yet. Five new southern/arid tiles not exported yet. |
| Notebook(s) with outputs | The five notebooks in `notebooks/`, saved with outputs; training curves and tables are in `03_train_validate_test.ipynb`. |
| Prototype source, README, requirements | Done and tested on real Heidelberg data. README and `requirements.txt` in the repo. |
| Trained weights | V2 seeds exist. Copy the final variant to `weights/`, with `ood_reference.json`. |
| Screen recording, urban area | Not done. Plan: Heidelberg, run locally. |
| Class distribution table | Check: needs a per-split table of class shares in the report. |
| Preprocessing text | S1_GRD in Earth Engine: calibrated, geometric terrain correction, dB; no radiometric terrain flattening, no spatial speckle filter (temporal median per month). State exactly this. |
| OA, mIoU, per-class IoU/F1 | Done in `evaluate.py` (bootstrap by city). |
| Variant comparison | V0-V4 done; V5-V8 pending. |
| Independent map | WorldCover and Esri in the prototype and `evaluate.py`. Heidelberg 2021 screenshot still needed. |
| Critical discussion | Material exists: unseen regions (Great Basin), grass/shrub confusion, December 2021 anomaly, false change. Write it with numbers. |
| Imbalance | Class-weighted CE + Dice (+ Lovasz in V6). |
| Early stopping | Done: patience 5 epochs on validation mIoU of the EMA weights, best checkpoint kept (max 12 epochs). |
