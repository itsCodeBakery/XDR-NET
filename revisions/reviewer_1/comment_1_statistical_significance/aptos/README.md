# Reviewer 1, Comment 1 — APTOS Statistical Significance Workflow

This directory contains the reproducible APTOS 2019 workflow prepared to address the reviewer request for confidence intervals and statistical significance testing.

## Files

- `01_aptos_data_pipeline.py` — dataset verification, fixed five-fold stratification, fundus preprocessing, CLAHE, augmentation, effective-number weighting, PyTorch datasets/loaders, and quality-control figures.
- `requirements.txt` — Python dependencies used by the workflow.

## Kaggle input paths

```text
/kaggle/input/competitions/aptos2019-blindness-detection/train.csv
/kaggle/input/competitions/aptos2019-blindness-detection/test.csv
/kaggle/input/competitions/aptos2019-blindness-detection/train_images
/kaggle/input/competitions/aptos2019-blindness-detection/test_images
```

## Output directory

```text
/kaggle/working/xdrnet_aptos
```

The script saves fixed fold assignments, class distributions, effective-number class weights, preprocessing previews, and class-distribution figures. All figure titles and data labels are rendered in bold.

## Important methodological note

APTOS 2019 does not provide patient identifiers. Therefore, the workflow uses stratified image-level folds. The manuscript should not describe these folds as patient-level splits unless an external patient-linkage file is available.

## Run

```bash
python 01_aptos_data_pipeline.py
```

The same `aptos_5fold_assignments.csv` file must be used for XDR-Net and every baseline to support paired statistical comparisons.
