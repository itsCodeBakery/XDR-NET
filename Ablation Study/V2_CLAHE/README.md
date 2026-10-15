# V2_CLAHE

## Purpose

EfficientNet-B0 with contrast-limited adaptive histogram equalization (CLAHE) added during preprocessing.

## Relationship to the manuscript

This directory records the **V2** ablation variant. The updated manuscript reports an input size of **384×384**, whereas some archived `README.txt` records for early variants specify **224×224**. Treat these as historical run artifacts until rerun/profiling equivalence is confirmed; do not silently conflate configurations.

## Files in this directory

- `README.txt`
- `acc_curves.png`
- `efficiency_v2.json`
- `loss_curves.png`
- `metrics.json`
- `metrics_v2.csv`

## Reproducibility guidance

These descriptions reflect the **files actually present in GitHub**. A README is not evidence that saved checkpoints, five-seed logs, test predictions, or final manuscript-corrected artifacts have been verified. See the [repository-level README](https://github.com/ItsCodeBakery/XDR-NET#readme) for corrected classification results, caveats, and the manuscript-supporting release plan.
