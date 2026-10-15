# V3_ClassReweighting

## Purpose

Backbone with CLAHE and class reweighting; compare against V2 to evaluate the effect of the imbalance strategy.

## Relationship to the manuscript

This directory records the **V3** ablation variant. The updated manuscript reports an input size of **384×384**, whereas some archived `README.txt` records for early variants specify **224×224**. Treat these as historical run artifacts until rerun/profiling equivalence is confirmed; do not silently conflate configurations.

## Files in this directory

- `README.txt`
- `acc_curves.png`
- `confusion_matrix_v3.png`
- `efficiency_v3.json`
- `f1_auc_curves.png`
- `loss_curves.png`
- `metrics.json`
- `metrics_v3.csv`

## Reproducibility guidance

These descriptions reflect the **files actually present in GitHub**. A README is not evidence that saved checkpoints, five-seed logs, test predictions, or final manuscript-corrected artifacts have been verified. See the [repository-level README](https://github.com/ItsCodeBakery/XDR-NET#readme) for corrected classification results, caveats, and the manuscript-supporting release plan.
