# V5_TTA

## Purpose

The final XDR-Net variant using four test-time augmentation views, aggregated into one prediction per input image.

## Relationship to the manuscript

This directory records the **V5** ablation variant. The updated manuscript reports an input size of **384×384**, whereas some archived `README.txt` records for early variants specify **224×224**. Treat these as historical run artifacts until rerun/profiling equivalence is confirmed; do not silently conflate configurations.

**Efficiency note:** Four TTA forward passes correspond to **3.32 GFLOPs** when a single pass is **0.83 GFLOPs**. Check whether reported **39.44 ms** APTOS V5 latency was measured end-to-end or extrapolated from V4.

## Files in this directory

- `README.txt`
- `efficiency_v5.json`
- `f1_auc_curves.png`
- `loss_curves.png`
- `metrics_v5.csv`

## Reproducibility guidance

These descriptions reflect the **files actually present in GitHub**. A README is not evidence that saved checkpoints, five-seed logs, test predictions, or final manuscript-corrected artifacts have been verified. See the [repository-level README](https://github.com/ItsCodeBakery/XDR-NET#readme) for corrected classification results, caveats, and the manuscript-supporting release plan.
