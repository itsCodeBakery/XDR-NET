# splits

## Purpose

Split summaries and, where committed, train/validation manifests; verify fixed cohort membership before replication. Parent: `BaseLineExperement/resnet18`.

**Partition warning:** Image-level split summaries are not proof of patient-level independence. APTOS 2019 uses a selected holdout validation cohort, not an independent public-label test set.

## Files in this directory

- `.gitkeep`
- `split_summary.json`

## Reproducibility guidance

These descriptions reflect the **files actually present in GitHub**. A README is not evidence that saved checkpoints, five-seed logs, test predictions, or final manuscript-corrected artifacts have been verified. See the [repository-level README](https://github.com/ItsCodeBakery/XDR-NET#readme) for corrected classification results, caveats, and the manuscript-supporting release plan.
