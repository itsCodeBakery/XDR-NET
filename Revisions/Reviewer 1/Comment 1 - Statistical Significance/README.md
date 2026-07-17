# Reviewer 1 — Comment 1: Statistical Significance

## Reviewer comment
The reported performance improvements over the baseline methods should be supported by appropriate statistical significance analysis (e.g., confidence intervals or statistical tests) to demonstrate that the improvements are not due to random variation.

## Decision
Additional statistical analysis is required. Full model retraining is not automatically required if saved per-image predictions from XDR-Net and the comparison models are available for the same held-out split. Otherwise, inference or targeted retraining will be performed for XDR-Net and the strongest baseline models under the identical split and preprocessing protocol.

## Planned analysis
1. Save per-image ground-truth labels and class probabilities for every compared model.
2. Compute 95% stratified bootstrap confidence intervals for accuracy, macro-F1, macro-precision, macro-recall, and macro-AUC.
3. Compare XDR-Net against the strongest baselines using paired stratified bootstrap differences.
4. Apply McNemar's exact test to paired correctness outcomes.
5. Correct multiple pairwise p-values using Holm's method.
6. Save all configurations, predictions, confidence intervals, pairwise tests, and publication-ready tables in this folder.

## Required prediction format
Each model must produce a CSV containing:

`image_id,true_label,pred_label,prob_0,prob_1,prob_2,prob_3,prob_4`

All models must be evaluated on exactly the same held-out images.
