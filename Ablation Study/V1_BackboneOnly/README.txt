Ablation Study - Variant V1: Backbone Only (EfficientNet-B0)

Description:
This variant uses only EfficientNet-B0 backbone with ImageNet initialization.
No CLAHE preprocessing, no class reweighting, no attention bridge, no TTA.

Hyperparameters:
- Epochs: 20
- Batch Size: 32
- Learning Rate: 1e-4
- Image Size: 224x224

Metrics Tracked:
- Training/Validation Accuracy
- Validation Macro-F1
- Validation AUC
- Model Parameters
- Inference Latency (to be added)

Mathematical Formulations:
Cross-Entropy Loss:
L = - Σ y_i log( p_i )
where y_i is the true label (one-hot) and p_i is predicted probability.

Macro-F1:
F1_macro = (1/C) Σ (2 * Precision_c * Recall_c) / (Precision_c + Recall_c)
where C is number of classes.

Multi-class AUC (One-vs-Rest):
AUC = (1/C) Σ AUC(class c)

Results are stored in metrics_v1.csv, metrics.json, and training plots.
