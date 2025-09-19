Ablation Study - Variant V2: +CLAHE (EfficientNet-B0)

Description:
EfficientNet-B0 backbone with Contrast Limited Adaptive Histogram Equalization (CLAHE) applied to the L-channel in LAB space before resizing/normalization.
No class reweighting, no attention bridge, no TTA.

Preprocessing (CLAHE):
- clip_limit = 2.0
- tile_grid_size = (8, 8)

Hyperparameters:
- Epochs: 20
- Batch Size: 32
- Learning Rate: 1e-4
- Image Size: 224x224

Metrics Tracked:
- Training/Validation Accuracy & Loss (per epoch)
- Validation Macro-F1 (per epoch)
- Validation AUC (one-vs-rest, macro, per epoch)
- Model Parameters and (recorded separately) Efficiency Metrics

Mathematical Formulations:
Cross-Entropy Loss:
  L = - \sum_{i=1}^{C} y_i \log p_i
  where y_i \in {0,1} is one-hot target and p_i is predicted probability.

Macro-F1:
  F1_{macro} = \frac{1}{C} \sum_{c=1}^{C} \frac{2\,\mathrm{Prec}_c\,\mathrm{Rec}_c}{\mathrm{Prec}_c+\mathrm{Rec}_c}

Multi-class AUC (OvR, macro-average):
  AUC = \frac{1}{C} \sum_{c=1}^{C} \mathrm{AUC}(c\;\text{vs.}\;\text{rest})

Files:
- model_v2.pth       : trained weights
- metrics.json       : summary of losses/accuracy/params
- metrics_v2.csv     : per-epoch Acc/Loss/F1/AUC
- loss_curves.png, acc_curves.png
- efficiency_v2.json : (after running efficiency block)

Efficiency Metrics (V2 + CLAHE):
- Latency (batch=1): mean=10.938 ms (p50=10.870, p90=11.267, p95=11.557)
- Throughput (batch=32): 1179.59 img/s
- Params: 4.014 M
- MACs: 413.87M | FLOPs (≈2*MACs): 827.74M
Note: FLOPs estimated as 2×MACs (conv ops assumption). Measured with thop if available.
