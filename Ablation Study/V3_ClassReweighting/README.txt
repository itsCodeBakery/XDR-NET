Ablation Study - Variant V3: +Class Reweighting (EfficientNet-B0)

Description:
EfficientNet-B0 with class-imbalance mitigation via loss reweighting. We compute weights from the training split
as w_c = N / (C * n_c), where n_c is count of class c.

Hyperparameters:
- Epochs: 20
- Batch Size: 32
- Learning Rate: 0.0001
- Image Size: 224x224

Loss:
  L = - sum_{i=1}^{C} w_i * y_i * log(p_i)
  where w_i are class weights, y_i one-hot target, p_i predicted probability.

Evaluation:
  Macro-F1 = (1/C) * sum_c ( 2 * Prec_c * Rec_c / (Prec_c + Rec_c) )
  AUC (macro OvR) = (1/C) * sum_c AUC(c vs rest)

Artifacts:
- model_v3.pth
- metrics.json (per-epoch curves + params + class weights)
- metrics_v3.csv (per-epoch Acc/Loss/F1/AUC + efficiency row)
- loss_curves.png, acc_curves.png, f1_auc_curves.png
- confusion_matrix_v3.png
- efficiency_v3.json (latency/throughput/MACs/FLOPs)
