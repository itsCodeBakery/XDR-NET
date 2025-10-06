# XDR-NET: Hybrid CNN–Transformer Network for Diabetic Retinopathy Detection

A lightweight, explainable DR grader that couples an EfficientNet backbone with a single self-attention bridge, delivering strong performance and clear Grad-CAM evidence on benchmarks.

---

##  Overview

Automated diabetic retinopathy (DR) screening demands models that are **accurate, efficient, and transparent**. Pure CNNs capture local lesion cues (microaneurysms, hemorrhages, exudates) but can miss **global retinal context**; attention models capture global relations but are often heavy. **XDR-Net** balances both: it keeps EfficientNet’s compact convolutional features and adds a single token-level self-attention block before pooling. We pair this with a pragmatic training recipe for imbalanced grades and clinician-oriented visual explanations.

---

## Preprocessing (APTOS 2019)

We apply a reproducible, screening-oriented preprocessing pipeline:

- Circular crop of the fundus and background removal  
- Resize to **384×384**, per-channel standardization  
- **CLAHE** on luminance to enhance local lesion contrast while limiting noise

**CLAHE example**

![CLAHE](https://github.com/ItsCodeBakery/XDR-NET/blob/main/Proposed%20Methodology/plots/ClahePrep.png)

---

## 🏗Proposed Methodology (XDR-Net)

**Backbone:** EfficientNet (from `timm`) for compact, high-quality feature extraction  
**Token-Attention bridge:** a **single Multi-Head Self-Attention (MHA)** applied on the final feature map (tokenized) to inject **global retinal context** with minimal overhead  
**Head:** Global Average Pooling → LayerNorm → Dropout → Linear (320→5) for 5-class DR grading  
**Training/Inference:** AdamW (cosine schedule), class-weighted CE with label smoothing; test-time augmentation and temperature scaling for calibrated probabilities  
**Explainability:** Grad-CAM on the last conv block; class-wise grids and error panels

**Architecture sketch**

![Methodology](https://github.com/ItsCodeBakery/XDR-NET/blob/main/Proposed%20Methodology/plots/METHODLOGY.png)

---

## Results (APTOS 2019)

**Validation split:** 1,805 images (five classes)

- **XDR-Net:** **Accuracy 97.40%**, **Macro-F1 97.38%**  
- **Baselines (same pipeline):**
  - ResNet-18 — Acc **81.82%**, Macro-F1 **68.39%**
  - ResNet-50 — Acc **82.73%**, Macro-F1 **70.32%**
  - ConvNeXt-Base — Acc **81.04%**, Macro-F1 **66.18%**

**Confusion matrix (APTOS)**

![Confusion Matrix](https://github.com/ItsCodeBakery/XDR-NET/blob/main/Proposed%20Methodology/plots/confusion_matrix_counts_vs_normalized.png)

**Confusion matrix (EyePACS)**

![Confusion Matrix](https://github.com/ItsCodeBakery/XDR-NET/blob/main/Proposed%20Methodology/plots/cmEyePacs.png)

**Confusion matrix (IDRiD)**
![Confusion Matrix](https://github.com/ItsCodeBakery/XDR-NET/blob/main/Proposed%20Methodology/plots/cmIDriD.png)


**Confusion matrix (Messidor)**

![Confusion Matrix](https://github.com/ItsCodeBakery/XDR-NET/blob/main/Proposed%20Methodology/plots/cmMessidor.png)


---

## 📉 Learning Curves

The learning curves indicate stable optimization and good generalization: validation **accuracy** and **macro-F1** steadily improve and closely track training metrics, while the **loss** decreases without divergence. The absence of large gaps between training and validation curves suggests that class rebalancing, label smoothing, and cosine-scheduled AdamW effectively mitigate overfitting on APTOS.

<div align="center">

**Accuracy (train/val)**  
<img src="https://github.com/ItsCodeBakery/XDR-NET/blob/main/Proposed%20Methodology/plots/accuracy_curve.png" alt="Accuracy Curve" width="48%">

**Macro-F1 (train/val)**  
<img src="https://github.com/ItsCodeBakery/XDR-NET/blob/main/Proposed%20Methodology/plots/f1_curve.png" alt="F1 Curve" width="48%">

**Loss (train/val)**  
<img src="https://github.com/ItsCodeBakery/XDR-NET/blob/main/Proposed%20Methodology/plots/loss_curve.png" alt="Loss Curve" width="60%">

</div>

---
## Explainability

We generate class-discriminative **Grad-CAM** overlays from the final convolutional block, enabling graders to visualize lesion evidence and failure modes.

**Class-wise Grad-CAM montage**

![Grad-CAM](https://github.com/ItsCodeBakery/XDR-NET/blob/main/Proposed%20Methodology/plots/EyesPacs.JPG)
![Grad-CAM](https://github.com/ItsCodeBakery/XDR-NET/blob/main/Proposed%20Methodology/plots/APTOS.JPG)
![Grad-CAM](https://github.com/ItsCodeBakery/XDR-NET/blob/main/Proposed%20Methodology/plots/IRDiD.JPG)

---

---

## Reproducibility (Kaggle / Local)

- **Dataset:** APTOS 2019 (add on Kaggle and point `train_images/` + `train.csv`)  
- **Env:** Python 3.10+, PyTorch 2.x, `timm`, `torchvision`, `scikit-learn`, `pandas`, `matplotlib`  
- **Run:** See `Proposed Methodology/code/` for XDR-Net and `BaseLineExperement/*/code/` for baselines  
- **Explainability:** Grad-CAM scripts reproduce the class-wise grids shown above

---

## Contact

Questions or collaboration: **shayan.ali@imsciences.edu.pk**


