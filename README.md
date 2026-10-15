# XDR–Net: A Hybrid Convolutional Network with a Single-Block Attention Bridge and Balanced Optimization for Diabetic Retinopathy Detection

[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg)](https://pytorch.org/)
![Research status](https://img.shields.io/badge/Status-Manuscript%20revision-blue)
![Evaluation](https://img.shields.io/badge/Evaluation-Retrospective%20benchmarks-lightgrey)

**XDR–Net** is an attention-augmented convolutional framework for diabetic retinopathy (DR) severity grading from color fundus photographs. It combines an **EfficientNet-B0** feature extractor, a **single multi-head self-attention bridge**, effective-number-based class balancing, CLAHE preprocessing, four-view test-time augmentation (TTA), and Grad-CAM visualization.

> **Manuscript alignment (October 2026 revision).** This README summarizes the revised manuscript titled above, including corrected confusion-matrix-derived classification metrics and the author's updated **4.425 M** V4/V5 parameter figure. This README update **does not** certify that every checked-in notebook, prediction file, split manifest, checkpoint, or profiling artifact has been synchronized to the revision. Unresolved evidence items are identified below.

## Main results

The following accuracy, macro-precision, macro-recall, and macro-F1 scores agree with the confusion matrices and Table 7 in the revised manuscript. Metrics are percentages.

| Evaluation cohort | Accuracy | Macro-F1 | Macro-Precision | Macro-Recall | Reported macro-AUC |
|---|---:|---:|---:|---:|---:|
| **APTOS 2019** — selected holdout validation, N=550 | **97.45** | **97.07** | **96.54** | **97.67** | 97.99 |
| **EyePACS** — test, N=3,512 | **97.44** | **97.03** | **96.74** | **97.36** | 98.90 |
| **Messidor** — four-grade test, N=120 | **96.67** | **96.73** | **96.79** | **96.70** | 97.50 |
| **IDRiD** — test, N=103 | **96.12** | **95.60** | **94.50** | **96.85** | 96.75 |

The unweighted arithmetic mean across these four evaluation cohorts is **96.92% accuracy** and **96.61% macro-F1**. This is a descriptive cross-dataset mean, **not** a pooled sample-weighted clinical performance estimate. The reported mean macro-AUC is **97.79%**; AUCs require predicted probabilities and cannot be independently inferred from confusion-matrix counts.

**Evaluation qualification:** APTOS 2019 uses the same 550-image holdout for checkpoint selection and final reporting because labeled official competition test images were unavailable. Consequently, its scores are **selected validation estimates**, not independent test-set measurements, and may be optimistically biased. All datasets use image-level partitions; patient-level independence has not been verified.

## Dataset cohorts and grading definitions

| Dataset | Total images used | Training | Validation | Test | Classification labels |
|---|---:|---:|---:|---:|---|
| APTOS 2019 | 3,662 | 3,112 | 550 | — | Five grades (0–4) |
| EyePACS | 35,126 | 28,102 | 3,512 | 3,512 | Five grades (0–4) |
| Messidor | 1,200 | 960 | 120 | 120 | Four native categories (0–3) |
| IDRiD | 516 | 330 | 83 | 103 | Five grades (0–4) |

Partitions are described as generated before augmentation and class rebalancing, with a fixed split seed of **42**. The manuscript lists separate evaluation cohorts for EyePACS and Messidor. The fixed split manifests and exact image identifiers should be used when reproducing results; dataset class-distribution alignment and final manifest completeness still require source-data checks.

**Messidor is not five-class in this study.** Its evaluation follows the native four-category scale (Grades 0–3) described in the revised manuscript. Do not directly equate its fourth category with the separate Grade 4 of the other three datasets.

### Confusion-matrix cohort sizes

Each confusion matrix contains **one predicted class per original evaluation image** (not four copies from TTA). The displayed matrices total **550 APTOS**, **3,512 EyePACS**, **120 Messidor**, and **103 IDRiD** images. Under four-view TTA, view scores are aggregated to form the final image-level prediction. The exact aggregation implementation should be confirmed from the released evaluation code.

## Architecture

```text
Fundus photograph (RGB, 384 × 384)
    │
    ▼
Circular retinal cropping + CLAHE + normalization
    │
    ▼
EfficientNet-B0 convolutional backbone
    │
    ▼
Feature tensor (320 × 12 × 12)
    │
    ▼
Reshape to 144 spatial tokens, width 320
    │
    ▼
LayerNorm → single four-head multi-head self-attention
    │        (80 channels per head)
    ▼
Attention output projection + dropout + one residual addition
    │
    ▼
Mean pooling over 144 tokens
    │
    ▼
LayerNorm → dropout → linear classifier → softmax
```

The **implemented bridge described by the revised methodology has one attention sublayer and one residual connection, without a separate feedforward network (FFN) or extra residual sublayer**. The architectural figure in the manuscript should be reconciled with this description and the actual code.

For standard biased query/key/value and output projections with `d = 320`:

```text
MHA projection parameters = 4 × d² + 4 × d
                          = 4 × 320² + 4 × 320
                          = 410,880
```

The updated manuscript parameter figure for V4/V5 is **approximately 4.425 million** parameters, compared with **4.014 million** for V1–V3. This is arithmetically consistent with adding approximately 0.411 M attention parameters to the reported base figure. **The exact instantiated model and checkpoint parameter counts should still be verified**, including layer normalization, the classification head, and any replaced components.

## Preprocessing and training

The reported pipeline uses center/circular retinal cropping, **384 × 384** resizing, per-channel normalization, luminance-channel **CLAHE**, and mild geometric/photometric augmentation during training. The model uses effective-number-derived class weights for both **class-aware sampling** and **class-weighted, label-smoothed cross-entropy**.

| Setting | Manuscript configuration |
|---|---|
| Backbone | EfficientNet-B0 |
| Input size | 384 × 384 |
| Training batch size | 32 |
| Optimizer | AdamW, β₁=0.9, β₂=0.999 |
| Initial learning rate | 3e-4 |
| Weight decay | 1e-4 |
| Schedule | Cosine decay |
| Maximum epochs | 20 |
| Early stopping | Patience 3, selected by validation macro-F1 |
| Gradient clipping | ℓ2 norm 1.0 |
| Label smoothing | ε=0.1 |
| Attention | Single MHA, d=320, 4 heads |
| Test-time augmentation | 4 views |

Reported effective-number hyperparameters are approximately β=0.999 (APTOS), 0.9999 (EyePACS), and 0.995 (Messidor, IDRiD). In the revised ablation study, class reweighting **alone (V3) reduces macro-F1 compared with V2**, so the effect of this component should not be characterized as uniformly beneficial in isolation.

### Repeated-run evidence

The manuscript describes training initialization seeds **42, 43, 44, 45, 46** for repeated-run analyses, with the intention of retaining the dataset partitions from split seed **42**. Its APTOS V3→V4 ablation discussion reports:

| Configuration | Reported mean macro-F1 | Reported range |
|---|---:|---|
| V3: + class reweighting | 59.71 ± 1.18% | 58.20–61.10% |
| V4: + attention bridge | 95.12 ± 0.34% | 94.65–95.50% |

The difference of reported means is **+35.41 percentage points**. These are **manuscript-reported summaries**, not independently revalidated five-seed logs. Individual per-seed results and matched controls should accompany a final reproducibility release. The manuscript's statistical significance claims, paired McNemar tests, permutation tests, and bootstrap intervals also require reconciliation with the corresponding prediction-level evidence; this README does **not** assert blanket `p < 0.001` significance.

## Computational efficiency and ablation

The revised manuscript specifies an **NVIDIA Tesla T4 (16 GB)**, image size **384 × 384**, and timing batch size **1**, with **50 warm-up** and **1,000 timing iterations**.

| APTOS configuration | Params (M) | GFLOPs | Reported latency (ms/image) | Macro-F1 |
|---|---:|---:|---:|---:|
| V4: single pass | **4.425** | 0.83 | 9.86 | 95.12% |
| V5: four-view TTA | **4.425** | 3.32 | 39.44 | 97.07% |

Four-view GFLOPs are the cumulative sum of four 0.83-GFLOP network forward passes. TTA reuses model weights, so its parameter count does not change. The stated V5 latency is exactly four times the V4 latency; whether it was independently measured end-to-end or estimated from single-pass timing, and whether preprocessing/aggregation overhead was included, **still requires verification**.

APTOS 2019 component ablation, as reported in the manuscript:

| Variant | Evaluation accuracy (%) | Macro-F1 (%) | Reported macro-AUC (%) | Params (M) | GFLOPs |
|---|---:|---:|---:|---:|---:|
| V1: EfficientNet-B0 only | 80.90 | 63.94 | 91.87 | 4.014 | 0.79 |
| V2: + CLAHE | 80.08 | 64.55 | 89.98 | 4.014 | 0.79 |
| V3: + class reweighting | 78.04 | 59.71 | 90.49 | 4.014 | 0.79 |
| V4: + single-block attention | 94.95 | 95.12 | 95.60 | **4.425** | 0.83 |
| **V5: + four-view TTA** | **97.45** | **97.07** | **97.99** | **4.425** | **3.32** |

V5 improves macro-F1 over V4 by **1.95 percentage points** on APTOS, **1.18** on EyePACS, **1.93** on Messidor, and **1.70** on IDRiD. These are comparisons under the reported experimental configuration; they do not establish clinical transferability.

## Grad-CAM explainability

Grad-CAM is calculated using the final convolutional feature map before the attention bridge. Its overlays offer **post-hoc spatial attribution**, not evidence of causal or independently clinician-validated reasoning.

The manuscript describes an IDRiD lesion-mask comparison using pointing-game accuracy, lesion-region energy, and thresholded IoU. **The exact localization cohort and per-grade sample counts must be reconciled with the 103-image IDRiD grading test split**, including image IDs, lesion-mask availability, eligibility criteria, and split membership. For this reason, the previously stated **82-image** quantitative localization summary is **not presented here as a verified reproducibility result**. Pixel-level lesion masks were not used to train the grading model.

## Code and available repository content

The repository currently contains these directories and artifacts:

```text
XDR-NET/
├── Ablation Study/
│   ├── V1_BackboneOnly/
│   ├── V2_CLAHE/
│   ├── V3_ClassReweighting/
│   ├── V4_AttentionBridge/
│   └── V5_TTA/
├── BaseLineExperement/
├── Proposed Methodology/
│   ├── README.md
│   ├── code/
│   │   └── xdr-net (2).ipynb
│   ├── plots/
│   └── splits/
│       └── split_summary.json
└── README.md
```

This tree is a **selected overview**, not a guarantee that every run-specific checkpoint, full split manifest, saved probability array, or experiment log is published. The repository name `BaseLineExperement` is retained verbatim to match the existing folder.

### Getting started

```bash
git clone https://github.com/ItsCodeBakery/XDR-NET.git
cd XDR-NET
```

Review `Proposed Methodology/code/xdr-net (2).ipynb` for the current notebook workflow. **Do not assume the default-branch notebook reproduces the corrected manuscript metrics without validating its configuration and inputs.** A verified end-to-end installation and training command is not supplied here because the current repository root does not contain a published `requirements.txt`.

Public dataset providers:

- [APTOS 2019 (Kaggle)](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data)
- [EyePACS DR Detection (Kaggle)](https://www.kaggle.com/competitions/diabetic-retinopathy-detection/data)
- [Messidor (ADCIS)](https://www.adcis.net/en/third-party/messidor/)
- [IDRiD (IEEE DataPort)](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid)

Datasets are **not redistributed** here. Follow each provider's access and usage conditions.

## Reproducibility status and release plan

For a manuscript-supporting **immutable release**, the repository should include:

1. The exact V1–V5 model definitions, checkpoints or checksum-linked weights, preprocessing code, and four-view TTA aggregation.
2. Fixed image-ID split manifests, severity-label mappings (especially four-category Messidor), and evaluation-cohort counts.
3. Final class predictions and class probabilities for every evaluated image, plus scripts regenerating Figure 4 and Tables 7 and 11.
4. GPU profiling code/logs, explicitly distinguishing direct end-to-end TTA latency from a four-times-single-pass estimate.
5. Five-seed experiment records, paired predictions for statistical tests, and version-locked environment dependencies.
6. Verified Grad-CAM localization image/mask membership and regenerated per-grade quantitative summaries, if reported.

**A README commit alone is not proof of complete result reproducibility.** Once the materials are synchronized, cite the full Git commit SHA or archive a release with a permanent DOI. No final results-supported release is asserted by this README.

## Experimental software environment (as stated in manuscript)

```text
GPU:            NVIDIA Tesla T4, 16 GB
Python:         3.12.13
PyTorch:        2.10.0
CUDA:           12.8
cuDNN:          9.10.2
torchvision:    0.25.0
timm:           1.0.26
Albumentations: 2.0.8
OpenCV:         4.13.0
scikit-learn:   1.6.1
NumPy:          2.0.2
pandas:         2.3.3
```

These are manuscript-reported versions and should be compared with the actual saved training environment.

## Scope, ethics, and limitations

XDR–Net is a **research-stage, retrospective benchmark method**. The reported results are not independent prospective clinical validation. Patient-level separation could not be consistently verified, the APTOS holdout is used for model selection, and performance under institution/camera shifts and low-quality real-world imaging remains unestablished. Grad-CAM does not provide a validated clinical rationale, and additional external, patient-separated, and prospective evaluation is needed before considering screening deployment.

## Citation

**Manuscript:** *XDR–Net: A Hybrid Convolutional Network with a Single-Block Attention Bridge and Balanced Optimization for Diabetic Retinopathy Detection*. Cite the manuscript or its publication record when available; no DOI or formal publication status is asserted here.

## Contact

For reproducibility questions, please open a [GitHub issue](https://github.com/ItsCodeBakery/XDR-NET/issues) or use the correspondence information in the manuscript.
