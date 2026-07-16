# ============================================================
# APTOS 2019: EfficientNet-B0 baseline, 5-fold cross-validation
# Reviewer 1, Comment 1: statistical significance analysis
# ============================================================

from __future__ import annotations

import gc
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm

# ============================================================
# Configuration
# ============================================================

SEED = 42
N_FOLDS = 5
NUM_CLASSES = 5
IMAGE_SIZE = 384
BATCH_SIZE = 16
NUM_WORKERS = 2
EPOCHS = 20
PATIENCE = 5
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.10
BETA_EFFECTIVE_NUMBER = 0.999
MODEL_NAME = "efficientnet_b0"
USE_PRETRAINED = True

FOLD_FILE = Path("/kaggle/working/xdrnet_aptos/aptos_5fold_assignments.csv")
OUTPUT_ROOT = Path("/kaggle/working/xdrnet_aptos/efficientnet_b0_5fold")
CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"
PREDICTION_DIR = OUTPUT_ROOT / "predictions"
METRIC_DIR = OUTPUT_ROOT / "metrics"
FIGURE_DIR = OUTPUT_ROOT / "figures"

for directory in [OUTPUT_ROOT, CHECKPOINT_DIR, PREDICTION_DIR, METRIC_DIR, FIGURE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = torch.cuda.is_available()

CLASS_NAMES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ============================================================
# Reproducibility
# ============================================================


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


set_seed(SEED)

# ============================================================
# Image preprocessing
# ============================================================


def read_rgb_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def remove_black_border(image: np.ndarray, threshold: int = 7) -> np.ndarray:
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    valid_mask = (grayscale > threshold).astype(np.uint8)
    coordinates = cv2.findNonZero(valid_mask)
    if coordinates is None:
        return image
    x, y, width, height = cv2.boundingRect(coordinates)
    return image[y : y + height, x : x + width]


def circular_crop(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    radius = int(0.98 * min(center))
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, thickness=-1)
    return cv2.bitwise_and(image, image, mask=mask)


def apply_clahe_lab(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    luminance, channel_a, channel_b = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_luminance = clahe.apply(luminance)
    enhanced_lab = cv2.merge((enhanced_luminance, channel_a, channel_b))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)


def preprocess_fundus(image: np.ndarray) -> np.ndarray:
    image = remove_black_border(image)
    image = circular_crop(image)
    image = apply_clahe_lab(image)
    return image


# ============================================================
# Transformations
# ============================================================


def build_train_transform() -> A.Compose:
    return A.Compose(
        [
            A.Resize(IMAGE_SIZE, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR),
            A.HorizontalFlip(p=0.5),
            A.Rotate(
                limit=15,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5,
            ),
            A.RandomResizedCrop(
                size=(IMAGE_SIZE, IMAGE_SIZE),
                scale=(0.90, 1.00),
                ratio=(0.95, 1.05),
                interpolation=cv2.INTER_LINEAR,
                p=0.35,
            ),
            A.ColorJitter(
                brightness=0.10,
                contrast=0.10,
                saturation=0.08,
                hue=0.02,
                p=0.30,
            ),
            A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0.1, 1.0), p=0.10),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )


def build_valid_transform() -> A.Compose:
    return A.Compose(
        [
            A.Resize(IMAGE_SIZE, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )


# ============================================================
# Dataset
# ============================================================


class AptosDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform: A.Compose) -> None:
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.dataframe.iloc[index]
        image = preprocess_fundus(read_rgb_image(row["image_path"]))
        image = self.transform(image=image)["image"]
        return {
            "image": image,
            "label": torch.tensor(int(row["diagnosis"]), dtype=torch.long),
            "id_code": row["id_code"],
            "fold": int(row["fold"]),
        }


# ============================================================
# Class weights and loaders
# ============================================================


def effective_number_weights(labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float64)
    effective_numbers = (1.0 - np.power(BETA_EFFECTIVE_NUMBER, counts)) / (
        1.0 - BETA_EFFECTIVE_NUMBER
    )
    weights = 1.0 / effective_numbers
    weights = weights / weights.mean()
    return weights.astype(np.float32)


def make_loaders(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    fold_seed: int,
) -> Tuple[DataLoader, DataLoader, np.ndarray]:
    labels = train_df["diagnosis"].to_numpy(dtype=np.int64)
    class_weights = effective_number_weights(labels)
    sample_weights = torch.as_tensor(class_weights[labels], dtype=torch.double)
    sampler_generator = torch.Generator().manual_seed(fold_seed)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=sampler_generator,
    )

    train_dataset = AptosDataset(train_df, build_train_transform())
    valid_dataset = AptosDataset(valid_df, build_valid_transform())
    loader_generator = torch.Generator().manual_seed(fold_seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=AMP_ENABLED,
        persistent_workers=NUM_WORKERS > 0,
        worker_init_fn=seed_worker,
        generator=loader_generator,
        drop_last=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=AMP_ENABLED,
        persistent_workers=NUM_WORKERS > 0,
        worker_init_fn=seed_worker,
        generator=loader_generator,
        drop_last=False,
    )

    return train_loader, valid_loader, class_weights


# ============================================================
# Model and metrics
# ============================================================


def build_model() -> nn.Module:
    model = timm.create_model(
        MODEL_NAME,
        pretrained=USE_PRETRAINED,
        num_classes=NUM_CLASSES,
        drop_rate=0.30,
    )
    return model.to(DEVICE)


def compute_metrics(labels: np.ndarray, probabilities: np.ndarray) -> Dict[str, float]:
    predictions = probabilities.argmax(axis=1)
    one_hot_labels = label_binarize(labels, classes=np.arange(NUM_CLASSES))
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "macro_precision": float(
            precision_score(labels, predictions, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(labels, predictions, average="macro", zero_division=0)
        ),
        "macro_auc": float(
            roc_auc_score(one_hot_labels, probabilities, average="macro", multi_class="ovr")
        ),
    }


# ============================================================
# Training and validation loops
# ============================================================


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    epoch_index: int,
    fold_index: int,
) -> float:
    model.train()
    running_loss = 0.0

    progress = tqdm(
        loader,
        desc=f"Fold {fold_index} | Epoch {epoch_index:02d} | Train",
        leave=False,
    )

    for batch in progress:
        images = batch["image"].to(DEVICE, non_blocking=True)
        labels = batch["label"].to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=AMP_ENABLED):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / len(loader.dataset)


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    epoch_index: int,
    fold_index: int,
) -> Tuple[float, np.ndarray, np.ndarray, List[str]]:
    model.eval()
    running_loss = 0.0
    all_probabilities: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_ids: List[str] = []

    progress = tqdm(
        loader,
        desc=f"Fold {fold_index} | Epoch {epoch_index:02d} | Valid",
        leave=False,
    )

    for batch in progress:
        images = batch["image"].to(DEVICE, non_blocking=True)
        labels = batch["label"].to(DEVICE, non_blocking=True)

        with autocast(enabled=AMP_ENABLED):
            logits = model(images)
            loss = criterion(logits, labels)

        probabilities = torch.softmax(logits, dim=1)
        running_loss += loss.item() * images.size(0)
        all_probabilities.append(probabilities.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_ids.extend(batch["id_code"])

    return (
        running_loss / len(loader.dataset),
        np.concatenate(all_labels),
        np.concatenate(all_probabilities),
        all_ids,
    )


# ============================================================
# Plotting utilities: all titles and data labels are bold
# ============================================================


def apply_bold_axis_style(axis: plt.Axes) -> None:
    axis.title.set_fontweight("bold")
    axis.xaxis.label.set_fontweight("bold")
    axis.yaxis.label.set_fontweight("bold")
    for tick in axis.get_xticklabels() + axis.get_yticklabels():
        tick.set_fontweight("bold")


def plot_training_history(history_df: pd.DataFrame, fold_index: int) -> None:
    plots = [
        ("train_loss", "valid_loss", "Cross-Entropy Loss", "Loss"),
        ("valid_accuracy", None, "Validation Accuracy", "Accuracy"),
        ("valid_macro_f1", None, "Validation Macro-F1", "Macro-F1"),
        ("valid_macro_auc", None, "Validation Macro-AUC", "Macro-AUC"),
    ]

    for first_column, second_column, title, y_label in plots:
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.plot(history_df["epoch"], history_df[first_column], marker="o", label=first_column)
        if second_column is not None:
            axis.plot(
                history_df["epoch"],
                history_df[second_column],
                marker="o",
                label=second_column,
            )
        axis.set_title(f"EfficientNet-B0 Fold {fold_index}: {title}", fontweight="bold")
        axis.set_xlabel("Epoch", fontweight="bold")
        axis.set_ylabel(y_label, fontweight="bold")
        axis.grid(alpha=0.3)
        if second_column is not None:
            legend = axis.legend()
            for text in legend.get_texts():
                text.set_fontweight("bold")
        apply_bold_axis_style(axis)
        figure.tight_layout()
        figure.savefig(
            FIGURE_DIR / f"fold_{fold_index}_{first_column}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(figure)


def plot_confusion_matrix(labels: np.ndarray, predictions: np.ndarray) -> None:
    matrix = confusion_matrix(labels, predictions, labels=np.arange(NUM_CLASSES))
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix)
    axis.set_title("APTOS 2019 EfficientNet-B0 Out-of-Fold Confusion Matrix", fontweight="bold")
    axis.set_xlabel("Predicted DR Grade", fontweight="bold")
    axis.set_ylabel("True DR Grade", fontweight="bold")
    axis.set_xticks(np.arange(NUM_CLASSES), [CLASS_NAMES[i] for i in range(NUM_CLASSES)], rotation=30, ha="right")
    axis.set_yticks(np.arange(NUM_CLASSES), [CLASS_NAMES[i] for i in range(NUM_CLASSES)])

    threshold = matrix.max() / 2.0
    for row in range(NUM_CLASSES):
        for column in range(NUM_CLASSES):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:d}",
                ha="center",
                va="center",
                fontweight="bold",
                color="white" if matrix[row, column] > threshold else "black",
            )

    for tick in axis.get_xticklabels() + axis.get_yticklabels():
        tick.set_fontweight("bold")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "oof_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_roc_curves(labels: np.ndarray, probabilities: np.ndarray) -> None:
    one_hot_labels = label_binarize(labels, classes=np.arange(NUM_CLASSES))
    figure, axis = plt.subplots(figsize=(8, 7))

    for class_index in range(NUM_CLASSES):
        false_positive_rate, true_positive_rate, _ = roc_curve(
            one_hot_labels[:, class_index], probabilities[:, class_index]
        )
        class_auc = auc(false_positive_rate, true_positive_rate)
        axis.plot(
            false_positive_rate,
            true_positive_rate,
            linewidth=2,
            label=f"{CLASS_NAMES[class_index]} (AUC = {class_auc:.3f})",
        )

    axis.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5)
    axis.set_title("APTOS 2019 EfficientNet-B0 Out-of-Fold ROC Curves", fontweight="bold")
    axis.set_xlabel("False Positive Rate", fontweight="bold")
    axis.set_ylabel("True Positive Rate", fontweight="bold")
    legend = axis.legend(loc="lower right")
    for text in legend.get_texts():
        text.set_fontweight("bold")
    axis.grid(alpha=0.3)
    apply_bold_axis_style(axis)
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "oof_roc_curves.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_fold_metric_bars(fold_metrics_df: pd.DataFrame) -> None:
    metric_columns = ["accuracy", "macro_f1", "macro_auc"]
    metric_titles = {
        "accuracy": "Accuracy",
        "macro_f1": "Macro-F1",
        "macro_auc": "Macro-AUC",
    }

    for metric in metric_columns:
        figure, axis = plt.subplots(figsize=(8, 5))
        values = 100.0 * fold_metrics_df[metric].to_numpy()
        bars = axis.bar(fold_metrics_df["fold"].astype(str), values)
        axis.set_title(
            f"APTOS 2019 EfficientNet-B0: {metric_titles[metric]} Across Folds",
            fontweight="bold",
        )
        axis.set_xlabel("Fold", fontweight="bold")
        axis.set_ylabel(f"{metric_titles[metric]} (%)", fontweight="bold")
        axis.set_ylim(0, max(100.0, values.max() + 5.0))
        axis.grid(axis="y", alpha=0.3)

        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{value:.2f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        apply_bold_axis_style(axis)
        figure.tight_layout()
        figure.savefig(FIGURE_DIR / f"fold_{metric}_comparison.png", dpi=300, bbox_inches="tight")
        plt.close(figure)


# ============================================================
# Main cross-validation procedure
# ============================================================


def main() -> None:
    if not FOLD_FILE.is_file():
        raise FileNotFoundError(f"Fold assignment file not found: {FOLD_FILE}")

    dataframe = pd.read_csv(FOLD_FILE)
    required_columns = {"id_code", "diagnosis", "image_path", "fold"}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")

    dataframe["diagnosis"] = dataframe["diagnosis"].astype(int)
    dataframe["fold"] = dataframe["fold"].astype(int)

    print("=" * 80)
    print("APTOS 2019 EFFICIENTNET-B0 FIVE-FOLD TRAINING")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Images: {len(dataframe):,}")
    print(f"Folds: {N_FOLDS}")
    print(f"Epochs per fold: {EPOCHS}")
    print(f"Output directory: {OUTPUT_ROOT}")

    all_fold_predictions: List[pd.DataFrame] = []
    all_fold_metrics: List[Dict[str, float]] = []

    for fold_index in range(N_FOLDS):
        fold_seed = SEED + fold_index
        set_seed(fold_seed)

        print("\n" + "=" * 80)
        print(f"STARTING FOLD {fold_index}")
        print("=" * 80)

        train_df = dataframe.loc[dataframe["fold"] != fold_index].reset_index(drop=True)
        valid_df = dataframe.loc[dataframe["fold"] == fold_index].reset_index(drop=True)
        train_loader, valid_loader, class_weights = make_loaders(
            train_df, valid_df, fold_seed
        )

        model = build_model()
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32, device=DEVICE),
            label_smoothing=LABEL_SMOOTHING,
        )
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
        scaler = GradScaler(enabled=AMP_ENABLED)

        best_macro_f1 = -np.inf
        best_epoch = -1
        epochs_without_improvement = 0
        history_rows: List[Dict[str, float]] = []
        checkpoint_path = CHECKPOINT_DIR / f"efficientnet_b0_fold_{fold_index}.pt"
        fold_start_time = time.time()

        for epoch_index in range(1, EPOCHS + 1):
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler, epoch_index, fold_index
            )
            valid_loss, labels, probabilities, sample_ids = validate_one_epoch(
                model, valid_loader, criterion, epoch_index, fold_index
            )
            metrics = compute_metrics(labels, probabilities)
            scheduler.step()

            history_row = {
                "fold": fold_index,
                "epoch": epoch_index,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
                **{f"valid_{key}": value for key, value in metrics.items()},
            }
            history_rows.append(history_row)

            print(
                f"Fold {fold_index} | Epoch {epoch_index:02d}/{EPOCHS} | "
                f"Train Loss {train_loss:.4f} | Valid Loss {valid_loss:.4f} | "
                f"Acc {metrics['accuracy']:.4f} | F1 {metrics['macro_f1']:.4f} | "
                f"AUC {metrics['macro_auc']:.4f}"
            )

            if metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = metrics["macro_f1"]
                best_epoch = epoch_index
                epochs_without_improvement = 0
                torch.save(
                    {
                        "fold": fold_index,
                        "epoch": epoch_index,
                        "model_name": MODEL_NAME,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_macro_f1": best_macro_f1,
                        "class_weights": class_weights.tolist(),
                        "configuration": {
                            "image_size": IMAGE_SIZE,
                            "batch_size": BATCH_SIZE,
                            "learning_rate": LEARNING_RATE,
                            "weight_decay": WEIGHT_DECAY,
                            "label_smoothing": LABEL_SMOOTHING,
                            "seed": fold_seed,
                        },
                    },
                    checkpoint_path,
                )
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping at epoch {epoch_index}; best epoch was {best_epoch}.")
                break

        history_df = pd.DataFrame(history_rows)
        history_df.to_csv(METRIC_DIR / f"fold_{fold_index}_training_history.csv", index=False)
        plot_training_history(history_df, fold_index)

        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        _, labels, probabilities, sample_ids = validate_one_epoch(
            model, valid_loader, criterion, best_epoch, fold_index
        )
        predictions = probabilities.argmax(axis=1)
        fold_metrics = compute_metrics(labels, probabilities)
        fold_metrics.update(
            {
                "fold": fold_index,
                "best_epoch": best_epoch,
                "training_seconds": time.time() - fold_start_time,
            }
        )
        all_fold_metrics.append(fold_metrics)

        fold_prediction_df = pd.DataFrame(
            {
                "id_code": sample_ids,
                "true_label": labels,
                "predicted_label": predictions,
                "fold": fold_index,
                "model": "EfficientNet-B0",
            }
        )
        for class_index in range(NUM_CLASSES):
            fold_prediction_df[f"probability_class_{class_index}"] = probabilities[:, class_index]

        fold_prediction_df.to_csv(
            PREDICTION_DIR / f"efficientnet_b0_fold_{fold_index}_predictions.csv",
            index=False,
        )
        all_fold_predictions.append(fold_prediction_df)

        report = classification_report(
            labels,
            predictions,
            labels=np.arange(NUM_CLASSES),
            target_names=[CLASS_NAMES[index] for index in range(NUM_CLASSES)],
            output_dict=True,
            zero_division=0,
        )
        with open(METRIC_DIR / f"fold_{fold_index}_classification_report.json", "w") as file:
            json.dump(report, file, indent=2)

        print(
            f"Fold {fold_index} completed | Best epoch {best_epoch} | "
            f"Accuracy {fold_metrics['accuracy']:.4f} | "
            f"Macro-F1 {fold_metrics['macro_f1']:.4f} | "
            f"Macro-AUC {fold_metrics['macro_auc']:.4f}"
        )

        del model, optimizer, scheduler, scaler, train_loader, valid_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    oof_predictions = pd.concat(all_fold_predictions, ignore_index=True)
    oof_predictions = oof_predictions.sort_values("id_code").reset_index(drop=True)
    oof_predictions.to_csv(PREDICTION_DIR / "efficientnet_b0_oof_predictions.csv", index=False)

    probability_columns = [f"probability_class_{index}" for index in range(NUM_CLASSES)]
    oof_labels = oof_predictions["true_label"].to_numpy(dtype=np.int64)
    oof_probabilities = oof_predictions[probability_columns].to_numpy(dtype=np.float64)
    oof_predictions_array = oof_predictions["predicted_label"].to_numpy(dtype=np.int64)
    oof_metrics = compute_metrics(oof_labels, oof_probabilities)

    fold_metrics_df = pd.DataFrame(all_fold_metrics).sort_values("fold")
    fold_metrics_df.to_csv(METRIC_DIR / "efficientnet_b0_fold_metrics.csv", index=False)

    metric_summary_rows = []
    for metric in ["accuracy", "macro_f1", "macro_precision", "macro_recall", "macro_auc"]:
        values = fold_metrics_df[metric].to_numpy(dtype=np.float64)
        metric_summary_rows.append(
            {
                "metric": metric,
                "mean": values.mean(),
                "standard_deviation": values.std(ddof=1),
                "minimum": values.min(),
                "maximum": values.max(),
                "oof_value": oof_metrics[metric],
            }
        )

    metric_summary_df = pd.DataFrame(metric_summary_rows)
    metric_summary_df.to_csv(METRIC_DIR / "efficientnet_b0_metric_summary.csv", index=False)

    with open(METRIC_DIR / "efficientnet_b0_oof_metrics.json", "w") as file:
        json.dump(oof_metrics, file, indent=2)

    plot_confusion_matrix(oof_labels, oof_predictions_array)
    plot_roc_curves(oof_labels, oof_probabilities)
    plot_fold_metric_bars(fold_metrics_df)

    print("\n" + "=" * 80)
    print("FIVE-FOLD EFFICIENTNET-B0 TRAINING COMPLETED")
    print("=" * 80)
    print(f"OOF Accuracy:        {100 * oof_metrics['accuracy']:.2f}%")
    print(f"OOF Macro-F1:        {100 * oof_metrics['macro_f1']:.2f}%")
    print(f"OOF Macro-Precision: {100 * oof_metrics['macro_precision']:.2f}%")
    print(f"OOF Macro-Recall:    {100 * oof_metrics['macro_recall']:.2f}%")
    print(f"OOF Macro-AUC:       {100 * oof_metrics['macro_auc']:.2f}%")
    print(f"Predictions: {PREDICTION_DIR / 'efficientnet_b0_oof_predictions.csv'}")
    print(f"Metrics:     {METRIC_DIR}")
    print(f"Figures:     {FIGURE_DIR}")


if __name__ == "__main__":
    main()
