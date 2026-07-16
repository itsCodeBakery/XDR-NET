"""APTOS 2019 data preparation for Reviewer 1 statistical analysis.

This script:
1. verifies the Kaggle APTOS files;
2. creates fixed stratified five-fold assignments;
3. implements circular cropping and CLAHE preprocessing;
4. builds PyTorch datasets and data loaders;
5. calculates effective-number class weights;
6. generates quality-control figures with bold titles and bold data labels;
7. saves all revision outputs under /kaggle/working/xdrnet_aptos.

APTOS does not provide patient identifiers, so the generated folds are
stratified image-level folds rather than patient-level folds.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, Tuple

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SEED = 42
N_FOLDS = 5
SELECTED_FOLD = 0
IMAGE_SIZE = 384
BATCH_SIZE = 16
NUM_WORKERS = 2
NUM_CLASSES = 5
EFFECTIVE_NUMBER_BETA = 0.999

TRAIN_CSV = Path(
    "/kaggle/input/competitions/aptos2019-blindness-detection/train.csv"
)
TEST_CSV = Path(
    "/kaggle/input/competitions/aptos2019-blindness-detection/test.csv"
)
TRAIN_IMAGE_DIR = Path(
    "/kaggle/input/competitions/aptos2019-blindness-detection/train_images"
)
TEST_IMAGE_DIR = Path(
    "/kaggle/input/competitions/aptos2019-blindness-detection/test_images"
)
OUTPUT_DIR = Path("/kaggle/working/xdrnet_aptos")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES: Dict[int, str] = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


set_seed(SEED)
LOADER_GENERATOR = torch.Generator().manual_seed(SEED)


# -----------------------------------------------------------------------------
# Dataset verification and folds
# -----------------------------------------------------------------------------
def verify_required_paths() -> None:
    required = {
        "Training CSV": TRAIN_CSV,
        "Test CSV": TEST_CSV,
        "Training images": TRAIN_IMAGE_DIR,
        "Test images": TEST_IMAGE_DIR,
    }
    print("=" * 78)
    print("VERIFYING APTOS DATASET PATHS")
    print("=" * 78)
    for name, path in required.items():
        status = "FOUND" if path.exists() else "MISSING"
        print(f"{name:<25}: {status} — {path}")
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")


def load_metadata() -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    if not {"id_code", "diagnosis"}.issubset(train_df.columns):
        raise ValueError("train.csv must contain id_code and diagnosis columns.")
    if "id_code" not in test_df.columns:
        raise ValueError("test.csv must contain an id_code column.")

    train_df["diagnosis"] = train_df["diagnosis"].astype(int)
    train_df["image_path"] = train_df["id_code"].map(
        lambda image_id: str(TRAIN_IMAGE_DIR / f"{image_id}.png")
    )
    test_df["image_path"] = test_df["id_code"].map(
        lambda image_id: str(TEST_IMAGE_DIR / f"{image_id}.png")
    )
    return train_df, test_df


def verify_images(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    tqdm.pandas(desc="Checking APTOS training images")
    train_exists = train_df["image_path"].progress_apply(
        lambda path: Path(path).is_file()
    )
    tqdm.pandas(desc="Checking APTOS test images")
    test_exists = test_df["image_path"].progress_apply(
        lambda path: Path(path).is_file()
    )

    if not train_exists.all():
        missing = train_df.loc[~train_exists, "image_path"].tolist()
        raise FileNotFoundError(f"Missing training images: {missing[:10]}")
    if not test_exists.all():
        missing = test_df.loc[~test_exists, "image_path"].tolist()
        raise FileNotFoundError(f"Missing test images: {missing[:10]}")


def create_or_load_folds(train_df: pd.DataFrame) -> pd.DataFrame:
    fold_path = OUTPUT_DIR / "aptos_5fold_assignments.csv"
    if fold_path.exists():
        print(f"Loading existing fixed folds: {fold_path}")
        folds = pd.read_csv(fold_path)
        required = {"id_code", "diagnosis", "image_path", "fold"}
        if not required.issubset(folds.columns):
            raise ValueError("Existing fold file has missing columns.")
        return folds

    train_df = train_df.copy()
    train_df["fold"] = -1
    splitter = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=SEED,
    )
    iterator = splitter.split(train_df, train_df["diagnosis"])
    for fold, (_, valid_indices) in tqdm(
        enumerate(iterator), total=N_FOLDS, desc="Assigning stratified folds"
    ):
        train_df.loc[valid_indices, "fold"] = fold

    train_df["fold"] = train_df["fold"].astype(int)
    train_df[["id_code", "diagnosis", "image_path", "fold"]].to_csv(
        fold_path, index=False
    )
    return train_df


# -----------------------------------------------------------------------------
# Bold plotting utilities
# -----------------------------------------------------------------------------
def add_bold_bar_labels(axis: plt.Axes, decimals: int = 0) -> None:
    """Attach bold numeric labels above every bar in an axis."""
    for container in axis.containers:
        labels = []
        for bar in container:
            height = bar.get_height()
            if np.isnan(height):
                labels.append("")
            elif decimals == 0:
                labels.append(f"{height:.0f}")
            else:
                labels.append(f"{height:.{decimals}f}")
        axis.bar_label(
            container,
            labels=labels,
            padding=3,
            fontsize=10,
            fontweight="bold",
        )


def style_axis_bold(axis: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    axis.set_title(title, fontsize=14, fontweight="bold")
    axis.set_xlabel(xlabel, fontsize=11, fontweight="bold")
    axis.set_ylabel(ylabel, fontsize=11, fontweight="bold")
    axis.tick_params(axis="both", labelsize=10)
    for tick in axis.get_xticklabels() + axis.get_yticklabels():
        tick.set_fontweight("bold")


def save_class_distribution_figure(fold_df: pd.DataFrame) -> Path:
    counts = fold_df["diagnosis"].value_counts().sort_index()
    labels = [CLASS_NAMES[index] for index in counts.index]

    figure, axis = plt.subplots(figsize=(11, 6))
    axis.bar(labels, counts.values)
    style_axis_bold(
        axis,
        "APTOS 2019 Class Distribution",
        "Diabetic Retinopathy Grade",
        "Number of Images",
    )
    add_bold_bar_labels(axis)
    plt.setp(axis.get_xticklabels(), rotation=15, ha="right", fontweight="bold")
    figure.tight_layout()

    output = OUTPUT_DIR / "aptos_class_distribution_bold.png"
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(figure)
    return output


def save_fold_distribution_figure(fold_df: pd.DataFrame) -> Path:
    table = pd.crosstab(fold_df["fold"], fold_df["diagnosis"])
    figure, axis = plt.subplots(figsize=(12, 7))
    table.plot(kind="bar", ax=axis)
    style_axis_bold(
        axis,
        "APTOS 2019 Stratified Class Distribution Across Five Folds",
        "Fold",
        "Number of Images",
    )
    axis.legend(
        [CLASS_NAMES[index] for index in table.columns],
        title="DR Grade",
        prop={"weight": "bold", "size": 9},
        title_fontproperties={"weight": "bold", "size": 10},
    )
    add_bold_bar_labels(axis)
    figure.tight_layout()

    output = OUTPUT_DIR / "aptos_fold_distribution_bold.png"
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(figure)
    return output


# -----------------------------------------------------------------------------
# Fundus preprocessing
# -----------------------------------------------------------------------------
def read_rgb_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def remove_black_border(image: np.ndarray, threshold: int = 7) -> np.ndarray:
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = (grayscale > threshold).astype(np.uint8)
    coordinates = cv2.findNonZero(mask)
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
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    luminance, channel_a, channel_b = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size,
    )
    enhanced = clahe.apply(luminance)
    merged = cv2.merge((enhanced, channel_a, channel_b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def preprocess_fundus(image: np.ndarray) -> np.ndarray:
    image = remove_black_border(image)
    image = circular_crop(image)
    return apply_clahe_lab(image)


TRAIN_TRANSFORM = A.Compose(
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
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]
)

VALID_TRANSFORM = A.Compose(
    [
        A.Resize(IMAGE_SIZE, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]
)


class AptosDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        transform: A.Compose,
        apply_preprocessing: bool = True,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.transform = transform
        self.apply_preprocessing = apply_preprocessing

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> dict:
        row = self.dataframe.iloc[index]
        image = read_rgb_image(row["image_path"])
        if self.apply_preprocessing:
            image = preprocess_fundus(image)
        image = self.transform(image=image)["image"]
        return {
            "image": image,
            "label": torch.tensor(int(row["diagnosis"]), dtype=torch.long),
            "id_code": row["id_code"],
            "fold": int(row["fold"]),
        }


def effective_number_weights(
    labels: np.ndarray,
    number_of_classes: int = NUM_CLASSES,
    beta: float = EFFECTIVE_NUMBER_BETA,
) -> np.ndarray:
    class_counts = np.bincount(labels, minlength=number_of_classes).astype(float)
    if np.any(class_counts == 0):
        raise ValueError("At least one class has no training examples.")
    effective_numbers = (1.0 - np.power(beta, class_counts)) / (1.0 - beta)
    weights = 1.0 / effective_numbers
    weights /= weights.mean()
    return weights.astype(np.float32)


def build_loaders(fold_df: pd.DataFrame) -> Tuple[DataLoader, DataLoader, np.ndarray]:
    train_df = fold_df.loc[fold_df["fold"] != SELECTED_FOLD].reset_index(drop=True)
    valid_df = fold_df.loc[fold_df["fold"] == SELECTED_FOLD].reset_index(drop=True)

    train_dataset = AptosDataset(train_df, TRAIN_TRANSFORM)
    valid_dataset = AptosDataset(valid_df, VALID_TRANSFORM)

    labels = train_df["diagnosis"].to_numpy(dtype=np.int64)
    class_weights = effective_number_weights(labels)
    sample_weights = torch.as_tensor(class_weights[labels], dtype=torch.double)
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=NUM_WORKERS > 0,
        worker_init_fn=seed_worker,
        generator=LOADER_GENERATOR,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=NUM_WORKERS > 0,
        worker_init_fn=seed_worker,
        generator=LOADER_GENERATOR,
        drop_last=False,
    )
    return train_loader, valid_loader, class_weights


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
    std = np.asarray(IMAGENET_STD, dtype=np.float32)
    return np.clip(image * std + mean, 0.0, 1.0)


def save_preprocessing_preview(valid_loader: DataLoader) -> Path:
    dataset = valid_loader.dataset
    dataframe = dataset.dataframe
    figure, axes = plt.subplots(1, NUM_CLASSES, figsize=(20, 4.5))

    for class_index, axis in enumerate(axes):
        matching = dataframe.index[dataframe["diagnosis"] == class_index].tolist()
        if not matching:
            axis.axis("off")
            continue
        sample = dataset[matching[0]]
        axis.imshow(denormalize_image(sample["image"]))
        axis.set_title(
            f"{class_index}: {CLASS_NAMES[class_index]}",
            fontsize=11,
            fontweight="bold",
        )
        axis.axis("off")

    figure.suptitle(
        "APTOS 2019 Deterministically Preprocessed Validation Images",
        fontsize=16,
        fontweight="bold",
    )
    figure.tight_layout()
    output = OUTPUT_DIR / "aptos_preprocessing_preview_bold.png"
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(figure)
    return output


def save_tables(fold_df: pd.DataFrame, class_weights: np.ndarray) -> None:
    class_distribution = (
        fold_df["diagnosis"]
        .value_counts()
        .sort_index()
        .rename_axis("diagnosis")
        .reset_index(name="count")
    )
    class_distribution["class_name"] = class_distribution["diagnosis"].map(
        CLASS_NAMES
    )
    class_distribution["percentage"] = (
        100.0 * class_distribution["count"] / len(fold_df)
    )
    class_distribution.to_csv(
        OUTPUT_DIR / "aptos_class_distribution.csv", index=False
    )

    fold_distribution = pd.crosstab(fold_df["fold"], fold_df["diagnosis"])
    fold_distribution.to_csv(OUTPUT_DIR / "aptos_fold_distribution.csv")

    weights_df = pd.DataFrame(
        {
            "class": range(NUM_CLASSES),
            "class_name": [CLASS_NAMES[index] for index in range(NUM_CLASSES)],
            "effective_number_weight": class_weights,
        }
    )
    weights_df.to_csv(
        OUTPUT_DIR / f"aptos_fold_{SELECTED_FOLD}_class_weights.csv", index=False
    )


def main() -> None:
    verify_required_paths()
    train_df, test_df = load_metadata()
    verify_images(train_df, test_df)
    fold_df = create_or_load_folds(train_df)

    class_figure = save_class_distribution_figure(fold_df)
    fold_figure = save_fold_distribution_figure(fold_df)

    train_loader, valid_loader, class_weights = build_loaders(fold_df)
    preview = save_preprocessing_preview(valid_loader)
    save_tables(fold_df, class_weights)

    train_batch = next(iter(train_loader))
    valid_batch = next(iter(valid_loader))

    print("\n" + "=" * 78)
    print("APTOS DATA PIPELINE COMPLETED")
    print("=" * 78)
    print(f"Training images in fold {SELECTED_FOLD} setup: {len(train_loader.dataset):,}")
    print(f"Validation images: {len(valid_loader.dataset):,}")
    print(f"Training batch shape: {tuple(train_batch['image'].shape)}")
    print(f"Validation batch shape: {tuple(valid_batch['image'].shape)}")
    print("Effective-number class weights:")
    for class_index, weight in enumerate(class_weights):
        print(f"  {class_index} ({CLASS_NAMES[class_index]}): {weight:.6f}")
    print("Saved figures:")
    print(f"  {class_figure}")
    print(f"  {fold_figure}")
    print(f"  {preview}")
    print(f"All revision outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
