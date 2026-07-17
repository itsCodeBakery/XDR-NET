# Reviewer 1, Comment 1: train XDR-Net on the common leakage-free APTOS split.
# This script is intended for Kaggle and assumes Blocks 1-8 have already defined
# model, train_loader, validation_loader, criterion, device, and output directories.

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm.auto import tqdm

NUM_EPOCHS = 20
INITIAL_LR = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 6
GRADIENT_CLIP_NORM = 1.0

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=INITIAL_LR,
    weight_decay=WEIGHT_DECAY,
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=NUM_EPOCHS,
)

scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())


def run_epoch(model, loader, criterion, optimizer=None, scaler=None, training=False):
    model.train(training)
    running_loss = 0.0
    all_true, all_pred = [], []

    progress = tqdm(
        loader,
        desc="Training" if training else "Validation",
        leave=False,
        dynamic_ncols=True,
    )

    for batch in progress:
        if training:
            images, labels = batch
        else:
            images, labels, _ = batch

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(images)
                loss = criterion(logits, labels)

            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=GRADIENT_CLIP_NORM,
                )
                scaler.step(optimizer)
                scaler.update()

        predictions = logits.argmax(dim=1)
        running_loss += loss.item() * images.size(0)
        all_true.extend(labels.detach().cpu().numpy())
        all_pred.extend(predictions.detach().cpu().numpy())

        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{accuracy_score(all_true, all_pred):.4f}",
            macro_f1=f"{f1_score(all_true, all_pred, average='macro', zero_division=0):.4f}",
        )

    epoch_loss = running_loss / len(loader.dataset)
    metrics = {
        "loss": epoch_loss,
        "accuracy": accuracy_score(all_true, all_pred),
        "macro_f1": f1_score(all_true, all_pred, average="macro", zero_division=0),
        "macro_precision": precision_score(all_true, all_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(all_true, all_pred, average="macro", zero_division=0),
    }
    return metrics


history = []
best_macro_f1 = -np.inf
best_epoch = -1
epochs_without_improvement = 0
best_checkpoint_path = CHECKPOINT_DIR / "xdrnet_common_split_best.pth"

outer_progress = tqdm(
    range(1, NUM_EPOCHS + 1),
    desc="XDR-Net epochs",
    dynamic_ncols=True,
)

for epoch in outer_progress:
    epoch_start = time.time()

    train_metrics = run_epoch(
        model=model,
        loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        scaler=scaler,
        training=True,
    )

    validation_metrics = run_epoch(
        model=model,
        loader=validation_loader,
        criterion=criterion,
        training=False,
    )

    scheduler.step()
    elapsed_seconds = time.time() - epoch_start

    row = {
        "epoch": epoch,
        "learning_rate": optimizer.param_groups[0]["lr"],
        "elapsed_seconds": elapsed_seconds,
        **{f"train_{key}": value for key, value in train_metrics.items()},
        **{f"val_{key}": value for key, value in validation_metrics.items()},
    }
    history.append(row)

    pd.DataFrame(history).to_csv(
        LOG_DIR / "xdrnet_training_history.csv",
        index=False,
    )

    current_macro_f1 = validation_metrics["macro_f1"]

    if current_macro_f1 > best_macro_f1:
        best_macro_f1 = current_macro_f1
        best_epoch = epoch
        epochs_without_improvement = 0

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_validation_macro_f1": best_macro_f1,
                "configuration": {
                    "num_epochs": NUM_EPOCHS,
                    "initial_lr": INITIAL_LR,
                    "weight_decay": WEIGHT_DECAY,
                    "patience": PATIENCE,
                    "gradient_clip_norm": GRADIENT_CLIP_NORM,
                    "seed": SEED,
                    "image_size": IMAGE_SIZE,
                    "batch_size": BATCH_SIZE,
                    "effective_number_beta": EFFECTIVE_NUMBER_BETA,
                    "label_smoothing": LABEL_SMOOTHING,
                },
            },
            best_checkpoint_path,
        )
    else:
        epochs_without_improvement += 1

    outer_progress.set_postfix(
        best_epoch=best_epoch,
        best_macro_f1=f"{best_macro_f1:.4f}",
        val_acc=f"{validation_metrics['accuracy']:.4f}",
        val_macro_f1=f"{current_macro_f1:.4f}",
    )

    if epochs_without_improvement >= PATIENCE:
        tqdm.write(
            f"Early stopping at epoch {epoch}; best epoch was {best_epoch}."
        )
        break

summary = {
    "best_epoch": int(best_epoch),
    "best_validation_macro_f1": float(best_macro_f1),
    "checkpoint_path": str(best_checkpoint_path),
    "epochs_completed": int(len(history)),
}

with open(LOG_DIR / "xdrnet_training_summary.json", "w") as file:
    json.dump(summary, file, indent=4)

print("Training completed.")
print(json.dumps(summary, indent=4))
