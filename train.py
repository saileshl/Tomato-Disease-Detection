"""
train.py — End-to-End Training Pipeline for Tomato Disease Detection
====================================================================
• Loads images from  PlantVillage/  (class-per-folder layout)
• Applies real-time data augmentation via ImageDataGenerator
• Trains a MobileNetV2-based model (from model.py) in two phases:
      Phase 1 – Feature extraction (frozen backbone)
      Phase 2 – Fine-tuning (last N layers unfrozen)
• Saves:
      tomato_model.h5              – final trained weights
      training_history.png         – accuracy & loss curves
      confusion_matrix.png         – test-set confusion matrix
      classification_report.txt    – precision / recall / F1
"""

from __future__ import annotations

import io
import os
import sys
import json
from pathlib import Path

# Force UTF-8 output on Windows to avoid cp1252 emoji crashes
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend — no GUI needed
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from model import build_model, unfreeze_and_finetune, IMG_SIZE, NUM_CLASSES, CLASS_NAMES

# ── Paths ────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "PlantVillage"
MODEL_PATH = BASE_DIR / "tomato_model.h5"
HISTORY_PLOT = BASE_DIR / "training_history.png"
CM_PLOT = BASE_DIR / "confusion_matrix.png"
REPORT_TXT = BASE_DIR / "classification_report.txt"

# ── Hyper-parameters ────────────────────────────────────────
BATCH_SIZE = 64
EPOCHS_PHASE1 = 3           # feature-extraction phase
EPOCHS_PHASE2 = 2           # fine-tuning phase
VALIDATION_SPLIT = 0.2
SEED = 42


# ═══════════════════════════════════════════════════════════
#  1. DATA LOADING & AUGMENTATION
# ═══════════════════════════════════════════════════════════
def load_data():
    """Create augmented training and validation generators."""
    if not DATA_DIR.is_dir():
        sys.exit(
            f"❌  Dataset not found at {DATA_DIR}\n"
            "   Run `python data_downloader.py` first."
        )

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
        validation_split=VALIDATION_SPLIT,
    )

    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=VALIDATION_SPLIT,
    )

    common_args = dict(
        directory=str(DATA_DIR),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        seed=SEED,
    )

    train_gen = train_datagen.flow_from_directory(
        **common_args, subset="training", shuffle=True
    )
    val_gen = val_datagen.flow_from_directory(
        **common_args, subset="validation", shuffle=False
    )

    # Print discovered classes
    idx_to_class = {v: k for k, v in train_gen.class_indices.items()}
    print("\n📂  Discovered classes:")
    for i in sorted(idx_to_class):
        print(f"    {i}: {idx_to_class[i]}")
    print(f"\n  Training samples  : {train_gen.samples}")
    print(f"  Validation samples: {val_gen.samples}\n")

    return train_gen, val_gen, idx_to_class


# ═══════════════════════════════════════════════════════════
#  2. TRAINING
# ═══════════════════════════════════════════════════════════
def train():
    train_gen, val_gen, idx_to_class = load_data()
    num_classes = len(idx_to_class)

    # ── Phase 1: Feature extraction (frozen backbone) ───────
    print("=" * 60)
    print("  PHASE 1 — Feature Extraction")
    print("=" * 60)
    model = build_model(num_classes=num_classes, img_size=IMG_SIZE, freeze_base=True)

    callbacks_p1 = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=1
        ),
    ]

    history_p1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_PHASE1,
        callbacks=callbacks_p1,
        verbose=1,
    )

    # ── Phase 2: Fine-tuning ───────────────────────────────
    print("\n" + "=" * 60)
    print("  PHASE 2 — Fine-Tuning (last 30 layers)")
    print("=" * 60)
    model = unfreeze_and_finetune(model, num_layers_to_unfreeze=30, learning_rate=1e-5)

    callbacks_p2 = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history_p2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_PHASE2,
        callbacks=callbacks_p2,
        verbose=1,
    )

    # Merge histories for plotting
    history = {}
    for key in history_p1.history:
        history[key] = history_p1.history[key] + history_p2.history[key]

    # Save the final model (if checkpoint didn't already)
    model.save(str(MODEL_PATH))
    print(f"\n💾  Model saved → {MODEL_PATH}")

    # Save class-index mapping alongside the model
    class_map_path = BASE_DIR / "class_indices.json"
    with open(class_map_path, "w") as f:
        json.dump(idx_to_class, f, indent=2)
    print(f"📋  Class indices saved → {class_map_path}")

    return model, history, val_gen, idx_to_class


# ═══════════════════════════════════════════════════════════
#  3. EVALUATION & PLOTS
# ═══════════════════════════════════════════════════════════
def plot_history(history: dict) -> None:
    """Plot training vs validation accuracy & loss and save to disk."""
    epochs_range = range(1, len(history["accuracy"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    ax1.plot(epochs_range, history["accuracy"], label="Train Accuracy", linewidth=2)
    ax1.plot(epochs_range, history["val_accuracy"], label="Val Accuracy", linewidth=2)
    ax1.set_title("Model Accuracy", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # Loss
    ax2.plot(epochs_range, history["loss"], label="Train Loss", linewidth=2)
    ax2.plot(epochs_range, history["val_loss"], label="Val Loss", linewidth=2)
    ax2.set_title("Model Loss", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(HISTORY_PLOT), dpi=150)
    plt.close()
    print(f"📈  Training curves saved → {HISTORY_PLOT}")


def evaluate(model, val_gen, idx_to_class: dict) -> None:
    """Generate confusion matrix, classification report, and print metrics."""
    print("\n" + "=" * 60)
    print("  EVALUATION")
    print("=" * 60)

    # Reset the generator and predict
    val_gen.reset()
    y_pred_proba = model.predict(val_gen, verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = val_gen.classes[: len(y_pred)]

    class_names = [idx_to_class[i] for i in sorted(idx_to_class.keys())]

    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print("\n" + report)
    with open(REPORT_TXT, "w") as f:
        f.write("Tomato Disease Detection — Classification Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
    print(f"📝  Report saved → {REPORT_TXT}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, values_format="d")
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(CM_PLOT), dpi=150)
    plt.close()
    print(f"🔢  Confusion matrix saved → {CM_PLOT}")

    # Overall accuracy
    acc = np.mean(y_pred == y_true)
    print(f"\n🎯  Overall Validation Accuracy: {acc:.4f}  ({acc*100:.2f}%)")


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════
def main() -> None:
    print("=" * 62)
    print("  🍅 Tomato Disease Detection — Training Pipeline")
    print("=" * 62)

    # Detect GPU
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"  🟢 GPU detected: {gpus[0].name}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("  🟡 No GPU detected — training on CPU (will be slow)")

    model, history, val_gen, idx_to_class = train()
    plot_history(history)
    evaluate(model, val_gen, idx_to_class)

    print("\n" + "=" * 62)
    print("  ✅ Pipeline complete!")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    main()
