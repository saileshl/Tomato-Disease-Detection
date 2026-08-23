"""
model.py — CNN Architecture for Tomato Disease Detection
=========================================================
Uses MobileNetV2 (transfer-learning) as the feature extractor,
topped with custom dense layers for 10-class classification.

Classes
-------
    0  Bacterial_spot
    1  Early_blight
    2  Late_blight
    3  Leaf_Mold
    4  Septoria_leaf_spot
    5  Spider_mites_(Two-spotted_spider_mite)
    6  Target_Spot
    7  Tomato_Yellow_Leaf_Curl_Virus
    8  Tomato_mosaic_virus
    9  healthy
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

# ── Constants ────────────────────────────────────────────────
IMG_SIZE: int = 224
NUM_CLASSES: int = 10

CLASS_NAMES: list[str] = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy",
]


# ── Model builder ───────────────────────────────────────────
def build_model(
    num_classes: int = NUM_CLASSES,
    img_size: int = IMG_SIZE,
    freeze_base: bool = True,
) -> tf.keras.Model:
    """
    Build a MobileNetV2-based classifier.

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    img_size : int
        Spatial dimension of input images (square).
    freeze_base : bool
        If True the MobileNetV2 backbone is frozen (feature-extraction mode).

    Returns
    -------
    tf.keras.Model  –  compiled model ready for `.fit()`.
    """
    base_model = MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = not freeze_base

    model = models.Sequential(
        [
            layers.InputLayer(input_shape=(img_size, img_size, 3)),
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def unfreeze_and_finetune(
    model: tf.keras.Model,
    num_layers_to_unfreeze: int = 30,
    learning_rate: float = 1e-5,
) -> tf.keras.Model:
    """
    Unfreeze the last *n* layers of the MobileNetV2 backbone and
    recompile with a lower learning rate for fine-tuning.
    """
    # Find the MobileNetV2 base dynamically (it's a Functional model inside Sequential)
    base = None
    for layer in model.layers:
        if hasattr(layer, "layers") and len(getattr(layer, "layers", [])) > 10:
            base = layer
            break

    if base is None:
        raise RuntimeError("Could not find MobileNetV2 base in model.layers")

    base.trainable = True
    for layer in base.layers[: -num_layers_to_unfreeze]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── Quick sanity check ──────────────────────────────────────
if __name__ == "__main__":
    m = build_model()
    m.summary()
    print(f"\n✅  Model built — {m.count_params():,} total parameters")
