"""
main.py — FastAPI Backend for Tomato Disease Detection
======================================================
Exposes:
    GET  /          → serves the frontend UI
    POST /predict   → accepts an image upload, returns disease + confidence

The main prediction class is named **Hello** per project spec.
"""

from __future__ import annotations

import io
import os
import sys
import json
from pathlib import Path
from typing import Optional

# Force UTF-8 output on Windows to avoid cp1252 emoji crashes
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Suppress noisy TF logs before import
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf  # noqa: E402

# ── Paths ────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "tomato_model.h5"
CLASS_MAP_PATH = BASE_DIR / "class_indices.json"
TEMPLATE_PATH = BASE_DIR / "templates" / "index.html"

# ── Default class names (used if class_indices.json not found) ─
DEFAULT_CLASS_NAMES: dict[int, str] = {
    0: "Tomato_Bacterial_spot",
    1: "Tomato_Early_blight",
    2: "Tomato_Late_blight",
    3: "Tomato_Leaf_Mold",
    4: "Tomato_Septoria_leaf_spot",
    5: "Tomato_Spider_mites_Two_spotted_spider_mite",
    6: "Tomato__Target_Spot",
    7: "Tomato__Tomato_YellowLeaf__Curl_Virus",
    8: "Tomato__Tomato_mosaic_virus",
    9: "Tomato_healthy",
}

# Human-readable disease descriptions + basic remedies
DISEASE_INFO: dict[str, dict] = {
    "Tomato_Bacterial_spot": {
        "display": "Bacterial Spot",
        "severity": "Medium",
        "remedy": "Apply copper-based bactericides. Remove and destroy affected leaves. Avoid overhead watering.",
    },
    "Tomato_Early_blight": {
        "display": "Early Blight",
        "severity": "High",
        "remedy": "Apply chlorothalonil or mancozeb fungicide. Practice crop rotation. Remove lower infected leaves.",
    },
    "Tomato_Late_blight": {
        "display": "Late Blight",
        "severity": "Critical",
        "remedy": "Apply metalaxyl-based fungicide immediately. Remove infected plants. Ensure good air circulation.",
    },
    "Tomato_Leaf_Mold": {
        "display": "Leaf Mold",
        "severity": "Medium",
        "remedy": "Improve ventilation in greenhouses. Apply fungicides containing chlorothalonil. Reduce humidity.",
    },
    "Tomato_Septoria_leaf_spot": {
        "display": "Septoria Leaf Spot",
        "severity": "Medium",
        "remedy": "Apply mancozeb or copper-based fungicide. Remove infected foliage. Mulch around plants.",
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "display": "Spider Mites",
        "severity": "Medium",
        "remedy": "Spray neem oil or insecticidal soap. Increase humidity. Introduce predatory mites for biological control.",
    },
    "Tomato__Target_Spot": {
        "display": "Target Spot",
        "severity": "Medium",
        "remedy": "Apply chlorothalonil-based fungicide. Improve air flow. Practice crop rotation.",
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "display": "Yellow Leaf Curl Virus",
        "severity": "Critical",
        "remedy": "Remove and destroy infected plants. Control whitefly populations with insecticides or sticky traps.",
    },
    "Tomato__Tomato_mosaic_virus": {
        "display": "Mosaic Virus",
        "severity": "High",
        "remedy": "Remove infected plants immediately. Disinfect tools with 10% bleach. Use resistant varieties.",
    },
    "Tomato_healthy": {
        "display": "Healthy",
        "severity": "None",
        "remedy": "No disease detected. Continue with regular care -- proper watering, sunlight, and balanced fertilisation.",
    },
}


# ═══════════════════════════════════════════════════════════
#  Hello — Main Prediction Class
# ═══════════════════════════════════════════════════════════
class Hello:
    """
    Central inference engine for Tomato Disease Detection.

    Responsibilities
    ----------------
    • Load the trained Keras model once (singleton pattern).
    • Pre-process uploaded images to model-ready tensors.
    • Run inference and return top prediction + confidence.
    """

    IMG_SIZE: int = 224
    _instance: Optional["Hello"] = None
    _model: Optional[tf.keras.Model] = None
    _class_names: dict[int, str] = {}

    def __new__(cls) -> "Hello":
        """Singleton — ensures the model is loaded only once."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load()
        return cls._instance

    # ── Internal loader ──────────────────────────────────────
    def _load(self) -> None:
        """Load model weights and class-index mapping."""
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Train the model first with `python train.py`."
            )
        print(f"🔄  Loading model from {MODEL_PATH} …")
        self._model = tf.keras.models.load_model(str(MODEL_PATH))
        print("✅  Model loaded successfully.")

        # Load class indices if available
        if CLASS_MAP_PATH.exists():
            with open(CLASS_MAP_PATH) as f:
                raw = json.load(f)
            self._class_names = {int(k): v for k, v in raw.items()}
        else:
            self._class_names = DEFAULT_CLASS_NAMES.copy()

    # ── Image preprocessing ──────────────────────────────────
    def preprocess(self, image_bytes: bytes) -> np.ndarray:
        """
        Convert raw upload bytes → model-ready float32 tensor.

        Steps: decode → RGB → resize → normalize → add batch dim.
        """
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((self.IMG_SIZE, self.IMG_SIZE), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        return np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)

    # ── Prediction ───────────────────────────────────────────
    def predict(self, image_bytes: bytes) -> dict:
        """
        Run inference on raw image bytes.

        Returns
        -------
        dict with keys:
            class_name, display_name, confidence, severity,
            remedy, all_predictions
        """
        tensor = self.preprocess(image_bytes)
        probabilities = self._model.predict(tensor, verbose=0)[0]

        top_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[top_idx])
        class_name = self._class_names.get(top_idx, f"class_{top_idx}")

        info = DISEASE_INFO.get(class_name, {
            "display": class_name,
            "severity": "Unknown",
            "remedy": "Consult an agronomist.",
        })

        # Build full probability map for the frontend
        all_preds = {
            self._class_names.get(i, f"class_{i}"): round(float(p), 4)
            for i, p in enumerate(probabilities)
        }

        return {
            "class_name": class_name,
            "display_name": info["display"],
            "confidence": round(confidence, 4),
            "confidence_pct": f"{confidence * 100:.2f}%",
            "severity": info["severity"],
            "remedy": info["remedy"],
            "all_predictions": dict(
                sorted(all_preds.items(), key=lambda x: x[1], reverse=True)
            ),
        }


# ═══════════════════════════════════════════════════════════
#  FastAPI Application
# ═══════════════════════════════════════════════════════════
app = FastAPI(
    title="Tomato Disease Detection API",
    description="Upload a tomato leaf image and get an instant disease diagnosis.",
    version="1.0.0",
)

# CORS — allow the frontend (any origin during dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Lazy-load the predictor on first request ────────────────
_predictor: Optional[Hello] = None


def _get_predictor() -> Hello:
    global _predictor
    if _predictor is None:
        _predictor = Hello()
    return _predictor


# ── Routes ──────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the single-page frontend."""
    if not TEMPLATE_PATH.exists():
        return HTMLResponse("<h1>Frontend not found.</h1>", status_code=404)
    return HTMLResponse(TEMPLATE_PATH.read_text(encoding="utf-8"))


@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    """
    Accept an image upload and return the predicted disease.

    Parameters
    ----------
    file : UploadFile
        JPEG / PNG image of a tomato leaf.

    Returns
    -------
    JSONResponse with prediction details.
    """
    # Validate content type
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Expected an image file, got '{file.content_type}'.",
        )

    try:
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        predictor = _get_predictor()
        result = predictor.predict(image_bytes)
        return JSONResponse(content={"status": "success", "prediction": result})

    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        )


@app.get("/health")
async def health_check():
    """Lightweight health-check endpoint."""
    return {"status": "ok", "model_loaded": _predictor is not None}


# ═══════════════════════════════════════════════════════════
#  Local dev server
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn

    print("🍅 Starting Tomato Disease Detection API …")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
