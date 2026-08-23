# 🍅 Tomato Disease Detection

> **DLT Mini Project** — Deep Learning-based identification of 10 tomato leaf diseases using MobileNetV2 transfer learning, served via a FastAPI backend and a sleek web UI.

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?logo=tensorflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Disease Classes](#disease-classes)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
  - [1 — Clone & Install](#1--clone--install)
  - [2 — Download the Dataset](#2--download-the-dataset)
  - [3 — Train the Model](#3--train-the-model)
  - [4 — Run the Web App](#4--run-the-web-app)
- [Google Colab Notebook](#google-colab-notebook)
- [Deploy to Vercel](#deploy-to-vercel)
- [API Reference](#api-reference)
- [Results](#results)
- [Tech Stack](#tech-stack)
- [License](#license)

---

## Overview

This project builds an end-to-end pipeline for **automated tomato disease detection** from leaf images:

1. **Data** — The PlantVillage dataset (tomato subset, ~18 000 images, 10 classes).
2. **Model** — MobileNetV2 backbone with custom classification head, trained in two phases (feature extraction → fine-tuning).
3. **API** — FastAPI server exposing a `/predict` endpoint that returns the disease name, confidence score, severity level, and recommended treatment.
4. **Frontend** — A modern single-page UI (Tailwind CSS) with drag-and-drop upload, animated results, and a full probability breakdown.

---

## Architecture

```
                 ┌────────────────────┐
                 │   Browser / UI     │
                 │   (Tailwind CSS)   │
                 └────────┬───────────┘
                          │  HTTP POST /predict
                          ▼
                 ┌────────────────────┐
                 │    FastAPI Server   │
                 │    (main.py)        │
                 │                    │
                 │  ┌──────────────┐  │
                 │  │  Hello class │  │  ← prediction engine
                 │  │  • preprocess│  │
                 │  │  • predict   │  │
                 │  └──────┬───────┘  │
                 │         │          │
                 │  ┌──────▼───────┐  │
                 │  │ tomato_model │  │
                 │  │    .h5       │  │
                 │  └──────────────┘  │
                 └────────────────────┘
```

---

## Disease Classes

| # | Class Name | Severity |
|---|-----------|----------|
| 0 | Bacterial Spot | Medium |
| 1 | Early Blight | High |
| 2 | Late Blight | Critical |
| 3 | Leaf Mold | Medium |
| 4 | Septoria Leaf Spot | Medium |
| 5 | Spider Mites (Two-spotted) | Medium |
| 6 | Target Spot | Medium |
| 7 | Yellow Leaf Curl Virus | Critical |
| 8 | Mosaic Virus | High |
| 9 | Healthy ✅ | None |

---

## Repository Structure

```
DLT MINI PROJ/
├── README.md                        ← You are here
├── requirements.txt                 ← Python dependencies
├── .gitignore                       ← Git exclusions
├── vercel.json                      ← Vercel deployment config
│
├── data_downloader.py               ← Downloads & organises the dataset
├── model.py                         ← CNN architecture (MobileNetV2)
├── train.py                         ← Full training + evaluation pipeline
├── main.py                          ← FastAPI backend (Hello class)
│
├── templates/
│   └── index.html                   ← Frontend UI (Tailwind CSS)
│
├── Tomato_Disease_Detection.ipynb   ← Standalone Colab notebook
│
├── tomato_model.h5                  ← Trained model (generated)
├── class_indices.json               ← Label mapping  (generated)
├── training_history.png             ← Acc/Loss plots  (generated)
├── confusion_matrix.png             ← Confusion matrix (generated)
└── classification_report.txt        ← Precision/Recall/F1 (generated)
```

---

## Quick Start

### Prerequisites

- Python **3.9+**
- (Optional) NVIDIA GPU with CUDA for faster training

### 1 — Clone & Install

```bash
git clone https://github.com/<your-username>/tomato-disease-detection.git
cd tomato-disease-detection

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

> **GPU users**: Replace `tensorflow-cpu` with `tensorflow` in `requirements.txt` before installing.

### 2 — Download the Dataset

```bash
python data_downloader.py
```

The script tries multiple public mirrors (Google Drive, HTTP) — no API keys needed. If all automated methods fail, follow the on-screen instructions to download from [Kaggle](https://www.kaggle.com/datasets/arjuntejaswi/plant-village) and place the 10 `Tomato_*` folders into `PlantVillage/`.

A backup archive `Tomato_Disease_Dataset.zip` is automatically created.

### 3 — Train the Model

```bash
python train.py
```

This will:

1. Load & augment the dataset (80/20 train/val split)
2. **Phase 1** — Train with frozen MobileNetV2 backbone (15 epochs)
3. **Phase 2** — Fine-tune the last 30 layers (10 epochs)
4. Save `tomato_model.h5`, training plots, confusion matrix, and classification report

Training time: ~15 min on GPU, ~2 h on CPU.

### 4 — Run the Web App

```bash
python main.py
# or
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000** in your browser. Upload a leaf image → get an instant diagnosis.

---

## Google Colab Notebook

The file `Tomato_Disease_Detection.ipynb` is a **fully self-contained** notebook that runs the entire pipeline in the cloud (no local setup needed):

1. Downloads & zips the dataset
2. Trains the MobileNetV2 model
3. Outputs all evaluation metrics
4. Auto-downloads `tomato_model.h5` to your machine

**To use:**
1. Upload `Tomato_Disease_Detection.ipynb` to [Google Colab](https://colab.research.google.com/)
2. Set runtime to **GPU** (`Runtime → Change runtime type → T4 GPU`)
3. Click **Run All**

---

## Deploy to Vercel

This project includes a `vercel.json` for serverless deployment.

```bash
# Install the Vercel CLI
npm i -g vercel

# Deploy (make sure tomato_model.h5 is present)
vercel --prod
```

> ⚠️ **Note**: Vercel serverless functions have a **50 MB** size limit. The model + TensorFlow runtime may exceed this. For production deployment, consider hosting the model on cloud storage and downloading it at cold-start, or use a dedicated server (Railway, Render, AWS Lambda with container images).

---

## API Reference

### `POST /predict`

**Request**: `multipart/form-data` with a `file` field containing an image (JPEG/PNG).

**Response** (JSON):
```json
{
  "status": "success",
  "prediction": {
    "class_name": "Early_blight",
    "display_name": "Early Blight",
    "confidence": 0.9734,
    "confidence_pct": "97.34%",
    "severity": "High",
    "remedy": "Apply chlorothalonil or mancozeb fungicide. Practice crop rotation.",
    "all_predictions": {
      "Early_blight": 0.9734,
      "Late_blight": 0.0121,
      "...": "..."
    }
  }
}
```

### `GET /health`

Returns `{"status": "ok", "model_loaded": true}`.

---

## Results

After training, you can expect:

| Metric | Value |
|--------|-------|
| Validation Accuracy | ~95-97% |
| Macro F1-Score | ~0.95 |
| Best Class | healthy (~99%) |
| Hardest Class | Septoria vs Early Blight overlap |

Training curves and the confusion matrix are saved as `training_history.png` and `confusion_matrix.png`.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Deep Learning | TensorFlow / Keras, MobileNetV2 |
| Backend | FastAPI, Uvicorn |
| Frontend | HTML5, Tailwind CSS, Vanilla JS |
| Data Science | scikit-learn, matplotlib, NumPy, Pillow |
| Deployment | Vercel (serverless) |
| Notebook | Google Colab |

---

## License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.

---

<p align="center">
  Made with 🍅 and ☕ for the DLT Mini Project
</p>
