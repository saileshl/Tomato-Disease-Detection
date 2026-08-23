<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:059669,100:06b6d4&height=220&section=header&text=🍅%20LeafScan%20AI&fontSize=52&fontColor=ffffff&fontAlignY=38&desc=Tomato%20Disease%20Detection%20|%20Deep%20Learning&descSize=16&descAlignY=55&animation=fadeIn" width="100%" />
</p>

<p align="center">
  <a href="https://tomato-disease-detection-dlt.vercel.app"><img src="https://img.shields.io/badge/🌐_LIVE_DEMO-Visit_App-10b981?style=for-the-badge&labelColor=0c1017" /></a>
  &nbsp;
  <a href="https://colab.research.google.com/github/saileshl/Tomato-Disease-Detection/blob/master/Tomato_Disease_Detection.ipynb"><img src="https://img.shields.io/badge/📓_COLAB-Run_Notebook-F9AB00?style=for-the-badge&labelColor=0c1017" /></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-0.109+-009688?style=flat-square&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/MobileNetV2-Transfer_Learning-6366f1?style=flat-square" />
  <img src="https://img.shields.io/badge/Deployed-Vercel-000?style=flat-square&logo=vercel&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-22c55e?style=flat-square" />
</p>

<p align="center">
  <em>Deep learning-powered identification of 10 tomato leaf diseases using MobileNetV2 transfer learning,<br/>served via a FastAPI backend with a premium glassmorphic web UI.</em>
</p>

---

## ⚡ Quick Links

| | |
|:---:|:---:|
| 🌐 **[Live Demo](https://tomato-disease-detection-dlt.vercel.app)** | 📓 **[Colab Notebook](https://colab.research.google.com/github/saileshl/Tomato-Disease-Detection/blob/master/Tomato_Disease_Detection.ipynb)** |
| 📦 **[Releases](https://github.com/saileshl/Tomato-Disease-Detection/releases)** | 📋 **[API Docs](#-api-reference)** |

---

## 📌 Overview

This project builds a **production-ready, end-to-end pipeline** for automated tomato disease detection from leaf images:

```
📥 Data Collection → 🧠 Model Training → 🚀 API Server → 🖥 Web Interface → ☁️ Cloud Deploy
```

| Component | Description |
|:---|:---|
| **Dataset** | PlantVillage (tomato subset) — ~18,000 images across 10 classes |
| **Model** | MobileNetV2 backbone + custom classification head (Transfer Learning) |
| **Training** | Two-phase: frozen feature extraction → fine-tuning last 30 layers |
| **Backend** | FastAPI with the `Hello` prediction class (singleton pattern) |
| **Frontend** | Premium dark UI with glassmorphism, animations & drag-drop upload |
| **Deployment** | Vercel serverless (TFLite model, ~2.8 MB) |

---

## 🏗 Architecture

```
┌──────────────────────────────────────────────────────┐
│                    Browser / UI                       │
│         Dark Glassmorphic Interface (Vanilla CSS)     │
└─────────────────────┬────────────────────────────────┘
                      │ POST /predict (multipart/form)
                      ▼
┌──────────────────────────────────────────────────────┐
│                  FastAPI Server                       │
│                                                      │
│   ┌────────────────────────────────────┐             │
│   │         Hello  (Singleton)         │             │
│   │  ┌──────────┐  ┌───────────────┐  │             │
│   │  │preprocess│→ │  TFLite / TF  │  │             │
│   │  │ (PIL)    │  │  Inference    │  │             │
│   │  └──────────┘  └───────┬───────┘  │             │
│   │                        │          │             │
│   │              ┌─────────▼────────┐ │             │
│   │              │ tomato_model     │ │             │
│   │              │ .tflite (2.8 MB) │ │             │
│   │              └──────────────────┘ │             │
│   └────────────────────────────────────┘             │
└──────────────────────────────────────────────────────┘
```

---

## 🦠 Disease Classes

| # | Disease | Severity | Samples |
|:-:|:---|:---:|:---:|
| 0 | Bacterial Spot | 🟡 Medium | 2,127 |
| 1 | Early Blight | 🟠 High | 1,000 |
| 2 | Late Blight | 🔴 Critical | 1,909 |
| 3 | Leaf Mold | 🟡 Medium | 952 |
| 4 | Septoria Leaf Spot | 🟡 Medium | 1,771 |
| 5 | Spider Mites (Two-spotted) | 🟡 Medium | 1,676 |
| 6 | Target Spot | 🟡 Medium | 1,404 |
| 7 | Yellow Leaf Curl Virus | 🔴 Critical | 3,209 |
| 8 | Mosaic Virus | 🟠 High | 373 |
| 9 | Healthy ✅ | 🟢 None | 1,591 |

---

## 📁 Repository Structure

```
Tomato-Disease-Detection/
│
├── 📄 README.md                        ← Project documentation
├── 📄 requirements.txt                 ← Python dependencies
├── 📄 vercel.json                      ← Vercel serverless config
├── 📄 .gitignore                       ← Git exclusions
├── 📄 .vercelignore                    ← Vercel upload exclusions
│
├── 🧠 model.py                         ← CNN architecture (MobileNetV2)
├── ⚙️ train.py                         ← Training + evaluation pipeline
├── 🚀 main.py                          ← FastAPI backend (Hello class)
├── 📥 data_downloader.py               ← Dataset acquisition script
│
├── 🖥 templates/
│   └── index.html                      ← Premium web UI
│
├── 📓 Tomato_Disease_Detection.ipynb   ← Colab notebook (standalone)
│
├── 🤖 tomato_model.tflite              ← Optimised model (2.8 MB)
├── 📋 class_indices.json               ← Label mapping
│
└── 📊 [Generated after training]
    ├── tomato_model.h5                 ← Full Keras model
    ├── training_history.png            ← Accuracy/Loss curves
    ├── confusion_matrix.png            ← Confusion matrix
    └── classification_report.txt       ← Precision/Recall/F1
```

---

## 🚀 Quick Start

### Prerequisites

- Python **3.9+**
- (Optional) NVIDIA GPU with CUDA for faster training

### 1 — Clone & Install

```bash
git clone https://github.com/saileshl/Tomato-Disease-Detection.git
cd Tomato-Disease-Detection

python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### 2 — Download the Dataset

```bash
python data_downloader.py
```

> The script auto-downloads from multiple sources (Kaggle, HuggingFace, HTTP mirrors). If automated methods fail, follow the on-screen instructions to place the 10 `Tomato_*` folders into `PlantVillage/`.

### 3 — Train the Model

```bash
python train.py
```

| Phase | Description | Epochs |
|:---:|:---|:---:|
| 1 | Feature extraction (frozen backbone) | 10 |
| 2 | Fine-tuning (last 30 layers unfrozen) | 5 |

> ⏱ **Training time**: ~10 min on GPU (T4), ~30 min on CPU

### 4 — Run the Web App

```bash
python main.py
```

Open **http://localhost:8000** → upload a leaf image → instant diagnosis.

---

## 📓 Google Colab Notebook

A **fully self-contained** notebook that runs the complete pipeline in the cloud:

<p align="center">
  <a href="https://colab.research.google.com/github/saileshl/Tomato-Disease-Detection/blob/master/Tomato_Disease_Detection.ipynb">
    <img src="https://img.shields.io/badge/Open_in-Google_Colab-F9AB00?style=for-the-badge&logo=google-colab&logoColor=white" />
  </a>
</p>

**Steps:**
1. Open the link above
2. Set runtime to **GPU** (`Runtime → Change runtime type → T4 GPU`)
3. Click **Run All** — the notebook handles everything

---

## ☁️ Deployment

The app is deployed on **Vercel** using a lightweight TFLite model (2.8 MB) instead of full TensorFlow (~500 MB).

| Component | Detail |
|:---|:---|
| Runtime | Python 3.12 (Vercel serverless) |
| Inference | `ai-edge-litert` (Google's TFLite successor) |
| Model size | 2.8 MB (quantised TFLite) |
| Cold start | ~3 seconds |

```bash
# Deploy yourself
npm i -g vercel
vercel --prod
```

---

## 📡 API Reference

### `POST /predict`

Upload a leaf image and receive a diagnosis.

**Request:**
```bash
curl -X POST https://tomato-disease-detection-dlt.vercel.app/predict \
  -F "file=@leaf_image.jpg"
```

**Response:**
```json
{
  "status": "success",
  "prediction": {
    "class_name": "Tomato_Early_blight",
    "display_name": "Early Blight",
    "confidence": 0.9734,
    "confidence_pct": "97.34%",
    "severity": "High",
    "remedy": "Apply chlorothalonil or mancozeb fungicide. Practice crop rotation.",
    "all_predictions": { "...": "..." }
  }
}
```

### `GET /health`

```json
{ "status": "ok", "model_loaded": true }
```

---

## 📊 Results

| Metric | Value |
|:---|:---:|
| Validation Accuracy | **~80%** |
| Macro Precision | 0.81 |
| Macro Recall | 0.77 |
| Macro F1-Score | 0.74 |
| Best Class | YellowLeaf Curl Virus (~97%) |
| Model Size (TFLite) | 2.8 MB |

> 💡 Higher accuracy (~92%+) achievable with more training epochs and GPU-based training via the Colab notebook.

---

## 🛠 Tech Stack

<p align="center">
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/Pillow-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Vercel-000?style=for-the-badge&logo=vercel&logoColor=white" />
  <img src="https://img.shields.io/badge/Google_Colab-F9AB00?style=for-the-badge&logo=google-colab&logoColor=white" />
</p>

| Layer | Technology |
|:---|:---|
| **Deep Learning** | TensorFlow / Keras, MobileNetV2 (Transfer Learning) |
| **Inference** | TFLite (`ai-edge-litert`) for serverless, full TF for local |
| **Backend** | FastAPI, Uvicorn |
| **Frontend** | HTML5, Vanilla CSS (glassmorphism), JavaScript |
| **Data Science** | scikit-learn, matplotlib, NumPy, Pillow |
| **Deployment** | Vercel (serverless Python) |
| **Notebook** | Google Colab (GPU runtime) |

---

## 📜 License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:059669,100:06b6d4&height=120&section=footer" width="100%" />
</p>

<p align="center">
  <strong>Built with 🍅 and ☕ for the DLT Mini Project</strong><br/>
  <sub>Deep Learning Techniques — Tomato Disease Detection</sub>
</p>
