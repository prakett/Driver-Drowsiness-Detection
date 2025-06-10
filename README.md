# 🚗 Driver Drowsiness Detection System

This project implements a **real-time driver monitoring system** using deep learning to detect signs of **drowsiness** from webcam video. It leverages **EfficientNet** for classification and **MediaPipe** for face detection. The system also includes a prediction **smoothing mechanism** to reduce noise from individual frames.

---

## 📁 Dataset

We use the [Driver Drowsiness Dataset (DDD)](https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd) from Kaggle.

- The dataset contains images labeled as "Drowsy" and "Non Drowsy".
- Place your raw dataset in a folder (e.g., `raw_dataset/`) with subfolders:
  - `Drowsy/`
  - `Non Drowsy/`

---

## 🔧 Installation

1. Clone the repository.
2. (Optional) Create and activate a virtual environment.
3. Install the dependencies:

```bash
pip install -r requirements.txt
```





## 🧠 Model

- We use **EfficientNet-B0** as the base model, modified for **binary classification**.
- Model weights are stored in the `model/` folder (e.g., `drowsiness_detector_model.pth`).
- The model outputs a **sigmoid probability score**.



## 🧪 Scripts
🔹 `preprocess.py`
- Splits the dataset into **train/val/test**.
- Normalizes and saves images to `processed_dataset/`.

🔹 `smooth_prediction.py`
- Applies a **moving average** on frame-level predictions to ensure stable output.

🔹 `app.py`
- Launches **webcam**.
- Detects faces using **MediaPipe**.
- Classifies **drowsiness in real-time**.
- Displays **prediction** and **FPS** on screen.


```bash
 🗂️ Project Structure

├── app.py                  # Main app for webcam-based real-time inference
├── preprocess.py           # Prepares dataset: splits, normalizes, and organizes
├── smooth_prediction.py    # Handles prediction smoothing over sequences
├── model/                  # Folder containing trained models (e.g., .pth file)
├── processed_dataset/      # Preprocessed train/val/test image directories
├── requirements.txt        # List of Python dependencies
└── README.md               # Project overview and instructions

```
## 🚀 Run Inference

After training or loading the model:

```bash

python app.py

```
Press q to quit the real-time window.

## 📦 Requirements

See requirements.txt for full list. Key packages:


















