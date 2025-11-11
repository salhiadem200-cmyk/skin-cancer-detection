# 🩺 Skin Cancer Detection Platform

A deep learning web application for **skin cancer detection** from dermoscopic images.  
This project was developed as part of a medical image analysis competition and won **1st place** 🥇 at the Bootcamp (15 Décembre 2024).

---

## 🚀 Project Overview
The platform uses **EfficientNetV2-B2** as a feature extractor to classify skin lesions and integrates patient metadata for improved diagnostic accuracy.  
A **Flask web interface** allows users to upload images and receive predictions in real time.

---

## 🧠 Model Architecture
- **Base model:** EfficientNetV2-B2 (pretrained on ImageNet)  
- **Additional layers:** GlobalAveragePooling → Dense(256, ReLU) → Dropout(0.3) → Dense(1, Sigmoid)  
- **Fusion module:** Combines image embeddings with tabular patient features  
- **Loss function:** Binary Cross-Entropy  
- **Optimizer:** Adam (lr = 1e-4)

---

## 📊 Dataset
- Source: **ISIC Challenge Dataset**
- Types of lesions: Melanoma, Nevus, Basal Cell Carcinoma, etc.
- Data split:
  - 70 % Training  
  - 15 % Validation  
  - 15 % Testing  

Each image is paired with patient metadata such as age, gender, and lesion location.

---

## ⚙️ Tech Stack
| Category | Tools |
|-----------|-------|
| **Languages** | Python |
| **Libraries** | TensorFlow, Keras, NumPy, Pandas, Matplotlib |
| **Web Framework** | Flask |
| **Deployment** | Docker, Gunicorn |
| **Visualization** | Seaborn, Plotly |

---

## 🧪 Training Pipeline
1. **Data Preprocessing** – Image resizing, normalization, metadata encoding  
2. **Model Training** – Fine-tuning EfficientNetV2-B2 on the preprocessed dataset  
3. **Evaluation** – Accuracy, AUC, F1-score, Confusion Matrix  
4. **Deployment** – Flask web app for interactive prediction

---

## 💻 Web Application
**Main Features**
- Upload a dermoscopic image (`.jpg` or `.png`)
- Input patient metadata
- Get instant prediction (`Benign` / `Malignant`)
- Visualize class probability and Grad-CAM heatmap

**Run Locally**
```bash
# Clone the repo
git clone https://github.com/yourusername/skin-cancer-detection.git
cd skin-cancer-detection

# Install dependencies
pip install -r requirements.txt

# Launch the app
python app/app.py
