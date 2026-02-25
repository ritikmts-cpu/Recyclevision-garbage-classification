# RecycleVision - Garbage Image Classification

RecycleVision is a deep learning–based web application that classifies garbage images into six categories:
**cardboard, glass, metal, paper, plastic, trash**.

---

## 1. Problem Statement

Manual waste sorting is time-consuming and often inaccurate, which reduces the efficiency of recycling systems.
This project aims to build a lightweight image classification model that can automatically predict the category
of a garbage item from an image, enabling faster and more consistent waste segregation. [web:57][web:62]

---

## 2. Dataset

- Source: Kaggle Garbage Classification dataset (6 classes).
- Total images: ~2500+ across six categories.
- Split: 80% training, 20% validation.
- Input size: 224 × 224 × 3.

The dataset contains real-world images with varying lighting conditions, backgrounds and orientations,
which makes the classification task challenging and realistic. [web:62][web:65]

---

## 3. Model Architecture

The model is built using **transfer learning** with **MobileNetV2**.

**Base model**

- `tf.keras.applications.MobileNetV2`
- `include_top=False`
- `weights="imagenet"`
- Input shape: `(224, 224, 3)`
- The base model is used as a fixed feature extractor:
  - `base_model.trainable = False` [web:38][web:65]

**Custom classification head**

- `GlobalAveragePooling2D`
- `Dropout(0.2)`
- `Dense(6, activation="softmax")`

**Preprocessing and training setup**

- Data augmentation:
  - Random horizontal flip, rotation, zoom, contrast.
- Normalization:
  - `tf.keras.applications.mobilenet_v2.preprocess_input` applied inside the model. [web:38][web:65]
- Loss: `sparse_categorical_crossentropy`
- Optimizer: `Adam (learning_rate=1e-4)`
- Batch size: 32
- Epochs: ~30 with early stopping on validation loss.

---

## 4. Training Results

Final performance on the validation set:

- **Validation accuracy:** ~**85.21%**
- **Validation loss:** ~0.46

This means the model correctly classifies roughly 85 out of 100 validation images.
For MobileNetV2-based waste classification systems, validation accuracies in the 80–85% range are commonly
reported in the literature, so these results are competitive with published work. [web:62][web:65][web:57]

---

## 5. Streamlit Web Application

### Features

- Upload of single garbage images (`.jpg`, `.jpeg`, `.png`).
- Server-side preprocessing:
  - Convert to RGB.
  - Resize to 224 × 224.
  - Add batch dimension.
  - All further preprocessing and softmax are handled **inside** the saved Keras model.
- Output:
  - Predicted class with confidence (%).
  - Top‑3 classes with their probabilities.

### Inference Pipeline

1. User uploads an image of a garbage item.
2. The app resizes and formats the image and passes it to the trained MobileNetV2 model.
3. The model outputs a probability distribution over the six classes.
4. The class with the highest probability is displayed as the final prediction, along with the top‑3 scores.

---

## 6. How to Run Locally

### 6.1. Clone the repository
```bash
git clone <repo-url>
cd "RecycleVision- Garbage Image Classification"

6.2. Create a virtual environment and install dependencies
python -m venv venv
venv\Scripts\activate 

pip install -r requirements.txt

6.3. Project structure
RecycleVision- Garbage Image Classification/
│
├─ Data/                
├─ Models/
│   └─ recyclevision_mobilenetv2.keras
├─ notebook/
│   └─ recyclevision.ipynb
└─ app.py

6.4. Launch the Streamlit app
streamlit run app.py


