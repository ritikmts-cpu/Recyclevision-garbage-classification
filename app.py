import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import os

# 1) Model load (same as Colab saved .keras)
MODEL_PATH = os.path.join("Models", "recyclevision_mobilenetv2.keras")
model = keras.models.load_model(MODEL_PATH)

CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
IMG_SIZE = (224, 224)

st.title("RecycleVision - Garbage Image Classifier")
st.write("Upload a garbage image, and the model will predict its category.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

def preprocess_image(image: Image.Image):
    # Sirf resize + RGB; **koi /255 ya extra preprocess yahan nahi**
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        img_batch = preprocess_image(image)

        # Model ke andar hi preprocessing + softmax hai,
        # isliye yahan direct predict + squeeze:
        preds = model.predict(img_batch)          # shape: (1, 6)
        probs = preds[0]                          # already probabilities

        top_idx = int(np.argmax(probs))
        predicted_class = CLASS_NAMES[top_idx]
        confidence = float(probs[top_idx] * 100.0)

        st.markdown(f"### Predicted: {predicted_class} ({confidence:.2f}% confidence)")

        st.write("Top-3 predictions:")
        top3_idx = probs.argsort()[-3:][::-1]
        for i in top3_idx:
            st.write(f"{CLASS_NAMES[i]}: {probs[i]*100:.2f}%")
