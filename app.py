import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

st.title("🧠 Brain Tumor Detection App")
st.write("Upload an MRI image to detect tumor type using CNN")

model = tf.keras.models.load_model("brain_tumor_cnn_model.h5")
categories = ['glioma', 'meningioma', 'pituitary', 'notumor']

uploaded_file = st.file_uploader("Choose an MRI image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_resized = cv2.resize(img, (150, 150))
    img_norm = img_resized / 255.0
    img_input = np.expand_dims(img_norm, axis=0)

    st.image(img, channels="BGR", caption="Uploaded MRI Image", use_column_width=True)

    prediction = model.predict(img_input)
    class_idx = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    st.subheader(f"Prediction: {categories[class_idx]}")
    st.write(f"Confidence: {confidence:.2f}%")
