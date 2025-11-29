import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

MODEL_PATH = "model/currency_note_classifier.h5"
model = tf.keras.models.load_model(MODEL_PATH)

labels = ["1Hundrednote", "2Hundrednote", "2Thousandnote",
          "5Hundrednote", "Fiftynote", "Tennote", "Twentynote"]

IMG_SIZE = 224

st.title("💵 Indian Currency Note Classifier")
st.write("Upload an image of a currency note to classify it")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

def predict_img(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)
    label_id = np.argmax(pred)
    return labels[label_id]

if uploaded_file is not None:
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(img, channels="BGR", width=300)

    result = predict_img(img)
    st.success(f"Predicted: {result}")
