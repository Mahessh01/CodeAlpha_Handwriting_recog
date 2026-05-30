import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import string

mnist_model = tf.keras.models.load_model("mnist_model.keras")
emnist_model = tf.keras.models.load_model("emnist_model.keras")

letters = list(string.ascii_uppercase)

st.title("Handwritten Recognition (Upload & Predict)")

option = st.selectbox(
    "Choose Model",
    ["Digit (MNIST)", "Character (EMNIST)"]
)

st.write("Upload a handwritten image (prefer black text on white background)")

uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

def preprocess_image(image):
    image = image.convert("L")          # grayscale
    image = image.resize((28, 28))      # MNIST/EMNIST size
    img_array = np.array(image)

    img_array = img_array.reshape(1, 28, 28, 1)
    img_array = img_array / 255.0

    return img_array

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", width=200)

    img_array = preprocess_image(image)

    if st.button("Predict"):

        if option == "Digit (MNIST)":
            prediction = mnist_model.predict(img_array)
            result = np.argmax(prediction)
            st.success(f"Predicted Digit: {result}")

        else:
            prediction = emnist_model.predict(img_array)
            result = np.argmax(prediction)
            st.success(f"Predicted Letter: {letters[result]}")