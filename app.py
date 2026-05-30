import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import string

from streamlit_drawable_canvas import st_canvas

mnist_model = tf.keras.models.load_model("mnist_model.keras")
emnist_model = tf.keras.models.load_model("emnist_model.keras")

letters = list(string.ascii_uppercase)


st.title("Handwritten Recognition (Draw & Predict)")

option = st.selectbox(
    "Choose Model",
    ["Digit (MNIST)", "Character (EMNIST)"]
)

st.write("Draw below and click Predict")

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas"
)


if st.button("Predict"):

    if canvas_result.image_data is not None:

        img = canvas_result.image_data

        # convert to grayscale
        img = Image.fromarray((img[:, :, 0]).astype(np.uint8))
        img = img.resize((28, 28))

        img_array = np.array(img)
        img_array = img_array.reshape(1, 28, 28, 1)
        img_array = img_array / 255.0

        if option == "Digit (MNIST)":

            prediction = mnist_model.predict(img_array)
            result = np.argmax(prediction)

            st.success(f"Predicted Digit: {result}")

        else:

            prediction = emnist_model.predict(img_array)
            result = np.argmax(prediction)

            st.success(f"Predicted Letter: {letters[result]}")