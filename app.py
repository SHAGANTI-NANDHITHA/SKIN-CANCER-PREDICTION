import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
import requests

# Function to download the model file from Google Drive
@st.cache_resource
def load_model_from_drive():
    try:
        # Direct download link
        url = "https://drive.google.com/uc?id=1ZuZCCybYsXt4c1F1WYNjuG0cszeGp3UA"
        model_path = "skin_cancer_model.h5"

        # Check if the model is already downloaded
        if not os.path.exists(model_path):
            # Download the model file
            st.info("Downloading the model file. Please wait...")
            response = requests.get(url, stream=True)
            if response.status_code != 200:
                st.error("Failed to download the model. Please check the link.")
                return None

            # Save the downloaded model file
            with open(model_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)

        # Load the model
        model = tf.keras.models.load_model(model_path)
        st.success("Model loaded successfully!")
        return model

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


# Load the model
model = load_model_from_drive()

# Define class labels
class_labels = ['Benign', 'Malignant']

# Streamlit App
st.title("Skin Cancer Classification")
st.write("Upload an image to classify it as benign or malignant.")

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Save the uploaded image to a temporary file
        temp_file_path = "temp_image.jpg"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load and preprocess the image
        image = load_img(temp_file_path, target_size=(150, 150))
        image_array = img_to_array(image) / 255.0  # Normalize
        image_array = np.expand_dims(image_array, axis=0)

        # Predict
        if model:
            st.image(image, caption="Uploaded Image", use_column_width=True)
            st.write("Classifying...")

            prediction = model.predict(image_array)
            class_idx = int(prediction[0] > 0.5)  # Binary classification
            class_label = class_labels[class_idx]

            # Display result
            st.write(f"### Prediction: {class_label}")
        else:
            st.error("Model could not be loaded. Please try again.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
    finally:
        # Remove the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
