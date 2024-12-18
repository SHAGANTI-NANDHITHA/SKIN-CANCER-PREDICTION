import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
import requests
import tensorflow as tf  # Import TensorFlow

# Function to download and load the model from Google Drive
@st.cache_resource
def load_model_from_drive():
    try:
        # Direct download link for the model file
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


# Define class labels
class_labels = ['Benign', 'Malignant']

# Streamlit App
st.title("Skin Cancer Classification")
st.write("Upload an image to classify it as benign or malignant.")

# Load the model
model = load_model_from_drive()

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    # Save the uploaded image to a temporary file
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Load and preprocess the image
    image = load_img("temp_image.jpg", target_size=(150, 150))
    image_array = img_to_array(image) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)
    
    # Predict
    prediction = model.predict(image_array)
    class_idx = int(prediction[0] > 0.5)  # Binary classification
    class_label = class_labels[class_idx]
    
    # Show the image and prediction
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write(f"### Prediction: {class_label}")

    # Remove the temporary file
    os.remove("temp_image.jpg")
