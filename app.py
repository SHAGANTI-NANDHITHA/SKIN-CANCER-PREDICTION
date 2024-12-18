import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
import requests

def load_model_from_drive():
    try:
        # Direct download link to the model file
        url = "https://drive.google.com/uc?id=1ZuZCCybYsXt4c1F1WYNjuG0cszeGp3UA"
        
        # Download the model file
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            st.error("Failed to download the model. Please check the link.")
            return None
        
        with open("skin_cancer_model.h5", "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        
        # Load the model
        model = tensorflow.keras.models.load_model("skin_cancer_model.h5")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


# Define class labels
class_labels = ['Benign', 'Malignant']

# Streamlit App
st.title("Skin Cancer Classification")
st.write("Upload an image to classify it as benign or malignant.")

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
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
