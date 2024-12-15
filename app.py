import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
import nbformat
from nbconvert import PythonExporter
import tempfile

# Function to extract and save model from the Jupyter notebook
def extract_model_from_notebook(notebook_path):
    try:
        # Load the Jupyter notebook
        with open(notebook_path, "r") as notebook_file:
            notebook_content = nbformat.read(notebook_file, as_version=4)
        
        # Convert notebook into Python code
        exporter = PythonExporter()
        python_code, _ = exporter.from_notebook_node(notebook_content)

        # Save Python code to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.py') as temp_script:
            temp_script.write(python_code.encode())
            temp_script_path = temp_script.name
        
        # Execute the Python script to load the model and save it
        exec(open(temp_script_path).read())

        # Return the path to the saved model (ensure you save the model as 'skin_cancer_model.h5' in the notebook)
        return "skin_cancer_model.h5"
    except Exception as e:
        st.error(f"Error extracting model from notebook: {e}")
        return None

# Define the path to the Jupyter notebook
NOTEBOOK_PATH = "Skin_cancer_(CNN).ipynb"

# Extract the model from the notebook
MODEL_PATH = extract_model_from_notebook(NOTEBOOK_PATH)

# If model path is valid, load the model
if MODEL_PATH and os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    st.success("Model loaded successfully!")
else:
    st.error("Model could not be loaded.")
    st.stop()

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
        temp_image_path = "temp_image.jpg"
        with open(temp_image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load and preprocess the image
        image = load_img(temp_image_path, target_size=(150, 150))
        image_array = img_to_array(image) / 255.0  # Normalize
        image_array = np.expand_dims(image_array, axis=0)
        
        # Predict
        prediction = model.predict(image_array)
        class_idx = int(prediction[0] > 0.5)  # Binary classification
        class_label = class_labels[class_idx]
        
        # Show the image and prediction
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write(f"### Prediction: {class_label}")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
    
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
