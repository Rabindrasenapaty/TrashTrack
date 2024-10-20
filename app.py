import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load your trained model
model = tf.keras.models.load_model('Model.keras')

# Define class names corresponding to your model's output
class_names = [
    "Battery",
    "Biological Waste",
    "Brown Glass",
    "Cardboard",
    "Clothes",
    "e-waste",
    "white glass",
    "green glass",
    "Metal",
    "Paper",
    "Plastic",
    "Shoes",
    "Trash"
]

# Function to preprocess the uploaded image (resize, normalize, etc.)
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to match your model's input size
    image = np.array(image)  # Convert the image to an array
    image = image.astype('float32') / 255.0  # Normalize pixel values between 0 and 1
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to make prediction on the image
def predict_class(image, model):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    class_idx = np.argmax(prediction)  # Get the index of the highest score
    return class_idx

# Streamlit Interface
st.title("Garbage Classification System")
st.write("Upload an image of garbage to classify its type.")

# Image upload widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image file
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Make prediction
    class_idx = predict_class(image, model)

    # Get the class name from the index
    class_name = class_names[class_idx]

    # Display the prediction
    st.write(f"The garbage is classified as: {class_name}")
