import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2

# Load your CNN model
model = tf.keras.models.load_model('model')

# Define function to handle file upload and prediction
def predict_image(image):
    # Convert image to numpy array
    img_array = np.array(image)
    # Resize image to (256, 256)
    img_resized = cv2.resize(img_array, (256, 256))
    # Reshape image for prediction
    img_reshaped = np.expand_dims(img_resized, axis=0)  # Add batch dimension
    # Normalize image
    img_normalized = img_reshaped / 255.0
    # Make prediction
    prediction = model.predict(img_normalized)
    return prediction

# Streamlit app
def main():
    # Set page title and favicon
    st.set_page_config(page_title="Dog or Cat Prediction", page_icon="🐾")

    # Header
    st.title('Dog or Cat Prediction')
    st.write('Upload an image of a dog or a cat to predict its label.')

    # File upload widget
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Make prediction when the 'Predict' button is clicked
        if st.button('Predict'):
            prediction = predict_image(image)
            if prediction[0][0] >= 0.5:
                st.write('Prediction: Dog')
            else:
                st.write('Prediction: Cat')

if __name__ == "__main__":
    main()
