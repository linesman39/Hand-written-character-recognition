from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
from PIL import Image

# Load model from GitHub repo directory
MODEL_PATH = "models/model_hand.h5"
model = load_model(MODEL_PATH)

# Prediction function
def predict(image):
    image = image.resize((28, 28))
    image = np.array(image.convert("L")) / 255.0
    image = image.reshape(1, 28, 28, 1)
    prediction = model.predict(image)
    return np.argmax(prediction)

# Streamlit UI
st.title("Handwritten Character Recognition")
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Predict"):
        result = predict(image)
        st.write(f"Predicted Character: {chr(result + 65)}")
