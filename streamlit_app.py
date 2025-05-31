import streamlit as st
import numpy as np
import cv2
from model import load_model
from preprocess import load_data, preprocess_data

st.title("Handwritten Character Recognition")
st.write("Upload a 28x28 grayscale image of a handwritten character (EMNIST format)")

# Load model and label encoder (cache for performance)
@st.cache_resource
def get_model_and_encoder():
    model = load_model('models/handwritten_char_recognition.h5')
    data = load_data('data/emnist-letters-train.csv')
    _, _, label_encoder = preprocess_data(data)
    return model, label_encoder

model, label_encoder = get_model_and_encoder()

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Read image as bytes, then decode
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        # Resize and preprocess
        img_resized = cv2.resize(img, (28, 28))
        img_inverted = 255 - img_resized
        img_norm = img_inverted / 255.0
        img_input = img_norm.reshape(1, 28, 28, 1)
        st.image(img_resized, caption='Uploaded Image (resized to 28x28)', width=150)
        # Predict
        prediction = model.predict(img_input)
        predicted_class = np.argmax(prediction[0])
        predicted_char = label_encoder.inverse_transform([predicted_class])[0]
        confidence = prediction[0][predicted_class]
        st.success(f"Predicted Character: {predicted_char}")
        st.info(f"Confidence: {confidence:.2%}")
    else:
        st.error("Could not read the image. Please upload a valid image file.")
else:
    st.info("Please upload an image file.") 