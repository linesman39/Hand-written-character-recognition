import cv2
import numpy as np
from model import load_model
from preprocess import load_data

def preprocess_image(image_path):
    """
    Preprocess a single image for prediction
    """
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize to 28x28
    img = cv2.resize(img, (28, 28))
    
    # Invert the image (EMNIST format)
    img = 255 - img
    
    # Normalize
    img = img / 255.0
    
    # Reshape for model input
    img = img.reshape(1, 28, 28, 1)
    
    return img

def predict_character(model, image_path, label_encoder):
    """
    Make a prediction on a single image
    """
    # Preprocess the image
    img = preprocess_image(image_path)
    
    # Make prediction
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction[0])
    
    # Convert numeric prediction to character
    predicted_char = label_encoder.inverse_transform([predicted_class])[0]
    confidence = prediction[0][predicted_class]
    
    return predicted_char, confidence

def main():
    # Load the model
    model = load_model('models/handwritten_char_recognition.h5')
    
    # Load the label encoder
    data = load_data('data/emnist-letters-train.csv')
    _, _, label_encoder = preprocess_data(data)
    
    # Example usage
    image_path = 'data/test_image.png'  # Replace with your test image path
    predicted_char, confidence = predict_character(model, image_path, label_encoder)
    
    print(f"Predicted character: {predicted_char}")
    print(f"Confidence: {confidence:.2%}")

if __name__ == "__main__":
    main() 