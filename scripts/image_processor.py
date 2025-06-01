import cv2
import numpy as np

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (28, 28)) / 255.0
    image = np.expand_dims(image, axis=(0, -1))  # Reshape for model input
    return image
