# Handwritten Character Recognition

This project implements a Convolutional Neural Network (CNN) for recognizing handwritten characters using the EMNIST dataset. The model is built using TensorFlow and Keras.

## Project Overview
- Handwritten character recognition system using CNN
- Trained on the EMNIST letters dataset
- Support for recognizing uppercase and lowercase letters
- Real-time prediction capabilities

## Project Structure
```
.
├── data/                   # Directory for dataset
├── models/                 # Directory for saved models
├── preprocess.py          # Data preprocessing utilities
├── model.py              # CNN model architecture
├── train.py              # Training script
├── predict.py            # Prediction script
└── requirements.txt      # Project dependencies
```

## Setup and Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the EMNIST dataset:
- Download the EMNIST letters dataset from [here](https://www.nist.gov/itl/products-and-services/emnist-dataset)
- Place the CSV files in the `data/` directory

## Usage

1. Training the model:
```bash
python train.py
```
This will:
- Load and preprocess the EMNIST dataset
- Train the CNN model
- Save the trained model to `models/handwritten_char_recognition.h5`
- Generate training history plots

2. Making predictions:
```bash
python predict.py
```
Make sure to:
- Place your test image in the `data/` directory
- Update the `image_path` in `predict.py` to point to your test image

## Model Architecture
- Input: 28x28 grayscale images
- 3 Convolutional layers with ReLU activation
- MaxPooling layers for dimensionality reduction
- Dense layers with dropout for classification
- Output: 62 classes (uppercase and lowercase letters)

## License
This project is licensed under the MIT License - see the LICENSE file for details. 