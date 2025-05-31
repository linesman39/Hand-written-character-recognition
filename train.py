import os
import numpy as np
import matplotlib.pyplot as plt
from preprocess import load_data, preprocess_data, split_data
from model import create_model, train_model, save_model

def plot_training_history(history):
    """
    Plot the training and validation accuracy/loss
    """
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def main():
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    data = load_data('data/emnist-letters-train.csv')
    X, y, label_encoder = preprocess_data(data)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Create and train model
    print("Creating and training model...")
    model = create_model()
    history = train_model(model, X_train, y_train, X_test, y_test, epochs=10)
    
    # Plot training history
    print("Plotting training history...")
    plot_training_history(history)
    
    # Save the model
    print("Saving model...")
    save_model(model, 'models/handwritten_char_recognition.h5')
    
    # Evaluate the model
    print("Evaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")

if __name__ == "__main__":
    main() 