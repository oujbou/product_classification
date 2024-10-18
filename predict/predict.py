import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os


def get_class_labels(data_dir):
    """
    Retrieve class labels based on the subfolder names in the dataset directory.
    Args:
        data_dir (str): Path to the directory containing class subfolders.
    Returns:
        class_labels (list): List of class labels (folder names).
    """
    # Get the class labels from the directory names
    class_labels = sorted(os.listdir(data_dir))
    return class_labels

def load_model_again(model_path):
    """
    Load the model once so it can be reused for multiple predictions.
    Args:
        model_path (str): Path to the saved model (h5 file).
    Returns:
        model: The loaded Keras model.
    """
    # Load and return the model
    return load_model(model_path)


def load_and_predict(model, img_path, class_labels):
    """
    Function to load the model and make a prediction on a single new image.
    Args:
        model: The loaded Keras model.
        img_path (str): Path to the new image.
        class_labels (list): List of class labels to map the predicted index to the actual label.
    Returns:
        predicted_class_label (str): Predicted label for the image.
    """

    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make a prediction
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class_label = class_labels[predicted_class_index]  # Map index to class label

    print(f"Predicted coat model: {predicted_class_label}")
    return predicted_class_label


if __name__ == "__main__":

    model_path = '/home/toujlakh/Projects/product_classification/efficientnet_coat_classifier.h5'
    data_dir = '/home/toujlakh/Projects/product_classification/preprocessed_data/train'

    img_paths  = [
        '/home/toujlakh/Projects/product_classification/new_images/1.png',
        '/home/toujlakh/Projects/product_classification/new_images/2.png',
        '/home/toujlakh/Projects/product_classification/new_images/3.png',
        '/home/toujlakh/Projects/product_classification/new_images/4.png',
        '/home/toujlakh/Projects/product_classification/new_images/5.png',
        '/home/toujlakh/Projects/product_classification/new_images/6.png'
        ]

    class_labels = get_class_labels(data_dir)

    model = load_model_again(model_path)

    for img_path in img_paths:
        print(f"Making prediction for image: {img_path}")
        load_and_predict(model, img_path, class_labels)