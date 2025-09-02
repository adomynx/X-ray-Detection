import cv2
import os
import numpy as np

def preprocess_image(image_path, target_size=(150, 150)):
    """
    Load an image in grayscale, resize, normalize, and return.
    Used for prediction and evaluation.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unable to load image at {image_path}")
    image = cv2.resize(image, target_size)
    image = image.astype("float32") / 255.0
    return image

def load_dataset(data_dir, target_size=(150, 150)):
    """
    Load and preprocess all images in a dataset directory.
    Returns (images, labels)
    - NORMAL = 0
    - PNEUMONIA = 1
    """
    images = []
    labels = []
    for label in ["NORMAL", "PNEUMONIA"]:
        class_dir = os.path.join(data_dir, label)
        if not os.path.isdir(class_dir):
            continue
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            try:
                image = preprocess_image(image_path, target_size)
                images.append(image)
                labels.append(0 if label == "NORMAL" else 1)
            except Exception as e:
                print(f"Skipping {image_path}: {e}")
    return np.array(images), np.array(labels)
