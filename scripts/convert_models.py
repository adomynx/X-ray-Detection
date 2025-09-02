import os

# Import standalone Keras loader (Keras 3)
from keras.models import load_model as k_load_model

MODELS = {
    "customcnn": "models/customcnn_model.keras",
    "mobilenetv2": "models/mobilenetv2_model.keras",
    "resnet50": "models/resnet50_model.keras",
    "vgg16": "models/vgg16_model.keras"
}

def convert_model(keras_path, h5_path):
    try:
        print(f"[INFO] Converting {keras_path} -> {h5_path}")
        model = k_load_model(keras_path, compile=False, safe_mode=False)
        model.save(h5_path)  # save in H5 format
        print(f"[SUCCESS] Saved H5 model: {h5_path}")
    except Exception as e:
        print(f"[ERROR] Failed to convert {keras_path}: {e}")

def main():
    for name, keras_path in MODELS.items():
        if not os.path.exists(keras_path):
            print(f"[WARNING] File not found: {keras_path}")
            continue
        h5_path = keras_path.replace(".keras", ".h5")
        convert_model(keras_path, h5_path)

if __name__ == "__main__":
    main()
