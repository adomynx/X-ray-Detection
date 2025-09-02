import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from tensorflow.keras.models import load_model

# Import project utility
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocessing import load_dataset

def evaluate_model(model_path, model_name, images, labels, grayscale=True):
    model = load_model(model_path)

    # Prepare input shape
    if grayscale:
        images_expanded = np.expand_dims(images, axis=-1)
    else:
        images_expanded = np.expand_dims(images, axis=-1)
        images_expanded = np.repeat(images_expanded, 3, axis=-1)

    # Predict
    proba = model.predict(images_expanded)
    predictions = (proba > 0.5).astype("int32")

    # Metrics
    acc = np.mean(predictions.flatten() == labels)
    report = classification_report(labels, predictions, output_dict=True)
    auc = roc_auc_score(labels, proba)

    # Create result folder for model
    result_dir = f"results/{model_name}"
    os.makedirs(result_dir, exist_ok=True)

    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"{result_dir}/confusion_matrix.png")
    plt.clf()

    # ROC Curve
    fpr, tpr, _ = roc_curve(labels, proba)
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.2f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend()
    plt.savefig(f"{result_dir}/roc_curve.png")
    plt.clf()

    return {
        "Model": model_name,
        "Accuracy": acc,
        "Precision": report['1']['precision'],
        "Recall": report['1']['recall'],
        "F1-Score": report['1']['f1-score'],
        "AUC": auc
    }

# Load test dataset
data_dir = "data/chest_xray/test"
images, labels = load_dataset(data_dir)
os.makedirs("results", exist_ok=True)

# Evaluate all models
metrics = []
models = [
    ("models/mobilenetv2_model.keras", "mobilenetv2", False),
    ("models/vgg16_model.keras", "vgg16", False),
    ("models/resnet50_model.keras", "resnet50", False),
    ("models/customcnn_model.keras", "customcnn", False)
]

for path, name, gray in models:
    metrics.append(evaluate_model(path, name, images, labels, grayscale=gray))

# Save comparison CSV
df = pd.DataFrame(metrics)
df.to_csv("results/metrics_comparison.csv", index=False)
print(df)
