import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print("GPUs:", tf.config.list_physical_devices('GPU'))

# Check for GPU availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU detected: {gpus[0].name}")
    # Enable memory growth to avoid VRAM allocation issues
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except:
        pass
else:
    print("No GPU detected, training on CPU")

# Fixed import - was "confusion_matraix" (typo)
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense,
                                     Dropout, GlobalAveragePooling2D)
from tensorflow.keras.applications import VGG16, MobileNetV2, ResNet50
from tensorflow.keras.optimizers import Adam

# Add utility module path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# For now, let's create a simple data loading function since utils might not exist
def load_dataset_simple(data_dir):
    """Simple dataset loader for chest X-ray data"""
    from PIL import Image
    
    normal_dir = os.path.join(data_dir, "NORMAL")
    pneumonia_dir = os.path.join(data_dir, "PNEUMONIA")
    
    images = []
    labels = []
    
    # Load normal images
    if os.path.exists(normal_dir):
        for img_name in os.listdir(normal_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(normal_dir, img_name)
                try:
                    img = Image.open(img_path).convert('L')  # Convert to grayscale
                    img = img.resize((150, 150))
                    img_array = np.array(img) / 255.0
                    images.append(img_array)
                    labels.append(0)  # Normal = 0
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
    
    # Load pneumonia images
    if os.path.exists(pneumonia_dir):
        for img_name in os.listdir(pneumonia_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(pneumonia_dir, img_name)
                try:
                    img = Image.open(img_path).convert('L')  # Convert to grayscale
                    img = img.resize((150, 150))
                    img_array = np.array(img) / 255.0
                    images.append(img_array)
                    labels.append(1)  # Pneumonia = 1
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
    
    return np.array(images), np.array(labels)

# Load and preprocess dataset
data_dir = "data/chest_xray/train"
print(f"Loading dataset from: {data_dir}")

# Check if data directory exists
if not os.path.exists(data_dir):
    print(f"Error: Data directory {data_dir} does not exist!")
    print("Please make sure your dataset is in the correct location.")
    sys.exit(1)

try:
    images, labels = load_dataset_simple(data_dir)
    print(f"Loaded {len(images)} images")
    print(f"Normal cases: {np.sum(labels == 0)}")
    print(f"Pneumonia cases: {np.sum(labels == 1)}")
except Exception as e:
    print(f"Error loading dataset: {e}")
    sys.exit(1)

# Convert grayscale to RGB by repeating channels
images = np.expand_dims(images, axis=-1)
images = np.repeat(images, 3, axis=-1)

# Split dataset
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42, stratify=labels)

print(f"Training set: {len(X_train)} images")
print(f"Validation set: {len(X_val)} images")

# Class weights to balance dataset
classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, class_weights))
print(f"Class weights: {class_weight_dict}")

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(X_train)

# Hyperparameter grid - reduced for faster testing
DROPOUTS = [0.3, 0.5]
LEARNING_RATES = [0.0001, 0.001]
DENSE_UNITS = [128, 256]

# Select model to train
model_name = "vgg16"  # Change to: "mobilenetv2", "vgg16", "customcnn", or "resnet50"
base_dir = f"results/{model_name}/hyperparameter_tuning"
os.makedirs(base_dir, exist_ok=True)

# Model builder
def build_model(name, dropout, lr, units):
    if name == "vgg16":
        base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
        for layer in base.layers:
            layer.trainable = False
        x = Flatten()(base.output)
        x = Dense(units, activation='relu')(x)
        x = Dropout(dropout)(x)
        output = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=base.input, outputs=output)
        
    elif name == "mobilenetv2":
        base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
        for layer in base.layers:
            layer.trainable = False
        x = GlobalAveragePooling2D()(base.output)
        x = Dense(units, activation='relu')(x)
        x = Dropout(dropout)(x)
        output = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=base.input, outputs=output)
        
    elif name == "resnet50":
        base = ResNet50(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
        for layer in base.layers:
            layer.trainable = False
        x = GlobalAveragePooling2D()(base.output)
        x = Dense(units, activation='relu')(x)
        x = Dropout(dropout)(x)
        output = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=base.input, outputs=output)
        
    elif name == "customcnn":
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(units, activation='relu'),
            Dropout(dropout),
            Dense(1, activation='sigmoid')
        ])
    else:
        raise ValueError(f"Unsupported model: {name}")

    model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Run tuning
summary = []
run_id = 1
total_runs = len(DROPOUTS) * len(LEARNING_RATES) * len(DENSE_UNITS)

print(f"\nStarting hyperparameter tuning: {total_runs} combinations")

for d in DROPOUTS:
    for lr in LEARNING_RATES:
        for u in DENSE_UNITS:
            print(f"\nRun {run_id}/{total_runs}: {model_name} | Dropout={d}, LR={lr}, Units={u}")
            
            try:
                model = build_model(model_name, d, lr, u)
                early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

                history = model.fit(
                    datagen.flow(X_train, y_train, batch_size=32),
                    validation_data=(X_val, y_val),
                    epochs=10,  # Reduced for faster testing
                    callbacks=[early_stop],
                    class_weight=class_weight_dict,
                    verbose=1
                )

                run_dir = os.path.join(base_dir, f"run_{run_id}")
                os.makedirs(run_dir, exist_ok=True)

                # Save model
                model.save(os.path.join(run_dir, f"{model_name}_d{d}_lr{lr}_u{u}.keras"))

                # Save training plot
                plt.figure(figsize=(10, 4))
                plt.subplot(1, 2, 1)
                plt.plot(history.history['accuracy'], label='Train Acc')
                plt.plot(history.history['val_accuracy'], label='Val Acc')
                plt.title(f"{model_name.upper()} Accuracy")
                plt.xlabel("Epoch")
                plt.ylabel("Accuracy")
                plt.legend()
                
                plt.subplot(1, 2, 2)
                plt.plot(history.history['loss'], label='Train Loss')
                plt.plot(history.history['val_loss'], label='Val Loss')
                plt.title(f"{model_name.upper()} Loss")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.legend()
                
                plt.tight_layout()
                plt.savefig(os.path.join(run_dir, "training_plot.png"))
                plt.close()

                # Save metrics
                pd.DataFrame(history.history).to_csv(os.path.join(run_dir, "metrics.csv"), index=False)
                with open(os.path.join(run_dir, "config.json"), "w") as f:
                    json.dump({"dropout": d, "learning_rate": lr, "units": u}, f, indent=4)

                # Evaluation
                proba = model.predict(X_val)
                pred = (proba > 0.5).astype("int32")

                auc = roc_auc_score(y_val, proba)
                
                # ROC Curve
                fpr, tpr, _ = roc_curve(y_val, proba)
                plt.figure()
                plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
                plt.plot([0, 1], [0, 1], 'k--', label='Random')
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("ROC Curve")
                plt.legend()
                plt.savefig(os.path.join(run_dir, "roc_curve.png"))
                plt.close()

                # Confusion Matrix
                cm = confusion_matrix(y_val, pred)
                plt.figure()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title("Confusion Matrix")
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                plt.savefig(os.path.join(run_dir, "confusion_matrix.png"))
                plt.close()

                # Summary
                best_val_acc = max(history.history['val_accuracy'])
                final_val_loss = min(history.history['val_loss'])
                
                summary.append({
                    "Run": run_id,
                    "Dropout": d,
                    "LearningRate": lr,
                    "Units": u,
                    "BestValAccuracy": best_val_acc,
                    "FinalValLoss": final_val_loss,
                    "AUC": auc
                })

                print(f"✓ Completed - Val Acc: {best_val_acc:.3f}, AUC: {auc:.3f}")
                
            except Exception as e:
                print(f"✗ Error in run {run_id}: {e}")
                summary.append({
                    "Run": run_id,
                    "Dropout": d,
                    "LearningRate": lr,
                    "Units": u,
                    "BestValAccuracy": 0,
                    "FinalValLoss": 999,
                    "AUC": 0,
                    "Error": str(e)
                })

            run_id += 1

# Save summary
summary_df = pd.DataFrame(summary)
summary_df.to_csv(os.path.join(base_dir, "tuning_summary.csv"), index=False)

# Display best results
print(f"\nBest results:")
best_run = summary_df.loc[summary_df['BestValAccuracy'].idxmax()]
print(f"Best validation accuracy: {best_run['BestValAccuracy']:.3f}")
print(f"Best configuration: Dropout={best_run['Dropout']}, LR={best_run['LearningRate']}, Units={best_run['Units']}")

# Create results visualization
if len(summary_df) > 0:
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.scatter(summary_df['Dropout'], summary_df['BestValAccuracy'])
    plt.xlabel('Dropout')
    plt.ylabel('Best Val Accuracy')
    plt.title('Dropout vs Accuracy')
    
    plt.subplot(1, 3, 2)
    plt.scatter(summary_df['LearningRate'], summary_df['BestValAccuracy'])
    plt.xlabel('Learning Rate')
    plt.ylabel('Best Val Accuracy')
    plt.title('Learning Rate vs Accuracy')
    plt.xscale('log')
    
    plt.subplot(1, 3, 3)
    plt.scatter(summary_df['Units'], summary_df['BestValAccuracy'])
    plt.xlabel('Dense Units')
    plt.ylabel('Best Val Accuracy')
    plt.title('Units vs Accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "hyperparameter_analysis.png"))
    plt.close()

print(f"\nTuning complete! Results saved in: {base_dir}")
print(f"Summary: {len(summary_df)} runs completed")