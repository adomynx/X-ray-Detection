import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# Add root utils to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocessing import load_dataset

# Load and preprocess data
data_dir = "data/chest_xray/train"
images, labels = load_dataset(data_dir)

# Preprocessing: ResNet expects 3-channel normalized inputs
images = np.expand_dims(images, axis=-1)
images = np.repeat(images, 3, axis=-1)
images = preprocess_input(images)  # ResNet50-specific preprocessing

# Split data
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Compute class weights
classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, class_weights))

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(X_train)

# Load ResNet50 base
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(150, 150, 3))
for layer in base_model.layers:
    layer.trainable = False  # Freeze base model initially

# Add custom classification head
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

# Compile
model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# First training phase: train only head
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_val, y_val),
    epochs=10,
    callbacks=[early_stop],
    class_weight=class_weight_dict
)

# Fine-tuning: unfreeze top layers
for layer in base_model.layers[-20:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

# Second training phase: fine-tune
fine_tune_history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_val, y_val),
    epochs=10,
    callbacks=[early_stop],
    class_weight=class_weight_dict
)

# Combine history
for key in fine_tune_history.history:
    history.history[key].extend(fine_tune_history.history[key])

# Create result folders
os.makedirs("models", exist_ok=True)
os.makedirs("results/resnet50", exist_ok=True)

# Save model
model.save("models/resnet50_model.keras")

# Save accuracy plot
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("ResNet50 Transfer Learning Accuracy (with Fine-tuning)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("results/resnet50/training_plot.png")
plt.close()

# Save metrics
pd.DataFrame(history.history).to_csv("results/resnet50/metrics.csv", index=False)
