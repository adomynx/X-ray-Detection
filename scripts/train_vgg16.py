import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# ───────────────────────────────────────────────────────────────
# Project path setup and imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocessing import load_dataset

# ───────────────────────────────────────────────────────────────
# CONFIGURABLE PARAMETERS
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
MODEL_NAME = "vgg16"
MODEL_PATH = f"models/{MODEL_NAME}_model.keras"
RESULTS_DIR = f"results/{MODEL_NAME}"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)

# ───────────────────────────────────────────────────────────────
# Load and prepare dataset
data_dir = "data/chest_xray/train"
images, labels = load_dataset(data_dir)
images = np.expand_dims(images, axis=-1)
images = np.repeat(images, 3, axis=-1)  # Convert to RGB

X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# ───────────────────────────────────────────────────────────────
# Class weights
classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, class_weights))

# ───────────────────────────────────────────────────────────────
# Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(X_train)

# ───────────────────────────────────────────────────────────────
# Load VGG16 base and build model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
for layer in base_model.layers:
    layer.trainable = False

x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ───────────────────────────────────────────────────────────────
# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# ───────────────────────────────────────────────────────────────
# Train
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    callbacks=[early_stop],
    class_weight=class_weight_dict
)

# ───────────────────────────────────────────────────────────────
# Save model
model.save(MODEL_PATH)

# ───────────────────────────────────────────────────────────────
# Save metrics
pd.DataFrame(history.history).to_csv(f"{RESULTS_DIR}/metrics.csv", index=False)

# Plot training accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title("VGG16 Transfer Learning Accuracy")
plt.legend()
plt.savefig(f"{RESULTS_DIR}/training_plot.png")
plt.close()
