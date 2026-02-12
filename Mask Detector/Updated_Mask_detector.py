import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

# ==============================
# Configuration
# ==============================
INIT_LR = 1e-4
FINE_TUNE_LR = 1e-5
EPOCHS = 10
FINE_TUNE_EPOCHS = 5
BATCH_SIZE = 32
IMG_SIZE = (224, 224)

DATASET_DIR = r"Mask Dataset/facemask-dataset/dataset"
for class_dir in ["with_mask", "without_mask"]:
    path = os.path.join(DATASET_DIR, class_dir)
    for filename in os.listdir(path):
        if filename.lower().endswith(".png"):
            img_path = os.path.join(path, filename)
            img = Image.open(img_path)
            img.save(img_path)
# ==============================
# Load Dataset (Memory Efficient)
# ==============================
train_ds = keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print("Classes:", class_names)

# Performance optimization
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# ==============================
# Data Augmentation
# ==============================
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.15),
    layers.RandomTranslation(0.1, 0.1),
])

# ==============================
# Build Model
# ==============================
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False  # freeze base model

inputs = keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(2, activation="softmax")(x)

model = keras.Model(inputs, outputs)

# ==============================
# Compile (Stage 1: Feature Extraction)
# ==============================
model.compile(
    optimizer=Adam(learning_rate=INIT_LR),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n[INFO] Training head...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# ==============================
# Stage 2: Fine Tuning
# ==============================
base_model.trainable = True

# Freeze earlier layers, unfreeze last 30 layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=FINE_TUNE_LR),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n[INFO] Fine-tuning model...")
history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS
)

# ==============================
# Save Model
# ==============================
model.save("mask_detector.keras")
print("Model saved successfully.")

# ==============================
# Plot Results
# ==============================
acc = history.history["accuracy"] + history_fine.history["accuracy"]
val_acc = history.history["val_accuracy"] + history_fine.history["val_accuracy"]
loss = history.history["loss"] + history_fine.history["loss"]
val_loss = history.history["val_loss"] + history_fine.history["val_loss"]

epochs_range = range(len(acc))

plt.figure()
plt.plot(epochs_range, acc, label="Train Accuracy")
plt.plot(epochs_range, val_acc, label="Validation Accuracy")
plt.plot(epochs_range, loss, label="Train Loss")
plt.plot(epochs_range, val_loss, label="Validation Loss")
plt.legend()
plt.title("Training and Validation Metrics")
plt.savefig("training_plot.png")
plt.show()
