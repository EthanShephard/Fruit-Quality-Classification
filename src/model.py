import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras import mixed_precision
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPool2D,
    Rescaling,
)
from tensorflow.keras.models import Sequential
from tensorflow.python.client import device_lib
from tqdm import tqdm

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["NVIDIA_VISIBLE_DEVICES"] = "all"
os.environ["NVIDIA_DRIVER_CAPABILITIES"] = "compute,utility"

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU detected and enabled:", gpus)
    except RuntimeError as e:
        print("GPU memory growth setup failed:", e)
else:
    print("No GPU detected — TensorFlow will use CPU.")

print("Num GPUs Available:", len(gpus))


mixed_precision.set_global_policy("mixed_float16")


print("\nTensorFlow Device List:")
print(device_lib.list_local_devices())

data_dir = "/home/dhruv/Downloads/Processed_Images_Fruits/"
class_labels = sorted(os.listdir(data_dir))
print("Detected Classes:", class_labels)

chart_data = []
for folder in class_labels:
    path = os.path.join(data_dir, folder)
    size = len(os.listdir(path))
    chart_data.append(size)
    print(f"The folder '{folder}' contains {size} samples")

img_size = (96, 96)
batch_size = 32

train_data = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=img_size,
    batch_size=batch_size,
    shuffle=True,
)

val_data = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=img_size,
    batch_size=batch_size,
    shuffle=True,
)


normalization_layer = Rescaling(1.0 / 255)
train_data = train_data.map(lambda x, y: (normalization_layer(x), y))
val_data = val_data.map(lambda x, y: (normalization_layer(x), y))

train_data = train_data.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
val_data = val_data.cache().prefetch(buffer_size=tf.data.AUTOTUNE)


cnn_model = Sequential(
    [
        Input(shape=(96, 96, 3)),
        Conv2D(32, 3, padding="same", activation="relu"),
        Conv2D(32, 3, padding="same", activation="relu"),
        BatchNormalization(),
        MaxPool2D(2),
        Conv2D(64, 3, padding="same", activation="relu"),
        Conv2D(64, 3, padding="same", activation="relu"),
        BatchNormalization(),
        MaxPool2D(2, padding="same"),
        Conv2D(128, 3, padding="same", activation="relu"),
        Conv2D(128, 3, padding="same", activation="relu"),
        BatchNormalization(),
        MaxPool2D(2, padding="same"),
        Flatten(),
        Dropout(0.4),
        Dense(256, activation="relu"),
        Dense(len(class_labels), activation="softmax", dtype="float32"),
    ]
)

cnn_model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

cnn_model.summary()

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "model/best_fruit_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1,
)

history = cnn_model.fit(
    train_data,
    epochs=10,
    validation_data=val_data,
    callbacks=[checkpoint],
)

val_loss, val_acc = cnn_model.evaluate(val_data)
print(f"Validation Accuracy: {val_acc*100:.2f}%")

y_true = np.concatenate([y for x, y in val_data], axis=0)
y_pred_probs = cnn_model.predict(val_data)
y_pred = np.argmax(y_pred_probs, axis=1)

print(classification_report(y_true, y_pred, target_names=class_labels))

plt.figure(figsize=(6, 6))
sns.heatmap(
    confusion_matrix(y_true, y_pred),
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_labels,
    yticklabels=class_labels,
)
plt.title("Confusion Matrix")
plt.show()

cnn_model.save("model/final_fruit_model.h5")
print("Model saved successfully.")
