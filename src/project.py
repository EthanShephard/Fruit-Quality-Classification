import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

model = load_model(filepath="/model/final_fruit_model.h5")


def prediction(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (96, 96))
    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    predict = model.predict(img)
    predicted = np.argmax(predict)

    types = [
        "Apple_Bad",
        "Apple_Good",
        "Banana_Bad",
        "Banana_Good",
        "Guava_Bad",
        "Guava_Good",
        "Lime_Bad",
        "Lime_Good",
        "Orange_Bad",
        "Orange_Good",
        "Pomegranate_Bad",
        "Pomegranate_Good",
    ]

    print("Predicted output shape:", predict.shape)
    print("Raw prediction:", predict)
    print("Predicted index:", predicted)
    print("Available classes:", len(types))
    return types[predicted]


result = prediction("apple.jpg")
print("The fruit is ", result)
