import numpy as np

# =====================
# CONFIG
# =====================
AVG_WEIGHT = 180  # grams
PRICE_PER_KG = 120


# =====================
# CORE PIPELINE
# =====================
def process_image(model, image):

    results = model.predict(
        image,
        conf=0.3,
        iou=0.45,
        imgsz=640,
        verbose=False
    )

    boxes = results[0].boxes

    if boxes is not None:
        count = len(boxes)
    else:
        count = 0

    # =====================
    # WEIGHT ESTIMATION
    # =====================
    weight_kg = (count * AVG_WEIGHT) / 1000

    # =====================
    # PRICE ESTIMATION
    # =====================
    price = weight_kg * PRICE_PER_KG

    annotated = results[0].plot()

    return annotated, count, weight_kg, price