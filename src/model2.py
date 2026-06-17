from ultralytics import YOLO
import cv2
import os
import numpy as np

# ===============================
# CONFIG
# ===============================

MODEL_PATH = "/home/dhruv/Documents/projects/Fruit-Quality-Classification/runs/apple_detector6/weights/best.pt"
IMAGE_FOLDER = "/home/dhruv/Documents/projects/Fruit-Quality-Classification/src/image_folder"

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

# weight assumptions
AVERAGE_APPLE_WEIGHT = 180  # grams per apple
MARKET_PRICE_PER_KG = 120   # example market rate (₹/kg)

model = YOLO(MODEL_PATH)


# ===============================
# DETECTION CLEANING
# ===============================

def get_detections(results):

    boxes = results[0].boxes

    if boxes is None:
        return []

    xyxy = boxes.xyxy.cpu().numpy()

    return xyxy


# ===============================
# WEIGHT ESTIMATION
# ===============================

def estimate_weight(count):

    total_weight_g = count * AVERAGE_APPLE_WEIGHT
    total_weight_kg = total_weight_g / 1000

    return total_weight_g, total_weight_kg


# ===============================
# PRICE ESTIMATION
# ===============================

def estimate_price(weight_kg):

    price = weight_kg * MARKET_PRICE_PER_KG

    return round(price, 2)


# ===============================
# IMAGE MODE
# ===============================

print("\n===== IMAGE COUNTING MODE =====\n")

for file in os.listdir(IMAGE_FOLDER):

    if not file.lower().endswith((".jpg",".jpeg",".png")):
        continue

    path = os.path.join(IMAGE_FOLDER, file)

    img = cv2.imread(path)

    if img is None:
        print(f"Could not load {file}")
        continue

    results = model.predict(
        img,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        imgsz=640,
        verbose=False
    )

    boxes = get_detections(results)

    apple_count = len(boxes)

    weight_g, weight_kg = estimate_weight(apple_count)

    price = estimate_price(weight_kg)

    print(f"\n{file}")
    print(f"Apples detected : {apple_count}")
    print(f"Estimated weight: {weight_g} g ({weight_kg:.2f} kg)")
    print(f"Estimated price : ₹{price}")


print("\nImage processing complete.\n")


# ===============================
# CAMERA MODE
# ===============================

print("===== CAMERA COUNTING MODE =====")

cap = cv2.VideoCapture(0)

stable_frames = 0
previous_count = -1
STABLE_REQUIRED = 20
final_count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model.predict(
        frame,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        imgsz=640,
        verbose=False
    )

    boxes = get_detections(results)

    current_count = len(boxes)

    if current_count == previous_count:
        stable_frames += 1
    else:
        stable_frames = 0

    previous_count = current_count

    annotated = frame.copy()

    for box in boxes:
        x1,y1,x2,y2 = map(int,box)
        cv2.rectangle(annotated,(x1,y1),(x2,y2),(0,255,0),2)

    weight_g, weight_kg = estimate_weight(current_count)
    price = estimate_price(weight_kg)

    cv2.putText(
        annotated,
        f"Apples: {current_count}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        3
    )

    cv2.putText(
        annotated,
        f"Weight: {weight_kg:.2f} kg",
        (20,80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,255,0),
        2
    )

    cv2.putText(
        annotated,
        f"Price: ₹{price}",
        (20,120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,200,255),
        2
    )

    cv2.imshow("Fruit Pricing System", annotated)

    if stable_frames >= STABLE_REQUIRED:
        final_count = current_count
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()


final_weight_g, final_weight_kg = estimate_weight(final_count)
final_price = estimate_price(final_weight_kg)

print("\n===== FINAL RESULT =====")
print(f"Apples detected : {final_count}")
print(f"Estimated weight: {final_weight_kg:.2f} kg")
print(f"Estimated price : ₹{final_price}")