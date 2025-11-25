from ultralytics import YOLO
import os

# load the model
MODEL_PATH = "models/best.pt"
model = YOLO(MODEL_PATH)

def detect_food(image_path):
    results = model(image_path)
    items = []

    for r in results:
        for cls_id in r.boxes.cls:
            label = model.names[int(cls_id)]
            items.append(label)

    return list(set(items))
