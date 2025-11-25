from ultralytics import YOLO
import os
import cv2
import numpy as np

# Load the model
MODEL_PATH = "models/best.pt"
model = YOLO(MODEL_PATH)

def detect_food(image_path):
    """Basic food detection - returns list of food names"""
    results = model(image_path, conf=0.25)
    items = []

    for r in results:
        for cls_id in r.boxes.cls:
            label = model.names[int(cls_id)]
            items.append(label)

    return list(set(items))

def detect_food_with_details(image_path):
    """Advanced food detection with confidence scores and bounding boxes"""
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return {"detected_foods": [], "detections": [], "error": "Cannot read image"}
        
        # Run detection
        results = model(image_path, conf=0.2)
        
        detections = []
        detected_foods = set()
        
        for r in results:
            boxes = r.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    label = model.names[class_id]
                    
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    detection = {
                        "label": label,
                        "confidence": confidence,
                        "bbox": [x1, y1, x2, y2],
                        "class_id": class_id
                    }
                    
                    detections.append(detection)
                    detected_foods.add(label)
        
        # Sort by confidence
        detections.sort(key=lambda x: x["confidence"], reverse=True)
        
        return {
            "detected_foods": list(detected_foods),
            "detections": detections,
            "total_detections": len(detections),
            "image_size": image.shape
        }
        
    except Exception as e:
        return {
            "detected_foods": [],
            "detections": [],
            "error": str(e)
        }

def get_model_info():
    """Get information about the loaded model"""
    return {
        "model_path": MODEL_PATH,
        "classes_count": len(model.names),
        "classes": list(model.names.values()),
        "input_size": model.overrides.get("imgsz", 640)
    }