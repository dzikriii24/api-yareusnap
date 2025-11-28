from ultralytics import YOLO
import os
import cv2
import numpy as np
import time

# Global model instance
model = None
model_loaded = False

def load_model():
    """Load model dengan optimasi"""
    global model, model_loaded
    
    if model_loaded:
        return model
    
    MODEL_PATH = "models/best.pt"
    
    try:
        print("⚡ Loading optimized YOLO model...")
        model = YOLO(MODEL_PATH)
        
        # Set optimal inference parameters
        model.overrides['conf'] = 0.25      # Confidence threshold
        model.overrides['iou'] = 0.45       # IOU threshold
        model.overrides['imgsz'] = 320      # Optimized image size
        model.overrides['agnostic_nms'] = False
        model.overrides['max_det'] = 10     # Max detections
        model.overrides['verbose'] = False  # Disable logging
        
        model_loaded = True
        print(f"✅ Model loaded with {len(model.names)} classes")
        
        return model
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        raise e

def warmup_model():
    """Warm up model dengan inference sample"""
    model = load_model()
    # Warm up dengan sample image
    sample_image = np.ones((320, 320, 3), dtype=np.uint8) * 255
    _ = model(sample_image, verbose=False)
    print("🔥 Model warmed up!")

def detect_food_optimized(image_path):
    """Optimized food detection dengan speed focus"""
    start_time = time.time()
    
    try:
        # Load image dengan optimasi
        image = cv2.imread(image_path)
        if image is None:
            return empty_result("Cannot read image")
        
        # Resize image jika terlalu besar (speed optimization)
        original_shape = image.shape
        if max(image.shape) > 640:
            scale = 640 / max(image.shape)
            new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
        
        # Load model jika belum loaded
        model = load_model()
        
        # Ultra-fast inference dengan optimized parameters
        results = model(
            image, 
            conf=0.2,           # Lower confidence for more detections
            iou=0.4,            # Slightly lower IOU
            imgsz=320,          # Fixed optimized size
            augment=False,      # No augmentation for speed
            verbose=False,      # No logging
            max_det=8,          # Limit detections
        )
        
        detections = []
        detected_foods = set()
        
        for r in results:
            boxes = r.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    label = model.names[class_id]
                    
                    # Skip jika confidence terlalu rendah
                    if confidence < 0.15:
                        continue
                    
                    # Fast bbox conversion
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    detection = {
                        "label": label,
                        "confidence": confidence,
                        "bbox": [x1, y1, x2, y2],
                        "class_id": class_id
                    }
                    
                    detections.append(detection)
                    detected_foods.add(label)
        
        # Fast sorting
        detections.sort(key=lambda x: x["confidence"], reverse=True)
        
        processing_time = time.time() - start_time
        
        return {
            "detected_foods": list(detected_foods),
            "detections": detections,
            "total_detections": len(detections),
            "processing_time": f"{processing_time:.3f}s",
            "image_size": original_shape
        }
        
    except Exception as e:
        print(f"❌ Detection error: {e}")
        return empty_result(str(e))

def detect_food_ultrafast(image_path):
    """Ultra-fast detection untuk real-time applications"""
    try:
        # Load image in grayscale for speed (convert back to BGR)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return empty_result("Cannot read image")
        
        # Convert to BGR (YOLO expects 3 channels)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Resize to fixed small size
        image = cv2.resize(image, (320, 320), interpolation=cv2.IMREAD_GRAYSCALE)
        
        model = load_model()
        
        # Minimal inference
        results = model(image, conf=0.25, imgsz=320, verbose=False, max_det=3)
        
        detected_foods = set()
        for r in results:
            if r.boxes is not None:
                for cls_id in r.boxes.cls:
                    label = model.names[int(cls_id)]
                    detected_foods.add(label)
        
        return {
            "detected_foods": list(detected_foods),
            "detections": [],  # Skip detailed detections for speed
            "total_detections": len(detected_foods),
            "processing_time": "ultrafast",
            "mode": "ultrafast"
        }
        
    except Exception as e:
        return empty_result(str(e))

def empty_result(error_msg=""):
    """Return empty result template"""
    return {
        "detected_foods": [],
        "detections": [],
        "total_detections": 0,
        "processing_time": "0s",
        "error": error_msg
    }

def get_model_info():
    """Get model information"""
    model = load_model()
    return {
        "model_path": "models/best.pt",
        "classes_count": len(model.names),
        "classes": list(model.names.values())[:10],  # First 10 only
        "input_size": 320,
        "optimized": True,
        "status": "loaded" if model_loaded else "loading"
    }

# Legacy function untuk compatibility
def detect_food(image_path):
    result = detect_food_optimized(image_path)
    return result["detected_foods"]

def detect_food_with_details(image_path):
    return detect_food_optimized(image_path)