from fastapi import FastAPI, UploadFile
import shutil
import os
from app.yolo_detector import detect_food
from app.mistral_service import ask_mistral
from app.nutrition_advisor import build_prompt

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/predict")
async def predict(file: UploadFile):

    save_path = f"{UPLOAD_DIR}/{file.filename}"
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # YOLO detect
    detected_foods = detect_food(save_path)

    # nutrisi + rekomendasi
    prompt = build_prompt(detected_foods)
    advice = ask_mistral(prompt)

    return {
        "detected_food": detected_foods,
        "nutrition_advice": advice
    }


@app.get("/")
def home():
    return {"status": "Food Detection API running!"}
