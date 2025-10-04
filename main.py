from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import io
from PIL import Image
import random
import json
from typing import List

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model & class labels
MODEL_PATH = "silkworm_model.h5"
CLASS_INDICES_PATH = "class_indices.json"

model = load_model(MODEL_PATH)

with open(CLASS_INDICES_PATH, "r") as f:
    class_indices = json.load(f)
class_labels = list(class_indices.keys())

DISEASES = [
    {
        "name": "Grasserie",
        "measures": [
            "Remove and destroy diseased larvae immediately",
            "Disinfect rearing house with 2% formalin",
            "Avoid overcrowding of larvae",
            "Maintain optimal temperature and humidity"
        ]
    },
    {
        "name": "Flacherie",
        "measures": [
            "Feed leaves free from pesticides",
            "Ensure proper ventilation",
            "Maintain hygiene in rearing trays",
            "Avoid overfeeding mulberry leaves"
        ]
    },
    {
        "name": "Pebrine",
        "measures": [
            "Use disease-free layings (DFLs)",
            "Destroy infected worms and moths",
            "Disinfect rearing equipment",
            "Conduct microscopic examination of moths"
        ]
    },
    {
        "name": "Muscardine",
        "measures": [
            "Dust larvae with 2% slaked lime",
            "Maintain dry and clean environment",
            "Avoid excess humidity",
            "Remove dead larvae immediately"
        ]
    }
]

# ===============================
# Prediction Endpoint (multiple images)
# ===============================
@app.post("/predict")
async def predict(files: List[UploadFile] = File(...)):
    results = []
    try:
        for file in files:
            contents = await file.read()
            img = Image.open(io.BytesIO(contents)).convert("RGB")
            img = img.resize((224, 224))  # match training input
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            predictions = model.predict(img_array)
            confidence = float(np.max(predictions))
            predicted_index = int(np.argmax(predictions))
            predicted_label = class_labels[predicted_index]

            result = {
                "filename": file.filename,
                "label": predicted_label,
                "confidence": confidence,
                "probabilities": dict(zip(class_labels, predictions[0].astype(float)))
            }

            if predicted_label == "diseased":
                disease = random.choice(DISEASES)
                result["disease_name"] = disease["name"]
                result["preventive_measures"] = disease["measures"]

            results.append(result)

        return {"success": True, "predictions": results}

    except Exception as e:
        return {"success": False, "error": str(e)}
