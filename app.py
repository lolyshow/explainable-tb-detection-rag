# app.py

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import shutil
import uuid

from pathlib import Path

from inference import run_inference


app = FastAPI()

# =========================
# DIRECTORIES
# =========================
UPLOADS = Path("uploads")
UPLOADS.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")


# =========================
# HOME PAGE
# =========================
@app.get("/")
def home():
    return FileResponse("static/index.html")


# =========================
# PREDICTION ENDPOINT
# =========================
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    question: str = Form(...)
):

    file_id = str(uuid.uuid4())
    image_path = UPLOADS / f"{file_id}.png"

    with open(image_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    result = run_inference(
        image_path=image_path,
        question=question,
        cam_method="gradcam"
    )

    return result