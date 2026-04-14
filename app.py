# app.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
import shutil
from pathlib import Path
import uuid

from inference import run_inference

app = FastAPI()

UPLOADS = Path("uploads")
UPLOADS.mkdir(exist_ok=True)


@app.post("/predict")
async def predict(file: UploadFile = File(...), question: str = Form(...)):

    file_id = str(uuid.uuid4())
    path = UPLOADS / f"{file_id}.png"

    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    result = run_inference(path, question)

    return result


@app.get("/")
def root():
    return FileResponse("static/index.html")