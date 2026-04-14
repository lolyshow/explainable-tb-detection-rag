# inference.py

import torch
import numpy as np
import cv2
import base64
import io

from pathlib import Path
from PIL import Image

from torchvision import transforms
from transformers import AutoTokenizer

from model import ResNet50MedVQA
from explainability import GradCAM, GradCAMPlusPlus, ScoreCAM
from rag import retrieve_rag_context, load_corpus


# =========================
# CONFIG
# =========================
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ANSWER_VOCAB = ["Yes", "No", "Uncertain"]

# =========================
# LOAD TOKENIZER
# =========================
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


# =========================
# IMAGE TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================
# LOAD MODEL (ONCE)
# =========================
model = ResNet50MedVQA(
    bert_model_name="bert-base-uncased",
    num_answers=len(ANSWER_VOCAB),
    cnn_trainable=False,
    dropout=0.3
).to(DEVICE)

ckpt = torch.load("resnet50_best.pt", map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])

model.eval()

# Enable gradients for CAM
for p in model.cnn_backbone.parameters():
    p.requires_grad = True


# =========================
# SETUP CAM
# =========================
target_layer = model.cnn_backbone[-1][-1].conv3

gradcam   = GradCAM(model, target_layer)
gradcampp = GradCAMPlusPlus(model, target_layer)
scorecam  = ScoreCAM(model, target_layer)


# =========================
# LOAD RAG CORPUS
# =========================
load_corpus([
    "statpearl_corpus.txt",
    "9789241511506-eng.txt"
])


# =========================
# HEATMAP OVERLAY
# =========================
def overlay_cam(image_path, cam):
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    cam = cv2.resize(cam, (img.shape[1], img.shape[0]))

    heatmap = cv2.applyColorMap(
        (cam * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    return overlay


def encode_image_base64(np_img):
    pil_img = Image.fromarray(np_img)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


# =========================
# MAIN INFERENCE FUNCTION
# =========================
def run_inference(
    image_path,
    question="Is tuberculosis present in the chest X-ray?",
    cam_method="gradcam",     # "gradcam", "gradcam++", "scorecam"
    top_k_rag=3
):
    image_path = Path(image_path)

    # ---- 1. LOAD IMAGE ----
    img_pil = Image.open(image_path).convert("RGB")
    img_t = transform(img_pil).unsqueeze(0).to(DEVICE)

    # ---- 2. TOKENIZE QUESTION ----
    enc = tokenizer(
        question,
        max_length=64,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    ids = enc["input_ids"].to(DEVICE)
    mask = enc["attention_mask"].to(DEVICE)

    # ---- 3. SELECT CAM ----
    if cam_method == "gradcam":
        cam_extractor = gradcam
    elif cam_method == "gradcam++":
        cam_extractor = gradcampp
    else:
        cam_extractor = scorecam   # default

    # ---- 4. COMPUTE CAM ----
    cam, class_idx = cam_extractor.compute(img_t, ids, mask)

    # ---- 5. PREDICTION ----
    model.eval()
    with torch.no_grad():
        logits, _, _ = model(img_t, ids, mask)
        probs = torch.softmax(logits, dim=1)[0]

    confidence = probs[class_idx].item()
    predicted_answer = ANSWER_VOCAB[class_idx]

    # ---- 6. RAG EXPLANATION ----
    rag_text = retrieve_rag_context(
        question=question,
        predicted_answer=predicted_answer,
        image_id=image_path.name,
        top_k=top_k_rag
    )

    # ---- 7. HEATMAP ----
    overlay = overlay_cam(image_path, cam)
    heatmap_base64 = encode_image_base64(overlay)

    # ---- 8. RETURN OUTPUT ----
    return {
        "answer": predicted_answer,
        "confidence": float(confidence),
        "cam_method": cam_method,
        "rag_context": rag_text,
        "heatmap": heatmap_base64
    }