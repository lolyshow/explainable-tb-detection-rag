# rag.py

import re
from pathlib import Path
import torch

from sentence_transformers import SentenceTransformer
from sentence_transformers import util as sbert_util
import numpy as np                                     # Core numerical operations


# =========================
# CONFIG
# =========================
SBERT_MODEL_NAME = "all-mpnet-base-v2"

# Will be set at runtime
corpus_passages = []
corpus_embeddings = None

sbert_model = SentenceTransformer(SBERT_MODEL_NAME)


# =========================
# PASSAGE CLEANING
# =========================
def clean_passage(text: str) -> str:
    text = text.strip()

    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^\d+[\.\)]\s*", "", text)
    text = re.sub(r"\b\d+\s+\d+\b", "", text)
    text = re.sub(r"\s{2,}", " ", text).strip()

    # Remove section headers at start
    text = re.sub(
        r"^\d+\s+[A-Z][a-zA-Z\s\(\)\-]+?(?=\s{2,}|\s[A-Z][a-z]|It |The |This |A |An |In |When |Although |For |Despite )",
        "",
        text
    ).strip()

    # Reject ALL CAPS lines
    if re.fullmatch(r"[A-Z0-9\s\(\)\-\/:,\.]+", text):
        return ""

    # Reject embedded ALL CAPS blocks
    if re.search(r"\b[A-Z]{4,}(?:\s+[A-Z]{2,}){2,}\b", text):
        return ""

    # Reject references / structural lines
    if re.search(r"\b(Reference:|Chapter\s+\d|Section\s+\d|Table\s+\d|Figure\s+\d)", text, re.IGNORECASE):
        return ""

    # Reject table-of-contents patterns
    if re.search(r"\d\s+\d", text):
        return ""

    # Reject low text density
    letter_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
    if letter_ratio < 0.55:
        return ""

    # Reject too short
    if len(text.split()) < 12:
        return ""

    # Reject no lowercase words
    if not re.search(r"[a-z]{3,}", text):
        return ""

    # Reject no verb-like structure
    if not re.search(
        r"\b(is|are|was|were|has|have|may|can|should|indicate|suggest|show|present|detect|diagnose|require|identify|include|occur|cause|result|demonstrate|confirm|reveal)\b",
        text,
        re.IGNORECASE
    ):
        return ""

    # Reject too long
    if len(text.split()) > 80:
        return ""

    return text


# =========================
# LOAD CORPUS
# =========================
def load_corpus(corpus_files):
    global corpus_passages, corpus_embeddings

    corpus_passages = []

    for corpus_file in corpus_files:
        corpus_file = Path(corpus_file)

        if not corpus_file.exists():
            print(f"Warning: missing corpus file: {corpus_file}")
            continue

        text = corpus_file.read_text(encoding="utf-8", errors="ignore")

        # Split into sentence-like chunks
        chunks = [s.strip() for s in text.split(".") if len(s.strip()) > 40]

        corpus_passages.extend(chunks)
        print(f"Loaded {corpus_file.name}: {len(chunks)} passages")

    print(f"Total passages: {len(corpus_passages)}")

    # Encode corpus
    corpus_embeddings = sbert_model.encode(
        corpus_passages,
        batch_size=64,
        show_progress_bar=True,
        convert_to_tensor=True
    )

    print("Corpus embeddings ready:", corpus_embeddings.shape)


# =========================
# RETRIEVAL
# =========================
def retrieve_rag_context(
    question: str,
    predicted_answer: str,
    image_id: str = "",
    top_k: int = 3
) -> str:
    """
    Retrieves and formats the top-k most semantically relevant corpus passages
    as a coherent clinical explanation for the given question and predicted answer.

    The query is made image-aware using the image_id so that different images
    with the same predicted answer do not always retrieve identical passages.

    Parameters:
        question (str)          , clinical question asked of the model
        predicted_answer (str)  , answer predicted by the Med-VQA model
        top_k (int)             , number of passages to retrieve
        image_id (str)          , image filename stem used to diversify retrieval

    Returns:
        str , a formatted, evidence-grounded clinical explanation paragraph
    """
    # Image-aware query
    query = (
        f"{question} "
        f"The model predicted: {predicted_answer}. "
        f"Image reference: {image_id}."
    )

    query_emb = sbert_model.encode(query, convert_to_tensor=True)

    # Similarity search
    scores = sbert_util.cos_sim(query_emb, corpus_embeddings)[0]

    top_indices = torch.topk(
        scores,
        k=min(top_k * 6, len(corpus_passages))
    ).indices.tolist()

    # Deterministic variation using image_id
    offset = hash(image_id) % max(len(top_indices) // 4, 1)
    rotated = top_indices[offset:] + top_indices[:offset]

    clean_passages = []

    for i in rotated:
        cleaned = clean_passage(corpus_passages[i])
        if cleaned:
            clean_passages.append(cleaned)

        if len(clean_passages) >= top_k:
            break

    # =========================
    # FORMAT OUTPUT
    # =========================
    if predicted_answer == "Yes":
        intro = (
            "The chest X-ray analysis indicates findings consistent with tuberculosis. "
            "Supporting clinical evidence is summarised below."
        )
    elif predicted_answer == "No":
        intro = (
            "No radiological findings consistent with active tuberculosis were identified. "
            "Supporting clinical context is provided below."
        )
    else:
        intro = (
            "The findings are inconclusive for tuberculosis diagnosis. "
            "Relevant clinical evidence is provided below."
        )

    evidence = "\n\n".join(
        f"[Evidence {i+1}]:\n{p.rstrip('.')}."
        for i, p in enumerate(clean_passages)
    )

    return f"{intro}\n\n{evidence}"


def interpret_heatmap_position(cam: np.ndarray, threshold: float = 0.65) -> str:
    """
    Interprets the spatial position of peak saliency regions on a chest X-ray
    and generates a clinically meaningful anatomical description of where
    the model is focusing its attention.

    The chest X-ray is divided into a 3x3 anatomical grid:
    - Vertically:   upper, middle, lower lung zones
    - Horizontally: right lung field, central/perihilar region, left lung field

    Parameters:
        cam (np.ndarray)    , normalized saliency heatmap of shape (H, W) in [0, 1]
        threshold (float)   , activation threshold above which regions are considered salient

    Returns:
        str , a structured anatomical report describing heatmap positioning
    """
    h, w = cam.shape

    # Thresholds the heatmap to isolate the most salient regions
    salient_mask     = (cam >= threshold).astype(np.uint8)                # Binary mask of high-attention regions
    salient_coverage = salient_mask.sum() / (h * w)                      # Fraction of total image area that is salient

    # Flags excessively diffuse attention , coverage above 50% is not clinically localised
    # This typically indicates the model has not converged on a specific anatomical finding
    if salient_coverage > 0.50:
        peak_y, peak_x = np.unravel_index(np.argmax(cam), cam.shape)
        return (
            f"The saliency map covers {salient_coverage * 100:.1f}% of the image area, "
            f"with peak activation at coordinates ({peak_x}, {peak_y}). "
            f"This level of diffuse activation does not correspond to a localised anatomical finding. "
            f"The attention pattern suggests the model has not converged on a specific region of pathological interest. "
            f"Interpretation of this heatmap as a clinical localisation should be treated with caution "
            f"and considered alongside quantitative IoU and Pointing-Game scores."
        )

    # Finds the peak activation coordinate , the single most attended location
    peak_y, peak_x = np.unravel_index(np.argmax(cam), cam.shape)         # Row and column of maximum activation

    # Determines vertical lung zone from peak position
    if peak_y < h / 3:
        vertical_zone = "upper lung zone"                                 # Top third , classical TB territory
    elif peak_y < 2 * h / 3:
        vertical_zone = "middle lung zone"                                # Middle third of the image
    else:
        vertical_zone = "lower lung zone"                                 # Bottom third of the image

    # Determines horizontal lung field from peak position
    # Note: image left corresponds to patient right lung and vice versa on standard PA CXR
    if peak_x < w / 3:
        horizontal_field = "right lung field"                             # Image left = patient right lung
    elif peak_x < 2 * w / 3:
        horizontal_field = "central/perihilar region"                     # Central region near hilum and mediastinum
    else:
        horizontal_field = "left lung field"                              # Image right = patient left lung

    # Classifies spread based on salient coverage percentage
    if salient_coverage < 0.05:
        spread = "focal"                                                  # Highly concentrated attention in a small area
    elif salient_coverage < 0.20:
        spread = "regional"                                               # Moderate spread across a lung zone
    else:
        spread = "multifocal"                                             # Multiple areas of attention within a zone

    # Checks whether salient regions are bilateral by comparing left and right halves
    left_activation  = cam[:, : w // 2].mean()                           # Mean activation in image left half
    right_activation = cam[:, w // 2 :].mean()                           # Mean activation in image right half
    activation_ratio = max(left_activation, right_activation) / (
        min(left_activation, right_activation) + 1e-8
    )

    if activation_ratio > 2.0:
        laterality = "predominantly unilateral"                           # One side dominates activation significantly
    else:
        laterality = "bilateral"                                          # Both lung fields show comparable activation

    # Computes upper vs lower zone dominance
    upper_activation = cam[: h // 2, :].mean()                           # Mean activation in upper lung zones
    lower_activation = cam[h // 2 :, :].mean()                           # Mean activation in lower lung zones

    if upper_activation > lower_activation * 1.3:
        zone_dominance = "upper zone predominant"                         # TB classically favours upper lobe involvement
    elif lower_activation > upper_activation * 1.3:
        zone_dominance = "lower zone predominant"
    else:
        zone_dominance = "evenly distributed across lung zones"

    # Assembles the anatomical report
    report = (
        f"The model's attention is concentrated in the {vertical_zone}, {horizontal_field}. "
        f"The saliency pattern is {spread} and {laterality}, with {zone_dominance} activation. "
        f"Peak attention is located at image coordinates ({peak_x}, {peak_y}), "
        f"covering approximately {salient_coverage * 100:.1f}% of the total image area. "
    )

    # Appends a clinical interpretation note based on the observed pattern
    if "upper" in zone_dominance and "unilateral" in laterality:
        report += (
            "This pattern is consistent with classical post-primary pulmonary tuberculosis, "
            "which characteristically involves the upper lobes and apical segments."
        )
    elif "bilateral" in laterality and "multifocal" in spread:
        report += (
            "This multifocal bilateral pattern may suggest miliary tuberculosis or advanced "
            "pulmonary disease with widespread parenchymal involvement."
        )
    elif "central" in horizontal_field or "perihilar" in horizontal_field:
        report += (
            "Central or perihilar attention may reflect hilar lymphadenopathy or "
            "primary tuberculosis complex involvement."
        )
    elif "lower" in zone_dominance:
        report += (
            "Lower zone predominance is atypical for classical tuberculosis and may "
            "warrant consideration of alternative diagnoses or post-primary complications."
        )
    elif "upper" in zone_dominance and "bilateral" in laterality:
        report += (
            "Bilateral upper zone involvement is consistent with advanced pulmonary tuberculosis "
            "or reactivation disease affecting both lung apices."
        )
    else:
        report += (
            "The attention pattern does not conform to a single classical TB distribution "
            "and should be interpreted alongside bacteriological and clinical findings."
        )

    return report