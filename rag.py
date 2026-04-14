# rag.py

import re
from pathlib import Path
import torch

from sentence_transformers import SentenceTransformer
from sentence_transformers import util as sbert_util


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