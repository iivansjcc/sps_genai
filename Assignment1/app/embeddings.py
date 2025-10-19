# app/embeddings.py
from __future__ import annotations
from functools import lru_cache
from typing import List
import os
import numpy as np
import spacy

# Default to the same model used in Module 2 (has real vectors)
SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_lg")

@lru_cache(maxsize=1)
def get_nlp():
    try:
        return spacy.load(SPACY_MODEL)
    except OSError as e:
        raise RuntimeError(
            f"Could not load spaCy model '{SPACY_MODEL}'. "
            "Install a model with vectors, e.g. en_core_web_lg."
        ) from e

def embed_word(text: str) -> np.ndarray:
    """Module-2 style: nlp(text).vector for a single token/word."""
    nlp = get_nlp()
    vec = nlp(text).vector
    if vec is None or vec.size == 0 or float(np.linalg.norm(vec)) == 0.0:
        raise ValueError(
            f"No usable vector for '{text}'. "
            "Ensure you're using a model with vectors (en_core_web_lg)."
        )
    return vec.astype("float32")

def embed_sentence(sentence: str) -> np.ndarray:
    """Sentence embedding = mean of (non-zero) token vectors (Module-2 approach)."""
    nlp = get_nlp()
    doc = nlp(sentence)
    if not len(doc):
        raise ValueError("Empty sentence.")
    token_vecs = [t.vector for t in doc if float(np.linalg.norm(t.vector)) > 0.0]
    if not token_vecs:
        raise ValueError(
            "No non-zero token vectors found in the sentence (check your model)."
        )
    return np.mean(token_vecs, axis=0).astype("float32")