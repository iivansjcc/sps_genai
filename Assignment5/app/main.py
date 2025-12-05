# Assignment5/app/main.py

from __future__ import annotations

import os
from typing import Optional

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = os.environ.get("GPT2_MODEL_DIR", "outputs/gpt2-finetuned-squad")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(
    title="Assignment5 GPT-2 QA API",
    version="1.0.0",
    description="Fine-tuned GPT-2 that answers questions in a fixed format.",
)


class QARequest(BaseModel):
    question: str
    context: Optional[str] = ""


class QAResponse(BaseModel):
    answer: str
    raw_text: str


# Load model at startup (like Module 9 style)
@app.on_event("startup")
def load_model():
    global tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
    model.to(DEVICE)
    model.eval()


@app.get("/", tags=["health"])
def health():
    return {"status": "ok", "model_dir": MODEL_DIR, "device": DEVICE}


def build_prompt(question: str, context: str) -> str:
    """Prompt template consistent with training."""
    prefix = "That is a great question.\n"
    ctx = f"Question: {question}\n"
    if context:
        ctx += f"Context: {context}\n"
    else:
        ctx += "Context: (no additional context provided)\n"
    suffix = "Answer:"
    return prefix + ctx + suffix


@app.post("/generate", response_model=QAResponse, tags=["generation"])
def generate_answer(req: QARequest):
    prompt = build_prompt(req.question, req.context or "")

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            temperature=0.8,
            eos_token_id=tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Extract only the part after "Answer:"
    answer_part = full_text.split("Answer:", 1)[-1].strip()

    opening = "That is a great question."
    closing = "Let me know if you have any other questions."

    # Ensure it starts with the required opening sentence
    if not answer_part.startswith(opening):
        answer_part = f"{opening} {answer_part}"

    # Ensure it ends with the required closing sentence
    if closing not in answer_part:
        answer_part = answer_part.rstrip(". ") + ". " + closing

    return QAResponse(answer=answer_part, raw_text=full_text)