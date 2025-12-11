import os
import json
from typing import List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi import HTTPException

from openai import OpenAI

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------------------------------------------------------
# Environment + OpenAI client
# -------------------------------------------------------------------
load_dotenv()  # looks for .env in your project root

_openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# -------------------------------------------------------------------
# Pydantic models shared with FastAPI
# -------------------------------------------------------------------

class QAGenerationRequest(BaseModel):
    passage: str
    num_questions: int = 3
    difficulty: str = "easy"


class QuizQuestion(BaseModel):
    question: str
    options: List[str]
    correct_index: int
    explanation: Optional[str] = None


class QuizResponse(BaseModel):
    questions: List[QuizQuestion]


class QARequest(BaseModel):
    context: str
    question: str
    max_new_tokens: int = 64
    temperature: float = 0.7


class QAResponse(BaseModel):
    answer: str


# -------------------------------------------------------------------
# ChatGPT: generate multiple-choice questions
# -------------------------------------------------------------------

def generate_mc_questions_with_chatgpt(req: QAGenerationRequest) -> List[QuizQuestion]:
    """
    Use ChatGPT to generate multiple-choice questions in JSON format.
    This uses chat.completions (NOT responses.create), so response_format is valid.
    """

    system_prompt = (
        "You are a helpful assistant that writes multiple-choice quiz questions. "
        "Given a passage, create a list of questions in strict JSON.\n\n"
        "Return JSON with this schema:\n"
        "{\n"
        '  "questions": [\n'
        "    {\n"
        '      "question": "text",\n'
        '      "options": ["A", "B", "C", "D"],\n'
        '      "correct_index": 0,\n'
        '      "explanation": "brief explanation"\n'
        "    }, ...\n"
        "  ]\n"
        "}\n"
        "Do not include any extra text outside the JSON."
    )

    user_prompt = (
        f"Passage:\n{req.passage}\n\n"
        f"Number of questions: {req.num_questions}\n"
        f"Difficulty: {req.difficulty}\n"
    )

    try:
        completion = _openai_client.chat.completions.create(
            model="gpt-4.1-mini",  # or the model your project specifies
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=800,
            response_format={"type": "json_object"},
        )

        content = completion.choices[0].message.content
        data = json.loads(content)

        questions_raw = data.get("questions", [])
        questions: List[QuizQuestion] = []

        for q in questions_raw:
            try:
                questions.append(
                    QuizQuestion(
                        question=q["question"],
                        options=q["options"],
                        correct_index=int(q["correct_index"]),
                        explanation=q.get("explanation"),
                    )
                )
            except KeyError as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Malformed question from ChatGPT (missing key {e})",
                )

        return questions

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Quiz generation failed: {e}",
        )


# -------------------------------------------------------------------
# Local fine-tuned GPT-2: QA model
# -------------------------------------------------------------------

class LocalQAModel:
    """
    Simple wrapper around a fine-tuned GPT-2 model saved in `model_dir`.
    """

    def __init__(self, model_dir: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir)

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)
        self.model.eval()

    def answer(
        self,
        context: str,
        question: str,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
    ) -> str:
        prompt = (
            "You are a question-answering model. Use the context to answer.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = self.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        # Clean up a bit
        return generated.strip()


# single shared instance used by FastAPI
qa_model = LocalQAModel(model_dir="outputs/qa-gpt2")


def answer_with_local_model(
    context: str,
    question: str,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
) -> str:
    """Thin helper so main.py doesn’t have to know about the class."""
    return qa_model.answer(
        context=context,
        question=question,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )