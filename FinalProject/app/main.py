from fastapi import FastAPI, HTTPException

from .quiz_app import (
    QAGenerationRequest,
    QuizResponse,
    QARequest,
    QAResponse,
    generate_mc_questions_with_chatgpt,
    answer_with_local_model,
)

app = FastAPI(
    title="Final Project Quiz & QA API",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "message": "Quiz & QA API is running. See /docs for interactive documentation."
    }


# -------------------------------------------------------------------
# Endpoint 1: generate_quiz (ChatGPT)
# -------------------------------------------------------------------
@app.post("/generate_quiz", response_model=QuizResponse)
def generate_quiz(payload: QAGenerationRequest):
    """
    Generate multiple-choice questions using ChatGPT given a passage.
    """
    try:
        questions = generate_mc_questions_with_chatgpt(payload)
        return QuizResponse(questions=questions)
    except HTTPException:
        # already has a proper status code + message
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Quiz generation failed: {e}",
        )


# -------------------------------------------------------------------
# Endpoint 2: model_answer (local GPT-2)
# -------------------------------------------------------------------
@app.post("/model_answer", response_model=QAResponse)
def model_answer(payload: QARequest):
    """
    Use the fine-tuned local GPT-2 model to answer a question given a context.
    """
    try:
        answer_text = answer_with_local_model(
            context=payload.context,
            question=payload.question,
            max_new_tokens=payload.max_new_tokens,
            temperature=payload.temperature,
        )
        return QAResponse(answer=answer_text)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model answer failed: {e}",
        )