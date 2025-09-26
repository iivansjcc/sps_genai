from typing import Union, List
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from app.bigram_model import BigramModel
from app.embeddings import embed_word, embed_sentence

app = FastAPI()
# Sample corpus for the bigram model
corpus = [
"The Count of Monte Cristo is a novel written by Alexandre Dumas. \
It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
"this is another example sentence",
"we are generating text based on bigram probabilities",
"bigram models are simple but effective"
]

bigram_model = BigramModel(corpus)

class TextGenerationRequest(BaseModel):
    start_word: str
    length: int = 1

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}





class EmbedWordResponse(BaseModel):
    text: str
    dim: int
    vector: List[float]

class EmbedSentenceResponse(BaseModel):
    sentence: str
    dim: int
    vector: List[float]

@app.get("/embed/word", response_model=EmbedWordResponse, tags=["embeddings"])
def api_embed_word(text: str = Query(..., description="Single word/token")):
    try:
        vec = embed_word(text)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"text": text, "dim": int(vec.shape[0]), "vector": [float(x) for x in vec]}

@app.get("/embed/sentence", response_model=EmbedSentenceResponse, tags=["embeddings"])
def api_embed_sentence(sentence: str = Query(..., description="Sentence to embed (mean of word vectors)")):
    try:
        vec = embed_sentence(sentence)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"sentence": sentence, "dim": int(vec.shape[0]), "vector": [float(x) for x in vec]}