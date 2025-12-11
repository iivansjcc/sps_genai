# Final project - Quiz Generation & Answering API

---
### Authors: Siliang Ma, Jingcheng Sun, Zhenxiao Luo, Chenyi Zhao

---
### Setup Instructions

- Step 1: prerequest for installing the lfs package:

```bash
brew install git-lfs
git lfs install
```
- Step 2: Code for clone: 

```bash
git clone --branch FinalProject --single-branch https://github.com/iivansjcc/sps_genai.git
```
  
- Step 3: Put the .env file (The Openai Key) into the directory ~/FinalProject

---

### Overview

ReadCheck is a FastAPI-based service that helps instructors generate **reading comprehension quizzes** and **model answers** from arbitrary text passages.

The system combines:
- **OpenAI GPT-4** (via API) for generating multiple-choice questions.
- A **fine-tuned local GPT-2 model** for short answer generation and answer explanation.

This project was developed as the final project for a Generative AI course.

---

## Features

- 📄 **Input**: arbitrary reading passage (text).
- ❓ **Quiz generation**: automatic multi-choice questions with answer options and correct labels.
- 🧠 **Local QA model**: fine-tuned GPT-2 that generates short answers based on the passage and question.
- 🧪 **FastAPI endpoints** with interactive Swagger docs (`/docs`).
- 🐳 **Dockerized** for reproducible deployment.

---
   


### Code to run:  (Locally Without Docker)  

Make sure you are in the ~/FinalProject directory
```bash
uv sync
uv run uvicorn app.main:app --reload --port 8000
``` 
Sample test for /generate_quiz in fastapi:
```angular2html
{
  "passage": "France is a country in Western Europe. Its capital and largest city is Paris. It is famous for landmarks such as the Eiffel Tower and the Louvre Museum. Millions of tourists visit Paris every year to experience its food, art and culture.",
  "num_questions": 3,
  "difficulty": "easy"
}
```
Sample test for /model_anwser in fastapi:
```angular2html
{
  "context": "France is a country in Western Europe. Its capital and largest city is Paris.",
  "question": "What is the capital of France?",
  "max_new_tokens": 40,
  "temperature": 0.7
}
```
The result will show up in link below:
http://localhost:8000/docs

### Code to run: (By Docker)

Build and Run the container and expose port 8000:  
```bash
docker build -t finalproject-api .
docker run --rm -p 8000:8000 finalproject-api
```

The result will show up in link below:
http://localhost:8000/docs




  

 

