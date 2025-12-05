# Base Python image
FROM python:3.12-slim-bookworm

# Workdir inside the container
WORKDIR /code

# Install Python dependencies
# (Only the runtime deps needed for the API)
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    transformers \
    datasets \
    torch \
    accelerate

# Copy application code
COPY ./app /code/app
COPY ./outputs /code/outputs

# Tell the app where the fine-tuned model lives
ENV GPT2_MODEL_DIR=/code/outputs/gpt2-finetuned-squad

# Expose FastAPI port
EXPOSE 80

# Start the FastAPI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]