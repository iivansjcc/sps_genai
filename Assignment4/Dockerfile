# =============================
#  Base image
# =============================
FROM python:3.12-slim-bookworm

# =============================
# Workdir
# =============================
WORKDIR /code

# =============================
# Install Python dependencies
# =============================
# We don't use uv inside the container: just install runtime deps via pip.
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    torch \
    torchvision \
    pillow \
    matplotlib \
    python-multipart

# =============================
# Copy application code
# =============================
COPY ./app /code/app
COPY ./helper_lib /code/helper_lib
COPY ./outputs /code/outputs

# =============================
# Expose port
# =============================
EXPOSE 80

# =============================
# Start FastAPI server
# =============================
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]