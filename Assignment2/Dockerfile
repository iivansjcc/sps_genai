# Dockerfile
FROM python:3.12-slim-bookworm

# OS deps for UV installer
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && rm -rf /var/lib/apt/lists/*

# Install UV
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

# Workdir
WORKDIR /code

# Copy dependency manifests first (better caching)
COPY pyproject.toml uv.lock /code/

# Install deps (frozen to lockfile)
RUN uv sync --frozen --no-install-project

# Copy your app & library & checkpoint
COPY ./app /code/app
COPY ./helper_lib /code/helper_lib
COPY ./checkpoints /code/checkpoints

# Expose port 80 inside the container
EXPOSE 80

# Run FastAPI app on port 80
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]