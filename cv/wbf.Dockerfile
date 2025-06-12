# syntax=docker/dockerfile:1.7-labs
FROM nvcr.io/nvidia/cuda-dl-base:25.03-cuda12.8-devel-ubuntu24.04 AS builder
LABEL stage=builder
ENV DEBIAN_FRONTEND=noninteractive
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    HF_HOME=/app/cache \
    HF_HUB_CACHE=/app/cache/hub
    
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 -y && rm -rf /var/lib/apt/lists/*
    
# Install uv by copying its binaries from the distroless uv image
COPY --from=ghcr.io/astral-sh/uv:0.7.9 /uv /bin/
RUN uv python install 3.12

WORKDIR /app


RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev --no-install-package torch
    
# RUN uv lock && uv sync --locked --no-dev --no-install-project
RUN uv pip install torch==2.7.0 --force-reinstall --index-url https://download.pytorch.org/whl/cu128
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev
    
COPY src/wbf .
ENV HF_HUB_OFFLINE=1

RUN uv run --no-dev init_models.py
RUN uv cache clean
# Starts your model server.
CMD ["uv", "run", "--no-dev", "uvicorn", "cv_server:app", "--port", "5002", "--host", "0.0.0.0"]


# FROM nvcr.io/nvidia/cuda-dl-base:25.03-cuda12.8-runtime-ubuntu24.04 AS runner
