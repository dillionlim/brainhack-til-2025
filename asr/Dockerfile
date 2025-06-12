FROM nvcr.io/nvidia/cuda-dl-base:25.03-cuda12.8-devel-ubuntu24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    CUDA_LAUNCH_BLOCKING=1 \
    TORCH_USE_CUDA_DSA=1

# RUN apt-get update && apt-get install libgl1 libglib2.0-0 -y && rm -rf /var/lib/apt/lists/*

# Add uv binary and install python
COPY --from=ghcr.io/astral-sh/uv:0.7.9 /uv /uvx /bin/
RUN uv python install 3.10

WORKDIR /app

COPY pyproject.toml uv.lock* ./
RUN uv lock && uv sync --locked --no-dev

# Copy the rest of the codebase
COPY ./src ./

RUN uv run init_model.py

EXPOSE 5001

CMD ["uv", "run", "granian", "--interface", "asgi", "--host", "0.0.0.0", "--port", "5001", "--workers", "1", "asr_server:app"]
