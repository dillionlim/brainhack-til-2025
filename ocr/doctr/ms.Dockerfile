FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1
    
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 -y && rm -rf /var/lib/apt/lists/*
    
# Install uv by copying its binaries from the distroless uv image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/
RUN uv python install 3.12

WORKDIR /app

COPY pyproject.toml uv.lock* ./
RUN uv lock && uv sync --locked --no-dev
COPY ./src ./src/

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS runner

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_LAUNCH_BLOCKING=1 \
    TORCH_USE_CUDA_DSA=1

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 -y && rm -rf /var/lib/apt/lists/*

COPY --from=builder /bin/uv /bin/uv

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src/* /app/pyproject.toml /app/uv.lock /app/

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH"
RUN uv python install 3.12
RUN uv lock && uv sync --locked --no-dev

RUN uv run --no-dev init_model.py

EXPOSE 5003

# Starts your model server.
CMD ["uv", "run", "--no-dev", "granian", "--interface", "asgi", "--host", "0.0.0.0", "--port", "5003", "--workers", "1", "ocr_server:app"]
