FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Install ffmpeg and Python 3.12
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_LAUNCH_BLOCKING=1
ENV TORCH_USE_CUDA_DSA=1

RUN apt-get update && \
    apt-get install -y ffmpeg software-properties-common curl --fix-missing && \
    # add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Add uv binary and install python
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN uv python install 3.12

WORKDIR /app
COPY pyproject.toml uv.lock* ./
# COPY pyproject-paddle2-5.toml uv.lock* ./
# RUN mv pyproject-paddle2-5.toml pyproject.toml
RUN uv sync --frozen --no-install-project --no-dev
RUN uv pip install transformers torch accelerate bitsandbytes sentencepiece

RUN mkdir en_PP-OCRv3_det && mkdir en_PP-OCRv4_rec

# Copy the rest of the codebase
COPY ./srcs/src ./
COPY ./srcs/en_PP-OCRv3_det ./en_PP-OCRv3_det
COPY ./srcs/en_PP-OCRv4_rec ./en_PP-OCRv4_rec

RUN mkdir google
RUN mkdir google/flan-t5-small
COPY ./flan-t5-small ./google/flan-t5-small

RUN uv run init_model-flan.py

EXPOSE 5003

CMD ["uv", "run", "granian", "--interface", "asgi", "--host", "0.0.0.0", "--port", "5003", "--workers", "1", "ocr_server-flan:app"]