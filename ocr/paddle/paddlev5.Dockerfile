# FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS builder


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
# COPY pyproject.toml uv.lock* ./
# COPY pyproject-paddle2-5.toml uv.lock* ./
# RUN mv pyproject-paddle2-5.toml pyproject.toml
COPY pyproject-paddle3.toml ./pyproject.toml
RUN uv lock --upgrade --refresh
RUN uv sync --frozen --no-install-project --no-dev

# 2a) Copy our replacement file into the image (assumes you have a local file named "custom_cache.py")
COPY custom_paddle_font.py /tmp/custom_paddle_font.py

# 2b) Use `uv run python` (to run inside the uv-managed venv) to print paddlex’s install location
#     We capture that path into an environment variable PADDLEX_ROOT, then move our file over.
ENV PADDLEX_ROOT=/app/.venv/lib/python3.12/site-packages/paddlex
COPY custom_paddle_font.py /app/.venv/lib/python3.12/site-packages/paddlex/utils/fonts/__init__.py

RUN mkdir PP-OCRv5_mobile_det && mkdir PP-OCRv5_mobile_rec

# Copy the rest of the codebase
COPY ./srcs/src ./
# COPY ./srcs/en_PP-OCRv3_det ./en_PP-OCRv3_det
# COPY ./srcs/en_PP-OCRv4_rec ./en_PP-OCRv4_rec
COPY ./srcs/PP-OCRv5_mobile_det ./PP-OCRv5_mobile_det
COPY ./srcs/PP-OCRv5_mobile_rec ./PP-OCRv5_mobile_rec

# Create PaddleX default cache under root’s home
RUN mkdir -p /root/.paddlex/fonts

# Copy the two font files into /root/.paddlex/fonts/
COPY ./srcs/PingFang-SC-Regular.ttf /root/.paddlex/fonts/
COPY ./srcs/simfang.ttf        /root/.paddlex/fonts/

# Make sure they are readable
RUN chmod 644 /root/.paddlex/fonts/PingFang-SC-Regular.ttf \
             /root/.paddlex/fonts/simfang.ttf

# RUN uv run init_model_v5.py

EXPOSE 5003

CMD ["uv", "run", "granian", "--interface", "asgi", "--host", "0.0.0.0", "--port", "5003", "--workers", "1", "ocr_server:app"]


