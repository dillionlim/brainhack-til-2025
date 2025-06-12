FROM ghcr.io/astral-sh/uv:0.7.9-python3.9-bookworm-slim
ENV DEBIAN_FRONTEND=noninteractive
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1
    
RUN apt-get update && apt-get install -y --no-install-recommends curl

COPY --from=ghcr.io/astral-sh/uv:0.7.9 /uv /uvx /bin/

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev
    
COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev
    
COPY . .


CMD ["uv", "run", "--no-dev", "uvicorn", "surprise_server:app", "--port", "5005", "--host", "0.0.0.0"]