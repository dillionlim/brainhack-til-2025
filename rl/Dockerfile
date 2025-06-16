# Dockerfile for building the RL image.


# The base image. Using a regular Python base because RL agents don't typically
# require much GPU. This will speed up your evaluation.
FROM python:3.12-slim

# Configures settings for the image.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_ROOT_USER_ACTION=ignore

WORKDIR /app

# Installs your dependencies.
RUN pip install -U pip
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip uninstall -y sb3-contrib
RUN pip uninstall -y stable-baselines3

# Copies your source files.
COPY src .

# Starts your model server.
CMD uvicorn rl_server:app --port 5004 --host 0.0.0.0
