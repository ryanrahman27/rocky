# Training image for Rocky. Needs an NVIDIA GPU at run time:
#
#   docker build -t rocky .
#   docker run --gpus all -v "$PWD/logs:/app/logs" -e WANDB_API_KEY rocky \
#          scripts/train.sh Mjlab-Velocity-Flat-Rocky 4096 3000
#
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    MUJOCO_GL=egl

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.11 python3.11-dev python3-pip git curl ca-certificates \
      libegl1 libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
# Dependency layer first, so source edits don't re-resolve the environment.
COPY pyproject.toml README.md ./
COPY src/rocky/__init__.py src/rocky/
RUN uv sync --no-install-project

COPY . .
RUN uv sync

CMD ["scripts/train.sh"]
