FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_HOME=/workspace/.cache/huggingface \
    HF_XET_HIGH_PERFORMANCE=1

WORKDIR /opt/lean-treepo

RUN apt-get update && \
    apt-get install -y --no-install-recommends git git-lfs curl ca-certificates && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY tests ./tests
COPY configs ./configs

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -e ".[tracking]" && \
    chmod +x scripts/runpod_train.sh scripts/run_all_experiments.sh

ENTRYPOINT ["/opt/lean-treepo/scripts/runpod_train.sh"]
