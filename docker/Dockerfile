FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/models
ENV HUGGINGFACE_HUB_CACHE=/models/hub
ENV TRANSFORMERS_CACHE=/models/transformers
ENV TOKENIZERS_PARALLELISM=false

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        ffmpeg \
        python3.11 \
        python3.11-venv \
        python3-pip \
    && python3.11 -m pip install --no-cache-dir --upgrade pip==24.3.1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt
RUN python3.11 -m pip install --no-cache-dir \
        torch==2.5.1 torchaudio==2.5.1 \
        --index-url https://download.pytorch.org/whl/cu121 \
    && python3.11 -m pip install --no-cache-dir -r /workspace/requirements.txt

COPY app /workspace/app

RUN mkdir -p /workspace/input /workspace/processing /workspace/output /workspace/archive /workspace/failed /workspace/logs /models

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD python3.11 -m app.healthcheck

CMD ["python3.11", "-m", "app.main", "--config", "/workspace/config/config.yaml", "watch"]
