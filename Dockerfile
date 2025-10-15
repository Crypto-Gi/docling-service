FROM python:3.11-slim

ARG ENABLE_CUDA=false
ARG LOAD_OPTIONAL_MODELS=false

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    wget \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt

RUN pip install --upgrade pip && \
    if [ "$ENABLE_CUDA" = "true" ]; then \
        pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu121 torch==2.3.1 torchvision==0.18.1; \
    else \
        pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch==2.3.1 torchvision==0.18.1; \
    fi && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

COPY app /app/app
COPY templates /app/app/templates
COPY static /app/app/static

# Download core models (always included)
RUN echo "Downloading core Docling models..." && \
    python -c "from docling.utils import model_downloader; model_downloader.download_models()" && \
    echo "Downloading Granite Docling 258M model..." && \
    python -c "from huggingface_hub import snapshot_download; snapshot_download('ibm-granite/granite-docling-258M')"

# Conditionally download optional VLM models
RUN if [ "$LOAD_OPTIONAL_MODELS" = "true" ]; then \
        echo "Downloading optional VLM models..." && \
        python -c "from huggingface_hub import snapshot_download; \
                   snapshot_download('HuggingFaceTB/SmolVLM-256M-Instruct'); \
                   snapshot_download('ibm-granite/granite-vision-3.1-2b-preview'); \
                   snapshot_download('ibm-granite/granite-vision-3.2-8b')"; \
    else \
        echo "Skipping optional models (set LOAD_OPTIONAL_MODELS=true to include)"; \
    fi

EXPOSE 5001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5001"]
