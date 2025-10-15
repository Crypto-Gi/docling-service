# Docling Markdown Converter with VLM Support

A powerful document conversion service that converts PDFs and images to Markdown using AI-powered OCR and Visual Language Models (VLM), including IBM's Granite Docling models.

## Features

- **PDF to Markdown**: Convert PDF documents with standard OCR
- **Image to Markdown**: Convert images using advanced VLM models
- **Multiple AI Models**: Choose from 5 different processing models
- **Smart Defaults**: Automatic model selection based on document type
- **GPU Acceleration**: Optional CUDA support for faster processing
- **Web UI**: Clean, modern interface with real-time progress tracking
- **REST API**: Full API access for programmatic usage

## Supported Models

### Core Models (Always Available)
- **Default (Docling OCR)**: Standard OCR without VLM - best for PDFs
- **Granite Docling 258M**: Specialized 258MB model optimized for document understanding - best for images

### Optional Models (Require `DOCLING_LOAD_OPTIONAL_MODELS=true`)
- **SmolVLM**: Lightweight general-purpose VLM
- **Granite Vision 2B**: IBM's 2B parameter vision model
- **Granite Vision 8B**: IBM's largest and most capable vision model

## Quick Start

### Basic Usage (Core Models Only)

```bash
# Build and start the service
docker compose build
docker compose up
```

Access the UI at `http://localhost:5001`

### With Optional VLM Models

```bash
# Build with all models
DOCLING_LOAD_OPTIONAL_MODELS=true docker compose build
docker compose up
```

**Note**: Building with optional models will download ~10GB of additional model weights.

## Configuration

### Environment Variables

Edit `.env` file or set environment variables:

```bash
# Storage settings
DOCLING_MAX_UPLOAD_MB=25
DOCLING_STORAGE_SOFT_LIMIT_MB=200
DOCLING_STORAGE_HARD_LIMIT_MB=250

# GPU settings
DOCLING_ENABLE_CUDA=false
DOCLING_GPU_PREFERRED=true

# Model loading
DOCLING_LOAD_OPTIONAL_MODELS=false  # Set to 'true' for optional models

# VLM configuration
DOCLING_ENABLE_VLM=true
DOCLING_VLM_IMAGES_SCALE=2.0
DOCLING_VLM_TEMPERATURE=0.0
```

### GPU Support

For CUDA GPU acceleration:

```bash
DOCLING_ENABLE_CUDA=true docker compose build
docker compose up
```

## Usage

### Web UI

1. Open `http://localhost:5001`
2. Select document type (PDF or Image)
3. Choose a model (or use smart defaults)
4. Upload file or provide URL
5. Click "Convert to Markdown"
6. View results with metadata, preview tabs, and download options

### API Endpoints

#### Convert Document

```bash
POST /api/convert
Content-Type: multipart/form-data

Parameters:
- file: File upload (PDF or image)
- source_url: Remote file URL (alternative to file)
- document_type: "pdf" or "image"
- model: "default", "granite_docling", "smolvlm", "granite_vision_2b", "granite_vision_8b"
- do_picture_description: boolean
- images_scale: float (1.0-3.0)
- ocr_language: string ("en", "es", "fr", "de")
- force_ocr: boolean
- table_mode: "fast" or "accurate"

Response:
{
  "task_id": "abc123..."
}
```

#### Check Status

```bash
GET /api/status/{task_id}

Response:
{
  "task_id": "abc123...",
  "status": "completed",
  "download_url": "/api/result/abc123...",
  "metadata": {
    "model_used": "Granite Docling 258M",
    "processing_time_seconds": 2.5,
    "pages_processed": 3,
    "images_extracted": 5
  }
}
```

#### Download Result

```bash
GET /api/result/{task_id}

Returns: Markdown file
```

## Model Selection Guide

| Document Type | Recommended Model | Use Case |
|--------------|-------------------|----------|
| PDF (text-heavy) | Default (Docling OCR) | Fast, accurate text extraction |
| PDF (complex layout) | Granite Docling 258M | Better structure understanding |
| Images | Granite Docling 258M | Optimized for image-to-text |
| Photos/Screenshots | SmolVLM | General-purpose vision |
| Complex visuals | Granite Vision 2B/8B | Maximum accuracy |

## Architecture

- **Backend**: FastAPI with async task processing
- **OCR Engine**: Docling + Tesseract
- **VLM Models**: Granite Docling, SmolVLM, Granite Vision
- **Storage**: Persistent volumes for uploads, outputs, and model cache
- **Frontend**: Vanilla JavaScript with modern UI

## Development

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run server
uvicorn app.main:app --reload --port 5001
```

### Building Custom Images

```bash
# CPU-only image
docker build -t docling-service:cpu .

# GPU-enabled image
docker build --build-arg ENABLE_CUDA=true -t docling-service:gpu .

# With optional models
docker build --build-arg LOAD_OPTIONAL_MODELS=true -t docling-service:full .
```

## Troubleshooting

### Models Not Found

If you see "model not found" errors for optional models, rebuild with:

```bash
DOCLING_LOAD_OPTIONAL_MODELS=true docker compose build --no-cache
```

### Out of Memory

For large documents or VLM models:
- Reduce `images_scale` to 1.0-1.5
- Use smaller models (Default or Granite Docling 258M)
- Enable GPU acceleration if available

### Slow Processing

- Use Default OCR for PDFs (fastest)
- Enable GPU acceleration
- Reduce image quality scale
- Use "fast" table mode

## License

MIT

## Credits

Built with:
- [Docling](https://github.com/docling-project/docling) - Document conversion library
- [IBM Granite](https://github.com/ibm-granite) - Granite Vision models
- [SmolVLM](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct) - Lightweight VLM
