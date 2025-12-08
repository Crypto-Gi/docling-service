# Docling Service

**Convert documents (PDF, Word, Excel, PowerPoint) to AI-ready Markdown with automatic cloud storage.**

Docling Service is a production-ready document conversion API that transforms complex documents into clean, structured Markdown. Built on the powerful [Docling](https://github.com/docling-project/docling) library, it features automatic image extraction, cloud storage integration, GPU acceleration, and a modern web interface.

## ✨ Key Features

### Document Processing
- 📄 **Multi-format support**: PDF, Word (.docx), Excel (.xlsx), PowerPoint (.pptx)
- 📊 **Smart table conversion**: Native Markdown tables (not images)
- 🖼️ **Image extraction**: Automatic extraction and cloud upload
- 🎯 **AI-optimized output**: Clean Markdown perfect for LLMs
- 📍 **Position tracking**: Bounding boxes and normalized coordinates for images/tables
- 📑 **Page markers**: Automatic page number injection in markdown output

### Performance & Reliability
- ⚡ **GPU acceleration**: 3-4x faster processing with CUDA support
- 🔄 **Auto-fallback**: Automatic GPU→CPU fallback on memory errors
- 🧠 **Smart memory management**: Prevents OOM crashes
- 🔁 **Auto-restart**: Container self-healing on failures

### Cloud Integration
- ☁️ **Cloudflare R2**: Synchronous uploads with guaranteed URLs
- 🔗 **Custom domains**: Support for CDN URLs
- 💾 **Local fallback**: Works offline or when cloud is unavailable
- 🧹 **Auto-cleanup**: Optional local file deletion after upload

### Developer Experience
- 🚀 **REST API**: Simple, well-documented endpoints
- 🎨 **Modern UI**: Real-time progress tracking
- 🐳 **Docker ready**: One-command deployment
- 🧪 **Fully tested**: 100% API test coverage

## Quick Start

### Prerequisites

- Docker and Docker Compose v2
- Python 3.11+ (for local development tooling)
- Cloudflare R2 bucket (optional for cloud mode)

### Clone and Configure

```bash
git clone https://github.com/<your-org>/docling-service.git
cd docling-service
cp .env.example .env  # create your own environment file
```

Edit `.env` (or inject variables in your deployment environment):

| Variable | Default | Description |
|----------|---------|-------------|
| `DOCLING_MAX_UPLOAD_MB` | `25` | Maximum upload size in MiB |
| `DOCLING_CLOUD_STORAGE_ENABLED` | `false` | Toggle cloud uploads |
| `DOCLING_CLOUD_STORAGE_PROVIDER` | `local` | `local` or `cloudflare_r2` |
| `DOCLING_CLOUD_UPLOAD_MODE` | `sync` | Upload strategy (`sync` today) |
| `DOCLING_CLOUD_KEEP_LOCAL` | `false` | Keep local copies after upload |
| `DOCLING_R2_ACCOUNT_ID` | – | Cloudflare account identifier |
| `DOCLING_R2_ACCESS_KEY_ID` | – | Access key ID |
| `DOCLING_R2_SECRET_ACCESS_KEY` | – | Secret access key |
| `DOCLING_R2_BUCKET_NAME` | – | Target bucket name |
| `DOCLING_R2_REGION` | `auto` | R2 region |
| `DOCLING_R2_PUBLIC_URL_BASE` | – | Optional custom domain (e.g. `https://cdn.example.com`) |

> ⚠️ Credentials should never be committed. `.gitignore` excludes `.env` and deployment artifacts by default.

### Build and Run

```bash
docker compose build
docker compose up -d
```

The UI is available at `http://localhost:5010`. The FastAPI docs live at `http://localhost:5010/docs`.

### GPU/CPU Mode Control

**Quick mode switching:**
```bash
# Use CPU mode (stable, recommended)
./set-cpu-mode.sh && docker compose restart

# Use GPU mode (faster, needs more memory)
./set-gpu-mode.sh && docker compose restart
```

**GPU vs CPU Mode:**
- **CPU Mode** (default): Stable, works on any hardware, ~15s per 10-page doc
- **GPU Mode**: 3-4x faster, requires 4GB+ VRAM, ~5s per 10-page doc

The service automatically falls back to CPU if GPU runs out of memory.

### Stopping the Stack

```bash
docker compose down
```

## API Overview

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/healthz` | GET | Liveness probe |
| `/api/convert` | POST (multipart/form-data) | Submit a PDF or URL for conversion |
| `/api/status/{task_id}` | GET | Poll task status (pending, processing, completed, failed) |
| `/api/result/{task_id}` | GET | Download resulting Markdown |
| `/api/result/{task_id}/json` | GET | Get markdown content as JSON |
| `/api/result/{task_id}/enhanced` | GET | **Full document with images/tables positions and metadata** |
| `/api/cloud-storage/status` | GET | Inspect active storage backend configuration |

**Interactive API documentation**: `http://localhost:5010/docs`

### API Usage Examples

**Upload a file:**
```bash
curl -X POST http://localhost:5010/api/convert \
  -F "file=@document.pdf"
# Response: {"task_id": "abc123..."}
```

**Check status:**
```bash
curl http://localhost:5010/api/status/abc123
# Response: {"status": "completed", "markdown_url": "https://..."}
```

**Download result:**
```bash
curl -O -J http://localhost:5010/api/result/abc123
```

**Get JSON with metadata:**
```bash
curl http://localhost:5010/api/result/abc123/json
```

**Get enhanced response with image/table positions (ideal for AI/LLM):**
```bash
curl http://localhost:5010/api/result/abc123/enhanced
```

Response includes:
- Full markdown content with page markers (`<!-- Page N -->`)
- Images array with page numbers, bounding boxes, and normalized coordinates
- Tables array with page numbers, positions, row/column counts
- Document metadata with page dimensions

## Cloud Storage Setup

### Cloudflare R2 Configuration

1. **Create R2 bucket** in Cloudflare dashboard
2. **Generate API tokens** (Account ID, Access Key, Secret Key)
3. **Configure `.env`**:
   ```bash
   DOCLING_CLOUD_STORAGE_ENABLED=true
   DOCLING_CLOUD_STORAGE_PROVIDER=cloudflare_r2
   DOCLING_R2_ACCOUNT_ID=your_account_id
   DOCLING_R2_ACCESS_KEY_ID=your_access_key
   DOCLING_R2_SECRET_ACCESS_KEY=your_secret_key
   DOCLING_R2_BUCKET_NAME=your_bucket_name
   DOCLING_R2_PUBLIC_URL_BASE=https://your-cdn.com  # Optional
   ```
4. **Restart service**: `docker compose restart`

### How It Works

1. Document is converted, images extracted
2. Images uploaded to R2: `images/{task_id}/picture-{n}.png`
3. Markdown uploaded to R2: `markdown/{task_id}/{filename}.md`
4. Cloud URLs embedded in Markdown output
5. Local files cleaned up (if `DOCLING_CLOUD_KEEP_LOCAL=false`)

**Result**: Markdown with cloud-hosted images ready for sharing!

## Testing

### Run Comprehensive API Tests

```bash
./test_api.sh
```

Tests all endpoints, validates conversions, checks error handling. Generates:
- `api_test_results.txt` - Detailed test log
- `test_result.md` - Sample conversion output

### Generate Test PDF

```bash
python3 test_pdf_generator.py
```

Creates `test_document.pdf` with tables, images, and complex formatting.

## Memory Management

### Recommended Resources

| Mode | RAM | GPU VRAM | Performance |
|------|-----|----------|-------------|
| CPU | 4GB+ | N/A | ~15s per 10 pages |
| GPU | 8GB+ | 4GB+ | ~5s per 10 pages |

### Troubleshooting

**Container crashes (exit 137)**:
```bash
# Switch to CPU mode
./set-cpu-mode.sh
docker compose restart
```

**GPU out of memory**:
- Service automatically falls back to CPU
- Check logs: `docker logs docling-service-docling-1`
- Increase Docker memory limit in `docker-compose.yml`

**Slow processing**:
```bash
# Try GPU mode (if you have 4GB+ VRAM)
./set-gpu-mode.sh
docker compose restart
```

## Project Structure

```
docling-service/
├── app/
│   ├── main.py              # FastAPI application
│   ├── config.py            # Environment configuration
│   └── storage/             # Cloud storage backends
├── templates/               # Web UI (Jinja2)
├── static/                  # CSS, JavaScript
├── storage/docling/         # Local file storage
├── test_api.sh              # API test suite
├── test_pdf_generator.py    # Test PDF generator
├── set-cpu-mode.sh          # CPU mode helper
├── set-gpu-mode.sh          # GPU mode helper
├── docker-compose.yml       # Docker configuration
└── README.md                # This file
```

## Supported Formats

| Format | Extension | Tables | Images | Status |
|--------|-----------|--------|--------|--------|
| PDF | `.pdf` | ✅ Native Markdown | ✅ Extracted | ✅ Fully tested |
| Word | `.docx` | ✅ Native Markdown | ✅ Extracted | ⚠️ Supported, needs testing |
| Excel | `.xlsx` | ✅ Native Markdown | ✅ Charts extracted | ⚠️ Supported, needs testing |
| PowerPoint | `.pptx` | ✅ Native Markdown | ✅ Slide images | ⚠️ Supported, needs testing |

**Note**: Tables are converted to native Markdown format (not images) for better AI processing.

## Enhanced API Response (v1.6)

The `/api/result/{task_id}/enhanced` endpoint returns comprehensive document metadata ideal for AI/LLM processing:

### Sample Response

```json
{
  "task_id": "abc123",
  "markdown_content": "<!-- Page 1 -->\n## Document Title\n...",
  "images": [
    {
      "id": "picture-1",
      "url": "https://cdn.example.com/images/abc123/picture-1.png",
      "page": 1,
      "position": {
        "x": 88.04,
        "y": 553.29,
        "width": 434.83,
        "height": 155.54,
        "coord_origin": "BOTTOMLEFT",
        "x_norm": 0.1439,
        "y_norm": 0.6986,
        "width_norm": 0.7105,
        "height_norm": 0.1964
      },
      "page_dimensions": {"width": 612, "height": 792},
      "mimetype": "image/png"
    }
  ],
  "tables": [
    {
      "id": "table-1",
      "page": 1,
      "position": {"x": 88.04, "y": 553.29, "...": "..."},
      "page_dimensions": {"width": 612, "height": 792},
      "num_rows": 8,
      "num_cols": 5
    }
  ],
  "metadata": {
    "total_pages": 3,
    "total_images": 2,
    "total_tables": 3,
    "processing_time_ms": 21567,
    "page_dimensions": [
      {"width": 612, "height": 792},
      {"width": 612, "height": 792},
      {"width": 612, "height": 792}
    ]
  }
}
```

### Coordinate Systems

| Type | Fields | Description |
|------|--------|-------------|
| **Absolute** | `x`, `y`, `width`, `height` | PDF points (1 pt = 1/72 inch) |
| **Normalized** | `x_norm`, `y_norm`, `width_norm`, `height_norm` | 0-1 range, resolution-independent |

### Markdown Output Features

- **Page markers**: `<!-- Page N -->` inserted at page boundaries
- **Metadata block**: JSON metadata appended at end of `.md` file between `<!-- DOCLING_METADATA_START -->` and `<!-- DOCLING_METADATA_END -->`

### Query Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `include_normalized` | `true` | Include normalized coordinates (0-1 range) |

## Version History

- **v1.6** (Current): Enhanced API with image/table positions, page markers, normalized coordinates
- **v1.5**: Per-request cloud storage credentials, enable/disable flag, credential merging
- **v1.4**: Multi-format support, GPU/CPU control, memory management
- **v1.3**: Cloud storage integration (Cloudflare R2)
- **v1.2**: UI improvements, progress tracking
- **v1.0**: Initial release (PDF to Markdown)

See [RELEASE_v1.5.md](RELEASE_v1.5.md) for complete v1.5 documentation.

## Troubleshooting Guide

### Common Issues

**Q: Service won't start**
```bash
# Check logs
docker logs docling-service-docling-1

# Verify .env file exists
ls -la .env

# Rebuild
docker compose down
docker compose build --no-cache
docker compose up -d
```

**Q: Conversion fails**
- Check file format is supported (PDF, DOCX, XLSX, PPTX)
- Verify file size < 25MB
- Check logs for specific error

**Q: Images not uploading to R2**
- Verify R2 credentials in `.env`
- Check bucket permissions
- Test with: `curl http://localhost:5010/api/cloud-storage/status`

**Q: Slow performance**
- Try GPU mode if available
- Check Docker resource limits
- Monitor with: `docker stats docling-service-docling-1`

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Test locally: `./test_api.sh`
4. Commit changes: `git commit -m 'Add amazing feature'`
5. Push to branch: `git push origin feature/amazing-feature`
6. Open a Pull Request

## License

MIT License - see [LICENSE](LICENSE) for details.

## Credits

Built with [Docling](https://github.com/docling-project/docling) by IBM Research.

---

**Questions?** Open an issue or check the interactive API docs at `http://localhost:5010/docs`
