# Docling Service

Docling Service converts PDFs into richly formatted Markdown with image extraction, cloud storage uploads, and a modern web UI. The project ships with production-ready synchronous uploads to Cloudflare R2, REST APIs for automation, and an in-browser experience designed for agents and humans alike.

## Features

- Cloud-aware pipeline with pluggable storage backends (`local`, `cloudflare_r2`).
- Synchronous upload mode that guarantees working URLs on completion.
- Background-safe fallback to local assets when cloud uploads fail.
- UI enhancements: bottom-right progress HUD, stage-based status, and active cloud storage badge.
- REST API endpoints for conversion submission, status tracking, result downloads, and storage status checks.
- Docker Compose stack for GPU-enabled or CPU-only deployments.

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

The UI is available at `http://localhost:5001`. The FastAPI docs live at `http://localhost:5001/docs`.

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
| `/api/cloud-storage/status` | GET | Inspect active storage backend configuration |

**📖 For detailed API usage, code examples, and integration patterns, see [docs/API_USAGE.md](docs/API_USAGE.md)**

Interactive API documentation is available at `http://localhost:5001/docs`

## Cloud Storage Workflow

1. Extracted images are saved locally under `images/{task_id}/`.
2. The configured storage backend uploads each asset with the same key.
3. Cloud URLs (or local fallbacks) are injected into the Markdown document.
4. Optional cleanup removes local copies when `DOCLING_CLOUD_KEEP_LOCAL=false`.

Cloudflare R2 is implemented via the S3-compatible `boto3` client and supports custom domains through `DOCLING_R2_PUBLIC_URL_BASE`.

## Development Notes

- Application code lives under `app/` (FastAPI entrypoint `app/main.py`).
- Storage backends reside in `app/storage/`.
- Front-end templates and styles live in `templates/` and `static/`.
- Documentation can be found inside `docs/` (setup guide, architecture plan, changelog).

Run linting or tests with your preferred tooling. Example (if pytest configured):

```bash
pytest
```

## Versioning

This repository uses annotated Git tags for releases, e.g. `v1.3.1`. See `docs/CHANGELOG_CLOUD_STORAGE.md` for notable updates.

## Contributing

1. Fork the repository and create a feature branch.
2. Run the Docker stack locally and verify changes.
3. Submit a pull request with screenshots or curl traces for UI/API updates.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
