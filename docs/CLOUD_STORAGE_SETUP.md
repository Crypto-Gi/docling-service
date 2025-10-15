# Cloud Storage Setup Guide

## Overview

The Docling service now supports uploading extracted images to cloud storage (Cloudflare R2, AWS S3, etc.) instead of storing them locally. This guide covers setup and configuration.

## Features

- **Modular architecture**: Easy to switch between storage providers
- **Synchronous upload mode**: Wait for uploads before returning results (reliable, production-ready)
- **Automatic fallback**: Falls back to local storage if cloud upload fails
- **API integration**: RESTful endpoints for status and configuration
- **UI indicators**: Visual feedback when cloud storage is active

## Quick Start (Local Storage - Default)

By default, the service uses local filesystem storage. No configuration needed.

## Cloudflare R2 Setup

### Step 1: Create R2 Bucket

1. Log in to [Cloudflare Dashboard](https://dash.cloudflare.com)
2. Navigate to **R2** → **Create bucket**
3. Name your bucket (e.g., `docling-images`)
4. Create an `images/` directory in the bucket (optional - will be created automatically on first upload)
5. Note your **Account ID** from the R2 overview page

**Bucket Structure**: Images will be organized as `images/{task_id}/picture-1.png`, matching the local storage structure.

### Step 2: Generate Access Keys

1. In R2 dashboard, go to **Manage R2 API Tokens**
2. Click **Create API Token**
3. Set permissions: **Object Read & Write**
4. Note the **Access Key ID** and **Secret Access Key**

### Step 3: Configure Public Access (Optional)

For public URLs without CDN:
1. Go to bucket **Settings**
2. Enable **Public Access** or configure **Custom Domain**
3. Note the public URL pattern

### Step 4: Update Environment Variables

Edit `.env` file:

```bash
# Enable cloud storage
DOCLING_CLOUD_STORAGE_ENABLED=true
DOCLING_CLOUD_STORAGE_PROVIDER=cloudflare_r2
DOCLING_CLOUD_UPLOAD_MODE=sync

# Cloudflare R2 credentials
DOCLING_R2_ACCOUNT_ID=your_account_id_here
DOCLING_R2_ACCESS_KEY_ID=your_access_key_id_here
DOCLING_R2_SECRET_ACCESS_KEY=your_secret_access_key_here
DOCLING_R2_BUCKET_NAME=docling-images
DOCLING_R2_REGION=auto

# Optional: Custom CDN URL
# DOCLING_R2_PUBLIC_URL_BASE=https://cdn.yourdomain.com

# Keep local copies after upload (optional)
DOCLING_CLOUD_KEEP_LOCAL=false
```

### Step 5: Rebuild and Restart

```bash
docker compose build
docker compose up -d
```

### Step 6: Verify

1. Open the web UI at `http://localhost:8000`
2. Look for **Cloud Storage: cloudflare_r2 (sync) ✓ Active** indicator
3. Upload a PDF and check that images appear in your R2 bucket
4. Download the Markdown and verify image URLs point to R2

**Expected Bucket Structure**:
```
your-bucket/
└── images/
    ├── abc123def456/          # Task ID folder
    │   ├── picture-1.png
    │   ├── picture-2.png
    │   └── table-1.png
    └── xyz789ghi012/          # Another task
        ├── picture-1.png
        └── picture-2.png
```

**Example Markdown Output**:
```markdown
![Image](https://your-bucket.account-id.r2.cloudflarestorage.com/images/abc123def456/picture-1.png)
```

## API Endpoints

### Check Cloud Storage Status

```bash
GET /api/cloud-storage/status
```

**Response**:
```json
{
  "enabled": true,
  "provider": "cloudflare_r2",
  "upload_mode": "sync",
  "backend_ready": true,
  "backend_type": "CloudflareR2Storage",
  "keep_local_copy": false
}
```

### Convert Document (Same as Before)

```bash
POST /api/convert
Content-Type: multipart/form-data

file=@document.pdf
```

**Response**:
```json
{
  "task_id": "abc123..."
}
```

### Check Task Status

```bash
GET /api/status/{task_id}
```

**Response**:
```json
{
  "task_id": "abc123...",
  "status": "completed",
  "download_url": "/api/result/abc123...",
  ...
}
```

### Download Result

```bash
GET /api/result/{task_id}
```

Returns Markdown file with cloud storage URLs for images.

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `DOCLING_CLOUD_STORAGE_ENABLED` | `false` | Enable cloud storage |
| `DOCLING_CLOUD_STORAGE_PROVIDER` | `local` | Provider: `local`, `cloudflare_r2` |
| `DOCLING_CLOUD_UPLOAD_MODE` | `sync` | Upload mode: `sync` (wait for upload) |
| `DOCLING_CLOUD_KEEP_LOCAL` | `false` | Keep local copies after upload |
| `DOCLING_R2_ACCOUNT_ID` | - | Cloudflare account ID |
| `DOCLING_R2_ACCESS_KEY_ID` | - | R2 access key ID |
| `DOCLING_R2_SECRET_ACCESS_KEY` | - | R2 secret access key |
| `DOCLING_R2_BUCKET_NAME` | - | R2 bucket name |
| `DOCLING_R2_REGION` | `auto` | R2 region (`wnam`, `enam`, `weur`, `eeur`, `apac`, `auto`) |
| `DOCLING_R2_PUBLIC_URL_BASE` | - | Optional CDN URL (e.g., `https://cdn.example.com`) |

## Troubleshooting

### Images still stored locally

- Check `DOCLING_CLOUD_STORAGE_ENABLED=true` in `.env`
- Verify R2 credentials are correct
- Check logs: `docker compose logs docling`
- Visit `/api/cloud-storage/status` to see backend status

### 404 errors on image URLs

- Ensure R2 bucket has public access enabled
- Verify `DOCLING_R2_PUBLIC_URL_BASE` is correct (if using CDN)
- Check bucket CORS settings if accessing from browser

### Upload failures

- Verify IAM permissions (Object Read & Write)
- Check network connectivity from container
- Review logs for specific error messages
- Service will fallback to local storage automatically

## Agent Integration

For automated/agent usage, query the status endpoint before conversion:

```python
import requests

# Check if cloud storage is active
status = requests.get("http://localhost:8000/api/cloud-storage/status").json()
if status["enabled"] and status["backend_ready"]:
    print(f"Cloud storage active: {status['provider']}")

# Submit conversion
response = requests.post(
    "http://localhost:8000/api/convert",
    files={"file": open("document.pdf", "rb")}
)
task_id = response.json()["task_id"]

# Poll for completion
while True:
    status = requests.get(f"http://localhost:8000/api/status/{task_id}").json()
    if status["status"] == "completed":
        markdown = requests.get(status["download_url"]).text
        # Images in markdown will have cloud URLs
        break
    time.sleep(3)
```

## Security Best Practices

1. **Never commit credentials** to version control
2. Use **environment variables** or secret managers
3. Limit **IAM permissions** to minimum required (Object Read & Write)
4. Enable **bucket encryption** in R2 settings
5. Use **CDN** with authentication for private content
6. Rotate **access keys** regularly

## Future Enhancements

- Asynchronous upload mode (background uploads)
- Additional providers (AWS S3, Google Cloud Storage, Azure Blob)
- Presigned URLs for private content
- Upload retry logic with exponential backoff
- Metrics and monitoring endpoints

## Support

For issues or questions:
- Check logs: `docker compose logs docling`
- Review plan: `docs/cloud-storage-plan.md`
- API docs: `http://localhost:8000/docs` (FastAPI auto-docs)
