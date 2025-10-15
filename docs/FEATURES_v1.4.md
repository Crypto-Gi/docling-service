# Version 1.4 Features

## New Features for Agentic AI Systems

### 1. Markdown Cloud Upload ☁️

**What**: Markdown files are now automatically uploaded to cloud storage alongside images.

**Structure**:
```
bucket/
├── images/{task_id}/
│   ├── picture-1.png
│   └── picture-2.png
└── markdown/{task_id}/
    └── 10-15-2025-203000.md
```

**Benefits**:
- ✅ Persistent storage across container restarts
- ✅ Distributed access for multiple agents
- ✅ Reduced local disk usage
- ✅ CDN delivery for fast global access
- ✅ Stable URLs for downstream systems

**API Response**:
```json
{
  "status": "completed",
  "markdown_url": "https://cdn.example.com/markdown/task_id/file.md",
  ...
}
```

---

### 2. JSON API Endpoint 🤖

**What**: New endpoint `/api/result/{task_id}/json` returns markdown content directly as JSON.

**Endpoint**: `GET /api/result/{task_id}/json`

**Response**:
```json
{
  "task_id": "abc123...",
  "markdown_content": "# Title\n\n![Image](https://...)\n\nContent...",
  "source_name": "document.pdf",
  "output_filename": "10-15-2025-203000.md",
  "markdown_url": "https://cdn.example.com/markdown/abc123/file.md",
  "created_at": "2025-10-15T20:30:00.000Z",
  "completed_at": "2025-10-15T20:30:15.000Z"
}
```

**Benefits**:
- ✅ No file download/handling required
- ✅ Single request for content + metadata
- ✅ Perfect for LLM/agent processing
- ✅ Includes cloud URLs when available
- ✅ Structured JSON response

---

## Usage Examples

### Python (JSON Endpoint)

```python
import requests
import time

BASE_URL = "http://localhost:5001"

# Submit PDF
response = requests.post(f"{BASE_URL}/api/convert", files={"file": open("doc.pdf", "rb")})
task_id = response.json()["task_id"]

# Poll until complete
while True:
    status = requests.get(f"{BASE_URL}/api/status/{task_id}").json()
    if status["status"] == "completed":
        break
    time.sleep(3)

# Get JSON result (no file download!)
result = requests.get(f"{BASE_URL}/api/result/{task_id}/json").json()

# Access markdown directly
markdown = result["markdown_content"]
cloud_url = result.get("markdown_url")  # If cloud enabled

print(f"Markdown: {len(markdown)} chars")
print(f"Cloud URL: {cloud_url}")
```

### cURL

```bash
# Get JSON result
curl http://localhost:5001/api/result/abc123/json

# Response includes markdown_content field with full text
```

---

## Cloud Storage Structure

When cloud storage is enabled:

### Images
```
https://cdn.example.com/images/{task_id}/picture-1.png
https://cdn.example.com/images/{task_id}/picture-2.png
```

### Markdown
```
https://cdn.example.com/markdown/{task_id}/10-15-2025-203000.md
```

### In Markdown Content
```markdown
# Document Title

![Image](https://cdn.example.com/images/task_id/picture-1.png)

Document content here...
```

---

## Migration Notes

### Existing Endpoints (Unchanged)
- `POST /api/convert` - Submit conversion
- `GET /api/status/{task_id}` - Check status (now includes `markdown_url`)
- `GET /api/result/{task_id}` - Download file (still works)

### New Endpoints
- `GET /api/result/{task_id}/json` - Get JSON response ⭐

### Breaking Changes
**None** - All existing functionality preserved. New features are additive.

---

## Configuration

No new environment variables required. Uses existing cloud storage settings:

```bash
DOCLING_CLOUD_STORAGE_ENABLED=true
DOCLING_CLOUD_STORAGE_PROVIDER=cloudflare_r2
DOCLING_R2_BUCKET_NAME=your-bucket
# ... other R2 settings
```

Markdown upload happens automatically when cloud storage is enabled.

---

## Why These Features?

### For Agentic AI Systems
1. **JSON Endpoint**: Agents can process markdown without file I/O
2. **Cloud URLs**: Stable references for multi-agent workflows
3. **Metadata**: Timestamps and filenames for tracking
4. **No Downloads**: Reduces complexity in agent code

### For Production Deployments
1. **Persistent Storage**: Results survive container restarts
2. **Scalability**: Multiple instances can share cloud storage
3. **Cost Efficiency**: Automatic cleanup of local files
4. **Global Access**: CDN delivery for distributed teams

---

## Documentation

- **Full API Guide**: [docs/API_USAGE.md](API_USAGE.md)
- **Quick Reference**: [docs/API_QUICK_REFERENCE.md](API_QUICK_REFERENCE.md)
- **Interactive Docs**: http://localhost:5001/docs

---

## Version History

- **v1.4** - Markdown cloud upload + JSON API endpoint
- **v1.3.1** - Compact upload HUD + repo scaffolding
- **v1.3** - Cloud storage integration (images)
- **v1.2** - Base conversion service
