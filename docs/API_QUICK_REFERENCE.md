# API Quick Reference

## 3-Step Workflow

```bash
# 1. Submit PDF
curl -X POST http://localhost:5001/api/convert -F "file=@doc.pdf"
# → {"task_id": "abc123..."}

# 2. Poll status (repeat until completed)
curl http://localhost:5001/api/status/abc123...
# → {"status": "completed", "download_url": "/api/result/abc123..."}

# 3. Download result
curl -O -J http://localhost:5001/api/result/abc123...
# → doc.md
```

## Python One-Liner

```python
import requests, time
task_id = requests.post("http://localhost:5001/api/convert", files={"file": open("doc.pdf", "rb")}).json()["task_id"]
while (s := requests.get(f"http://localhost:5001/api/status/{task_id}").json()["status"]) not in ["completed", "failed"]: time.sleep(3)
markdown = requests.get(f"http://localhost:5001/api/result/{task_id}").text
```

## Status Values

- `pending` → Queued
- `processing` → Converting
- `completed` → Ready (download available)
- `failed` → Error (check `detail` field)

## Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/convert` | POST | Submit PDF |
| `/api/status/{task_id}` | GET | Check progress |
| `/api/result/{task_id}` | GET | Download markdown |
| `/api/cloud-storage/status` | GET | Check cloud config |
| `/docs` | GET | Interactive API docs |

## Cloud Storage

When enabled, images in markdown use cloud URLs:
```markdown
![Image](https://cdn.example.com/images/task_id/picture-1.png)
```

Check status:
```bash
curl http://localhost:5001/api/cloud-storage/status
```

## Error Codes

- `202` - Task accepted
- `400` - Bad request (invalid file/format)
- `404` - Task/result not found
- `413` - File too large

---

**📖 Full documentation**: [API_USAGE.md](API_USAGE.md)
