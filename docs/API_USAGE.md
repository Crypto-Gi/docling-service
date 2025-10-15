# Docling Service API Usage Guide

This guide explains how to use the Docling Service API programmatically for agentic AI systems and automation workflows.

## API Workflow Overview

The API follows an **asynchronous task pattern**:

1. **Submit** a PDF for conversion → Get a `task_id`
2. **Poll** the status using `task_id` → Check if processing is complete
3. **Download** the Markdown result when status is `completed`

## Base URL

```
http://localhost:5001
```

For production, replace with your deployed domain.

---

## API Endpoints

### 1. Submit Conversion Job

**Endpoint**: `POST /api/convert`

**Description**: Submit a PDF file or URL for conversion. Returns immediately with a task ID.

#### Option A: Upload PDF File

```bash
curl -X POST http://localhost:5001/api/convert \
  -F "file=@document.pdf"
```

**Response**:
```json
{
  "task_id": "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6"
}
```

#### Option B: Provide PDF URL

```bash
curl -X POST http://localhost:5001/api/convert \
  -F "source_url=https://example.com/document.pdf"
```

**Response**:
```json
{
  "task_id": "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6"
}
```

**Status Codes**:
- `202 Accepted` - Task queued successfully
- `400 Bad Request` - Invalid input (no file/URL, wrong format, empty file)
- `413 Payload Too Large` - File exceeds size limit

---

### 2. Check Task Status

**Endpoint**: `GET /api/status/{task_id}`

**Description**: Poll the conversion status. Use this in a loop until status is `completed` or `failed`.

```bash
curl http://localhost:5001/api/status/a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6
```

**Response (Processing)**:
```json
{
  "task_id": "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6",
  "status": "processing",
  "detail": null,
  "download_url": null,
  "source_name": "document.pdf",
  "source_kind": "upload",
  "created_at": "2025-10-15T20:30:00.000Z",
  "updated_at": "2025-10-15T20:30:05.000Z",
  "output_filename": null
}
```

**Response (Completed)**:
```json
{
  "task_id": "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6",
  "status": "completed",
  "detail": null,
  "download_url": "/api/result/a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6",
  "source_name": "document.pdf",
  "source_kind": "upload",
  "created_at": "2025-10-15T20:30:00.000Z",
  "updated_at": "2025-10-15T20:30:15.000Z",
  "output_filename": "document.md"
}
```

**Status Values**:
- `pending` - Task queued, waiting to start
- `processing` - Conversion in progress
- `completed` - Conversion successful, result ready
- `failed` - Conversion failed (check `detail` field)

**Status Codes**:
- `200 OK` - Status retrieved
- `404 Not Found` - Task ID doesn't exist

---

### 3. Download Markdown Result

**Endpoint**: `GET /api/result/{task_id}`

**Description**: Download the converted Markdown file. Only available when status is `completed`.

```bash
curl -O -J http://localhost:5001/api/result/a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6
```

**Response**: Markdown file with `Content-Type: text/markdown`

**Status Codes**:
- `200 OK` - File download
- `404 Not Found` - Task not found or result not ready

---

### 4. Check Cloud Storage Status

**Endpoint**: `GET /api/cloud-storage/status`

**Description**: Inspect the active cloud storage configuration.

```bash
curl http://localhost:5001/api/cloud-storage/status
```

**Response**:
```json
{
  "enabled": true,
  "provider": "cloudflare_r2",
  "upload_mode": "sync",
  "backend_ready": true,
  "backend_type": "CloudflareR2Storage",
  "keep_local_copy": true
}
```

---

## Complete Workflow Examples

### Python Example (Synchronous)

```python
import requests
import time
from pathlib import Path

BASE_URL = "http://localhost:5001"

def convert_pdf(pdf_path: str) -> str:
    """Convert PDF and return markdown content."""
    
    # Step 1: Submit conversion
    with open(pdf_path, 'rb') as f:
        response = requests.post(
            f"{BASE_URL}/api/convert",
            files={"file": f}
        )
    response.raise_for_status()
    task_id = response.json()["task_id"]
    print(f"Task submitted: {task_id}")
    
    # Step 2: Poll status
    while True:
        response = requests.get(f"{BASE_URL}/api/status/{task_id}")
        response.raise_for_status()
        status_data = response.json()
        
        status = status_data["status"]
        print(f"Status: {status}")
        
        if status == "completed":
            break
        elif status == "failed":
            raise Exception(f"Conversion failed: {status_data.get('detail')}")
        
        time.sleep(3)  # Poll every 3 seconds
    
    # Step 3: Download result
    response = requests.get(f"{BASE_URL}/api/result/{task_id}")
    response.raise_for_status()
    
    markdown_content = response.text
    print(f"Markdown received: {len(markdown_content)} bytes")
    
    return markdown_content

# Usage
if __name__ == "__main__":
    markdown = convert_pdf("document.pdf")
    Path("output.md").write_text(markdown)
    print("Conversion complete!")
```

### Python Example (Async with aiohttp)

```python
import aiohttp
import asyncio
from pathlib import Path

BASE_URL = "http://localhost:5001"

async def convert_pdf_async(pdf_path: str) -> str:
    """Convert PDF asynchronously and return markdown content."""
    
    async with aiohttp.ClientSession() as session:
        # Step 1: Submit conversion
        with open(pdf_path, 'rb') as f:
            data = aiohttp.FormData()
            data.add_field('file', f, filename=Path(pdf_path).name)
            
            async with session.post(f"{BASE_URL}/api/convert", data=data) as resp:
                resp.raise_for_status()
                result = await resp.json()
                task_id = result["task_id"]
                print(f"Task submitted: {task_id}")
        
        # Step 2: Poll status
        while True:
            async with session.get(f"{BASE_URL}/api/status/{task_id}") as resp:
                resp.raise_for_status()
                status_data = await resp.json()
                
                status = status_data["status"]
                print(f"Status: {status}")
                
                if status == "completed":
                    break
                elif status == "failed":
                    raise Exception(f"Conversion failed: {status_data.get('detail')}")
                
                await asyncio.sleep(3)
        
        # Step 3: Download result
        async with session.get(f"{BASE_URL}/api/result/{task_id}") as resp:
            resp.raise_for_status()
            markdown_content = await resp.text()
            print(f"Markdown received: {len(markdown_content)} bytes")
            
            return markdown_content

# Usage
async def main():
    markdown = await convert_pdf_async("document.pdf")
    Path("output.md").write_text(markdown)
    print("Conversion complete!")

if __name__ == "__main__":
    asyncio.run(main())
```

### JavaScript/Node.js Example

```javascript
const fs = require('fs');
const FormData = require('form-data');
const axios = require('axios');

const BASE_URL = 'http://localhost:5001';

async function convertPdf(pdfPath) {
  // Step 1: Submit conversion
  const formData = new FormData();
  formData.append('file', fs.createReadStream(pdfPath));
  
  const submitResponse = await axios.post(`${BASE_URL}/api/convert`, formData, {
    headers: formData.getHeaders()
  });
  
  const taskId = submitResponse.data.task_id;
  console.log(`Task submitted: ${taskId}`);
  
  // Step 2: Poll status
  while (true) {
    const statusResponse = await axios.get(`${BASE_URL}/api/status/${taskId}`);
    const status = statusResponse.data.status;
    
    console.log(`Status: ${status}`);
    
    if (status === 'completed') {
      break;
    } else if (status === 'failed') {
      throw new Error(`Conversion failed: ${statusResponse.data.detail}`);
    }
    
    await new Promise(resolve => setTimeout(resolve, 3000)); // Wait 3 seconds
  }
  
  // Step 3: Download result
  const resultResponse = await axios.get(`${BASE_URL}/api/result/${taskId}`);
  const markdown = resultResponse.data;
  
  console.log(`Markdown received: ${markdown.length} bytes`);
  return markdown;
}

// Usage
convertPdf('document.pdf')
  .then(markdown => {
    fs.writeFileSync('output.md', markdown);
    console.log('Conversion complete!');
  })
  .catch(console.error);
```

### cURL Script (Bash)

```bash
#!/bin/bash

BASE_URL="http://localhost:5001"
PDF_FILE="document.pdf"

# Step 1: Submit conversion
echo "Submitting PDF for conversion..."
RESPONSE=$(curl -s -X POST "$BASE_URL/api/convert" -F "file=@$PDF_FILE")
TASK_ID=$(echo $RESPONSE | jq -r '.task_id')
echo "Task ID: $TASK_ID"

# Step 2: Poll status
while true; do
  STATUS_RESPONSE=$(curl -s "$BASE_URL/api/status/$TASK_ID")
  STATUS=$(echo $STATUS_RESPONSE | jq -r '.status')
  echo "Status: $STATUS"
  
  if [ "$STATUS" = "completed" ]; then
    break
  elif [ "$STATUS" = "failed" ]; then
    echo "Conversion failed!"
    exit 1
  fi
  
  sleep 3
done

# Step 3: Download result
echo "Downloading markdown..."
curl -o output.md "$BASE_URL/api/result/$TASK_ID"
echo "Conversion complete! Saved to output.md"
```

---

## Cloud Storage Behavior

When cloud storage is enabled (`DOCLING_CLOUD_STORAGE_ENABLED=true`):

1. **Images are uploaded** to the configured provider (e.g., Cloudflare R2)
2. **Markdown contains cloud URLs** like:
   ```markdown
   ![Image](https://pdf2md.mynetwork.ing/images/task_id/picture-1.png)
   ```
3. **Images are publicly accessible** via the custom domain or R2 public URL

### Checking Cloud Status Before Conversion

```python
response = requests.get(f"{BASE_URL}/api/cloud-storage/status")
cloud_status = response.json()

if cloud_status["enabled"] and cloud_status["backend_ready"]:
    print(f"Cloud storage active: {cloud_status['provider']}")
else:
    print("Using local storage")
```

---

## Error Handling

### Common Error Scenarios

**1. File Too Large**
```json
{
  "detail": "Uploaded file exceeds 25 MiB limit"
}
```
**Solution**: Reduce file size or increase `DOCLING_MAX_UPLOAD_MB`

**2. Invalid File Format**
```json
{
  "detail": "Only PDF uploads are supported"
}
```
**Solution**: Ensure file has `.pdf` extension

**3. Task Not Found**
```json
{
  "detail": "Task not found"
}
```
**Solution**: Verify task ID is correct

**4. Conversion Failed**
```json
{
  "status": "failed",
  "detail": "Error processing document"
}
```
**Solution**: Check server logs for details

---

## Best Practices for Agentic AI Systems

### 1. Implement Exponential Backoff

```python
import time

def poll_with_backoff(task_id, max_wait=300):
    """Poll with exponential backoff."""
    wait_time = 2
    elapsed = 0
    
    while elapsed < max_wait:
        response = requests.get(f"{BASE_URL}/api/status/{task_id}")
        status = response.json()["status"]
        
        if status in ["completed", "failed"]:
            return status
        
        time.sleep(wait_time)
        elapsed += wait_time
        wait_time = min(wait_time * 1.5, 30)  # Max 30 seconds
    
    raise TimeoutError("Conversion timeout")
```

### 2. Handle Rate Limits

```python
from time import sleep
from requests.exceptions import HTTPError

def submit_with_retry(pdf_path, max_retries=3):
    """Submit with retry logic."""
    for attempt in range(max_retries):
        try:
            with open(pdf_path, 'rb') as f:
                response = requests.post(
                    f"{BASE_URL}/api/convert",
                    files={"file": f}
                )
            response.raise_for_status()
            return response.json()["task_id"]
        except HTTPError as e:
            if e.response.status_code == 429:  # Rate limited
                sleep(2 ** attempt)  # Exponential backoff
            else:
                raise
    raise Exception("Max retries exceeded")
```

### 3. Batch Processing

```python
import asyncio
from typing import List

async def convert_batch(pdf_paths: List[str]) -> List[str]:
    """Convert multiple PDFs concurrently."""
    tasks = [convert_pdf_async(path) for path in pdf_paths]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results
```

---

## OpenAPI/Swagger Documentation

Interactive API documentation is available at:

```
http://localhost:5001/docs
```

This provides:
- Full API schema
- Request/response examples
- Interactive testing interface

---

## Support

For issues or questions:
- Check server logs: `docker compose logs -f docling`
- Review documentation in `docs/`
- Open an issue on GitHub
