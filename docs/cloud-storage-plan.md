# Cloud Storage Integration Plan

## Overview
Enable `_save_images()` in `app/main.py` to upload generated PNGs to cloud storage, replace local paths with public URLs, and have Markdown reference those URLs. The design must be modular so different providers (Cloudflare R2, AWS S3, Google Cloud Storage, Azure Blob, etc.) can be swapped with minimal changes.

## Phase 1: Modular Architecture Design

### Objectives
- Introduce a pluggable storage backend abstraction.
- Ensure provider-agnostic logic in the main pipeline.
- Allow future providers (e.g., `amazon.py`, `gcs.py`) without touching core code.

### Deliverables
- `app/storage/base.py`: Abstract base class defining required methods (`upload`, `delete`, `get_url`, `is_enabled`).
- `app/storage/local.py`: Current local filesystem implementation.
- `app/storage/cloudflare_r2.py`: Cloudflare R2 implementation.
- `app/storage/__init__.py`: Factory function `create_storage_backend(settings)` returning the appropriate backend.
- Module documentation (`docs/cloud-storage-plan.md`).

### Key Design Principles
- **Separation of concerns**: Keep provider-specific logic out of business logic.
- **Configuration-driven selection**: Environment variables specify the active provider.
- **Graceful fallback**: Fall back to local storage if cloud upload fails or is disabled.
- **Extensibility**: Adding a new provider requires only a new module implementing the base class.

## Phase 2: Upload Strategy Selection

### Synchronous vs Asynchronous Upload Modes

**Two approaches are available:**

#### Approach 1: Synchronous Upload (Recommended Default)
- **How it works**: Wait for all cloud uploads to complete before inserting URLs into Markdown.
- **Pros**:
  - Guaranteed data integrity - all links work immediately
  - Simpler error handling
  - No broken links or 404 errors for users
  - Easier to debug and maintain
- **Cons**:
  - Slower PDF processing (blocks on upload completion)
  - Higher perceived latency for users
- **Use case**: Production environments where reliability is critical

#### Approach 2: Asynchronous Background Upload (Advanced)
- **How it works**: 
  1. Generate predictable URLs immediately (before upload)
  2. Insert URLs into Markdown
  3. Return results to user quickly
  4. Upload files to cloud storage in background tasks
- **Pros**:
  - Faster PDF processing (non-blocking)
  - Better user experience (immediate results)
  - Parallel upload operations
- **Cons**:
  - Temporary broken links (404 errors until upload completes)
  - Complex error handling (what if background upload fails?)
  - Need upload status tracking
  - Requires retry logic for failed uploads
- **Use case**: High-throughput scenarios where speed > consistency

### Research Findings

Based on research into S3/R2 behavior:

1. **Pre-generated URLs are possible**: S3/R2 URLs follow predictable patterns (`https://bucket.endpoint/key`), so URLs can be constructed before upload.
2. **404 before upload completes**: Users accessing URLs before upload finishes will receive HTTP 404 errors.
3. **Background tasks in asyncio**: `asyncio.create_task()` is sufficient for background uploads without needing Celery for simple use cases.
4. **Eventual consistency patterns**: Need retry logic, status tracking, or placeholder mechanisms to handle the gap between URL insertion and upload completion.

### Configuration (`app/config.py`)
Add Cloudflare-specific settings and upload mode toggle:

- `DOCLING_CLOUD_STORAGE_ENABLED`
- `DOCLING_CLOUD_STORAGE_PROVIDER` (`local`, `cloudflare_r2`, future options)
- `DOCLING_CLOUD_UPLOAD_MODE` (`sync`, `async`) - **New: controls upload strategy**
- `DOCLING_R2_ACCOUNT_ID`
- `DOCLING_R2_ACCESS_KEY_ID`
- `DOCLING_R2_SECRET_ACCESS_KEY`
- `DOCLING_R2_BUCKET_NAME`
- `DOCLING_R2_REGION` (`wnam`, `enam`, `weur`, `eeur`, `apac`, `auto`)
- `DOCLING_R2_PUBLIC_URL_BASE`
- `DOCLING_CLOUD_KEEP_LOCAL`

Update `Settings` in `app/config.py` to expose these values and compute derived paths/flags.

### Cloudflare R2 Implementation (`app/storage/cloudflare_r2.py`)
- Initialize boto3 client with custom `endpoint_url = "https://<accountid>.r2.cloudflarestorage.com"`.
- Use `upload_fileobj` or `put_object` to upload PNGs.
- Set `ACL` to `public-read` (or use public bucket) to expose URLs.
- Construct public URLs using either R2 endpoint or configured CDN base.
- Provide `delete()` and `get_url()` implementations.
- Handle edge cases (missing configuration, client errors) gracefully.

### Local Storage Backend (`app/storage/local.py`)
- Provide baseline implementation storing files locally.
- Return relative paths (`images/<task_id>/picture-N.png`).
- When cloud storage disabled, this backend is used automatically.

### Storage Factory (`app/storage/__init__.py`)
- Inspect `Settings` to select backend.
- Default to `LocalStorage` when cloud storage is disabled or misconfigured.
- Easily expand to other providers.

## Phase 3: Pipeline Integration

### Implementation for Synchronous Mode (Default)

#### `TaskManager` updates (`app/main.py`)
- Instantiate storage backend during initialization: `self.storage_backend = create_storage_backend(settings)`.
- Modify `_save_images()` to:
  1. Save PNG locally to `images_dir`
  2. **Wait for upload to complete** via `await self.storage_backend.upload(local_path, remote_key)`
  3. Record final URL in `image_map` (cloud URL or local path fallback)
  4. Optionally delete local file if `DOCLING_CLOUD_KEEP_LOCAL=false`
- Update `_update_image_uris()` to use URL strings (cloud or local) when setting `ImageRef.uri`.
- Ensure Markdown serializer references the finalized URLs.
- Apply similar logic for tables if needed.

**Flow diagram (Synchronous)**:
```
PDF Processing → Extract Images → Save Locally → Upload to Cloud (WAIT) → Insert URLs → Return Markdown
```

### Implementation for Asynchronous Mode (Advanced)

#### `TaskManager` updates for async mode
- Add background task tracking: `self._upload_tasks: dict[str, list[asyncio.Task]] = {}`
- Modify `_save_images()` to:
  1. Save PNG locally to `images_dir`
  2. **Pre-generate URL** using `self.storage_backend.get_url(remote_key)` (before upload)
  3. Record pre-generated URL in `image_map` immediately
  4. Launch background upload task: `task = asyncio.create_task(self._background_upload(...))`
  5. Track task: `self._upload_tasks[task_id].append(task)`
  6. Return immediately without waiting
- Add new method `_background_upload()`:
  ```python
  async def _background_upload(self, local_path: Path, remote_key: str, task_id: str):
      try:
          await self.storage_backend.upload(local_path, remote_key)
          if not settings.cloud_keep_local_copy:
              local_path.unlink()
          _log.info(f"Background upload completed: {remote_key}")
      except Exception as e:
          _log.error(f"Background upload failed for {remote_key}: {e}")
          # Could implement retry logic here
  ```
- Add cleanup method to wait for pending uploads (optional):
  ```python
  async def wait_for_uploads(self, task_id: str, timeout: float = 60.0):
      tasks = self._upload_tasks.get(task_id, [])
      if tasks:
          await asyncio.wait(tasks, timeout=timeout)
  ```

**Flow diagram (Asynchronous)**:
```
PDF Processing → Extract Images → Save Locally → Pre-generate URLs → Insert URLs → Return Markdown
                                                ↓
                                         Background: Upload to Cloud (async)
```

### Storage Backend Interface Updates

Add method to `StorageBackend` base class for pre-generating URLs:

```python
@abstractmethod
def get_url(self, remote_key: str) -> str:
    """Get public URL for a key (works before upload for predictable URLs)."""
    pass
```

### Error Handling

#### Synchronous Mode
- Catch upload failures, log errors, and fall back to local paths.
- Avoid breaking existing functionality when cloud storage is unavailable.
- User always gets working links (or local fallback).

#### Asynchronous Mode
- Log background upload failures but don't block user response.
- Implement retry logic with exponential backoff (optional).
- Consider adding upload status endpoint for monitoring.
- Users may temporarily see 404 errors until upload completes.
- Could add JavaScript retry logic on frontend to handle 404s gracefully.

## Phase 4: Dependencies & Environment

### Dependencies
- Add `boto3>=1.34.0` and `botocore>=1.34.0` to `requirements.txt`.
- Rebuild Docker image after adding dependencies.

### Environment Variables
Update `.env` (and deployment manifests) with new variables:

```bash
DOCLING_CLOUD_STORAGE_ENABLED=false
DOCLING_CLOUD_STORAGE_PROVIDER=local
DOCLING_CLOUD_UPLOAD_MODE=sync  # sync or async
DOCLING_R2_ACCOUNT_ID=
DOCLING_R2_ACCESS_KEY_ID=
DOCLING_R2_SECRET_ACCESS_KEY=
DOCLING_R2_BUCKET_NAME=
DOCLING_R2_REGION=auto
DOCLING_R2_PUBLIC_URL_BASE=
DOCLING_CLOUD_KEEP_LOCAL=false
```

Provide documentation for secrets management and environment configuration.

## Phase 5: Testing & Validation

### Unit Tests
- Mock boto3 client to test R2 uploader (success, failure, retries).
- Test factory selection logic.
- Verify local backend paths.
- Confirm fallback behavior when configuration is incomplete.

### Integration Tests
- Use MinIO or Cloudflare R2 sandbox for end-to-end tests.
- Verify Markdown references point to accessible URLs.
- Confirm optional deletion of local files.
- Run tests with cloud storage disabled to ensure backward compatibility.

### Manual Verification Checklist

#### Synchronous Mode Tests
- Upload document with images; verify R2 objects exist.
- Open Markdown; ensure image URLs load immediately.
- Toggle `DOCLING_CLOUD_KEEP_LOCAL` and confirm behavior.
- Simulate credential failure; ensure fallback to local paths.
- Measure processing time with sync uploads enabled.

#### Asynchronous Mode Tests
- Upload document with images; verify Markdown returns quickly.
- Check that URLs are inserted immediately (before upload completes).
- Access image URLs immediately after Markdown generation (should get 404).
- Wait 5-10 seconds and retry URLs (should load successfully).
- Verify background tasks complete successfully (check logs).
- Test with multiple concurrent uploads to ensure task tracking works.
- Simulate upload failure in background; verify local files remain if configured.

## Phase 6: Documentation & Deployment

### Documentation
- Update project README with cloud storage setup instructions.
- Add provider-specific notes (Cloudflare R2 Access Keys, bucket creation, ACLs).
- Document environment variables and safety considerations.

### Deployment Steps
1. Add dependencies and rebuild Docker image.
2. Configure environment variables/secrets.
3. Deploy updated service.
4. Monitor logs for upload errors and performance.
5. Validate first upload on production/staging environment.

### Rollback Strategy
- Set `DOCLING_CLOUD_STORAGE_ENABLED=false` to revert to local storage without code changes.
- Keep local fallback in `_save_images()` for robustness.

## Phase 7: Future Enhancements
- Implement additional providers (`amazon.py`, `gcs.py`, `azure_blob.py`).
- Support presigned URLs for private content.
- Add retry/backoff logic for transient failures (especially for async mode).
- Collect metrics on upload success/failures.
- Implement lifecycle policies (delete old cloud objects when local cleanup runs).
- Add upload status API endpoint for monitoring background uploads.
- Implement frontend retry logic for handling 404s gracefully in async mode.

## Summary: Sync vs Async Decision Matrix

| Criteria | Synchronous Mode | Asynchronous Mode |
|----------|------------------|-------------------|
| **Reliability** | ✅ High - all links work immediately | ⚠️ Medium - temporary 404s possible |
| **Speed** | ⚠️ Slower - blocks on uploads | ✅ Fast - non-blocking |
| **Complexity** | ✅ Simple implementation | ⚠️ Complex - needs task tracking |
| **Error Handling** | ✅ Straightforward | ⚠️ Complex - background failures |
| **User Experience** | ⚠️ Higher latency | ✅ Immediate results |
| **Production Ready** | ✅ Yes - recommended default | ⚠️ Requires additional monitoring |
| **Best For** | Production, reliability-critical | High-throughput, speed-critical |

### Recommendation
- **Start with Synchronous Mode** (`DOCLING_CLOUD_UPLOAD_MODE=sync`) for initial deployment.
- **Consider Asynchronous Mode** only after:
  - Monitoring shows upload times are bottleneck
  - Frontend can handle temporary 404s gracefully
  - Background task monitoring is in place
  - Retry logic is implemented

## References
- Cloudflare R2 boto3 integration docs: https://developers.cloudflare.com/r2/examples/aws/boto3/
- Cloudflare R2 tutorials & examples: https://developers.cloudflare.com/r2/tutorials/
- Cloudflare R2 S3 API compatibility: https://developers.cloudflare.com/r2/api/s3/api/
- Boto3 S3 upload documentation: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/s3-uploading-files.html
- Perplexity research on S3/R2 URL generation and async patterns (conducted 2025-10-15)
