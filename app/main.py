from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional
from uuid import uuid4

import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import ConversionResult, ConversionStatus, DocumentConverter, PdfFormatOption
from docling_core.transforms.serializer.markdown import MarkdownDocSerializer, MarkdownParams
from docling_core.types.doc.base import ImageRefMode
from docling_core.types.doc.document import ImageRef, PictureItem, TableItem

from .config import settings
from .storage import create_storage_backend


class CloudStorageConfig(BaseModel):
    """Optional cloud storage configuration for per-request override."""
    enabled: bool = True
    provider: Literal["cloudflare_r2", "local"] = "cloudflare_r2"
    r2_account_id: Optional[str] = None
    r2_access_key_id: Optional[str] = None
    r2_secret_access_key: Optional[str] = None
    r2_bucket_name: Optional[str] = None
    r2_region: str = "auto"
    r2_public_url_base: Optional[str] = None


@dataclass
class ConversionSource:
    kind: Literal["upload", "url"]
    value: str
    original_name: Optional[str] = None
    file_path: Optional[Path] = None
    cloud_storage_config: Optional[dict] = None


@dataclass
class TaskState:
    task_id: str
    status: Literal["pending", "processing", "completed", "failed"]
    detail: Optional[str]
    output_path: Optional[Path]
    created_at: datetime
    updated_at: datetime
    source_name: Optional[str] = None
    source_kind: Literal["upload", "url"] = "upload"
    output_filename: Optional[str] = None
    markdown_url: Optional[str] = None
    cloud_storage_config: Optional[dict] = None


class TaskStatusResponse(BaseModel):
    task_id: str
    status: Literal["pending", "processing", "completed", "failed"]
    detail: Optional[str] = None
    download_url: Optional[str] = None
    source_name: Optional[str] = None
    source_kind: Literal["upload", "url"]
    created_at: datetime
    updated_at: datetime
    output_filename: Optional[str] = None
    markdown_url: Optional[str] = None


class ConverterManager:
    def __init__(self, prefer_gpu: bool) -> None:
        self.prefer_gpu = prefer_gpu
        self._converters: dict[str, DocumentConverter] = {}
        self._lock = asyncio.Lock()

    def _select_device(self) -> str:
        """Select the best available device based on GPU availability and memory."""
        if self.prefer_gpu:
            if torch.cuda.is_available():
                try:
                    # Check if GPU has enough memory (at least 2GB free)
                    gpu_mem_free = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                    if gpu_mem_free > 2 * 1024 * 1024 * 1024:  # 2GB
                        return "cuda"
                    else:
                        import logging
                        logging.getLogger(__name__).warning(
                            f"GPU has insufficient memory ({gpu_mem_free / 1024**3:.2f}GB free), using CPU"
                        )
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).warning(f"Error checking GPU memory, using CPU: {e}")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        return "cpu"

    async def _get_converter(self, device: str) -> DocumentConverter:
        async with self._lock:
            existing = self._converters.get(device)
            if existing is not None:
                return existing
            converter = await asyncio.to_thread(self._build_converter, device)
            self._converters[device] = converter
            return converter

    def _build_converter(self, device: str) -> DocumentConverter:
        accelerator = AcceleratorOptions(device=device)
        pipeline_options = PdfPipelineOptions(accelerator_options=accelerator)
        pipeline_options.generate_picture_images = True
        pipeline_options.generate_page_images = False
        format_options = {InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        return DocumentConverter(format_options=format_options)

    async def convert(self, source: str, force_device: Optional[str] = None) -> "ConversionResult":
        device = force_device or self._select_device()
        converter = await self._get_converter(device)
        try:
            result = await asyncio.to_thread(converter.convert, source)
        except torch.cuda.OutOfMemoryError as exc:
            # GPU out of memory - fallback to CPU
            import logging
            logging.getLogger(__name__).warning(f"GPU OOM error, falling back to CPU: {exc}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if device != "cpu":
                return await self.convert(source, force_device="cpu")
            raise RuntimeError("Docling conversion failed") from exc
        except MemoryError as exc:
            # System memory error - fallback to CPU if using GPU
            import logging
            logging.getLogger(__name__).warning(f"System memory error, falling back to CPU: {exc}")
            if device != "cpu":
                return await self.convert(source, force_device="cpu")
            raise RuntimeError("Docling conversion failed due to insufficient memory") from exc
        except RuntimeError as exc:
            # Generic runtime error - try CPU fallback
            import logging
            logging.getLogger(__name__).warning(f"Runtime error on {device}, attempting CPU fallback: {exc}")
            if device != "cpu":
                return await self.convert(source, force_device="cpu")
            raise RuntimeError("Docling conversion failed") from exc
        if result.status != ConversionStatus.SUCCESS:
            raise RuntimeError(f"Docling conversion returned status {result.status}")
        if result.document is None:
            raise RuntimeError("Docling conversion produced no document")
        return result


class TaskManager:
    def __init__(self, converter: ConverterManager, storage_root: Path, soft_limit: int, hard_limit: int) -> None:
        self.converter = converter
        self.storage_root = storage_root
        self.upload_dir = storage_root / "uploads"
        self.output_dir = storage_root / "outputs"
        self.images_dir = storage_root / "images"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self._tasks: dict[str, TaskState] = {}
        self._lock = asyncio.Lock()
        self._cleanup_lock = asyncio.Lock()
        self._soft_limit = soft_limit
        self._hard_limit = hard_limit
        self.storage_backend = create_storage_backend(settings)

    async def enqueue(self, task_id: str, source: ConversionSource) -> None:
        now = datetime.utcnow()
        state = TaskState(
            task_id=task_id,
            status="pending",
            detail=None,
            output_path=None,
            created_at=now,
            updated_at=now,
            source_name=source.original_name,
            source_kind=source.kind,
        )
        async with self._lock:
            self._tasks[task_id] = state
        asyncio.create_task(self._execute(task_id, source))

    async def _execute(self, task_id: str, source: ConversionSource) -> None:
        # Create storage backend based on request-specific config or default settings
        storage_backend = self._create_storage_backend(source.cloud_storage_config)
        await self._update(task_id, status="processing", detail=None)
        task_images_dir = self.images_dir / task_id
        task_images_dir.mkdir(parents=True, exist_ok=True)
        try:
            result = await self.converter.convert(source.value)
            image_map = await self._save_images(result, task_id, task_images_dir, storage_backend)
            await self._update_image_uris(result, image_map)

            serializer = MarkdownDocSerializer(
                doc=result.document,
                params=MarkdownParams(
                    image_mode=ImageRefMode.REFERENCED,
                ),
            )
            markdown = serializer.serialize().text
        except Exception as exc:  # noqa: BLE001
            detail = str(exc)
            await self._update(task_id, status="failed", detail=detail)
            if source.file_path and source.file_path.exists():
                await asyncio.to_thread(source.file_path.unlink)
            return
        timestamp = datetime.utcnow().strftime("%m-%d-%Y-%H%M%S")
        output_filename = f"{timestamp}.md"
        output_path = self.output_dir / output_filename
        await asyncio.to_thread(output_path.write_text, markdown, encoding="utf-8")
        await self.enforce_output_limit(exclude={output_path})
        
        # Upload markdown to cloud storage if enabled
        markdown_url = None
        if storage_backend.is_cloud_enabled():
            markdown_url = await self._upload_markdown(output_path, task_id, source.original_name, storage_backend)
        
        await self._update(
            task_id,
            status="completed",
            output_path=output_path,
            output_filename=output_filename,
            markdown_url=markdown_url,
        )
        if source.file_path and source.file_path.exists():
            await asyncio.to_thread(source.file_path.unlink)

    def _create_storage_backend(self, custom_config: Optional[dict]):
        """Create storage backend from custom config or default settings."""
        if custom_config:
            # Use custom configuration from request
            from .storage.cloudflare_r2 import CloudflareR2Storage
            from .storage.local import LocalStorage
            
            enabled = custom_config.get("enabled", True)
            
            if not enabled:
                return LocalStorage({"base_path": settings.storage_path / "images"})
            
            provider = custom_config.get("provider", "cloudflare_r2").lower()
            if provider == "cloudflare_r2":
                # Merge custom config with .env defaults for partial overrides
                return CloudflareR2Storage({
                    "enabled": True,
                    "account_id": custom_config.get("r2_account_id") or settings.r2_account_id,
                    "access_key_id": custom_config.get("r2_access_key_id") or settings.r2_access_key_id,
                    "secret_access_key": custom_config.get("r2_secret_access_key") or settings.r2_secret_access_key,
                    "bucket_name": custom_config.get("r2_bucket_name") or settings.r2_bucket_name,
                    "region": custom_config.get("r2_region") or settings.r2_region or "auto",
                    "public_url_base": custom_config.get("r2_public_url_base") or settings.r2_public_url_base,
                })
            return LocalStorage({"base_path": settings.storage_path / "images"})
        else:
            # Use default settings from .env
            return self.storage_backend

    async def _update_image_uris(self, result: ConversionResult, image_map: dict[str, str]) -> None:
        """Update image URIs in the document to reference saved file paths."""
        for element, _level in result.document.iterate_items():
            if isinstance(element, PictureItem):
                rel_path = image_map.get(element.self_ref)
                if rel_path and element.image is not None:
                    element.image = ImageRef(
                        mimetype=element.image.mimetype,
                        dpi=element.image.dpi,
                        size=element.image.size,
                        uri=rel_path,
                    )

    async def _save_images(self, result: ConversionResult, task_id: str, images_dir: Path, storage_backend) -> dict[str, str]:
        """Save picture images to disk and upload to cloud storage (sync mode).
        
        Tables are exported as native Markdown tables by the serializer, not as images.
        
        Returns:
            Dictionary mapping element.self_ref to final URL/path (cloud URL or local path)
        """
        import logging
        _log = logging.getLogger(__name__)
        
        picture_counter = 0
        image_map: dict[str, str] = {}
        
        for element, _level in result.document.iterate_items():
            if isinstance(element, PictureItem):
                picture_counter += 1
                filename = f"picture-{picture_counter}.png"
                local_path = images_dir / filename
                cloud_key = f"images/{task_id}/{filename}"
                
                # Save locally first
                img = element.get_image(result.document)
                if img:
                    await asyncio.to_thread(img.save, local_path, "PNG")
                    
                    # Upload to cloud storage if enabled (synchronous mode)
                    try:
                        url = await storage_backend.upload(local_path, cloud_key)
                        image_map[element.self_ref] = url
                        
                        # Optionally delete local copy after successful upload
                        if not settings.cloud_keep_local_copy and storage_backend.is_cloud_enabled():
                            if storage_backend.__class__.__name__ != "LocalStorage":
                                local_path.unlink()
                    
                    except Exception as e:
                        # Fallback to local path on upload failure
                        _log.error(f"Cloud upload failed for {filename}, using local path: {e}")
                        image_map[element.self_ref] = str(Path("images") / task_id / filename)
        
        return image_map

    def _upload_markdown(self, md_path: Path, task_id: str, original_name: Optional[str], storage_backend) -> Optional[str]:
        """Upload markdown file to cloud storage."""
        import logging
        logger = logging.getLogger(__name__)
        
        filename = original_name or f"{task_id}.md"
        if not filename.endswith(".md"):
            filename = f"{filename}.md"
        
        cloud_key = f"markdown/{task_id}/{filename}"
        url = storage_backend.upload(md_path, cloud_key)
        if url:
            logger.info(f"Uploaded markdown to {url}")
        return url

    async def _update(self, task_id: str, **updates) -> None:
        async with self._lock:
            state = self._tasks.get(task_id)
            if state is None:
                return
            for key, value in updates.items():
                setattr(state, key, value)
            state.updated_at = datetime.utcnow()

    async def get(self, task_id: str) -> Optional[TaskState]:
        async with self._lock:
            state = self._tasks.get(task_id)
            if state is None:
                return None
            return TaskState(
                task_id=state.task_id,
                status=state.status,
                detail=state.detail,
                output_path=state.output_path,
                created_at=state.created_at,
                updated_at=state.updated_at,
                source_name=state.source_name,
                source_kind=state.source_kind,
                output_filename=state.output_filename,
                markdown_url=state.markdown_url,
            )

    async def enforce_upload_limit(self, *, exclude: set[Path] | None = None) -> None:
        await self._enforce_limits(self.upload_dir, exclude=exclude)

    async def enforce_output_limit(self, *, exclude: set[Path] | None = None) -> None:
        await self._enforce_limits(self.output_dir, exclude=exclude)

    async def _enforce_limits(self, directory: Path, *, exclude: set[Path] | None = None) -> None:
        if self._hard_limit <= 0 or self._soft_limit <= 0:
            return
        async with self._cleanup_lock:
            await asyncio.to_thread(self._trim_directory, directory, exclude or set())

    def _trim_directory(self, directory: Path, exclude: set[Path]) -> None:
        if not directory.exists():
            return
        resolved_exclude = set()
        for path in exclude:
            try:
                resolved_exclude.add(path.resolve())
            except FileNotFoundError:
                continue
        entries: list[tuple[float, Path, int]] = []
        total_size = 0
        for item in directory.rglob("*"):
            if not item.is_file():
                continue
            try:
                resolved = item.resolve()
            except FileNotFoundError:
                continue
            if resolved in resolved_exclude:
                continue
            try:
                stat = resolved.stat()
            except FileNotFoundError:
                continue
            total_size += stat.st_size
            entries.append((stat.st_mtime, resolved, stat.st_size))
        if total_size <= self._hard_limit:
            return
        entries.sort(key=lambda entry: entry[0])
        target = min(self._soft_limit, self._hard_limit)
        for _mtime, path, size in entries:
            if total_size <= target:
                break
            try:
                path.unlink()
                total_size -= size
            except FileNotFoundError:
                continue


converter_manager = ConverterManager(prefer_gpu=settings.prefer_gpu)
manager = TaskManager(
    converter_manager,
    settings.storage_path,
    settings.storage_soft_limit_bytes,
    settings.storage_hard_limit_bytes,
)

app = FastAPI(title="Docling Markdown Converter")
base_path = Path(__file__).parent
templates = Jinja2Templates(directory=str(base_path / "templates"))
app.mount(
    "/static",
    StaticFiles(directory=str(base_path / "static")),
    name="static",
)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "max_upload_mb": settings.max_upload_mb,
        },
    )


class ConvertResponse(BaseModel):
    task_id: str


@app.post("/api/convert", response_model=ConvertResponse, status_code=status.HTTP_202_ACCEPTED)
async def submit_conversion(
    request: Request,
    file: UploadFile | None = File(None),
    source_url: str | None = Form(None),
    # Optional cloud storage configuration (overrides .env settings)
    cloud_storage_enabled: bool | None = Form(None),
    r2_account_id: str | None = Form(None),
    r2_access_key_id: str | None = Form(None),
    r2_secret_access_key: str | None = Form(None),
    r2_bucket_name: str | None = Form(None),
    r2_region: str | None = Form(None),
    r2_public_url_base: str | None = Form(None),
) -> ConvertResponse:
    if file is None and not source_url:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Provide a file or source_url")
    
    # Build custom cloud storage config if any parameters provided
    cloud_storage_config = None
    if any([cloud_storage_enabled is not None, r2_account_id, r2_access_key_id, r2_secret_access_key, r2_bucket_name]):
        # Parse boolean value (FastAPI converts "false" string to False boolean)
        enabled_value = cloud_storage_enabled if cloud_storage_enabled is not None else True
        
        cloud_storage_config = {
            "enabled": enabled_value,
            "provider": "cloudflare_r2",
            "r2_account_id": r2_account_id,
            "r2_access_key_id": r2_access_key_id,
            "r2_secret_access_key": r2_secret_access_key,
            "r2_bucket_name": r2_bucket_name,
            "r2_region": r2_region or "auto",
            "r2_public_url_base": r2_public_url_base,
        }
    
    task_id = uuid4().hex
    if file is not None:
        filename = file.filename or "document.pdf"
        suffix = Path(filename).suffix.lower() or ".pdf"
        # Supported formats: PDF and Microsoft Office documents
        allowed_extensions = {".pdf", ".docx", ".xlsx", ".pptx"}
        if suffix not in allowed_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file format. Allowed: {', '.join(sorted(allowed_extensions))}"
            )
        data = await file.read()
        if not data:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty")
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Uploaded file exceeds {settings.max_upload_mb} MiB limit",
            )
        upload_path = manager.upload_dir / f"{task_id}{suffix}"
        await asyncio.to_thread(upload_path.write_bytes, data)
        await manager.enforce_upload_limit(exclude={upload_path})
        source = ConversionSource(
            kind="upload",
            value=str(upload_path),
            original_name=filename,
            file_path=upload_path,
            cloud_storage_config=cloud_storage_config,
        )
    else:
        source = ConversionSource(
            kind="url",
            value=source_url.strip(),
            cloud_storage_config=cloud_storage_config,
        )
    await manager.enqueue(task_id, source)
    return ConvertResponse(task_id=task_id)


@app.get("/api/status/{task_id}", response_model=TaskStatusResponse)
async def get_status(request: Request, task_id: str) -> TaskStatusResponse:
    state = await manager.get(task_id)
    if state is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")
    download_url: Optional[str] = None
    if state.status == "completed" and state.output_path is not None:
        download_url = str(request.url_for("download_markdown", task_id=task_id))
    return TaskStatusResponse(
        task_id=state.task_id,
        status=state.status,
        detail=state.detail,
        download_url=download_url,
        source_name=state.source_name,
        source_kind=state.source_kind,
        created_at=state.created_at,
        updated_at=state.updated_at,
        output_filename=state.output_filename,
        markdown_url=state.markdown_url,
    )


@app.get("/api/result/{task_id}", response_class=FileResponse, name="download_markdown")
async def download_markdown(task_id: str) -> FileResponse:
    state = await manager.get(task_id)
    if state is None or state.output_path is None or state.status != "completed":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Result not available")
    filename = state.output_filename or state.source_name or f"{task_id}.md"
    return FileResponse(state.output_path, media_type="text/markdown", filename=filename)


class MarkdownResultResponse(BaseModel):
    """JSON response containing markdown content and metadata."""
    task_id: str
    markdown_content: str
    source_name: Optional[str]
    output_filename: str
    markdown_url: Optional[str]
    created_at: datetime
    completed_at: datetime


@app.get("/api/result/{task_id}/json", response_model=MarkdownResultResponse)
async def get_markdown_json(task_id: str) -> MarkdownResultResponse:
    """Get markdown content as JSON (ideal for agentic AI systems)."""
    state = await manager.get(task_id)
    if state is None or state.output_path is None or state.status != "completed":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Result not available")
    
    # Read markdown content
    markdown_content = await asyncio.to_thread(state.output_path.read_text, encoding="utf-8")
    
    return MarkdownResultResponse(
        task_id=task_id,
        markdown_content=markdown_content,
        source_name=state.source_name,
        output_filename=state.output_filename or f"{task_id}.md",
        markdown_url=state.markdown_url,
        created_at=state.created_at,
        completed_at=state.updated_at,
    )


class CloudStorageStatus(BaseModel):
    """Cloud storage configuration status."""
    enabled: bool
    provider: str
    upload_mode: str
    backend_ready: bool
    backend_type: str
    keep_local_copy: bool


@app.get("/api/cloud-storage/status", response_model=CloudStorageStatus)
async def get_cloud_storage_status() -> CloudStorageStatus:
    """Get current cloud storage configuration and status."""
    return CloudStorageStatus(
        enabled=settings.cloud_storage_enabled,
        provider=settings.cloud_storage_provider,
        upload_mode=settings.cloud_upload_mode,
        backend_ready=manager.storage_backend.is_enabled(),
        backend_type=manager.storage_backend.__class__.__name__,
        keep_local_copy=settings.cloud_keep_local_copy,
    )
