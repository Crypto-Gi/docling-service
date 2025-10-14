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
from docling.document_converter import ConversionStatus, DocumentConverter, PdfFormatOption
from docling_core.transforms.serializer.markdown import MarkdownDocSerializer, MarkdownParams
from docling_core.types.doc.base import ImageRefMode

from .config import settings


@dataclass
class ConversionSource:
    kind: Literal["upload", "url"]
    value: str
    original_name: Optional[str] = None
    file_path: Optional[Path] = None


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


class ConverterManager:
    def __init__(self, prefer_gpu: bool) -> None:
        self.prefer_gpu = prefer_gpu
        self._converters: dict[str, DocumentConverter] = {}
        self._lock = asyncio.Lock()

    def _select_device(self) -> str:
        if self.prefer_gpu:
            if torch.cuda.is_available():
                return "cuda"
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

    async def convert(self, source: str, force_device: Optional[str] = None) -> str:
        device = force_device or self._select_device()
        converter = await self._get_converter(device)
        try:
            result = await asyncio.to_thread(converter.convert, source)
        except torch.cuda.OutOfMemoryError as exc:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if device != "cpu":
                return await self.convert(source, force_device="cpu")
            raise RuntimeError("Docling conversion failed") from exc
        except RuntimeError as exc:
            if device != "cpu":
                return await self.convert(source, force_device="cpu")
            raise RuntimeError("Docling conversion failed") from exc
        if result.status != ConversionStatus.SUCCESS:
            raise RuntimeError(f"Docling conversion returned status {result.status}")
        if result.document is None:
            raise RuntimeError("Docling conversion produced no document")
        serializer = MarkdownDocSerializer(
            doc=result.document,
            params=MarkdownParams(image_mode=ImageRefMode.EMBEDDED),
        )
        markdown = serializer.serialize().text
        return markdown


class TaskManager:
    def __init__(self, converter: ConverterManager, storage_root: Path, soft_limit: int, hard_limit: int) -> None:
        self.converter = converter
        self.storage_root = storage_root
        self.upload_dir = storage_root / "uploads"
        self.output_dir = storage_root / "outputs"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._tasks: dict[str, TaskState] = {}
        self._lock = asyncio.Lock()
        self._cleanup_lock = asyncio.Lock()
        self._soft_limit = soft_limit
        self._hard_limit = hard_limit

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
        await self._update(task_id, status="processing", detail=None)
        try:
            markdown = await self.converter.convert(source.value)
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
        await self._update(
            task_id,
            status="completed",
            output_path=output_path,
            output_filename=output_filename,
        )
        if source.file_path and source.file_path.exists():
            await asyncio.to_thread(source.file_path.unlink)

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
) -> ConvertResponse:
    if file is None and not source_url:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Provide a file or source_url")
    task_id = uuid4().hex
    if file is not None:
        filename = file.filename or "document.pdf"
        suffix = Path(filename).suffix.lower() or ".pdf"
        if suffix != ".pdf":
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only PDF uploads are supported")
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
        )
    else:
        source = ConversionSource(kind="url", value=source_url.strip())
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
    )


@app.get("/api/result/{task_id}", response_class=FileResponse, name="download_markdown")
async def download_markdown(task_id: str) -> FileResponse:
    state = await manager.get(task_id)
    if state is None or state.output_path is None or state.status != "completed":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Result not available")
    filename = state.output_filename or state.source_name or f"{task_id}.md"
    return FileResponse(state.output_path, media_type="text/markdown", filename=filename)
