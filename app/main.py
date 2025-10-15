from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional
from uuid import uuid4

import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    PictureDescriptionVlmOptions,
    TableFormerMode,
    granite_picture_description,
    smolvlm_picture_description,
)
from docling.document_converter import ConversionStatus, DocumentConverter, PdfFormatOption
from docling_core.transforms.serializer.html import HTMLDocSerializer
from docling_core.transforms.serializer.markdown import MarkdownDocSerializer, MarkdownParams
from docling_core.types.doc.base import ImageRefMode

from .config import settings


class DocumentType(str, Enum):
    """Supported document types for conversion."""
    PDF = "pdf"
    IMAGE = "image"


class ModelChoice(str, Enum):
    """Available VLM models for document processing."""
    DEFAULT = "default"
    SMOLVLM = "smolvlm"
    GRANITE_DOCLING = "granite_docling"
    GRANITE_VISION_2B = "granite_vision_2b"
    GRANITE_VISION_8B = "granite_vision_8b"


@dataclass
class ConversionSource:
    kind: Literal["upload", "url"]
    value: str
    original_name: Optional[str] = None
    file_path: Optional[Path] = None
    document_type: DocumentType = DocumentType.PDF
    model_choice: ModelChoice = ModelChoice.DEFAULT
    table_mode: Literal["fast", "accurate"] = "fast"
    custom_prompt: str = ""


@dataclass
class ConversionMetadata:
    """Metadata about the conversion process."""
    model_used: str
    processing_time_seconds: float = 0.0
    pages_processed: int = 0
    images_extracted: int = 0


@dataclass
class PreviewArtifacts:
    """Paths to serialized preview artifacts."""
    html_path: Path
    json_path: Path


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
    metadata: Optional[ConversionMetadata] = None
    preview: Optional[PreviewArtifacts] = None


class ConversionMetadataResponse(BaseModel):
    """Response model for conversion metadata."""
    model_used: str
    processing_time_seconds: float
    pages_processed: int
    images_extracted: int


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
    metadata: Optional[ConversionMetadataResponse] = None


class PreviewResponse(BaseModel):
    markdown: str
    html: str
    json_data: dict


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

    async def _get_converter(self, device: str, model_choice: ModelChoice, doc_type: DocumentType, table_mode: str, custom_prompt: str = "") -> DocumentConverter:
        # Create unique key for converter cache based on device, model, table mode, and prompt
        # Only cache converters with default prompts to avoid cache explosion
        cache_key = f"{device}_{model_choice.value}_{doc_type.value}_{table_mode}_{custom_prompt}" if custom_prompt else f"{device}_{model_choice.value}_{doc_type.value}_{table_mode}"
        
        async with self._lock:
            existing = self._converters.get(cache_key)
            if existing is not None:
                return existing
            converter = await asyncio.to_thread(self._build_converter, device, model_choice, doc_type, table_mode, custom_prompt)
            # Only cache default-prompt converters to avoid memory bloat
            if not custom_prompt or len(self._converters) < 20:
                self._converters[cache_key] = converter
            return converter

    def _build_converter(self, device: str, model_choice: ModelChoice, doc_type: DocumentType, table_mode: str, custom_prompt: str = "") -> DocumentConverter:
        accelerator = AcceleratorOptions(device=device)
        pipeline_options = PdfPipelineOptions(accelerator_options=accelerator)
        pipeline_options.generate_picture_images = True
        pipeline_options.generate_page_images = False
        
        # Configure table extraction mode
        pipeline_options.do_table_structure = True
        if table_mode == "accurate":
            pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
        else:
            pipeline_options.table_structure_options.mode = TableFormerMode.FAST
        
        # Configure VLM-based picture description if using VLM models
        if model_choice == ModelChoice.SMOLVLM:
            pipeline_options.do_picture_description = True
            default_prompt = "Describe this image in detail, extracting all visible text and structure."
            prompt = custom_prompt if custom_prompt else default_prompt
            pipeline_options.picture_description_options = PictureDescriptionVlmOptions(
                repo_id="HuggingFaceTB/SmolVLM-256M-Instruct",
                prompt=prompt,
            )
            pipeline_options.images_scale = settings.vlm_images_scale
        elif model_choice == ModelChoice.GRANITE_DOCLING:
            # Granite Docling 258M: production VLM optimized for document understanding
            pipeline_options.do_picture_description = True
            default_prompt = "Convert this page to docling. Extract all text, tables, formulas, and code accurately."
            prompt = custom_prompt if custom_prompt else default_prompt
            pipeline_options.picture_description_options = PictureDescriptionVlmOptions(
                repo_id="ibm-granite/granite-docling-258M",
                prompt=prompt,
            )
            pipeline_options.images_scale = settings.vlm_images_scale
        elif model_choice == ModelChoice.GRANITE_VISION_2B:
            pipeline_options.do_picture_description = True
            if custom_prompt:
                pipeline_options.picture_description_options = PictureDescriptionVlmOptions(
                    repo_id="ibm-granite/granite-vision-3.1-2b-preview",
                    prompt=custom_prompt,
                )
            else:
                pipeline_options.picture_description_options = granite_picture_description
            pipeline_options.images_scale = settings.vlm_images_scale
        elif model_choice == ModelChoice.GRANITE_VISION_8B:
            # For 8B model, use custom configuration
            pipeline_options.do_picture_description = True
            default_prompt = "Convert this image to markdown. Provide detailed and accurate descriptions. Only output the markdown content."
            prompt = custom_prompt if custom_prompt else default_prompt
            pipeline_options.picture_description_options = PictureDescriptionVlmOptions(
                repo_id="ibm-granite/granite-vision-3.2-8b",
                prompt=prompt,
            )
            pipeline_options.images_scale = settings.vlm_images_scale
        # Default model uses standard OCR without VLM
        
        format_options = {InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        return DocumentConverter(format_options=format_options)

    async def convert(
        self,
        source: ConversionSource,
        force_device: Optional[str] = None,
    ) -> tuple[str, str, dict[str, Any], ConversionMetadata]:
        """Convert document and return markdown, html, json, and metadata."""
        import time
        start_time = time.time()
        
        device = force_device or self._select_device()
        converter = await self._get_converter(device, source.model_choice, source.document_type, source.table_mode, source.custom_prompt)
        
        try:
            result = await asyncio.to_thread(converter.convert, source.value)
        except torch.cuda.OutOfMemoryError as exc:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if device != "cpu":
                return await self.convert(source, force_device="cpu")
            raise RuntimeError("Docling conversion failed due to GPU OOM") from exc
        except RuntimeError as exc:
            if device != "cpu":
                return await self.convert(source, force_device="cpu")
            raise RuntimeError(f"Docling conversion failed: {exc}") from exc
        
        if result.status != ConversionStatus.SUCCESS:
            raise RuntimeError(f"Docling conversion returned status {result.status}")
        if result.document is None:
            raise RuntimeError("Docling conversion produced no document")
        
        # Extract metadata from conversion result
        pages_processed = len(result.document.pages) if hasattr(result.document, 'pages') else 0
        images_extracted = len(result.document.pictures) if hasattr(result.document, 'pictures') else 0
        
        # Serialize to markdown and HTML
        serializer = MarkdownDocSerializer(
            doc=result.document,
            params=MarkdownParams(image_mode=ImageRefMode.EMBEDDED),
        )
        markdown = serializer.serialize().text

        html_serializer = HTMLDocSerializer(doc=result.document)
        html_output = html_serializer.serialize().text

        json_export = result.document.export_to_dict()

        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Create metadata
        model_name_map = {
            ModelChoice.DEFAULT: "Docling OCR (Default)",
            ModelChoice.SMOLVLM: "SmolVLM",
            ModelChoice.GRANITE_DOCLING: "Granite Docling 258M",
            ModelChoice.GRANITE_VISION_2B: "Granite Vision 3.2 2B",
            ModelChoice.GRANITE_VISION_8B: "Granite Vision 3.2 8B",
        }
        
        metadata = ConversionMetadata(
            model_used=model_name_map.get(source.model_choice, "Unknown"),
            processing_time_seconds=processing_time,
            pages_processed=pages_processed,
            images_extracted=images_extracted,
        )
        
        return markdown, html_output, json_export, metadata


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
            markdown, html_output, json_export, metadata = await self.converter.convert(source)
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

        html_path = self.output_dir / f"{timestamp}.html"
        await asyncio.to_thread(html_path.write_text, html_output, encoding="utf-8")

        json_path = self.output_dir / f"{timestamp}.json"
        await asyncio.to_thread(
            json_path.write_text,
            json.dumps(json_export, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        await self.enforce_output_limit(exclude={output_path, html_path, json_path})
        await self._update(
            task_id,
            status="completed",
            output_path=output_path,
            output_filename=output_filename,
            metadata=metadata,
            preview=PreviewArtifacts(html_path=html_path, json_path=json_path),
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
                metadata=state.metadata,
                preview=state.preview,
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

app = FastAPI(
    title="Docling Markdown Converter",
    description="Convert documents (PDF, images) to Markdown, HTML, and JSON using AI-powered VLM models including Granite Docling 258M.",
    version="1.0.0",
    contact={
        "name": "Docling Service",
    },
)
base_path = Path(__file__).parent
templates = Jinja2Templates(directory=str(base_path / "templates"))
app.mount(
    "/static",
    StaticFiles(directory=str(base_path / "static")),
    name="static",
)


@app.get(
    "/healthz",
    summary="Health Check",
    description="Returns the health status of the service.",
    response_description="Service health status",
    tags=["System"],
)
async def healthz() -> dict[str, str]:
    """Check if the service is running and healthy."""
    return {"status": "ok"}


@app.get(
    "/",
    response_class=HTMLResponse,
    summary="Web UI",
    description="Serves the web interface for document conversion.",
    tags=["UI"],
)
async def index(request: Request) -> HTMLResponse:
    """Render the main web UI for uploading and converting documents."""
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "max_upload_mb": settings.max_upload_mb,
        },
    )


class ConvertResponse(BaseModel):
    task_id: str


@app.post(
    "/api/convert",
    response_model=ConvertResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit Document Conversion",
    description="""Submit a document (PDF or image) for conversion to Markdown, HTML, and JSON.
    
    **Supported Models:**
    - `default`: Standard Docling OCR (best for text-heavy PDFs)
    - `granite_docling`: Granite Docling 258M (recommended for images and complex layouts)
    - `smolvlm`: SmolVLM 256M (optional, requires LOAD_OPTIONAL_MODELS=true)
    - `granite_vision_2b`: Granite Vision 2B (optional)
    - `granite_vision_8b`: Granite Vision 8B (optional)
    
    **Table Modes:**
    - `fast`: Faster table extraction with good accuracy
    - `accurate`: Slower but more accurate table structure recognition
    """,
    response_description="Task ID for tracking conversion progress",
    tags=["Conversion"],
)
async def submit_conversion(
    request: Request,
    file: UploadFile | None = File(None, description="Document file to convert (PDF or image)"),
    source_url: str | None = Form(None, description="URL of document to convert (alternative to file upload)"),
    document_type: str = Form("pdf", description="Type of document: 'pdf' or 'image'"),
    model: str = Form("default", description="AI model to use for conversion"),
    do_picture_description: bool = Form(True, description="Enable picture description generation"),
    images_scale: float = Form(2.0, description="Image quality scale (1.0-3.0)"),
    ocr_language: str = Form("en", description="OCR language code (en, es, fr, de)"),
    force_ocr: bool = Form(False, description="Force OCR even for text-based PDFs"),
    table_mode: str = Form("fast", description="Table extraction mode: 'fast' or 'accurate'"),
    custom_prompt: str = Form("", description="Custom prompt for VLM models (optional, uses default if empty)"),
) -> ConvertResponse:
    if file is None and not source_url:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Provide a file or source_url")
    
    # Validate document type
    try:
        doc_type = DocumentType(document_type)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid document_type: {document_type}")
    
    # Validate model choice
    try:
        model_choice = ModelChoice(model)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid model: {model}")
    
    task_id = uuid4().hex
    if file is not None:
        filename = file.filename or "document"
        suffix = Path(filename).suffix.lower()
        
        # Validate file type based on document type
        if doc_type == DocumentType.PDF:
            if suffix not in [".pdf", ""]:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="PDF document type requires .pdf file")
            suffix = suffix or ".pdf"
        elif doc_type == DocumentType.IMAGE:
            if suffix not in [".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"]:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Image document type requires image file (.png, .jpg, .jpeg, .webp, .tiff, .bmp)")
        
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
            document_type=doc_type,
            model_choice=model_choice,
            table_mode=table_mode,
            custom_prompt=custom_prompt.strip() if custom_prompt else "",
        )
    else:
        source = ConversionSource(
            kind="url",
            value=source_url.strip(),
            document_type=doc_type,
            model_choice=model_choice,
            table_mode=table_mode,
            custom_prompt=custom_prompt.strip() if custom_prompt else "",
        )
    await manager.enqueue(task_id, source)
    return ConvertResponse(task_id=task_id)


@app.get(
    "/api/status/{task_id}",
    response_model=TaskStatusResponse,
    summary="Get Conversion Status",
    description="Retrieve the current status and metadata of a conversion task.",
    response_description="Task status including progress, metadata, and download URL if completed",
    tags=["Conversion"],
)
async def get_status(request: Request, task_id: str) -> TaskStatusResponse:
    state = await manager.get(task_id)
    if state is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")
    download_url: Optional[str] = None
    if state.status == "completed" and state.output_path is not None:
        download_url = str(request.url_for("download_markdown", task_id=task_id))
    # Convert metadata if present
    metadata_response = None
    if state.metadata:
        metadata_response = ConversionMetadataResponse(
            model_used=state.metadata.model_used,
            processing_time_seconds=state.metadata.processing_time_seconds,
            pages_processed=state.metadata.pages_processed,
            images_extracted=state.metadata.images_extracted,
        )
    
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
        metadata=metadata_response,
    )


@app.get(
    "/api/result/{task_id}",
    response_class=FileResponse,
    name="download_markdown",
    summary="Download Markdown Result",
    description="Download the converted document in Markdown format.",
    response_description="Markdown file",
    tags=["Results"],
)
async def download_markdown(task_id: str) -> FileResponse:
    state = await manager.get(task_id)
    if state is None or state.output_path is None or state.status != "completed":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Result not available")
    filename = state.output_filename or state.source_name or f"{task_id}.md"
    return FileResponse(state.output_path, media_type="text/markdown", filename=filename)


@app.get(
    "/api/result/{task_id}/json",
    response_class=FileResponse,
    name="download_json",
    summary="Download JSON Result",
    description="Download the converted document in JSON format (DoclingDocument structure).",
    response_description="JSON file",
    tags=["Results"],
)
async def download_json(task_id: str) -> FileResponse:
    state = await manager.get(task_id)
    if (
        state is None
        or state.status != "completed"
        or state.preview is None
        or not state.preview.json_path.exists()
    ):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="JSON result not available")

    filename = state.preview.json_path.name
    return FileResponse(state.preview.json_path, media_type="application/json", filename=filename)


@app.get(
    "/api/preview/{task_id}",
    response_model=PreviewResponse,
    summary="Get Preview Content",
    description="Retrieve Markdown, HTML, and JSON preview content for a completed conversion.",
    response_description="Preview content in all three formats",
    tags=["Results"],
)
async def get_preview(task_id: str) -> PreviewResponse:
    state = await manager.get(task_id)
    if state is None or state.status != "completed" or state.output_path is None or state.preview is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Preview not available")

    if not state.preview.html_path.exists() or not state.preview.json_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Preview artifacts missing")

    markdown_text = await asyncio.to_thread(state.output_path.read_text, encoding="utf-8")
    html_text = await asyncio.to_thread(state.preview.html_path.read_text, encoding="utf-8")
    json_text = await asyncio.to_thread(state.preview.json_path.read_text, encoding="utf-8")
    try:
        json_payload = json.loads(json_text)
    except json.JSONDecodeError as exc:  # pragma: no cover - unexpected
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to load JSON preview") from exc

    return PreviewResponse(markdown=markdown_text, html=html_text, json_data=json_payload)
