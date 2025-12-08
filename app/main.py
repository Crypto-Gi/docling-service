from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
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


# Enhanced response models for image metadata
class PageDimensions(BaseModel):
    """Page dimensions in points (PDF units)."""
    width: float
    height: float


class ImagePosition(BaseModel):
    """Bounding box position of an image on the page."""
    x: float  # left coordinate
    y: float  # top coordinate  
    width: float
    height: float
    coord_origin: str = "TOPLEFT"  # or BOTTOMLEFT depending on source
    # Normalized coordinates (0-1 range, resolution-independent)
    x_norm: Optional[float] = None
    y_norm: Optional[float] = None
    width_norm: Optional[float] = None
    height_norm: Optional[float] = None


class ImageInfo(BaseModel):
    """Detailed information about an extracted image."""
    id: str  # e.g., "picture-1"
    url: str  # Cloud URL or local path
    page: int  # 1-indexed page number
    position: Optional[ImagePosition] = None
    page_dimensions: Optional[PageDimensions] = None  # Page size for coordinate mapping
    alt_text: Optional[str] = None  # Caption or label if available
    description: Optional[str] = None  # For VLM enrichment later
    mimetype: str = "image/png"
    size: Optional[dict] = None  # width/height in pixels of extracted image


class TableInfo(BaseModel):
    """Detailed information about an extracted table."""
    id: str  # e.g., "table-1"
    page: int  # 1-indexed page number
    position: Optional[ImagePosition] = None
    page_dimensions: Optional[PageDimensions] = None
    num_rows: int = 0
    num_cols: int = 0
    caption: Optional[str] = None


class DocumentMetadata(BaseModel):
    """Document-level metadata."""
    total_pages: int
    total_images: int
    total_tables: int
    source_filename: Optional[str] = None
    source_type: str  # "upload" or "url"
    processing_time_ms: Optional[int] = None
    page_dimensions: Optional[list[PageDimensions]] = None  # Dimensions per page


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
    # Enhanced metadata
    images_metadata: Optional[list] = None  # List of ImageInfo dicts
    tables_metadata: Optional[list] = None  # List of TableInfo dicts
    document_metadata: Optional[dict] = None  # DocumentMetadata dict
    page_dimensions: Optional[list] = None  # List of {width, height} per page


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
        start_time = datetime.utcnow()
        try:
            result = await self.converter.convert(source.value)
            image_map, images_metadata = await self._save_images(result, task_id, task_images_dir, storage_backend)
            await self._update_image_uris(result, image_map)

            serializer = MarkdownDocSerializer(
                doc=result.document,
                params=MarkdownParams(
                    image_mode=ImageRefMode.REFERENCED,
                ),
            )
            markdown = serializer.serialize().text
            
            # Extract table metadata with positions
            tables_metadata = self._extract_tables(result)
            
            # Extract page dimensions
            page_dims = self._get_page_dimensions(result)
            page_dimensions_list = [
                {"page": pg, "width": dims["width"], "height": dims["height"]}
                for pg, dims in sorted(page_dims.items())
            ]
            
            # Extract document metadata
            total_pages = len(result.document.pages) if hasattr(result.document, 'pages') else 0
            processing_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            document_metadata = {
                "total_pages": total_pages,
                "total_images": len(images_metadata),
                "total_tables": len(tables_metadata),
                "source_filename": source.original_name,
                "source_type": source.kind,
                "processing_time_ms": processing_time_ms,
                "page_dimensions": page_dimensions_list,
            }
            
            # Inject page markers into markdown
            markdown = self._inject_page_markers(result, markdown)
            
            # Append metadata block to markdown
            metadata_block = self._create_metadata_block(
                images_metadata, tables_metadata, page_dimensions_list, document_metadata
            )
            markdown = markdown + metadata_block
            
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
            images_metadata=images_metadata,
            tables_metadata=tables_metadata,
            document_metadata=document_metadata,
            page_dimensions=page_dimensions_list,
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

    def _get_page_dimensions(self, result: ConversionResult) -> dict[int, dict]:
        """Extract page dimensions from the document.
        
        Returns:
            Dict mapping page_no (1-indexed) to {width, height}
        """
        page_dims: dict[int, dict] = {}
        if hasattr(result.document, 'pages') and result.document.pages:
            for page_no, page in result.document.pages.items():
                if hasattr(page, 'size') and page.size:
                    page_dims[page_no] = {
                        "width": float(page.size.width) if hasattr(page.size, 'width') else 0,
                        "height": float(page.size.height) if hasattr(page.size, 'height') else 0,
                    }
        return page_dims

    def _inject_page_markers(self, result: ConversionResult, markdown: str) -> str:
        """Inject page number markers into the markdown content.
        
        Inserts markers like '<!-- Page N -->' at appropriate positions based on
        where document elements appear on each page.
        """
        # Build a map of which elements appear on which page
        page_elements: dict[int, list[tuple[str, int]]] = {}  # page -> [(text_snippet, char_position)]
        
        for element, _level in result.document.iterate_items():
            if hasattr(element, 'prov') and element.prov and len(element.prov) > 0:
                prov = element.prov[0]
                page_no = prov.page_no if hasattr(prov, 'page_no') else (prov.page if hasattr(prov, 'page') else None)
                
                if page_no is not None:
                    # Get a text snippet to locate in markdown
                    text_snippet = None
                    if hasattr(element, 'text') and element.text:
                        # Take first 50 chars of text for matching
                        text_snippet = element.text[:50].strip()
                    
                    if text_snippet and len(text_snippet) > 10:
                        pos = markdown.find(text_snippet)
                        if pos >= 0:
                            if page_no not in page_elements:
                                page_elements[page_no] = []
                            page_elements[page_no].append((text_snippet, pos))
        
        if not page_elements:
            return markdown
        
        # Find the first occurrence position for each page
        page_positions: list[tuple[int, int]] = []  # (position, page_no)
        for page_no, elements in page_elements.items():
            if elements:
                # Use the earliest position for this page
                min_pos = min(pos for _, pos in elements)
                page_positions.append((min_pos, page_no))
        
        # Sort by position
        page_positions.sort(key=lambda x: x[0])
        
        # Insert page markers from end to beginning (to preserve positions)
        result_md = markdown
        inserted_pages = set()
        
        for pos, page_no in reversed(page_positions):
            if page_no not in inserted_pages:
                # Find the start of the line
                line_start = result_md.rfind('\n', 0, pos)
                if line_start == -1:
                    line_start = 0
                else:
                    line_start += 1
                
                marker = f"\n\n<!-- Page {page_no} -->\n\n"
                result_md = result_md[:line_start] + marker + result_md[line_start:]
                inserted_pages.add(page_no)
        
        # Ensure Page 1 marker exists at the beginning if not already
        if 1 not in inserted_pages:
            result_md = "<!-- Page 1 -->\n\n" + result_md
        
        return result_md

    def _create_metadata_block(
        self,
        images_metadata: list[dict],
        tables_metadata: list[dict],
        page_dimensions_list: list[dict],
        document_metadata: dict,
    ) -> str:
        """Create a fenced JSON metadata block to append to markdown.
        
        Returns a string like:
        ---
        <!-- DOCLING_METADATA_START -->
        ```json
        {...}
        ```
        <!-- DOCLING_METADATA_END -->
        """
        metadata_obj = {
            "images": [
                {
                    "id": img["id"],
                    "page": img["page"],
                    "position": img.get("position"),
                    "page_dimensions": img.get("page_dimensions"),
                }
                for img in images_metadata
            ],
            "tables": [
                {
                    "id": tbl["id"],
                    "page": tbl["page"],
                    "position": tbl.get("position"),
                    "page_dimensions": tbl.get("page_dimensions"),
                    "num_rows": tbl.get("num_rows", 0),
                    "num_cols": tbl.get("num_cols", 0),
                }
                for tbl in tables_metadata
            ],
            "pages": [
                {"page": i + 1, "width": pd["width"], "height": pd["height"]}
                for i, pd in enumerate(page_dimensions_list)
            ],
            "metadata": {
                "total_pages": document_metadata.get("total_pages", 0),
                "total_images": document_metadata.get("total_images", 0),
                "total_tables": document_metadata.get("total_tables", 0),
                "source_filename": document_metadata.get("source_filename"),
                "processing_time_ms": document_metadata.get("processing_time_ms"),
            }
        }
        
        json_str = json.dumps(metadata_obj, indent=2)
        
        return f"""

---

<!-- DOCLING_METADATA_START -->
```json
{json_str}
```
<!-- DOCLING_METADATA_END -->
"""

    async def _save_images(
        self, result: ConversionResult, task_id: str, images_dir: Path, storage_backend
    ) -> tuple[dict[str, str], list[dict]]:
        """Save picture images to disk and upload to cloud storage (sync mode).
        
        Tables are exported as native Markdown tables by the serializer, not as images.
        
        Returns:
            Tuple of:
            - Dictionary mapping element.self_ref to final URL/path (cloud URL or local path)
            - List of ImageInfo dicts with page numbers, positions, and page dimensions
        """
        import logging
        _log = logging.getLogger(__name__)
        
        picture_counter = 0
        image_map: dict[str, str] = {}
        images_metadata: list[dict] = []
        
        # Get page dimensions for normalized coordinates
        page_dims = self._get_page_dimensions(result)
        
        for element, _level in result.document.iterate_items():
            if isinstance(element, PictureItem):
                picture_counter += 1
                image_id = f"picture-{picture_counter}"
                filename = f"{image_id}.png"
                local_path = images_dir / filename
                cloud_key = f"images/{task_id}/{filename}"
                
                # Extract page number and position from provenance
                page_number = 1  # Default to page 1
                position_data = None
                page_dimensions = None
                alt_text = None
                
                # Get provenance data (page and bbox)
                if element.prov and len(element.prov) > 0:
                    prov = element.prov[0]  # Use first provenance item
                    page_number = prov.page_no if hasattr(prov, 'page_no') else (prov.page if hasattr(prov, 'page') else 1)
                    
                    # Get page dimensions for this page
                    page_dimensions = page_dims.get(page_number)
                    
                    # Extract bounding box
                    if hasattr(prov, 'bbox') and prov.bbox:
                        bbox = prov.bbox
                        # bbox has l (left), t (top), r (right), b (bottom)
                        coord_origin = "BOTTOMLEFT"  # PDF default
                        if hasattr(bbox, 'coord_origin') and bbox.coord_origin:
                            coord_origin = str(bbox.coord_origin.name) if hasattr(bbox.coord_origin, 'name') else str(bbox.coord_origin)
                        
                        x = float(bbox.l) if hasattr(bbox, 'l') else 0
                        y = float(bbox.t) if hasattr(bbox, 't') else 0
                        w = abs(float(bbox.r - bbox.l)) if hasattr(bbox, 'r') and hasattr(bbox, 'l') else 0
                        h = abs(float(bbox.b - bbox.t)) if hasattr(bbox, 'b') and hasattr(bbox, 't') else 0
                        
                        position_data = {
                            "x": x,
                            "y": y,
                            "width": w,
                            "height": h,
                            "coord_origin": coord_origin,
                        }
                        
                        # Calculate normalized coordinates if page dimensions available
                        if page_dimensions and page_dimensions.get("width", 0) > 0 and page_dimensions.get("height", 0) > 0:
                            pw, ph = page_dimensions["width"], page_dimensions["height"]
                            position_data["x_norm"] = round(x / pw, 4)
                            position_data["y_norm"] = round(y / ph, 4)
                            position_data["width_norm"] = round(w / pw, 4)
                            position_data["height_norm"] = round(h / ph, 4)
                
                # Get caption/alt text if available
                if hasattr(element, 'caption') and element.caption:
                    alt_text = element.caption.text if hasattr(element.caption, 'text') else str(element.caption)
                
                # Save locally first
                img = element.get_image(result.document)
                url = str(Path("images") / task_id / filename)  # Default local path
                image_size = None
                
                if img:
                    await asyncio.to_thread(img.save, local_path, "PNG")
                    image_size = {"width": img.width, "height": img.height}
                    
                    # Upload to cloud storage if enabled (synchronous mode)
                    try:
                        cloud_url = await storage_backend.upload(local_path, cloud_key)
                        image_map[element.self_ref] = cloud_url
                        url = cloud_url
                        
                        # Optionally delete local copy after successful upload
                        if not settings.cloud_keep_local_copy and storage_backend.is_cloud_enabled():
                            if storage_backend.__class__.__name__ != "LocalStorage":
                                local_path.unlink()
                    
                    except Exception as e:
                        # Fallback to local path on upload failure
                        _log.error(f"Cloud upload failed for {filename}, using local path: {e}")
                        image_map[element.self_ref] = url
                else:
                    image_map[element.self_ref] = url
                
                # Build ImageInfo dict
                image_info = {
                    "id": image_id,
                    "url": url,
                    "page": page_number,
                    "position": position_data,
                    "page_dimensions": page_dimensions,
                    "alt_text": alt_text,
                    "description": None,  # For VLM enrichment later
                    "mimetype": "image/png",
                    "size": image_size,
                }
                images_metadata.append(image_info)
        
        return image_map, images_metadata

    def _extract_tables(self, result: ConversionResult) -> list[dict]:
        """Extract table metadata including positions from the document.
        
        Returns:
            List of TableInfo dicts with page numbers and positions
        """
        tables_metadata: list[dict] = []
        table_counter = 0
        
        # Get page dimensions for normalized coordinates
        page_dims = self._get_page_dimensions(result)
        
        for element, _level in result.document.iterate_items():
            if isinstance(element, TableItem):
                table_counter += 1
                table_id = f"table-{table_counter}"
                
                # Extract page number and position from provenance
                page_number = 1
                position_data = None
                page_dimensions = None
                caption = None
                
                # Get provenance data (page and bbox)
                if element.prov and len(element.prov) > 0:
                    prov = element.prov[0]
                    page_number = prov.page_no if hasattr(prov, 'page_no') else (prov.page if hasattr(prov, 'page') else 1)
                    
                    # Get page dimensions for this page
                    page_dimensions = page_dims.get(page_number)
                    
                    # Extract bounding box
                    if hasattr(prov, 'bbox') and prov.bbox:
                        bbox = prov.bbox
                        coord_origin = "BOTTOMLEFT"
                        if hasattr(bbox, 'coord_origin') and bbox.coord_origin:
                            coord_origin = str(bbox.coord_origin.name) if hasattr(bbox.coord_origin, 'name') else str(bbox.coord_origin)
                        
                        x = float(bbox.l) if hasattr(bbox, 'l') else 0
                        y = float(bbox.t) if hasattr(bbox, 't') else 0
                        w = abs(float(bbox.r - bbox.l)) if hasattr(bbox, 'r') and hasattr(bbox, 'l') else 0
                        h = abs(float(bbox.b - bbox.t)) if hasattr(bbox, 'b') and hasattr(bbox, 't') else 0
                        
                        position_data = {
                            "x": x,
                            "y": y,
                            "width": w,
                            "height": h,
                            "coord_origin": coord_origin,
                        }
                        
                        # Calculate normalized coordinates
                        if page_dimensions and page_dimensions.get("width", 0) > 0 and page_dimensions.get("height", 0) > 0:
                            pw, ph = page_dimensions["width"], page_dimensions["height"]
                            position_data["x_norm"] = round(x / pw, 4)
                            position_data["y_norm"] = round(y / ph, 4)
                            position_data["width_norm"] = round(w / pw, 4)
                            position_data["height_norm"] = round(h / ph, 4)
                
                # Get caption if available
                if hasattr(element, 'caption') and element.caption:
                    caption = element.caption.text if hasattr(element.caption, 'text') else str(element.caption)
                
                # Get table dimensions
                num_rows = 0
                num_cols = 0
                if hasattr(element, 'data') and element.data:
                    if hasattr(element.data, 'num_rows'):
                        num_rows = element.data.num_rows
                    if hasattr(element.data, 'num_cols'):
                        num_cols = element.data.num_cols
                
                table_info = {
                    "id": table_id,
                    "page": page_number,
                    "position": position_data,
                    "page_dimensions": page_dimensions,
                    "num_rows": num_rows,
                    "num_cols": num_cols,
                    "caption": caption,
                }
                tables_metadata.append(table_info)
        
        return tables_metadata

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
                cloud_storage_config=state.cloud_storage_config,
                images_metadata=state.images_metadata,
                tables_metadata=state.tables_metadata,
                document_metadata=state.document_metadata,
                page_dimensions=state.page_dimensions,
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


class EnhancedResultResponse(BaseModel):
    """Enhanced JSON response with markdown, images, tables metadata, and document info.
    
    Includes:
    - Full markdown content
    - Images list with page numbers, bounding box positions, and normalized coordinates
    - Tables list with page numbers, positions, and dimensions
    - Document metadata (pages, page dimensions, processing time)
    """
    task_id: str
    markdown_content: str
    images: list[ImageInfo]
    tables: list[TableInfo]
    metadata: DocumentMetadata
    source_name: Optional[str] = None
    output_filename: str
    markdown_url: Optional[str] = None
    created_at: datetime
    completed_at: datetime


@app.get("/api/result/{task_id}/enhanced", response_model=EnhancedResultResponse)
async def get_enhanced_result(
    task_id: str,
    include_normalized: bool = True,
) -> EnhancedResultResponse:
    """Get full document result with markdown, images/tables with positions, and metadata.
    
    Args:
        task_id: The conversion task ID
        include_normalized: If True (default), include normalized coordinates (0-1 range)
    
    Returns structured JSON ideal for:
    - AI agents that need image/table locations for VLM processing
    - Document analysis workflows requiring precise element positioning
    - Integration with downstream systems
    
    Example response:
    ```json
    {
      "markdown_content": "# Document Title...",
      "images": [
        {
          "id": "picture-1",
          "url": "https://r2.../images/task123/picture-1.png",
          "page": 3,
          "position": {
            "x": 100, "y": 200, "width": 400, "height": 300,
            "coord_origin": "BOTTOMLEFT",
            "x_norm": 0.1667, "y_norm": 0.25, "width_norm": 0.6667, "height_norm": 0.375
          },
          "page_dimensions": {"width": 612, "height": 792},
          "alt_text": "Figure 1: Network Diagram"
        }
      ],
      "tables": [
        {
          "id": "table-1",
          "page": 2,
          "position": {"x": 50, "y": 100, ...},
          "num_rows": 5,
          "num_cols": 3
        }
      ],
      "metadata": {
        "total_pages": 12,
        "total_images": 5,
        "total_tables": 2,
        "processing_time_ms": 3500,
        "page_dimensions": [{"width": 612, "height": 792}, ...]
      }
    }
    ```
    """
    state = await manager.get(task_id)
    if state is None or state.output_path is None or state.status != "completed":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Result not available")
    
    # Read markdown content
    markdown_content = await asyncio.to_thread(state.output_path.read_text, encoding="utf-8")
    
    # Convert stored image dicts to Pydantic models
    images_list = []
    if state.images_metadata:
        for img_dict in state.images_metadata:
            position = None
            if img_dict.get("position"):
                pos_data = img_dict["position"].copy()
                # Optionally strip normalized coordinates
                if not include_normalized:
                    pos_data.pop("x_norm", None)
                    pos_data.pop("y_norm", None)
                    pos_data.pop("width_norm", None)
                    pos_data.pop("height_norm", None)
                position = ImagePosition(**pos_data)
            
            page_dims = None
            if img_dict.get("page_dimensions"):
                page_dims = PageDimensions(**img_dict["page_dimensions"])
            
            images_list.append(ImageInfo(
                id=img_dict["id"],
                url=img_dict["url"],
                page=img_dict["page"],
                position=position,
                page_dimensions=page_dims,
                alt_text=img_dict.get("alt_text"),
                description=img_dict.get("description"),
                mimetype=img_dict.get("mimetype", "image/png"),
                size=img_dict.get("size"),
            ))
    
    # Convert stored table dicts to Pydantic models
    tables_list = []
    if state.tables_metadata:
        for tbl_dict in state.tables_metadata:
            position = None
            if tbl_dict.get("position"):
                pos_data = tbl_dict["position"].copy()
                if not include_normalized:
                    pos_data.pop("x_norm", None)
                    pos_data.pop("y_norm", None)
                    pos_data.pop("width_norm", None)
                    pos_data.pop("height_norm", None)
                position = ImagePosition(**pos_data)
            
            page_dims = None
            if tbl_dict.get("page_dimensions"):
                page_dims = PageDimensions(**tbl_dict["page_dimensions"])
            
            tables_list.append(TableInfo(
                id=tbl_dict["id"],
                page=tbl_dict["page"],
                position=position,
                page_dimensions=page_dims,
                num_rows=tbl_dict.get("num_rows", 0),
                num_cols=tbl_dict.get("num_cols", 0),
                caption=tbl_dict.get("caption"),
            ))
    
    # Build document metadata with page dimensions
    doc_meta = state.document_metadata or {}
    page_dims_list = None
    if doc_meta.get("page_dimensions"):
        page_dims_list = [
            PageDimensions(width=pd["width"], height=pd["height"])
            for pd in doc_meta["page_dimensions"]
        ]
    
    metadata = DocumentMetadata(
        total_pages=doc_meta.get("total_pages", 0),
        total_images=doc_meta.get("total_images", len(images_list)),
        total_tables=doc_meta.get("total_tables", len(tables_list)),
        source_filename=doc_meta.get("source_filename") or state.source_name,
        source_type=doc_meta.get("source_type", state.source_kind),
        processing_time_ms=doc_meta.get("processing_time_ms"),
        page_dimensions=page_dims_list,
    )
    
    return EnhancedResultResponse(
        task_id=task_id,
        markdown_content=markdown_content,
        images=images_list,
        tables=tables_list,
        metadata=metadata,
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
