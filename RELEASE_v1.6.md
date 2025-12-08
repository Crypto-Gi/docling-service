# Docling Service v1.6 – Enhanced AI Metadata & Page Positioning

**Release Date:** December 8, 2025  
**Version:** 1.6.0  
**Status:** ✅ Production Ready

---

## 🎉 What’s New in v1.6

### Major Features

1. **Enhanced Result API (`/api/result/{task_id}/enhanced`)**
   - Returns full markdown content **and** rich structured metadata
   - Includes **images** and **tables** with page numbers and positions
   - Adds **absolute** (PDF points) and **normalized** (0–1) coordinates

2. **Page-Level Metadata & Markers**
   - Per-page width/height extracted from the PDF
   - **Page markers** inserted into markdown: `<!-- Page N -->`
   - Fenced JSON metadata block appended to every `.md` file

3. **LLM-Friendly Document Structure**
   - Single JSON response containing:
     - `markdown_content` (full .md as string)
     - `images[]` with coordinates, page, and dimensions
     - `tables[]` with coordinates and shape (rows/cols)
     - `metadata` with total pages/images/tables and page dimensions

---

## 📋 Table of Contents

- [Overview](#overview)
- [Enhanced API Reference](#enhanced-api-reference)
- [Markdown Output Format](#markdown-output-format)
- [Usage Examples](#usage-examples)
- [Migration Guide](#migration-guide)
- [Changelog](#changelog)

---

## Overview

### Key Capabilities (v1.6)

✅ **Page-aware markdown** with `<!-- Page N -->` markers  
✅ **Image and table positions** (absolute + normalized coordinates)  
✅ **Per-page dimensions** (width/height in PDF points)  
✅ **Fenced JSON metadata block** at end of every `.md` file  
✅ **LLM-ready JSON response** via `/api/result/{task_id}/enhanced`  

### Use Cases

1. **Vision-Language Models (VLMs)** – Feed page-localized image regions to an LLM.  
2. **Document UIs** – Overlay bounding boxes on rendered pages.  
3. **Layout-Aware QA** – Reason about which page/region a table or figure comes from.  
4. **Downstream pipelines** – Store structured metadata alongside markdown for indexing.

---

## Enhanced API Reference

### Endpoint: `GET /api/result/{task_id}/enhanced`

Returns the full document result with:
- `markdown_content`: Markdown string (including page markers)
- `images[]`: Image metadata with locations and sizes
- `tables[]`: Table metadata with locations and sizes
- `metadata`: Document-level info and page dimensions

#### Query Parameters

| Parameter           | Type    | Default | Description                                       |
|--------------------|---------|---------|---------------------------------------------------|
| `include_normalized` | bool  | `true`  | Include normalized coordinates (0–1 range) in `position` |

#### Sample Response

```json
{
  "task_id": "abc123",
  "markdown_content": "<!-- Page 1 -->\n## Document Title...",
  "images": [
    {
      "id": "picture-1",
      "url": "https://cdn.example.com/images/abc123/picture-1.png",
      "page": 1,
      "position": {
        "x": 88.04,
        "y": 553.29,
        "width": 434.83,
        "height": 155.54,
        "coord_origin": "BOTTOMLEFT",
        "x_norm": 0.1439,
        "y_norm": 0.6986,
        "width_norm": 0.7105,
        "height_norm": 0.1964
      },
      "page_dimensions": {"width": 612, "height": 792},
      "mimetype": "image/png",
      "size": {"width": 435, "height": 155}
    }
  ],
  "tables": [
    {
      "id": "table-1",
      "page": 1,
      "position": {
        "x": 88.04,
        "y": 553.29,
        "width": 434.83,
        "height": 155.54,
        "coord_origin": "BOTTOMLEFT",
        "x_norm": 0.1439,
        "y_norm": 0.6986,
        "width_norm": 0.7105,
        "height_norm": 0.1964
      },
      "page_dimensions": {"width": 612, "height": 792},
      "num_rows": 8,
      "num_cols": 5,
      "caption": null
    }
  ],
  "metadata": {
    "total_pages": 3,
    "total_images": 2,
    "total_tables": 3,
    "source_filename": "test_document.pdf",
    "source_type": "upload",
    "processing_time_ms": 21567,
    "page_dimensions": [
      {"width": 612, "height": 792},
      {"width": 612, "height": 792},
      {"width": 612, "height": 792}
    ]
  }
}
```

---

## Markdown Output Format

Every generated `.md` file (local or cloud) now includes:

1. **Page markers** – indicate where each PDF page begins:

```markdown
<!-- Page 1 -->
## Docling Service Test Document
...

<!-- Page 2 -->
- High Performance:
...

<!-- Page 3 -->
## 5. Employee Performance Matrix
...
```

2. **Fenced JSON metadata block** at the end of the file:

```markdown
---

<!-- DOCLING_METADATA_START -->
```json
{
  "images": [
    {
      "id": "picture-1",
      "page": 1,
      "position": {
        "x": 88.04,
        "y": 553.29,
        "width": 434.83,
        "height": 155.54,
        "coord_origin": "BOTTOMLEFT",
        "x_norm": 0.1439,
        "y_norm": 0.6986,
        "width_norm": 0.7105,
        "height_norm": 0.1964
      },
      "page_dimensions": {"width": 612.0, "height": 792.0}
    }
  ],
  "tables": [
    {
      "id": "table-1",
      "page": 1,
      "position": {"x": 88.04, "y": 553.29, "...": "..."},
      "page_dimensions": {"width": 612.0, "height": 792.0},
      "num_rows": 8,
      "num_cols": 5
    }
  ],
  "pages": [
    {"page": 1, "width": 612.0, "height": 792.0},
    {"page": 2, "width": 612.0, "height": 792.0},
    {"page": 3, "width": 612.0, "height": 792.0}
  ],
  "metadata": {
    "total_pages": 3,
    "total_images": 2,
    "total_tables": 3,
    "source_filename": "test_document.pdf",
    "processing_time_ms": 21567
  }
}
```
<!-- DOCLING_METADATA_END -->
```

This makes the `.md` file **self-contained**: human-readable markdown plus machine-readable metadata.

---

## Coordinate Systems

Two coordinate systems are exposed for each image/table:

| Type        | Fields                               | Description                                    |
|-------------|--------------------------------------|------------------------------------------------|
| Absolute    | `x`, `y`, `width`, `height`         | PDF points (1 pt = 1/72 inch), `BOTTOMLEFT` origin |
| Normalized  | `x_norm`, `y_norm`, `width_norm`, `height_norm` | 0–1 range, resolution-independent      |

This allows easy mapping to any rendered resolution:

```text
pixel_x      = x_norm      * rendered_width
pixel_y      = y_norm      * rendered_height
pixel_width  = width_norm  * rendered_width
pixel_height = height_norm * rendered_height
```

---

## Usage Examples

### Get Enhanced Result (with normalized coordinates)

```bash
curl "http://localhost:5010/api/result/abc123/enhanced"
```

### Get Enhanced Result Without Normalized Coordinates

```bash
curl "http://localhost:5010/api/result/abc123/enhanced?include_normalized=false"
```

### Typical LLM Workflow

1. Call `/api/result/{task_id}/enhanced`.
2. Use `markdown_content` as the main context for the LLM.
3. Use `images` / `tables` arrays to:
   - Crop regions from rendered pages.
   - Provide precise coordinates to a VLM.
   - Answer questions like “what does the chart on page 3 show?”.
4. Optionally parse the metadata block from the `.md` file when working with local copies only.

---

## Migration Guide

### From v1.5 to v1.6

#### Breaking Changes

**None.** v1.6 is fully backward compatible:
- Existing endpoints (`/api/convert`, `/api/status`, `/api/result/{task_id}`, `/api/result/{task_id}/json`) are unchanged.
- Enhanced behavior is opt-in via the new `/api/result/{task_id}/enhanced` endpoint.

#### New Behavior

- Markdown files saved to `storage/docling/outputs/` now include:
  - `<!-- Page N -->` markers.
  - The `DOCLING_METADATA_START/END` fenced JSON block.
- Task state now stores additional `images_metadata`, `tables_metadata`, and `page_dimensions` fields.

#### Migration Steps

1. **Update code / container** to v1.6:
   ```bash
   git pull
   docker compose build
   docker compose up -d
   ```

2. **Update clients** (optional) to use the new endpoint:
   - For AI/LLM workflows, prefer `/api/result/{task_id}/enhanced`.

3. **No configuration changes required** for storage or credentials.

#### Rollback Plan

If you need to roll back:

```bash
git checkout v1.5
docker compose build
docker compose up -d
```

---

## Changelog

### v1.6.0 (2025-12-08)

**Added:**
- `/api/result/{task_id}/enhanced` endpoint.
- Image and table metadata with page numbers, bounding boxes, and normalized coordinates.
- Per-page dimensions (`page_dimensions`) at document and element level.
- Markdown page markers: `<!-- Page N -->`.
- Fenced JSON metadata block appended to every `.md` output.

**Changed:**
- `TaskState` extended to include `images_metadata`, `tables_metadata`, and `page_dimensions`.
- Enhanced README documentation for AI/LLM-focused usage.

**Fixed:**
- Ensured width/height values are always positive, regardless of PDF coordinate orientation.

---

## Support

### Documentation

- Main README: `README.md`
- v1.6 Release Notes (this file): `RELEASE_v1.6.md`
- v1.5 Release Notes: `RELEASE_v1.5.md`

### Reporting Issues

Please include:
1. Docker logs
2. Request parameters used
3. Example input document (if possible)
4. Expected vs actual behavior
5. Environment details (CPU/GPU, Docker version)

---

**🚀 v1.6 is ready for production use in AI and document understanding pipelines.**
