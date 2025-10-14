# UI and Implementation Plan for VLM Integration

## Overview
- **Objective** Ensure the Docling service supports both PDF-to-Markdown and image-to-Markdown conversions using VLM/Granite models while maintaining the existing workflow.
- **Scope** Frontend/UI enhancements, backend API updates, model integration, testing, and deployment considerations.

## UI Changes
### Layout
- **Header** Keep current title and description, update subtitle to mention PDFs and images.
- **Main Panel** Split into input column and options column on desktop; stack vertically on mobile.

### Input Controls
- **Document Type Selector** Radio buttons or tabs for `PDF` (default) and `Image`.
- **File Upload** Dynamic accept attribute (`application/pdf` or image types). Label updates with document type.
- **Remote URL Input** Placeholder changes to match document type; supports PDF or image URLs.
- **File Limit Notice** Reflects size limits per document type.

### Options Panel
- **Model Picker** Dropdown: Default (Docling OCR), SmolVLM, Granite Vision 2B, Granite Vision 8B.
- **Picture Descriptions** Checkbox enabled automatically for VLM selections.
- **Image Scale** Slider (1.0–3.0) with default 2.0.
- **OCR Language** Dropdown (EN, ES, FR, DE, etc.).
- **Advanced Toggle** Reveals Force OCR checkbox and Table Mode dropdown (fast, accurate).

### Status & Results
- **Status Summary** Textual progress updates (Queued, Processing with elapsed seconds, Completed, Failed).
- **Detail Messages** Error/warning output area reused from current UI.
- **Results Card** Displays model used, processing time, pages/images processed.
- **Preview Tabs** Markdown viewer (monospace), HTML preview, JSON view.
- **Actions** Download Markdown (primary), Copy Markdown, Download JSON.
- **Image Captions** Thumbnail grid showing extracted images with VLM-generated descriptions when available.

### Styling
- **Color Accents** Introduce secondary accent for VLM features; keep dark theme.
- **Responsive Layout** Two-column desktop layout, stacked mobile view.
- **Components** Add styles for tabs, sliders, dropdowns, metadata cards, thumbnails.

## Backend/API Adjustments
- **Packages** Add `docling[vlm]` to `requirements.txt`; ensure Docker builds include model prerequisites.
- **Configuration** Extend `app/config.py` with model repository IDs, cache paths, and GPU toggles.
- **Enums** Introduce `DocumentType` (PDF, IMAGE) and `ModelChoice` enums.
- **Converter Setup** Update `ConverterManager._build_converter()` to configure VLM options based on request.
- **Request Handling** Accept new form fields: `document_type`, `model`, `do_picture_description`, `images_scale`, `ocr_language`, `force_ocr`, `table_mode`.
- **Metadata** Store model name, processing time, counts in `TaskState` and surface via `/api/status/{task_id}`.
- **Preview Endpoint** Add `/api/preview/{task_id}` returning Markdown, HTML, JSON, and image captions.

## Implementation Phases
### Phase 1: Backend Foundation (Week 1)
- **Dependencies** Update `requirements.txt`, Docker base image, and add model cache strategy.
- **API** Extend request validation, update status responses, and add preview endpoint.
- **Converter Logic** Integrate model selection and image pipeline handling.

### Phase 2: Frontend Enhancements (Week 2)
- **Template** Modify `templates/index.html` to include new inputs, options panel, preview tabs, and metadata card.
- **Styles** Update `static/styles.css` for new components and responsive layout.
- **JavaScript** Handle new form fields, preview tabs, clipboard actions, and metadata rendering.

### Phase 3: Integration & Testing (Week 3)
- **Model Download** Automate fetching SmolVLM and Granite weights; validate GPU/CPU paths.
- **Functional Testing** PDFs with default OCR, PDFs with VLM, standalone images, URL sources, error flows.
- **Performance** Benchmark processing times, add caching and rate limiting if needed.

### Phase 4: Deployment & Documentation (Week 4)
- **Docker Images** Build CPU/GPU variants, configure health checks, update `docker-compose.yml`.
- **Documentation** Refresh README, API reference, and add UI user guide with screenshots.
- **Monitoring** Ensure logging captures model selection, processing time, and errors.

## Deliverables
- **Frontend** Updated `templates/index.html`, `static/styles.css`, and client-side script.
- **Backend** Enhanced `app/main.py`, request/response models, and configuration.
- **Infrastructure** Model cache configuration and Docker image updates.
- **Docs** README, API docs, user guide, troubleshooting notes.

## Next Steps
- **Create Feature Branch** e.g., `feature/vlm-ui`.
- **Implement Phase 1** focusing on backend groundwork.
- **Schedule UI Design Review** once wireframes/early prototype are ready.
- **Plan QA Session** after integration testing to validate new workflows.
