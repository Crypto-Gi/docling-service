from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    """Runtime configuration for the Docling service."""

    prefer_gpu: bool = os.getenv("DOCLING_GPU_PREFERRED", "true").lower() in {"1", "true", "yes", "on"}
    storage_path: Path = Path(os.getenv("DOCLING_STORAGE_PATH", "/data"))
    max_upload_mb: int = int(os.getenv("DOCLING_MAX_UPLOAD_MB", "25"))
    storage_soft_limit_mb: int = int(os.getenv("DOCLING_STORAGE_SOFT_LIMIT_MB", "200"))
    storage_hard_limit_mb: int = int(os.getenv("DOCLING_STORAGE_HARD_LIMIT_MB", "250"))

    def __post_init__(self) -> None:
        self.storage_path.mkdir(parents=True, exist_ok=True)
        if self.storage_soft_limit_mb <= 0:
            self.storage_soft_limit_mb = self.storage_hard_limit_mb
        if self.storage_hard_limit_mb < self.storage_soft_limit_mb:
            self.storage_hard_limit_mb = self.storage_soft_limit_mb
        if self.max_upload_mb <= 0:
            self.max_upload_mb = 1

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def storage_soft_limit_bytes(self) -> int:
        return self.storage_soft_limit_mb * 1024 * 1024

    @property
    def storage_hard_limit_bytes(self) -> int:
        return self.storage_hard_limit_mb * 1024 * 1024


settings = Settings()
