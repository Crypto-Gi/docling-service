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
    
    # Cloud storage configuration
    cloud_storage_enabled: bool = os.getenv("DOCLING_CLOUD_STORAGE_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
    cloud_storage_provider: str = os.getenv("DOCLING_CLOUD_STORAGE_PROVIDER", "local")
    cloud_upload_mode: str = os.getenv("DOCLING_CLOUD_UPLOAD_MODE", "sync")  # sync or async
    cloud_keep_local_copy: bool = os.getenv("DOCLING_CLOUD_KEEP_LOCAL", "false").lower() in {"1", "true", "yes", "on"}
    
    # Cloudflare R2 specific settings
    r2_account_id: str | None = os.getenv("DOCLING_R2_ACCOUNT_ID")
    r2_access_key_id: str | None = os.getenv("DOCLING_R2_ACCESS_KEY_ID")
    r2_secret_access_key: str | None = os.getenv("DOCLING_R2_SECRET_ACCESS_KEY")
    r2_bucket_name: str | None = os.getenv("DOCLING_R2_BUCKET_NAME")
    r2_region: str = os.getenv("DOCLING_R2_REGION", "auto")
    r2_public_url_base: str | None = os.getenv("DOCLING_R2_PUBLIC_URL_BASE")

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
