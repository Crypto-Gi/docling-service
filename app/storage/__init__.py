"""Cloud storage backend abstraction layer."""

from __future__ import annotations

from .base import StorageBackend
from .cloudflare_r2 import CloudflareR2Storage
from .local import LocalStorage


def create_storage_backend(settings) -> StorageBackend:
    """Factory to create storage backend based on configuration.
    
    Args:
        settings: Application settings object with cloud storage configuration
        
    Returns:
        StorageBackend instance (LocalStorage, CloudflareR2Storage, etc.)
    """
    if not settings.cloud_storage_enabled:
        return LocalStorage({"base_path": settings.storage_path / "images"})
    
    provider = settings.cloud_storage_provider.lower()
    
    if provider == "cloudflare_r2":
        return CloudflareR2Storage({
            "enabled": True,
            "account_id": settings.r2_account_id,
            "access_key_id": settings.r2_access_key_id,
            "secret_access_key": settings.r2_secret_access_key,
            "bucket_name": settings.r2_bucket_name,
            "region": settings.r2_region,
            "public_url_base": settings.r2_public_url_base,
        })
    
    # Fallback to local storage for unknown providers
    return LocalStorage({"base_path": settings.storage_path / "images"})


__all__ = ["StorageBackend", "LocalStorage", "CloudflareR2Storage", "create_storage_backend"]
