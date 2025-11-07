"""Local filesystem storage backend."""

from __future__ import annotations

from pathlib import Path

from .base import StorageBackend


class LocalStorage(StorageBackend):
    """Local filesystem storage backend.
    
    Stores files on local disk and returns relative paths.
    This is the default fallback when cloud storage is disabled.
    """
    
    def __init__(self, config: dict):
        """Initialize local storage backend.
        
        Args:
            config: Configuration dict with 'base_path' key
        """
        self.base_path = Path(config.get("base_path", "./storage/docling/images"))
        self.enabled = True
    
    async def upload(self, local_path: Path, remote_key: str) -> str:
        """Return relative path (no actual upload needed for local storage).
        
        Args:
            local_path: Path to local file (already saved)
            remote_key: Remote key (e.g., "task_id/picture-1.png")
            
        Returns:
            Relative path for Markdown references
        """
        # For local storage, file is already saved, just return relative path
        return str(Path("images") / remote_key)
    
    async def delete(self, remote_key: str) -> bool:
        """Delete local file.
        
        Args:
            remote_key: Remote key (e.g., "task_id/picture-1.png")
            
        Returns:
            True if file was deleted, False otherwise
        """
        file_path = self.base_path / remote_key
        if file_path.exists():
            try:
                file_path.unlink()
                return True
            except OSError:
                return False
        return False
    
    def get_url(self, remote_key: str) -> str:
        """Get relative path for local file.
        
        Args:
            remote_key: Remote key (e.g., "task_id/picture-1.png")
            
        Returns:
            Relative path for Markdown references
        """
        return str(Path("images") / remote_key)
    
    def is_enabled(self) -> bool:
        """Check if local storage is enabled.
        
        Returns:
            Always True for local storage
        """
        return True
    
    def is_cloud_enabled(self) -> bool:
        """Check if this is a cloud storage backend.
        
        Returns:
            False - this is local storage, not cloud
        """
        return False
