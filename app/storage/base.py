"""Abstract base class for storage backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class StorageBackend(ABC):
    """Abstract base class for storage backends.
    
    All storage providers (local, Cloudflare R2, AWS S3, etc.) must implement this interface.
    """
    
    @abstractmethod
    def __init__(self, config: dict):
        """Initialize storage backend with provider-specific configuration.
        
        Args:
            config: Dictionary containing provider-specific settings
        """
        pass
    
    @abstractmethod
    async def upload(self, local_path: Path, remote_key: str) -> str:
        """Upload file to storage and return public URL.
        
        Args:
            local_path: Path to local file to upload
            remote_key: Remote object key/path (e.g., "task_id/picture-1.png")
            
        Returns:
            Public URL or relative path to access the uploaded file
            
        Raises:
            RuntimeError: If upload fails
        """
        pass
    
    @abstractmethod
    async def delete(self, remote_key: str) -> bool:
        """Delete file from storage.
        
        Args:
            remote_key: Remote object key/path to delete
            
        Returns:
            True if deletion succeeded, False otherwise
        """
        pass
    
    @abstractmethod
    def get_url(self, remote_key: str) -> str:
        """Get public URL for a remote key.
        
        This method can be called before upload for predictable URL generation.
        
        Args:
            remote_key: Remote object key/path
            
        Returns:
            Public URL or relative path to access the file
        """
        pass
    
    @abstractmethod
    def is_enabled(self) -> bool:
        """Check if backend is enabled and properly configured.
        
        Returns:
            True if backend is ready to use, False otherwise
        """
        pass
    
    @abstractmethod
    def is_cloud_enabled(self) -> bool:
        """Check if this is a cloud storage backend (not local).
        
        Returns:
            True if this is a cloud backend (R2, S3, etc.), False for local storage
        """
        pass
