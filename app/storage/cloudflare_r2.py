"""Cloudflare R2 storage backend using S3-compatible API."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

try:
    import boto3
    from botocore.exceptions import ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

from .base import StorageBackend

_log = logging.getLogger(__name__)


class CloudflareR2Storage(StorageBackend):
    """Cloudflare R2 storage backend using boto3 S3 API.
    
    Cloudflare R2 is S3-compatible, so we use boto3 with a custom endpoint.
    """
    
    def __init__(self, config: dict):
        """Initialize Cloudflare R2 storage backend.
        
        Args:
            config: Configuration dict with R2-specific settings:
                - enabled: bool
                - account_id: str
                - access_key_id: str
                - secret_access_key: str
                - bucket_name: str
                - region: str (wnam, enam, weur, eeur, apac, auto)
                - public_url_base: Optional[str] (CDN URL)
        """
        self.account_id = config.get("account_id")
        self.access_key_id = config.get("access_key_id")
        self.secret_access_key = config.get("secret_access_key")
        self.bucket_name = config.get("bucket_name")
        self.region = config.get("region", "auto")
        self.public_url_base = config.get("public_url_base")
        self.enabled = config.get("enabled", False)
        self.client: Optional[object] = None
        
        if not BOTO3_AVAILABLE:
            _log.warning("boto3 not installed; Cloudflare R2 storage disabled")
            self.enabled = False
            return
        
        if self.enabled and self._validate_config():
            try:
                self.client = boto3.client(
                    service_name="s3",
                    endpoint_url=f"https://{self.account_id}.r2.cloudflarestorage.com",
                    aws_access_key_id=self.access_key_id,
                    aws_secret_access_key=self.secret_access_key,
                    region_name=self.region,
                )
                _log.info(f"Cloudflare R2 storage initialized: bucket={self.bucket_name}, region={self.region}")
            except Exception as e:
                _log.error(f"Failed to initialize Cloudflare R2 client: {e}")
                self.enabled = False
                self.client = None
        else:
            self.client = None
    
    def _validate_config(self) -> bool:
        """Validate required configuration.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        required = [self.account_id, self.access_key_id, self.secret_access_key, self.bucket_name]
        if not all(required):
            _log.warning("Cloudflare R2 configuration incomplete; storage disabled")
            return False
        return True
    
    async def upload(self, local_path: Path, remote_key: str) -> str:
        """Upload file to Cloudflare R2 and return public URL.
        
        Args:
            local_path: Path to local file to upload
            remote_key: R2 object key (e.g., "task_id/picture-1.png")
            
        Returns:
            Public URL to access the uploaded file
            
        Raises:
            RuntimeError: If upload fails or R2 is not enabled
        """
        if not self.is_enabled():
            raise RuntimeError("Cloudflare R2 storage not enabled or configured")
        
        try:
            def _upload():
                with open(local_path, "rb") as f:
                    self.client.upload_fileobj(
                        f,
                        self.bucket_name,
                        remote_key,
                        ExtraArgs={"ACL": "public-read"}  # Make publicly readable
                    )
            
            await asyncio.to_thread(_upload)
            url = self.get_url(remote_key)
            _log.info(f"Uploaded {local_path.name} to R2: {url}")
            return url
        
        except ClientError as e:
            _log.error(f"R2 upload failed for {local_path}: {e}")
            raise RuntimeError(f"R2 upload failed: {e}") from e
        except Exception as e:
            _log.error(f"Unexpected error during R2 upload: {e}")
            raise RuntimeError(f"R2 upload failed: {e}") from e
    
    async def delete(self, remote_key: str) -> bool:
        """Delete object from Cloudflare R2.
        
        Args:
            remote_key: R2 object key to delete
            
        Returns:
            True if deletion succeeded, False otherwise
        """
        if not self.is_enabled():
            return False
        
        try:
            await asyncio.to_thread(
                self.client.delete_object,
                Bucket=self.bucket_name,
                Key=remote_key
            )
            _log.info(f"Deleted from R2: {remote_key}")
            return True
        except ClientError as e:
            _log.error(f"R2 delete failed for {remote_key}: {e}")
            return False
        except Exception as e:
            _log.error(f"Unexpected error during R2 delete: {e}")
            return False
    
    def get_url(self, remote_key: str) -> str:
        """Get public URL for R2 object.
        
        Args:
            remote_key: R2 object key
            
        Returns:
            Public URL (CDN if configured, otherwise R2 public bucket URL)
        """
        if self.public_url_base:
            # Use custom CDN/public URL
            return f"{self.public_url_base.rstrip('/')}/{remote_key}"
        else:
            # Use R2 public bucket URL
            # Note: This assumes bucket is configured as public
            return f"https://{self.bucket_name}.{self.account_id}.r2.cloudflarestorage.com/{remote_key}"
    
    def is_enabled(self) -> bool:
        """Check if R2 is enabled and configured.
        
        Returns:
            True if R2 client is initialized and ready, False otherwise
        """
        return self.enabled and self.client is not None
    
    def is_cloud_enabled(self) -> bool:
        """Check if this is a cloud storage backend.
        
        Returns:
            True - this is cloud storage (Cloudflare R2)
        """
        return True
