#!/usr/bin/env python3
"""
R2 Migration Script: Migrate existing images to xxhash-based keys with deduplication.

This script:
1. Lists all existing images in R2 bucket
2. Downloads each image and computes its xxhash
3. Copies to new key (images/{xxhash}.{ext}) if unique
4. Deletes duplicates (same content = same hash)
5. Outputs a mapping file (old_key -> new_key)

Usage:
    # Dry run (preview changes without modifying R2)
    python scripts/r2_migrate_xxhash.py --dry-run

    # Execute migration with default batch size (10)
    python scripts/r2_migrate_xxhash.py

    # Execute with custom batch size
    python scripts/r2_migrate_xxhash.py --batch 50

    # Specify custom output mapping file
    python scripts/r2_migrate_xxhash.py --output migration_mapping.json
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import boto3
import xxhash
from botocore.config import Config
from dotenv import load_dotenv


def get_r2_client():
    """Create and return an S3-compatible client for Cloudflare R2."""
    account_id = os.getenv("DOCLING_R2_ACCOUNT_ID")
    access_key = os.getenv("DOCLING_R2_ACCESS_KEY_ID")
    secret_key = os.getenv("DOCLING_R2_SECRET_ACCESS_KEY")
    region = os.getenv("DOCLING_R2_REGION", "auto")

    if not all([account_id, access_key, secret_key]):
        raise ValueError(
            "Missing R2 credentials. Ensure DOCLING_R2_ACCOUNT_ID, "
            "DOCLING_R2_ACCESS_KEY_ID, and DOCLING_R2_SECRET_ACCESS_KEY are set."
        )

    endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"

    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
        config=Config(
            signature_version="s3v4",
            retries={"max_attempts": 3, "mode": "adaptive"},
        ),
    )


def list_all_objects(client, bucket: str, prefix: str = "images/") -> list[dict]:
    """List all objects in the bucket with the given prefix."""
    objects = []
    continuation_token = None

    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if continuation_token:
            kwargs["ContinuationToken"] = continuation_token

        response = client.list_objects_v2(**kwargs)

        if "Contents" in response:
            objects.extend(response["Contents"])

        if response.get("IsTruncated"):
            continuation_token = response.get("NextContinuationToken")
        else:
            break

    return objects


def get_extension(key: str) -> str:
    """Extract file extension from key."""
    path = Path(key)
    return path.suffix.lower() if path.suffix else ".png"


def compute_xxhash(data: bytes) -> str:
    """Compute xxhash64 of the given data and return hex digest."""
    return xxhash.xxh64(data).hexdigest()


def is_already_migrated(key: str) -> bool:
    """Check if a key is already in xxhash format (images/{16-char-hex}.ext)."""
    path = Path(key)
    name = path.stem
    # xxhash64 produces 16 hex characters
    if len(name) == 16:
        try:
            int(name, 16)
            return True
        except ValueError:
            pass
    return False


def migrate_images(
    dry_run: bool = True,
    batch_size: int = 10,
    output_file: str = "migration_mapping.json",
    limit: int = 0,
):
    """
    Migrate existing R2 images to xxhash-based keys with deduplication.

    Args:
        dry_run: If True, only preview changes without modifying R2
        batch_size: Number of images to process before logging progress
        output_file: Path to save the migration mapping JSON
        limit: Maximum number of objects to process (0 = no limit)
    """
    # Load environment variables
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)

    bucket = os.getenv("DOCLING_R2_BUCKET_NAME")
    public_url_base = os.getenv("DOCLING_R2_PUBLIC_URL_BASE", "")

    if not bucket:
        raise ValueError("DOCLING_R2_BUCKET_NAME not set in environment")

    print(f"{'=' * 60}")
    print(f"R2 Migration Script - {'DRY RUN' if dry_run else 'LIVE EXECUTION'}")
    print(f"{'=' * 60}")
    print(f"Bucket: {bucket}")
    print(f"Batch size: {batch_size}")
    print(f"Output file: {output_file}")
    print(f"Limit: {limit if limit > 0 else 'no limit'}")
    print()

    client = get_r2_client()

    # List all existing images
    print("Listing all objects in R2...")
    objects = list_all_objects(client, bucket, prefix="images/")
    total_objects = len(objects)
    print(f"Found {total_objects} objects")
    
    # Apply limit if specified
    if limit > 0 and total_objects > limit:
        objects = objects[:limit]
        print(f"Limiting to first {limit} objects for testing")
    
    total_objects = len(objects)
    print()

    if total_objects == 0:
        print("No images found. Nothing to migrate.")
        return

    # Track migration results
    hash_to_new_key: dict[str, str] = {}  # {xxhash: new_key}
    mapping: dict[str, dict] = {}  # {old_key: {new_key, new_url, action}}
    stats = {
        "total": total_objects,
        "migrated": 0,
        "deduplicated": 0,
        "already_migrated": 0,
        "errors": 0,
    }
    
    output_path = Path(__file__).parent.parent / output_file
    run_id = uuid4().hex
    started_at = datetime.utcnow().isoformat()
    
    def save_progress():
        """Save current progress to mapping file."""
        processed = (
            stats["migrated"]
            + stats["deduplicated"]
            + stats["already_migrated"]
            + stats["errors"]
        )
        session_data = {
            "run_id": run_id,
            "migration_date": started_at,
            "updated_at": datetime.utcnow().isoformat(),
            "dry_run": dry_run,
            "bucket": bucket,
            "limit": limit if limit > 0 else None,
            "batch_size": batch_size,
            "stats": stats.copy(),
            "mapping": mapping.copy(),
            "status": "in_progress" if processed < total_objects else "completed",
        }
        
        # Load existing sessions if file exists
        sessions = []
        if output_path.exists():
            try:
                with open(output_path, "r") as f:
                    existing = json.load(f)
                    if isinstance(existing, list):
                        # Replace the session for this run_id (keep other sessions intact)
                        sessions = [s for s in existing if s.get("run_id") != run_id]
                    elif isinstance(existing, dict):
                        sessions = [existing]
            except (json.JSONDecodeError, IOError):
                sessions = []
        
        sessions.append(session_data)
        
        with open(output_path, "w") as f:
            json.dump(sessions, f, indent=2)

    # Create an initial session entry immediately
    save_progress()

    # Process images
    for i, obj in enumerate(objects, 1):
        old_key = obj["Key"]

        try:
            # Check if already migrated
            if is_already_migrated(old_key):
                stats["already_migrated"] += 1
                # Still need to track the hash for dedup
                hash_value = Path(old_key).stem
                hash_to_new_key[hash_value] = old_key
                mapping[old_key] = {
                    "old_url": f"{public_url_base}/{old_key}",
                    "new_key": old_key,
                    "new_url": f"{public_url_base}/{old_key}",
                    "action": "skip_already_migrated",
                }
                continue

            # Download image
            response = client.get_object(Bucket=bucket, Key=old_key)
            data = response["Body"].read()

            # Compute hash
            hash_value = compute_xxhash(data)
            extension = get_extension(old_key)
            new_key = f"images/{hash_value}{extension}"

            if hash_value in hash_to_new_key:
                # DUPLICATE: same content already exists
                existing_key = hash_to_new_key[hash_value]
                stats["deduplicated"] += 1

                mapping[old_key] = {
                    "old_url": f"{public_url_base}/{old_key}",
                    "new_key": existing_key,
                    "new_url": f"{public_url_base}/{existing_key}",
                    "action": "deduplicated",
                    "duplicate_of": existing_key,
                }

                if not dry_run:
                    # Delete the duplicate
                    client.delete_object(Bucket=bucket, Key=old_key)

                print(f"  [{i}/{total_objects}] DEDUP: {old_key} -> {existing_key}")

            else:
                # UNIQUE: migrate to new key
                if not dry_run:
                    if old_key != new_key:
                        # Upload with new key
                        content_type = "image/png"
                        if extension in [".jpg", ".jpeg"]:
                            content_type = "image/jpeg"
                        elif extension == ".gif":
                            content_type = "image/gif"
                        elif extension == ".webp":
                            content_type = "image/webp"

                        client.put_object(
                            Bucket=bucket,
                            Key=new_key,
                            Body=data,
                            ContentType=content_type,
                        )
                        # Delete old key
                        client.delete_object(Bucket=bucket, Key=old_key)

                # Only mark as migrated after successful upload/delete (or during dry-run)
                hash_to_new_key[hash_value] = new_key
                stats["migrated"] += 1

                mapping[old_key] = {
                    "old_url": f"{public_url_base}/{old_key}",
                    "new_key": new_key,
                    "new_url": f"{public_url_base}/{new_key}",
                    "action": "migrated",
                }

                print(f"  [{i}/{total_objects}] MIGRATE: {old_key} -> {new_key}")

        except Exception as e:
            stats["errors"] += 1
            mapping[old_key] = {
                "old_url": f"{public_url_base}/{old_key}",
                "new_key": None,
                "new_url": None,
                "action": "error",
                "error": str(e),
            }
            print(f"  [{i}/{total_objects}] ERROR: {old_key} - {e}")

        # Progress update and save every batch_size items
        if i % batch_size == 0:
            print(f"\n  Progress: {i}/{total_objects} processed")
            print(f"    Migrated: {stats['migrated']}, Deduplicated: {stats['deduplicated']}, Errors: {stats['errors']}")
            save_progress()
            print(f"    (Progress saved to {output_file})\n")

    # Final save with completed status
    save_progress()

    # Print summary
    print()
    print(f"{'=' * 60}")
    print("Migration Summary")
    print(f"{'=' * 60}")
    print(f"Total objects:      {stats['total']}")
    print(f"Migrated:           {stats['migrated']}")
    print(f"Deduplicated:       {stats['deduplicated']}")
    print(f"Already migrated:   {stats['already_migrated']}")
    print(f"Errors:             {stats['errors']}")
    print()
    print(f"Mapping saved to: {output_path}")

    if dry_run:
        print()
        print("This was a DRY RUN. No changes were made to R2.")
        print("Run without --dry-run to execute the migration.")


def main():
    parser = argparse.ArgumentParser(
        description="Migrate R2 images to xxhash-based keys with deduplication"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without modifying R2",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=10,
        help="Batch size for progress logging (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="migration_mapping.json",
        help="Output file for migration mapping (default: migration_mapping.json)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of objects to process for testing (default: 0 = no limit)",
    )

    args = parser.parse_args()

    try:
        migrate_images(
            dry_run=args.dry_run,
            batch_size=args.batch,
            output_file=args.output,
            limit=args.limit,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
