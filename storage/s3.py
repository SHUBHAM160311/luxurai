"""
LuxurAI — storage/s3.py
─────────────────────────────────────────────────────────────────
AWS S3 Image Storage Helper

Kaam kya karta hai:
  1. Image generation API se aayi temporary URL → download karo
  2. AWS S3 bucket mein permanently upload karo
  3. Permanent public S3 URL return karo → DB mein save hota hai

Flow:
  jobs.py → generate_fn() → temp_url
          → s3.upload_from_url(temp_url, job_id)
          → permanent_url → job_service.complete_job(job_id, permanent_url)

.env mein ye daalo:
  AWS_ACCESS_KEY_ID=your_access_key
  AWS_SECRET_ACCESS_KEY=your_secret_key
  AWS_S3_BUCKET=luxurai-images
  AWS_REGION=ap-south-1         # Mumbai region (India ke liye fast)
  AWS_CDN_URL=                  # Optional: CloudFront URL if using CDN
                                # e.g. https://cdn.luxurai.in
─────────────────────────────────────────────────────────────────
"""

import os
import io
import logging
import mimetypes
from datetime import datetime, timezone
from typing import Optional

import httpx
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("luxurai.s3")


# ─────────────────────────────────────────────
# Config — from .env
# ─────────────────────────────────────────────
AWS_ACCESS_KEY_ID     = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_S3_BUCKET         = os.getenv("AWS_S3_BUCKET", "luxurai-images")
AWS_REGION            = os.getenv("AWS_REGION", "ap-south-1")   # Mumbai

# Optional: CloudFront CDN URL (faster delivery, custom domain)
# Agar CDN nahi hai toh S3 URL directly use hoga
AWS_CDN_URL           = os.getenv("AWS_CDN_URL", "").rstrip("/")

# Folder structure inside S3 bucket:
#   generated/2025/01/job_abc123.png
#   generated/2025/01/job_xyz456.webp
S3_PREFIX = "generated"


# ─────────────────────────────────────────────
# Exceptions
# ─────────────────────────────────────────────
class S3Error(Exception):
    """Base S3 exception."""

class S3UploadError(S3Error):
    """Upload failed."""

class S3DownloadError(S3Error):
    """Could not fetch image from source URL."""

class S3NotConfiguredError(S3Error):
    """AWS credentials missing in .env"""


# ─────────────────────────────────────────────
# S3 Client
# ─────────────────────────────────────────────
def _get_client():
    """
    Create boto3 S3 client.
    Raises S3NotConfiguredError if credentials missing.
    """
    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
        raise S3NotConfiguredError(
            "AWS credentials missing! "
            "Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in .env"
        )

    return boto3.client(
        "s3",
        region_name          = AWS_REGION,
        aws_access_key_id    = AWS_ACCESS_KEY_ID,
        aws_secret_access_key = AWS_SECRET_ACCESS_KEY,
    )


# ─────────────────────────────────────────────
# Key Generator
# ─────────────────────────────────────────────
def _make_s3_key(job_id: str, extension: str = "png") -> str:
    """
    Generate S3 object key with date-based folder.

    Example output:
        generated/2025/01/job_abc123def456.png
    """
    now   = datetime.now(timezone.utc)
    year  = now.strftime("%Y")
    month = now.strftime("%m")
    ext   = extension.lstrip(".")
    return f"{S3_PREFIX}/{year}/{month}/{job_id}.{ext}"


def _make_public_url(key: str) -> str:
    """
    Build the public URL for a given S3 key.

    If CDN configured:
        https://cdn.luxurai.in/generated/2025/01/job_abc.png
    Else S3 direct:
        https://luxurai-images.s3.ap-south-1.amazonaws.com/generated/2025/01/job_abc.png
    """
    if AWS_CDN_URL:
        return f"{AWS_CDN_URL}/{key}"
    return f"https://{AWS_S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{key}"


# ─────────────────────────────────────────────
# Core: Download image bytes from URL
# ─────────────────────────────────────────────
async def _download_image(url: str) -> tuple[bytes, str]:
    """
    Download image from a URL (e.g. Replicate/fal.ai temp URL).

    Returns:
        (image_bytes, content_type)
        e.g. (b'\x89PNG...', 'image/png')
    """
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.get(url, follow_redirects=True)

    if resp.status_code != 200:
        raise S3DownloadError(
            f"Failed to download image from {url} "
            f"(status {resp.status_code})"
        )

    content_type = resp.headers.get("content-type", "image/png").split(";")[0]
    return resp.content, content_type


# ─────────────────────────────────────────────
# Core: Upload bytes to S3
# ─────────────────────────────────────────────
def _upload_bytes_to_s3(
    data:         bytes,
    key:          str,
    content_type: str = "image/png",
) -> str:
    """
    Upload raw bytes to S3.

    Returns:
        Public URL of uploaded object
    """
    client = _get_client()

    try:
        client.put_object(
            Bucket      = AWS_S3_BUCKET,
            Key         = key,
            Body        = data,
            ContentType = content_type,
            # Public read — so frontend can display images directly
            ACL         = "public-read",
            # Cache for 1 year (images don't change)
            CacheControl = "public, max-age=31536000, immutable",
            # Metadata for debugging
            Metadata    = {
                "source": "luxurai-generation",
            }
        )
        logger.info(f"✓ Uploaded to S3: {key} ({len(data)} bytes)")
        return _make_public_url(key)

    except NoCredentialsError:
        raise S3NotConfiguredError("Invalid AWS credentials")
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        raise S3UploadError(f"S3 upload failed [{error_code}]: {e}")


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────
async def upload_from_url(
    image_url:  str,
    job_id:     str,
    extension:  str = "png",
) -> str:
    """
    Main function — jobs.py ya generator.py se call karo.

    Steps:
      1. image_url se image download karo
      2. S3 mein upload karo
      3. Permanent public URL return karo

    Args:
        image_url:  Replicate/fal.ai ka temporary image URL
        job_id:     Job ID (S3 key ke liye use hoga)
        extension:  File extension (default: 'png')

    Returns:
        Permanent S3/CDN URL
        e.g. "https://luxurai-images.s3.ap-south-1.amazonaws.com/generated/2025/01/job_abc.png"

    Raises:
        S3DownloadError:      Source URL se image download nahi hua
        S3UploadError:        S3 upload fail hua
        S3NotConfiguredError: AWS credentials .env mein nahi hain

    Usage in generator.py:
        from storage.s3 import upload_from_url

        async def my_generator(job: Job) -> str:
            # Step 1: Image generate karo
            temp_url = await replicate.run(...)

            # Step 2: S3 pe permanently save karo
            permanent_url = await upload_from_url(temp_url, job.id)
            return permanent_url
    """
    logger.info(f"Uploading job {job_id} image to S3...")

    # Download from generation API
    image_bytes, content_type = await _download_image(image_url)

    # Detect extension from content type if needed
    if content_type == "image/jpeg":
        extension = "jpg"
    elif content_type == "image/webp":
        extension = "webp"
    elif content_type == "image/png":
        extension = "png"

    # Build S3 key
    key = _make_s3_key(job_id, extension)

    # Upload
    permanent_url = _upload_bytes_to_s3(image_bytes, key, content_type)

    logger.info(f"✅ Job {job_id} image stored: {permanent_url}")
    return permanent_url


async def upload_from_bytes(
    image_bytes:  bytes,
    job_id:       str,
    content_type: str = "image/png",
    extension:    str = "png",
) -> str:
    """
    Agar generation API directly bytes return kare (URL nahi) toh ye use karo.

    Args:
        image_bytes:  Raw image bytes
        job_id:       Job ID
        content_type: MIME type (default: 'image/png')
        extension:    File extension (default: 'png')

    Returns:
        Permanent S3/CDN URL
    """
    key = _make_s3_key(job_id, extension)
    permanent_url = _upload_bytes_to_s3(image_bytes, key, content_type)
    logger.info(f"✅ Job {job_id} image stored: {permanent_url}")
    return permanent_url


def delete_image(job_id: str, extension: str = "png") -> bool:
    """
    S3 se image delete karo (user account delete hone pe use karo).

    Returns True if deleted, False if not found.
    """
    key = _make_s3_key(job_id, extension)
    client = _get_client()

    try:
        client.delete_object(Bucket=AWS_S3_BUCKET, Key=key)
        logger.info(f"Deleted S3 object: {key}")
        return True
    except ClientError:
        return False


def get_presigned_url(job_id: str, extension: str = "png", expires: int = 3600) -> str:
    """
    Private bucket ke liye temporary signed URL generate karo.
    (Agar bucket public nahi hai toh ye use karo instead of public URL)

    Args:
        job_id:    Job ID
        extension: File extension
        expires:   URL expiry seconds (default: 1 hour)

    Returns:
        Temporary signed URL
    """
    key    = _make_s3_key(job_id, extension)
    client = _get_client()

    url = client.generate_presigned_url(
        "get_object",
        Params  = {"Bucket": AWS_S3_BUCKET, "Key": key},
        ExpiresIn = expires,
    )
    return url


# ─────────────────────────────────────────────
# Health Check
# ─────────────────────────────────────────────
def check_s3_connection() -> dict:
    """
    S3 connection test karo — startup pe call karo.

    Returns:
        {"ok": True, "bucket": "luxurai-images", "region": "ap-south-1"}
        ya
        {"ok": False, "error": "...reason..."}
    """
    try:
        client = _get_client()
        # Bucket exist karta hai check karo
        client.head_bucket(Bucket=AWS_S3_BUCKET)
        return {
            "ok":     True,
            "bucket": AWS_S3_BUCKET,
            "region": AWS_REGION,
            "cdn":    AWS_CDN_URL or "none (using S3 direct URL)",
        }
    except S3NotConfiguredError as e:
        return {"ok": False, "error": str(e)}
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code == "404":
            return {"ok": False, "error": f"Bucket '{AWS_S3_BUCKET}' does not exist"}
        if error_code == "403":
            return {"ok": False, "error": "Access denied — check IAM permissions"}
        return {"ok": False, "error": str(e)}
    except Exception as e:
        return {"ok": False, "error": str(e)}
