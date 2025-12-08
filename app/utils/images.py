# app/utils/images.py
import imghdr
import os
from typing import Tuple
from urllib.parse import urlparse
from io import BytesIO
from PIL import Image

import httpx
from fastapi import HTTPException, UploadFile, status

from app.config import settings


async def _fetch_url_bytes(url: str) -> Tuple[bytes, str]:
    """Fetch image from URL and return bytes with filename."""
    try:
        async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:
            resp = await client.get(url, follow_redirects=True)
            resp.raise_for_status()
            
    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"Timeout fetching image from URL after {settings.request_timeout_seconds}s",
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to download image from URL: HTTP {exc.response.status_code}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error fetching image from URL: {str(exc)}",
        ) from exc

    body = resp.content
    
    if not body:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="URL returned empty response",
        )

    # Validate it's actually an image
    content_type = resp.headers.get("content-type", "")
    guessed_type = imghdr.what(None, h=body)
    
    if not guessed_type and not content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="URL did not return valid image content",
        )

    # Extract filename from URL
    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)
    if not filename or "." not in filename:
        # Use extension from detected type if available
        ext = guessed_type or "jpg"
        filename = f"image_from_url.{ext}"
    
    return body, filename


def _validate_image(data: bytes, name: str) -> None:
    """Validate that bytes represent a valid image."""
    if len(data) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"The image '{name}' is empty (0 bytes)",
        )
    
    # Check file size (optional: prevent extremely large files)
    max_size = 10 * 1024 * 1024  # 10 MB
    if len(data) > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image '{name}' exceeds maximum size of {max_size // (1024*1024)}MB",
        )
    
    # Verify it's a valid image using PIL
    try:
        img = Image.open(BytesIO(data))
        img.verify()  # Verify it's a valid image
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image file '{name}': {str(exc)}",
        ) from exc


async def read_image_input(
    file: UploadFile | None, url: str | None, role: str
) -> Tuple[bytes, str]:
    """
    Returns (bytes, name) from either UploadFile or URL.
    
    Args:
        file: Uploaded file object
        url: URL to fetch image from
        role: Description of image role (e.g., "citizen", "drone") for error messages
        
    Returns:
        Tuple of (image_bytes, filename)
    """
    if file and url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Provide either a {role} file or a {role}_url, not both",
        )
    
    if not file and not url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Missing required parameter: provide either {role}_image file or {role}_url",
        )

    if file:
        # Read uploaded file
        try:
            data = await file.read()
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to read uploaded {role} image: {str(exc)}",
            ) from exc
        finally:
            await file.close()
        
        filename = file.filename or f"{role}_image.jpg"
        _validate_image(data, filename)
        return data, filename

    # URL path
    data, filename = await _fetch_url_bytes(url)  # type: ignore[arg-type]
    _validate_image(data, filename)
    return data, filename