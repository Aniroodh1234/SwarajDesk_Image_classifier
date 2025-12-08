# app/clients/embedding_client.py
from typing import List
import httpx
import numpy as np
from fastapi import HTTPException
from app.config import settings

class EmbeddingClient:
    """Client wrapper for CLIP embeddings on Hugging Face Inference API."""

    def __init__(self):
        self.base_url = settings.embedding_api_url
        self.api_key = settings.embedding_api_key
        self.timeout = settings.request_timeout_seconds

    async def embed_image(self, file_bytes: bytes, filename: str) -> List[float]:
        """
        Send image bytes to the CLIP embedding API and return normalized embedding vector.
        
        Args:
            file_bytes: Raw image bytes
            filename: Name of the file (for logging/debugging)
            
        Returns:
            List of float values representing the image embedding
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Send raw image bytes directly
                resp = await client.post(
                    str(self.base_url),
                    headers=headers,
                    content=file_bytes
                )
            
            resp.raise_for_status()
            
        except httpx.TimeoutException as exc:
            raise HTTPException(
                status_code=504,
                detail=f"Embedding API timeout after {self.timeout}s. The model might be loading, please try again."
            ) from exc
        except httpx.HTTPStatusError as exc:
            error_detail = f"Embedding API error {resp.status_code}"
            try:
                error_body = resp.text
                
                # Handle specific error cases
                if resp.status_code == 503:
                    error_detail = "Model is currently loading. Please wait 20-30 seconds and try again."
                elif resp.status_code == 401:
                    error_detail = "Invalid API key. Please check your EMBEDDING_API_KEY in .env file."
                elif resp.status_code == 429:
                    error_detail = "Rate limit exceeded. Please wait a moment and try again."
                elif "410" in error_body or "no longer supported" in error_body.lower():
                    error_detail = "API endpoint deprecated. Please update EMBEDDING_API_URL in .env to: https://api-inference.huggingface.co/pipeline/feature-extraction/openai/clip-vit-large-patch14"
                else:
                    # Try to extract error message
                    try:
                        error_json = resp.json()
                        if "error" in error_json:
                            error_detail += f": {error_json['error']}"
                    except:
                        error_detail += f": {error_body[:200]}"
            except Exception:
                pass
            
            raise HTTPException(
                status_code=502,
                detail=error_detail
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Embedding service connection error: {str(exc)}"
            ) from exc

        try:
            # Parse the response
            response_data = resp.json()
            
            # The HuggingFace feature-extraction endpoint returns embeddings directly
            # Format can be: [[embedding_values]] or [embedding_values]
            if isinstance(response_data, list):
                # Flatten if nested
                if len(response_data) > 0 and isinstance(response_data[0], list):
                    # Multiple embeddings or patches - take the first one or average
                    arr = np.array(response_data[0], dtype=np.float32)
                else:
                    arr = np.array(response_data, dtype=np.float32)
            else:
                raise ValueError(f"Unexpected response format: {type(response_data)}")
            
            # Ensure it's 1D
            if arr.ndim > 1:
                arr = arr.flatten()
            
            # L2 normalize the embedding vector
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr = arr / norm
            
            return arr.astype(float).tolist()
            
        except (ValueError, KeyError) as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Failed to parse embedding response: {str(exc)}"
            ) from exc