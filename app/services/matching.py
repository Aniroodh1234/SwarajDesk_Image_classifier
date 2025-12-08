# app/services/matching.py
from typing import Tuple, Sequence
import numpy as np

from app.clients.embedding_client import EmbeddingClient
from app.config import settings

client = EmbeddingClient()   # reuse instead of creating new client every time

def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)

async def compare_images(
    user_image: bytes, drone_image: bytes, user_name: str, drone_name: str
) -> Tuple[bool, float, float]:
    user_embedding = await client.embed_image(user_image, user_name)
    drone_embedding = await client.embed_image(drone_image, drone_name)

    similarity = _cosine_similarity(user_embedding, drone_embedding)
    threshold = settings.similarity_threshold

    return similarity >= threshold, similarity, threshold
