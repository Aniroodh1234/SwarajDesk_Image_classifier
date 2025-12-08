from pydantic import BaseModel, Field


class CompareResponse(BaseModel):
    match: bool = Field(..., description="True if similarity >= threshold")
    similarity: float = Field(..., ge=0.0, le=1.0, description="Cosine similarity score")
    threshold: float = Field(..., ge=0.0, le=1.0, description="Similarity threshold used")

