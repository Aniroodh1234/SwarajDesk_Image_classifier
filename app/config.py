# app/config.py
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    embedding_api_url: str = Field(
        ...,
        description="Base URL for the hosted image embedding endpoint",
    )
    embedding_api_key: str = Field(
        ...,
        description="Bearer token or API key for the embedding service",
    )
    similarity_threshold: float = Field(
        0.78,
        ge=0.0,
        le=1.0,
        description="Cosine similarity threshold for declaring a match",
    )
    request_timeout_seconds: float = Field(
        30.0,  # Increased timeout for image processing
        gt=0.0,
        description="HTTP timeout for upstream embedding calls",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
        case_sensitive=False,
        extra="ignore",
    )

settings = Settings()