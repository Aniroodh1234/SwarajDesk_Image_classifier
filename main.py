from fastapi import FastAPI

from app.api.routes import router
from app.config import settings


def create_app() -> FastAPI:
    app = FastAPI(
        title="Civic Issue Image Matcher",
        description=(
            "Uses hosted DINOv2 embeddings to verify if a citizen image and a drone "
            "image depict the same civic issue."
        ),
        version="1.0.0",
    )
    app.include_router(router, prefix="/api")
    return app


app = create_app()


@app.get("/config", include_in_schema=False)
async def config():
    # Exposed for quick debugging; remove or secure in production.
    return {
        "embedding_api_url": str(settings.embedding_api_url),
        "similarity_threshold": settings.similarity_threshold,
        "timeout": settings.request_timeout_seconds,
    }

