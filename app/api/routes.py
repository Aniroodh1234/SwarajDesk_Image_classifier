from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.schemas import CompareResponse
from app.services.matching import compare_images
from app.utils.images import read_image_input

router = APIRouter()


@router.get("/health", summary="Health check")
async def health():
    return {"status": "ok"}


@router.post(
    "/compare",
    response_model=CompareResponse,
    summary="Compare citizen image with drone image (file upload or URL)",
    status_code=status.HTTP_200_OK,
)
async def compare(
    citizen_image: UploadFile | None = File(
        None, description="Citizen complaint image file (optional if citizen_url provided)"
    ),
    citizen_url: str | None = Form(
        None, description="Citizen complaint image URL (optional if citizen_image provided)"
    ),
    drone_image: UploadFile | None = File(
        None, description="Drone inspection image file (optional if drone_url provided)"
    ),
    drone_url: str | None = Form(
        None, description="Drone inspection image URL (optional if drone_image provided)"
    ),
):
    citizen_bytes, citizen_name = await read_image_input(citizen_image, citizen_url, role="citizen")
    drone_bytes, drone_name = await read_image_input(drone_image, drone_url, role="drone")

    try:
        match, similarity, threshold = await compare_images(
            citizen_bytes, drone_bytes, citizen_name, drone_name
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Embedding service error: {exc}") from exc

    return CompareResponse(match=match, similarity=similarity, threshold=threshold)

