import base64
from pathlib import Path
import binascii
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.face_service import FaceAnalysisError, analyze_image, image_size

MAX_IMAGE_BYTES = 10 * 1024 * 1024

app = FastAPI(
    title="Clear Face Detection API",
    description="Accepts a base64-encoded image and reports whether its face is acceptable.",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


class VerifyRequest(BaseModel):
    image: str = Field(description="Base64 image data, optionally as a data URL")


def _response(accepted: bool, flaws: str, result: dict[str, Any], status_code: int = 200) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "accepted": accepted,
            "flaws": flaws,
            "face_detected": result.get("face_detected", False),
            "face_count": result.get("face_count", 0),
            "image_size": result.get("image_size"),
            "facial_area": result.get("facial_area"),
            "landmarks": result.get("landmarks"),
            "detection_confidence": result.get("detection_confidence"),
            "quality": result.get("quality"),
        },
    )


@app.exception_handler(RequestValidationError)
async def invalid_request_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    return _response(False, "Invalid JSON request. Provide an 'image' field containing base64 image data.", {}, 400)


@app.post("/api/verify")
def verify_face(payload: VerifyRequest) -> JSONResponse:
    encoded_image = payload.image.strip()

    if encoded_image.startswith("data:"):
        header, separator, encoded_image = encoded_image.partition(",")
        if not separator or ";base64" not in header.lower():
            return _response(False, "Image data URL must use base64 encoding.", {}, 400)

    # Reject oversized requests before allocating a decoded copy.
    if len(encoded_image) > ((MAX_IMAGE_BYTES + 2) // 3) * 4:
        return _response(False, "Image is too large (maximum 10 MB).", {}, 413)

    try:
        image_bytes = base64.b64decode(encoded_image, validate=True)
    except (binascii.Error, ValueError):
        return _response(False, "Image must be valid base64 data.", {}, 400)

    if len(image_bytes) > MAX_IMAGE_BYTES:
        return _response(False, "Image is too large (maximum 10 MB).", {}, 413)

    try:
        result: dict[str, Any] = analyze_image(image_bytes)
    except FaceAnalysisError as exc:
        return _response(False, str(exc), {"image_size": image_size(image_bytes)}, exc.status_code)

    if result.get("acceptable_for_enrollment"):
        return _response(True, "", result)

    quality = result.get("quality") or {}
    issues = quality.get("issues") or []
    if issues:
        flaws = "; ".join(issues)
    else:
        flaws = result.get("message") or "The image was not accepted."
    return _response(False, flaws, result)


# The page that uploads a photo or opens the camera is served from this same
# process, so the project root is the only thing a user has to start.
app.mount("/", StaticFiles(directory=str(Path(__file__).resolve().parent.parent / "static"), html=True), name="static")
