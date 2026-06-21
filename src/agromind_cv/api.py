from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .inference import ModelUnavailableError, load_artifacts, predict_image

MAX_UPLOAD_BYTES = 10 * 1024 * 1024


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Warm-load if artifacts exist. Missing artifacts are reported cleanly by /health.
    try:
        load_artifacts()
    except ModelUnavailableError:
        pass
    yield


app = FastAPI(title="AgroMind CV Service", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    try:
        load_artifacts()
    except ModelUnavailableError:
        return {"status": "not_ready", "reason": "model_not_trained"}
    return {"status": "ok"}


@app.post("/predict")
async def predict(
    file: Annotated[UploadFile, File()],
    plantId: Annotated[str | None, Form()] = None,
) -> dict[str, object]:
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload must be an image.")
    image_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 10 MB limit.")
    try:
        result = predict_image(image_bytes)
    except ModelUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    if plantId:
        result["plantId"] = plantId
    return result
