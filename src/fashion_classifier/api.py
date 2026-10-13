"""Serve a trusted local checkpoint through a small, validated HTTP interface."""

from __future__ import annotations

import io
import logging
import os
import threading
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Protocol

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
from python_multipart.multipart import parse_options_header
from starlette.concurrency import run_in_threadpool

LOGGER = logging.getLogger(__name__)
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
# Allow ordinary multipart headers while bounding requests before form parsing.
MAX_REQUEST_BYTES = MAX_IMAGE_BYTES + 64 * 1024
SUPPORTED_FORMATS = {"JPEG", "PNG"}


class ImagePredictor(Protocol):
    @property
    def model_info(self) -> dict[str, Any]: ...

    def predict(self, image: Image.Image) -> dict[str, Any]: ...


class Prediction(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)


class PredictionResponse(BaseModel):
    predictions: dict[str, Prediction]
    latency_ms: float = Field(ge=0)
    model_info: dict[str, Any]


class _RequestTooLarge(HTTPException):
    pass


class RequestSizeLimitMiddleware:
    """Limit request bytes even when a client omits Content-Length."""

    def __init__(self, app: Any, max_bytes: int = MAX_REQUEST_BYTES) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        declared_length = headers.get(b"content-length")
        if declared_length is not None:
            try:
                length = int(declared_length)
                if length < 0:
                    raise ValueError("negative content length")
                oversized = length > self.max_bytes
            except ValueError:
                response = JSONResponse({"detail": "Invalid Content-Length header."}, 400)
                await response(scope, receive, send)
                return
            if oversized:
                response = JSONResponse({"detail": "Upload exceeds the 10 MiB image limit."}, 413)
                await response(scope, receive, send)
                return

        received = 0
        response_started = False
        content_type, options = parse_options_header(headers.get(b"content-type", b""))
        boundary = (
            options.get(b"boundary") if content_type.lower() == b"multipart/form-data" else None
        )
        closing_boundary = b"--" + boundary + b"--" if boundary else None
        trailing = b""

        async def limited_receive() -> dict:
            nonlocal received, trailing
            message = await receive()
            if message["type"] == "http.request":
                body = message.get("body", b"")
                received += len(body)
                if received > self.max_bytes:
                    raise _RequestTooLarge(413, "Upload exceeds the 10 MiB image limit.")
                if closing_boundary:
                    trailing = (trailing + body)[-(len(closing_boundary) + 64) :]
                    if not message.get("more_body", False) and not trailing.rstrip(
                        b"\r\n"
                    ).endswith(closing_boundary):
                        # Raise while the parser is active so it closes unfinished temp uploads.
                        raise HTTPException(400, "The multipart upload is incomplete.")
            return message

        async def tracked_send(message: dict) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracked_send)
        except _RequestTooLarge:
            if response_started:
                raise
            response = JSONResponse({"detail": "Upload exceeds the 10 MiB image limit."}, 413)
            await response(scope, receive, send)


def _public_model_info(predictor: ImagePredictor) -> dict[str, Any]:
    # Serving does not need to disclose local checkpoint paths to clients.
    return {key: value for key, value in predictor.model_info.items() if key != "checkpoint_path"}


def _decode_image(content: bytes) -> Image.Image:
    if not content:
        raise HTTPException(400, "The uploaded file is empty.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content)) as source:
                if source.format not in SUPPORTED_FORMATS:
                    raise HTTPException(415, "Upload a JPEG or PNG image.")
                if source.width * source.height > MAX_IMAGE_PIXELS:
                    raise HTTPException(413, "Image exceeds the 20 megapixel limit.")
                source.verify()
            # verify() closes the decoding state; reopen and fully decode before inference.
            with Image.open(io.BytesIO(content)) as source:
                source.load()
                return source.convert("RGB")
    except HTTPException:
        raise
    except (Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
        raise HTTPException(413, "Image dimensions are too large.") from exc
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise HTTPException(400, "The file is not a valid, complete JPEG or PNG image.") from exc


def create_app(
    checkpoint_path: str | Path | None = None, predictor: ImagePredictor | None = None
) -> FastAPI:
    """Create the app; injection keeps HTTP tests independent of torch and checkpoints."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.predictor = predictor
        app.state.unavailable_reason = None
        app.state.prediction_lock = threading.Lock()
        if predictor is None:
            path = Path(
                checkpoint_path or os.getenv("FASHION_CHECKPOINT", "artifacts/demo/best.pt")
            )
            if not path.is_file():
                app.state.unavailable_reason = "checkpoint_missing"
            else:
                try:
                    from fashion_classifier.inference import Predictor

                    app.state.predictor = await run_in_threadpool(
                        Predictor, path, device=os.getenv("FASHION_DEVICE", "cpu")
                    )
                except Exception as exc:
                    app.state.unavailable_reason = "checkpoint_unusable"
                    LOGGER.exception("Checkpoint initialization failed (%s).", type(exc).__name__)
        yield
        app.state.predictor = None

    app = FastAPI(
        title="Fashion image classifier",
        version="0.1.0",
        description="Predict gender, article type, base colour, and season from one fashion image.",
        lifespan=lifespan,
    )
    app.add_middleware(RequestSizeLimitMiddleware)

    @app.get("/health")
    async def health(request: Request):
        active_predictor = request.app.state.predictor
        if active_predictor is None:
            return JSONResponse(
                {
                    "status": "unavailable",
                    "ready": False,
                    "reason": request.app.state.unavailable_reason,
                    "message": "Set FASHION_CHECKPOINT to a trusted, compatible local checkpoint.",
                },
                status_code=503,
            )
        return {
            "status": "ready",
            "ready": True,
            "model_info": _public_model_info(active_predictor),
        }

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(request: Request, file: Annotated[UploadFile, File()]):
        try:
            active_predictor = request.app.state.predictor
            if active_predictor is None:
                raise HTTPException(
                    503, "No model is ready. Check /health and configure a checkpoint."
                )
            chunks = []
            total = 0
            while chunk := await file.read(64 * 1024):
                total += len(chunk)
                if total > MAX_IMAGE_BYTES:
                    raise HTTPException(413, "Upload exceeds the 10 MiB image limit.")
                chunks.append(chunk)
            decoded_image = _decode_image(b"".join(chunks))

            def run_prediction() -> dict[str, Any]:
                with request.app.state.prediction_lock:
                    return active_predictor.predict(decoded_image)

            try:
                result = await run_in_threadpool(run_prediction)
                return PredictionResponse(
                    predictions=result["predictions"],
                    latency_ms=result["latency_ms"],
                    model_info=_public_model_info(active_predictor),
                )
            except Exception as exc:
                LOGGER.exception("Prediction failed.")
                raise HTTPException(
                    500, "Prediction failed. Verify the checkpoint configuration."
                ) from exc
            finally:
                decoded_image.close()
        finally:
            await file.close()

    return app


app = create_app()
