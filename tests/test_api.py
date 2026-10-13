"""HTTP boundary tests do not depend on model downloads or training."""

import asyncio
import io

import httpx
import pytest
import starlette.formparsers
from fastapi.testclient import TestClient
from PIL import Image

from fashion_classifier import api
from fashion_classifier.api import create_app


class StubPredictor:
    def __init__(self):
        self.calls = 0
        self.model_info = {
            "demo": True,
            "architecture": "resnet50_shared_backbone_four_heads",
            "dataset_source": "synthetic_demo",
            "checkpoint_path": "/private/checkpoint.pt",
        }

    def predict(self, image):
        assert image.mode == "RGB"
        assert image.size == (16, 12)
        self.calls += 1
        return {
            "predictions": {
                "gender": {"label": "Unisex", "confidence": 0.8},
                "articleType": {"label": "Tshirts", "confidence": 0.9},
                "baseColour": {"label": "Blue", "confidence": 0.7},
                "season": {"label": "Summer", "confidence": 0.6},
            },
            "latency_ms": 12.5,
            "model_info": self.model_info,
        }


def image_bytes(image_format="PNG", mode="RGB"):
    output = io.BytesIO()
    Image.new(mode, (16, 12)).save(output, format=image_format)
    return output.getvalue()


@pytest.fixture
def predictor():
    return StubPredictor()


@pytest.fixture
def client(predictor):
    with TestClient(create_app(predictor=predictor)) as test_client:
        yield test_client


def test_health_reports_demo_provenance_without_local_path(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["ready"] is True
    assert response.json()["model_info"]["demo"] is True
    assert "checkpoint_path" not in response.json()["model_info"]


@pytest.mark.parametrize("image_format,mode", [("JPEG", "RGB"), ("PNG", "RGBA"), ("PNG", "L")])
def test_predict_decodes_supported_image_and_converts_rgb(client, predictor, image_format, mode):
    response = client.post("/predict", files={"file": ("image", image_bytes(image_format, mode))})
    assert response.status_code == 200
    payload = response.json()
    assert set(payload["predictions"]) == {"gender", "articleType", "baseColour", "season"}
    assert payload["predictions"]["articleType"]["label"] == "Tshirts"
    assert payload["latency_ms"] == 12.5
    assert payload["model_info"]["demo"] is True
    assert "checkpoint_path" not in payload["model_info"]
    assert predictor.calls == 1


@pytest.mark.parametrize("content", [b"", b"not an image", image_bytes()[:40]])
def test_rejects_empty_invalid_and_truncated_images(client, predictor, content):
    response = client.post("/predict", files={"file": ("image.png", content, "image/png")})
    assert response.status_code == 400
    assert predictor.calls == 0


def test_rejects_unsupported_image_format(client, predictor):
    response = client.post("/predict", files={"file": ("image.gif", image_bytes("GIF"))})
    assert response.status_code == 415
    assert predictor.calls == 0


def test_rejects_image_above_pixel_limit(client, predictor, monkeypatch):
    monkeypatch.setattr(api, "MAX_IMAGE_PIXELS", 100)
    response = client.post("/predict", files={"file": ("image.png", image_bytes())})
    assert response.status_code == 413
    assert predictor.calls == 0


def test_rejects_decompression_bomb_warning(client, predictor, monkeypatch):
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)
    response = client.post("/predict", files={"file": ("image.png", image_bytes())})
    assert response.status_code == 413
    assert predictor.calls == 0


def test_rejects_image_above_byte_limit(client, predictor, monkeypatch):
    monkeypatch.setattr(api, "MAX_IMAGE_BYTES", 50)
    response = client.post("/predict", files={"file": ("image.png", image_bytes())})
    assert response.status_code == 413
    assert predictor.calls == 0


def test_rejects_oversized_request_before_predictor(client, predictor):
    response = client.post(
        "/predict", content=b"x", headers={"Content-Length": str(api.MAX_REQUEST_BYTES + 1)}
    )
    assert response.status_code == 413
    assert predictor.calls == 0


def test_bounds_chunked_request_without_content_length(client, predictor):
    boundary = "fashion-test-boundary"
    prefix = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
        'filename="image.png"\r\nContent-Type: image/png\r\n\r\n'
    ).encode()
    chunks = iter([prefix, b"x" * (api.MAX_REQUEST_BYTES + 1), f"\r\n--{boundary}--\r\n".encode()])
    response = client.post(
        "/predict",
        content=chunks,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    assert response.status_code == 413
    assert predictor.calls == 0


def test_stream_limit_closes_partially_spooled_upload(predictor, monkeypatch):
    original_factory = starlette.formparsers.SpooledTemporaryFile
    temporary_files = []

    def tracked_temporary_file(*args, **kwargs):
        file = original_factory(*args, **kwargs)
        temporary_files.append(file)
        return file

    monkeypatch.setattr(starlette.formparsers, "SpooledTemporaryFile", tracked_temporary_file)
    boundary = "fashion-stream-boundary"
    prefix = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
        'filename="image.png"\r\nContent-Type: image/png\r\n\r\n'
    ).encode()

    async def body():
        yield prefix
        for _ in range(11):
            yield b"x" * (1024 * 1024)

    async def request():
        app = create_app(predictor=predictor)
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client,
        ):
            return await client.post(
                "/predict",
                content=body(),
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            )

    response = asyncio.run(request())
    assert response.status_code == 413
    assert predictor.calls == 0
    assert temporary_files
    assert all(file.closed for file in temporary_files)


def test_truncated_multipart_closes_unfinished_upload(predictor, monkeypatch):
    original_factory = starlette.formparsers.SpooledTemporaryFile
    temporary_files = []

    def tracked_temporary_file(*args, **kwargs):
        file = original_factory(*args, **kwargs)
        temporary_files.append(file)
        return file

    monkeypatch.setattr(starlette.formparsers, "SpooledTemporaryFile", tracked_temporary_file)
    boundary = "fashion-truncated-boundary"

    async def body():
        yield (
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
            'filename="image.png"\r\nContent-Type: image/png\r\n\r\n'
        ).encode()
        yield image_bytes()
        # Deliberately end before the multipart closing boundary.

    async def request():
        app = create_app(predictor=predictor)
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client,
        ):
            return await client.post(
                "/predict",
                content=body(),
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            )

    response = asyncio.run(request())
    assert response.status_code == 400
    assert predictor.calls == 0
    assert temporary_files
    assert all(file.closed for file in temporary_files)


@pytest.mark.parametrize("length", ["-1", "invalid"])
def test_invalid_content_length_has_client_error(client, length):
    response = client.post("/predict", content=b"x", headers={"Content-Length": length})
    assert response.status_code == 400


def test_malformed_multipart_boundary_is_rejected(client, predictor):
    response = client.post(
        "/predict", content=b"invalid form", headers={"Content-Type": "multipart/form-data"}
    )
    assert response.status_code == 400
    assert predictor.calls == 0


def test_missing_checkpoint_has_actionable_health_and_predict_error(tmp_path):
    with TestClient(create_app(checkpoint_path=tmp_path / "missing.pt")) as client:
        response = client.get("/health")
        assert response.status_code == 503
        assert response.json()["ready"] is False
        assert response.json()["reason"] == "checkpoint_missing"
        assert "FASHION_CHECKPOINT" in response.json()["message"]
        prediction = client.post("/predict", files={"file": ("image.png", image_bytes())})
        assert prediction.status_code == 503


def test_requires_image_upload(client):
    assert client.post("/predict").status_code == 422


def test_prediction_failure_does_not_expose_traceback(client, predictor):
    def fail(image):
        raise RuntimeError("private checkpoint detail")

    predictor.predict = fail
    response = client.post("/predict", files={"file": ("image.png", image_bytes())})
    assert response.status_code == 500
    assert "private checkpoint detail" not in response.text
    assert "Traceback" not in response.text
