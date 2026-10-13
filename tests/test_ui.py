"""Exercise Streamlit's actual render loop with an offline inference service."""

import io
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests
import streamlit
from PIL import Image
from streamlit.testing.v1 import AppTest

UI_PATH = Path(__file__).parents[1] / "src" / "fashion_classifier" / "ui.py"


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setenv("FASHION_API_URL", "http://127.0.0.1:8000")
    response = SimpleNamespace(
        ok=True,
        json=lambda: {"ready": True, "model_info": {"demo": True}},
    )
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: response)
    return response


def test_ready_demo_renders_warning_and_upload_control(api):
    page = AppTest.from_file(UI_PATH).run()
    assert not page.exception
    assert page.title[0].value == "Fashion image classifier"
    assert len(page.get("file_uploader")) == 1
    assert "synthetic images" in page.warning[0].value
    assert "do not measure performance" in page.warning[0].value


def test_real_checkpoint_does_not_show_synthetic_demo_warning(api):
    api.json = lambda: {"ready": True, "model_info": {"demo": False}}
    page = AppTest.from_file(UI_PATH).run()
    assert not page.exception
    assert not page.warning
    assert len(page.get("file_uploader")) == 1


def test_missing_checkpoint_stops_before_upload(api):
    api.ok = False
    api.json = lambda: {"ready": False, "reason": "checkpoint_missing"}
    page = AppTest.from_file(UI_PATH).run()
    assert not page.exception
    assert "No model is ready" in page.error[0].value
    assert not page.get("file_uploader")


def test_unavailable_service_has_helpful_error(api, monkeypatch):
    def unavailable(*args, **kwargs):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(requests, "get", unavailable)
    page = AppTest.from_file(UI_PATH).run()
    assert not page.exception
    assert "service is unavailable" in page.error[0].value


def test_uploaded_image_predicts_four_attributes_and_latency(api, monkeypatch):
    image = io.BytesIO()
    Image.new("RGB", (32, 32), color="blue").save(image, format="PNG")
    uploaded = SimpleNamespace(getvalue=image.getvalue, name="shirt.png", type="image/png")
    monkeypatch.setattr(streamlit, "file_uploader", lambda *args, **kwargs: uploaded)
    predictions = {
        "gender": {"label": "Unisex", "confidence": 0.8},
        "articleType": {"label": "Tshirts", "confidence": 0.9},
        "baseColour": {"label": "Blue", "confidence": 0.7},
        "season": {"label": "Summer", "confidence": 0.6},
    }
    posted = []

    def post(url, files, timeout):
        posted.append((url, files, timeout))
        return SimpleNamespace(
            ok=True, json=lambda: {"predictions": predictions, "latency_ms": 125}
        )

    monkeypatch.setattr(requests, "post", post)
    page = AppTest.from_file(UI_PATH).run()
    assert not page.exception
    page.button[0].click().run()
    assert not page.exception
    assert not page.error
    assert {metric.value for metric in page.metric} == {"Unisex", "Tshirts", "Blue", "Summer"}
    assert any("0.125 seconds" in caption.value for caption in page.caption)
    assert len(posted) == 1
    assert posted[0][0] == "http://127.0.0.1:8000/predict"
    assert posted[0][1]["file"][1] == image.getvalue()


def test_invalid_uploaded_image_shows_error_without_prediction(api, monkeypatch):
    uploaded = SimpleNamespace(
        getvalue=lambda: b"invalid image", name="image.png", type="image/png"
    )
    monkeypatch.setattr(streamlit, "file_uploader", lambda *args, **kwargs: uploaded)
    page = AppTest.from_file(UI_PATH).run()
    assert not page.exception
    assert "cannot be decoded" in page.error[0].value
    assert not page.button
