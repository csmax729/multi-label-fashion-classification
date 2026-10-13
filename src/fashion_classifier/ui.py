"""Streamlit client: streamlit run src/fashion_classifier/ui.py."""

from __future__ import annotations

import io
import os
import warnings

import requests
import streamlit as st
from PIL import Image, UnidentifiedImageError

API_URL = os.getenv("FASHION_API_URL", "http://127.0.0.1:8000").rstrip("/")
MAX_IMAGE_BYTES = 10 * 1024 * 1024
TASK_LABELS = {
    "gender": "Gender",
    "articleType": "Article type",
    "baseColour": "Base colour",
    "season": "Season",
}


def main() -> None:
    st.set_page_config(page_title="Fashion image classifier", page_icon="👕", layout="wide")
    st.title("Fashion image classifier")
    st.write("Upload a product photo to predict four fashion attributes from one image.")

    try:
        health_response = requests.get(f"{API_URL}/health", timeout=5)
        health = health_response.json()
    except (requests.RequestException, ValueError):
        st.error("The prediction service is unavailable. Start the inference service and refresh.")
        st.stop()
    if not health_response.ok or not health.get("ready"):
        st.error("No model is ready. Load a compatible trained checkpoint and refresh this page.")
        st.stop()

    model_info = health.get("model_info", {})
    if model_info.get("demo", False):
        st.warning(
            "DEMO: This model was trained on synthetic images. It demonstrates the app workflow; "
            "these predictions do not measure performance on real fashion products."
        )
    st.caption("Model: ResNet50 with four classification heads")

    uploaded = st.file_uploader(
        "Choose a JPEG or PNG image (maximum 10 MiB)", type=["jpg", "jpeg", "png"]
    )
    if uploaded is None:
        return
    content = uploaded.getvalue()
    if len(content) > MAX_IMAGE_BYTES:
        st.error("The image exceeds the 10 MiB upload limit.")
        return
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content)) as source:
                if source.format not in {"JPEG", "PNG"}:
                    st.error("Choose a complete JPEG or PNG image.")
                    return
                if source.width * source.height > 20_000_000:
                    st.error("Choose an image smaller than 20 megapixels.")
                    return
                source.load()
                preview = source.convert("RGB")
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ):
        st.error("The file cannot be decoded as a complete JPEG or PNG image.")
        return

    image_column, result_column = st.columns([1, 1.5])
    with image_column:
        st.image(preview, caption="Uploaded product image", width="stretch")
    with result_column:
        if st.button("Classify image", type="primary"):
            with st.spinner("Reading the image…"):
                try:
                    response = requests.post(
                        f"{API_URL}/predict",
                        files={
                            "file": (
                                uploaded.name,
                                content,
                                uploaded.type or "application/octet-stream",
                            )
                        },
                        timeout=(5, 60),
                    )
                    result = response.json()
                except (requests.RequestException, ValueError):
                    st.error("The prediction service did not respond. Please try again.")
                    return
            if not response.ok:
                st.error(result.get("detail", "The image could not be classified."))
                return
            prediction_columns = st.columns(2)
            for index, (task, title) in enumerate(TASK_LABELS.items()):
                prediction = result["predictions"].get(task)
                if prediction is None:
                    st.error(f"The model did not return a {title.lower()} prediction.")
                    continue
                with prediction_columns[index % 2]:
                    st.metric(title, prediction["label"])
                    st.caption(f"Confidence: {prediction['confidence']:.1%}")
            st.caption(f"Model prediction time: {result['latency_ms'] / 1000:.3f} seconds")
            st.caption("Confidence scores are not calibrated guarantees of correctness.")
    preview.close()


if __name__ == "__main__":
    main()
