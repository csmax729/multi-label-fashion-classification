FROM ghcr.io/astral-sh/uv:0.12.23 AS uv
FROM python:3.12-slim

COPY --from=uv /uv /uvx /usr/local/bin/
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=secret,id=ca_bundle --mount=type=cache,target=/root/.cache/uv \
    if [ -f /run/secrets/ca_bundle ]; then export SSL_CERT_FILE=/run/secrets/ca_bundle; fi \
    && UV_LINK_MODE=copy uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN --mount=type=secret,id=ca_bundle --mount=type=cache,target=/root/.cache/uv \
    if [ -f /run/secrets/ca_bundle ]; then export SSL_CERT_FILE=/run/secrets/ca_bundle; fi \
    && UV_LINK_MODE=copy uv sync --frozen --no-dev \
    && chmod -R a+rX /app/src \
    && useradd --create-home --uid 10001 fashion \
    && mkdir -p /app/artifacts \
    && chown fashion:fashion /app/artifacts

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    FASHION_DEVICE=cpu \
    FASHION_CHECKPOINT=/app/artifacts/demo/best.pt
USER fashion
EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1
CMD ["uvicorn", "fashion_classifier.api:app", "--host", "0.0.0.0", "--port", "8000"]
