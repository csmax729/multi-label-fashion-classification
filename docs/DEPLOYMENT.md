# Serve the model in a local container

The Docker image packages the CPU inference API and Streamlit dependencies. It
installs exactly the dependencies in `uv.lock`. Checkpoints and datasets stay
outside the image. No image is pushed and no public service is created by these
commands.

## Build and start the API

Run these commands from the repository root after producing a checkpoint:

```bash
docker build -t fashion-classifier:smoke .

docker run --rm --name fashion-api \
  --user "$(id -u):$(id -g)" \
  -p 127.0.0.1:8000:8000 \
  --mount "type=bind,src=$(pwd)/artifacts,dst=/app/artifacts,readonly" \
  -e FASHION_CHECKPOINT=/app/artifacts/demo/best.pt \
  fashion-classifier:smoke
```

The numeric user override lets the process read your checkpoint without changing
its ownership or permissions. The image otherwise defaults to a dedicated user
with UID 10001. Its application source is readable by either user. The mounted
artifacts directory is read-only.

For a model trained on real fashion images, replace the checkpoint environment
variable with the matching path under `/app/artifacts`. The API accepts trusted
local checkpoints produced by this project; it does not accept model uploads.
The demo checkpoint was trained on synthetic images and demonstrates the serving
workflow. It does not establish accuracy on the public fashion dataset.

If the cloud workspace has a read-only Docker configuration directory, use a
writable local directory for the build client:

```bash
mkdir -p /tmp/fashion-docker-config
DOCKER_CONFIG=/tmp/fashion-docker-config docker build -t fashion-classifier:smoke .
```

Some cloud workspaces provide internet access through injected proxy bindings and
a trusted proxy CA. Docker does not automatically inherit the machine's hostname
mapping or certificate configuration. For package-download DNS or certificate
errors in such a workspace, this build reuses its existing bindings:

```bash
python - <<'PY'
import os
import socket
import subprocess
from pathlib import Path
from urllib.parse import urlsplit

config = Path("/tmp/fashion-docker-config")
config.mkdir(exist_ok=True)
command = ["docker", "build", "--network=host", "-t", "fashion-classifier:smoke"]
proxy_hosts = set()
for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    if os.environ.get(name):
        command.extend(["--build-arg", name])
        proxy_hosts.add(urlsplit(os.environ[name]).hostname)
for host in sorted(proxy_hosts - {None}):
    address = socket.getaddrinfo(host, None, socket.AF_INET, socket.SOCK_STREAM)[0][4][0]
    command.extend(["--add-host", f"{host}:{address}"])
bundle = Path(os.environ.get("SSL_CERT_FILE") or "/etc/ssl/certs/ca-certificates.crt")
if not bundle.is_file():
    raise SystemExit("The existing trusted CA bundle is unavailable.")
command.extend(["--secret", f"id=ca_bundle,src={bundle}", "."])
environment = os.environ.copy()
environment["DOCKER_CONFIG"] = str(config)
subprocess.run(command, env=environment, check=True)
PY
```

The CA bundle is mounted only during dependency installation. Proxy and CA
bindings are absent from the runtime image; their values are never printed.
Package checksums and TLS verification remain enabled. Ordinary direct internet
builds do not need this optional configuration. Dependency installation has a
separate Docker layer and a uv cache, so source edits reuse installed dependencies.

## Check predictions and open the interface

In another terminal:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail -F file=@data/demo/images/1.png http://127.0.0.1:8000/predict

FASHION_API_URL=http://127.0.0.1:8000 \
  uv run --frozen streamlit run src/fashion_classifier/ui.py \
  --server.headless true
```

Open `http://127.0.0.1:8501`. The browser talks to Streamlit; Streamlit sends the
image to FastAPI. The API loads one checkpoint at startup and reuses it across
requests. `/health` returns HTTP 200 when the checkpoint is ready and HTTP 503
when it is missing or incompatible. The container health check uses this endpoint.

The API accepts JPEG and PNG images up to 10 MiB and 20 megapixels. It reports four
labels, their confidence scores, model provenance, and model prediction time. That
time excludes HTTP transport and image decoding. Use the benchmark command in
the README to measure latency consistently.

Stop the API with `Ctrl+C` in its terminal, or run `docker stop fashion-api`. The
`--rm` option removes that container after it stops; the built image remains
available locally.

For a public deployment, configure HTTPS, authentication, request rate limits,
monitoring, and the platform's checkpoint storage. These local commands alone do
not support a claim that the project was deployed publicly.

## Validation performed

The final image was built and tested in a temporary local CPU container. Its
mounted synthetic-demo checkpoint returned HTTP 200 for `/health` and `/predict`,
with all four prediction tasks. The test verified a nonroot process, a read-only
checkpoint mount, and no build proxy or CA bindings in the runtime image. The
temporary container was then stopped and removed; the local image remains.

`reports/docker_smoke.json` records the image ID and test results. This is a
functional test with one synthetic image, separate from the warmed latency
benchmark and real-dataset validation.
