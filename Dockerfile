# KL ERP Backend — FastAPI + Gunicorn + AWS API Gateway IP rotation
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

# curl: container healthcheck, ffmpeg: HLS live radio streaming
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY main.py gateway_proxy.py gunicorn_conf.py ./
COPY model/ ./model/

# Run as non-root; pre-create the log dir owned by appuser (only effective
# when no host bind mount shadows it)
RUN useradd --create-home appuser \
    && mkdir -p /app/logs \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/" || exit 1

# Gunicorn master pre-creates/reuses the API Gateways (on_starting hook),
# then forks Uvicorn workers whose lifespan attaches to the same gateways.
CMD ["gunicorn", "-c", "gunicorn_conf.py", "main:app"]
