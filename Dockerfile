# KL ERP Backend — FastAPI + Gunicorn (direct ERP connection)
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

# curl: container healthcheck
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY main.py gunicorn_conf.py ./
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

CMD ["gunicorn", "-c", "gunicorn_conf.py", "main:app"]
