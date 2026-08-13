"""Gunicorn config — KL ERP backend.

Run:  gunicorn -c gunicorn_conf.py main:app
"""

import os

bind = f"0.0.0.0:{os.environ.get('PORT', '8000')}"

# I/O-bound scraping workload: 2 async workers is plenty on a t3.micro/small.
# Override with WEB_CONCURRENCY.
workers = int(os.environ.get("WEB_CONCURRENCY", "2"))
worker_class = "uvicorn.workers.UvicornWorker"

# ERP flows are multi-step (login -> captcha -> POST); never kill a worker
# mid-flight. 120s comfortably covers worst-case retry loops.
timeout = 120
graceful_timeout = 30
keepalive = 5

# Recycle workers periodically to bound memory (ONNX runtime et al).
max_requests = 2000
max_requests_jitter = 200

# Log to stdout/stderr -> captured by journald under systemd.
accesslog = "-"
errorlog = "-"
loglevel = "info"


def on_starting(server):
    """Pre-create / locate the AWS API Gateway proxies ONCE in the gunicorn
    master, before forking workers.

    Without this, N workers starting simultaneously could race in
    requests-ip-rotator's start() and create duplicate gateways per region
    (both check "does one exist?" before either finishes creating). Each
    worker's own lifespan still calls start() afterwards, which then finds
    the existing gateways by name and just populates its endpoint list.
    """
    from requests_ip_rotator import ApiGateway

    site = os.environ.get("ERP_BASE_URL", "https://newerp.kluniversity.in")
    regions = [
        r.strip()
        for r in os.environ.get("AWS_GATEWAY_REGIONS", "ap-south-1,ap-south-2").split(",")
        if r.strip()
    ]

    server.log.info(f"[BOOT] Ensuring API Gateways exist for {site} in {regions} ...")
    gateway = ApiGateway(
        site,
        regions=regions,
        access_key_id=os.environ.get("AWS_ACCESS_KEY_ID") or None,
        access_key_secret=os.environ.get("AWS_SECRET_ACCESS_KEY") or None,
    )
    endpoints = gateway.start()
    if not endpoints:
        raise RuntimeError(
            "[BOOT] No API Gateway endpoints initialised — aborting boot. "
            "Check AWS credentials / IAM (apigateway:*) / region access."
        )
    server.log.info(f"[BOOT] API Gateway endpoints ready: {endpoints}")
