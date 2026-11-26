"""Gunicorn config — KL ERP backend.

Run:  gunicorn -c gunicorn_conf.py main:app
"""

import multiprocessing
import os

bind = f"0.0.0.0:{os.environ.get('PORT', '8000')}"

workers = int(os.environ.get("WEB_CONCURRENCY", str(multiprocessing.cpu_count() * 2 + 1)))
worker_class = "uvicorn.workers.UvicornWorker"

timeout = 120
graceful_timeout = 30
keepalive = 5

max_requests = int(os.environ.get("MAX_REQUESTS", "0"))
max_requests_jitter = 0

accesslog = "-"
errorlog = "-"
loglevel = "info"
