from fastapi import FastAPI, Form, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import httpx
import logging
import os
import sys
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlparse
import json
import numpy as np
from PIL import Image
import onnxruntime as ort
import io
import time
import re
import random
import asyncio
import threading
import hmac
import hashlib
import base64
import secrets
import datetime
import functools
from bs4 import BeautifulSoup
import sqlite3

# ------------------ DATA FLYWHEEL INIT ------------------
DATASET_DIR = "dataset"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
DB_PATH = os.path.join(DATASET_DIR, "captchas.db")

os.makedirs(IMAGES_DIR, exist_ok=True)

try:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS captchas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_hash TEXT UNIQUE,
                text_label TEXT,
                filename TEXT,
                created_at DATETIME
            )
        """)
except Exception as e:
    pass  # Rely on basic logging below if needed, or silently continue if DB init fails locally

from requests_ip_rotator import ApiGateway
from gateway_proxy import ApiGatewayTransport, GatewayUnavailableError, LockedEndpointTransport


class GatewayRequestError(Exception):
    """Transient network/gateway failure while talking to the ERP through
    AWS API Gateway. Triggers a full endpoint retry (each attempt egresses
    from a fresh gateway endpoint / AWS IP)."""


def with_gateway_retries(max_retries: int = 3):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except GatewayRequestError:
                    if attempt == max_retries - 1:
                        raise HTTPException(
                            status_code=503,
                            detail="University ERP portal is down or under maintenance. Please try again later."
                        )
                    logger.info(f"Retrying endpoint {func.__name__} (attempt {attempt + 2}/{max_retries})")
        return wrapper
    return decorator

# Optional MongoDB (observatory & latency telemetry). Falls back to in-memory if unavailable.
try:
    from pymongo import MongoClient, ASCENDING
    _pymongo_available = True
except ImportError:
    _pymongo_available = False
    logger = logging.getLogger(__name__)
    logger.warning("pymongo not installed — latency observatory will use in-memory store.")

# ------------------ LOGGING WITH TIME SEEDS & FILE PERSISTENCE ------------------
log_format_string = "%(asctime)s.%(munit)s [%(levelname)s] %(message)s"
log_date_format = "%Y-%m-%d %H:%M:%S"

log_formatter = logging.Formatter(fmt=log_format_string, datefmt=log_date_format)

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)

_log_handlers = [console_handler]
try:
    os.makedirs("logs", exist_ok=True)
    file_handler = logging.FileHandler("logs/production_api.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(log_formatter)
    _log_handlers.append(file_handler)
except OSError as log_err:
    # Never let log-file permissions kill the app (e.g. a root-owned Docker
    # bind mount) — degrade to console-only logging instead.
    print(f"[WARN] Log file unavailable, console-only logging: {log_err}")

logging.basicConfig(
    level=logging.INFO,
    handlers=_log_handlers
)

old_factory = logging.getLogRecordFactory()
def record_factory(*args, **kwargs):
    record = old_factory(*args, **kwargs)
    record.munit = f"{int(record.msecs):03d}"
    return record
logging.setLogRecordFactory(record_factory)

logger = logging.getLogger(__name__)

# ------------------ CAPTCHA SOLVER INIT ------------------
try:
    with open("model/crnn.json", "r") as f:
        _captcha_meta = json.load(f)
    _captcha_alphabet = _captcha_meta["alphabet"]
    _captcha_img_w = _captcha_meta["img_w"]
    _captcha_img_h = _captcha_meta["img_h"]
    _captcha_session = ort.InferenceSession("model/crnn.onnx")
except Exception as e:
    logger.error(f"Warning: Failed to load captcha model: {e}")

def solve_captcha(image_bytes: bytes) -> str:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    img = img.resize((_captcha_img_w, _captcha_img_h))
    alpha = np.array(img)[:, :, 3].astype(np.float32) / 255.0
    tensor = alpha[np.newaxis, np.newaxis, :, :]
    ort_inputs = {_captcha_session.get_inputs()[0].name: tensor}
    logits = _captcha_session.run(None, ort_inputs)[0][0]
    T, C = logits.shape
    out = []
    last = -1
    for t in range(T):
        best = int(np.argmax(logits[t]))
        if best != last and best != 0:
            out.append(_captcha_alphabet[best - 1])
        last = best
    return "".join(out)

# ------------------ STRUCTURAL STATICS (BROWSER FINGERPRINT) ------------------
BASE_URL = os.environ.get("ERP_BASE_URL", "https://newerp.kluniversity.in")

CHROME_VERSIONS = ["124.0.0.0", "125.0.0.0", "126.0.0.0"]
SELECTED_VERSION = random.choice(CHROME_VERSIONS)

DEFAULT_HEADERS = {
    "User-Agent": f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{SELECTED_VERSION} Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-IN,en-GB;q=0.9,en-US;q=0.8",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Connection": "keep-alive",
    "Cache-Control": "max-age=0",
    "Sec-Ch-Ua": f'"Google Chrome";v="{SELECTED_VERSION.split(".")[0]}", "Chromium";v="{SELECTED_VERSION.split(".")[0]}", "Not-A.Brand";v="99"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1"
}

# ------------------ AWS API GATEWAY IP-ROTATION PROXY ------------------
# The API Gateway infrastructure is managed by requests-ip-rotator and is
# started ONCE in the FastAPI lifespan below. Gateways are reused by name
# across restarts (start() finds pre-existing APIs instead of recreating).
#
# Auth resolution order (boto3 default chain): these env vars, then the EC2
# instance IAM role. An instance role is recommended — leave the env vars
# unset when one is attached.
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID") or None
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY") or None

# India regions only — keeps gateway egress geographically close to the ERP.
GATEWAY_REGIONS = [
    r.strip()
    for r in os.environ.get("AWS_GATEWAY_REGIONS", "ap-south-1,ap-south-2").split(",")
    if r.strip()
]

# Gateways cost ~$0 idle (pay-per-request) and start() reuses them by name,
# so the production default is to KEEP them on shutdown for fast restarts.
DELETE_GATEWAYS_ON_SHUTDOWN = (
    os.environ.get("DELETE_GATEWAYS_ON_SHUTDOWN", "false").strip().lower() == "true"
)

gateway_manager: ApiGateway | None = None


def get_gateway_endpoints() -> list[str]:
    """Live list of gateway endpoint hosts, read lazily per request."""
    if gateway_manager is None:
        return []
    return gateway_manager.endpoints


async def log_rate_limit(response: httpx.Response):
    """httpx response event hook. Logs every 429 the ERP returns, tagged with
    the gateway endpoint that egressed the request, so throttling is
    attributable to a specific API Gateway instead of guessed."""
    if response.status_code == 429:
        logger.warning(
            f"[RATE_LIMIT] status=429 via_gateway={response.url.host} url={response.url}"
        )


def make_erp_client(http2: bool = True, **overrides) -> httpx.AsyncClient:
    """One AsyncClient per logical operation. Every request is routed through
    a random AWS API Gateway endpoint (rotating AWS egress IPs) by the
    ApiGatewayTransport. TLS terminates at AWS with a valid ACM cert, so
    verification stays enabled."""
    options = dict(
        headers=DEFAULT_HEADERS,
        transport=ApiGatewayTransport(get_gateway_endpoints, http2=http2),
        event_hooks={"response": [log_rate_limit]},
        timeout=30.0,
    )
    options.update(overrides)
    return httpx.AsyncClient(**options)

# ------------------ GLOBAL CONNECTION LIFESPAN ------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global gateway_manager, _main_asyncio_loop
    try:
        _main_asyncio_loop = asyncio.get_running_loop()
    except Exception:
        pass
    logger.info("✅ FastAPI app starting (AWS API Gateway IP-Rotation Engine)...")

    # Start gateways ONCE for the whole process lifetime. boto3 calls are
    # blocking, so they run in a thread — never inside a request handler.
    gateway_manager = ApiGateway(
        BASE_URL,
        regions=GATEWAY_REGIONS,
        access_key_id=AWS_ACCESS_KEY_ID,
        access_key_secret=AWS_SECRET_ACCESS_KEY,
    )
    endpoints = await asyncio.to_thread(gateway_manager.start)
    if not endpoints:
        raise RuntimeError(
            f"No AWS API Gateway endpoints could be initialised in {GATEWAY_REGIONS}. "
            "Check AWS credentials / IAM permissions (apigateway:*) and region access."
        )
    logger.info(
        f"🚀 {len(endpoints)} API Gateway endpoint(s) live in {GATEWAY_REGIONS}: {endpoints}"
    )

    yield

    if DELETE_GATEWAYS_ON_SHUTDOWN:
        logger.info("🛑 Deleting API Gateways (DELETE_GATEWAYS_ON_SHUTDOWN=true)...")
        await asyncio.to_thread(gateway_manager.shutdown)
    else:
        logger.info("🛑 App stopped. API Gateways left in place (reused by name on next start).")

app = FastAPI(title="TimeTable & Attendance Backend", version="8.0.0", lifespan=lifespan)

# ------------------ CORS ------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Session-ID"],
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    body = await request.body()
    logger.error(f"422 Validation Error on {request.url.path}. Body: {body.decode('utf-8', 'ignore')}. Errors: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "message": "App version outdated. Please clear cache and refresh the page to update.",
            "detail": "App version outdated. Please clear cache and refresh the page to update."
        },
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"500 Internal Error on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "detail": f"Internal server fault: {str(exc)}",
            "message": "Temporary server error. Please try again."
        },
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )

# ------------------ HEALTH ------------------
@app.get("/")
def health():
    endpoints = get_gateway_endpoints()
    return {
        "message": "Backend running high-speed concurrent loops ✅",
        "status": "healthy",
        "gateway": {
            "regions": GATEWAY_REGIONS,
            "endpoints_live": len(endpoints),
        },
    }

# ------------------ DATA FLYWHEEL STATS ------------------
@app.get("/flywheel-stats")
def get_flywheel_stats():
    """Secure endpoint to check how many CAPTCHAs have been collected.
    Returns only the total count to protect privacy and dataset security."""
    try:
        with sqlite3.connect(DB_PATH, timeout=5.0) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM captchas")
            count = cursor.fetchone()[0]
        return {
            "success": True,
            "total_collected": count,
            "message": "Data Flywheel is actively deduplicating and collecting."
        }
    except Exception as e:
        logger.error(f"[FLYWHEEL] Stats endpoint error: {e}")
        return JSONResponse(status_code=500, content={"success": False, "detail": "Stats unavailable at the moment."})

# ------------------ UTILS ------------------
def is_login_failed(response: httpx.Response) -> bool:
    url_str = str(response.url)
    if "site%2Flogin" in url_str or "site/login" in url_str:
        return True
    if "LoginForm[username]" in response.text or "LoginForm[password]" in response.text:
        return True
    if "<h4" in response.text and "Login" in response.text:
        return True
    return False

def extract_csrf(html: str) -> str:
    m = re.search(r'name="csrf-token"\s+content="([^"]+)"', html)
    if m:
        return m.group(1)
    m = re.search(r'<input[^>]+name="_csrf"[^>]+value="([^"]+)"', html)
    if m:
        return m.group(1)
    m = re.search(r'<input[^>]+value="([^"]+)"[^>]+name="_csrf"', html)
    if m:
        return m.group(1)
    return ""

def collect_cookies(response: httpx.Response, base: dict) -> dict:
    merged = dict(base)
    for header_val in response.headers.get_list("set-cookie"):
        part = header_val.split(";")[0].strip()
        if "=" in part:
            k, v = part.split("=", 1)
            merged[k.strip()] = v.strip()
    return merged

async def _follow_redirects_collecting_cookies(
    client: httpx.AsyncClient, method: str, url: str, step_cookies: dict, timeout: int = 30, **kwargs
) -> tuple[httpx.Response, dict]:
    current_url = url
    current_cookies = dict(step_cookies)
    max_redirects = 10

    for _ in range(max_redirects):
        if method == "POST":
            resp = await client.post(
                current_url, cookies=current_cookies,
                follow_redirects=False, timeout=timeout, **kwargs
            )
        else:
            resp = await client.get(
                current_url, cookies=current_cookies,
                follow_redirects=False, timeout=timeout, **kwargs
            )

        current_cookies = collect_cookies(resp, current_cookies)

        if resp.status_code in (301, 302, 303, 307, 308):
            location = resp.headers.get("location", "")
            if not location:
                break
            if location.startswith("/"):
                parsed = urlparse(current_url)
                location = f"{parsed.scheme}://{parsed.netloc}{location}"
            elif not location.startswith("http"):
                location = BASE_URL + "/" + location

            # Normalise scheme in case the ERP emits absolute http:// redirect targets.
            if location.startswith("http://newerp.kluniversity.in:443"):
                location = location.replace("http://newerp.kluniversity.in:443", "https://newerp.kluniversity.in")
            elif location.startswith("http://newerp.kluniversity.in"):
                location = location.replace("http://newerp.kluniversity.in", "https://newerp.kluniversity.in")

            current_url = location
            method = "GET"
            kwargs = {}
        else:
            return resp, current_cookies

    return resp, current_cookies

# ------------------ AUTO LOGIN (FIREWALL BYPASS ENGINE) ------------------
async def auto_login(client: httpx.AsyncClient, username: str, password: str, seed_cookies: dict) -> tuple[httpx.Response, dict]:
    login_url = f"{BASE_URL}/index.php?r=site%2Flogin"
    logger.info(f"[LOGIN] Running thread-isolated ONNX auto-login for user={username}")

    local_headers = dict(DEFAULT_HEADERS)

    # Step 1: Initial Cold Handshake
    res, step_cookies = await _follow_redirects_collecting_cookies(client, "GET", login_url, {}, headers=local_headers)
    res.raise_for_status()

    csrf = extract_csrf(res.text)
    if not csrf:
        raise Exception("CSRF token not found on login page.")

    # Step 2: Mimic human browser delay processing layout
    await asyncio.sleep(random.uniform(0.1, 0.2))

    dummy_data = {"_csrf": csrf, "LoginForm[username]": "", "LoginForm[password]": ""}

    local_headers["Origin"] = BASE_URL
    local_headers["Referer"] = login_url
    local_headers["Sec-Fetch-Site"] = "same-origin"
    local_headers["Sec-Fetch-Mode"] = "cors"
    local_headers["Sec-Fetch-Dest"] = "empty"

    res_post, step_cookies = await _follow_redirects_collecting_cookies(
        client, "POST", login_url, step_cookies, data=dummy_data, headers=local_headers
    )
    res_post.raise_for_status()

    captcha_match = re.search(r'src="([^"]*?r=site%2Fcaptcha[^"]*?)"', res_post.text)
    if not captcha_match:
        raise Exception("CAPTCHA image locator missing from layout.")

    # Step 3: Pull Captcha Image Layer
    captcha_url = BASE_URL + captcha_match.group(1).replace("&amp;", "&")

    local_headers["Accept"] = "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8"
    local_headers["Sec-Fetch-Mode"] = "no-cors"
    local_headers["Sec-Fetch-Dest"] = "image"
    local_headers["Referer"] = login_url

    captcha_response, step_cookies = await _follow_redirects_collecting_cookies(
        client, "GET", captcha_url, step_cookies, headers=local_headers
    )
    captcha_response.raise_for_status()

    # Step 4: Run High-Speed ONNX Calculation
    captcha_text = solve_captcha(captcha_response.content)
    logger.info(f"[LOGIN] Captcha solved: {captcha_text}")

    # Step 5: Final Submission with Re-calibrated Headers
    payload = {
        "_csrf": csrf,
        "LoginForm[username]": username,
        "LoginForm[password]": password,
        "LoginForm[captcha]": captcha_text,
        "LoginForm[rememberMe]": "0",
        "LoginForm[qr_code]": "",
    }

    local_headers["Accept"] = "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8"
    local_headers["Sec-Fetch-Mode"] = "navigate"
    local_headers["Sec-Fetch-Dest"] = "document"
    local_headers["Sec-Fetch-User"] = "?1"

    # Anti-bot computational delay padding signature
    await asyncio.sleep(random.uniform(0.05, 0.15))

    response, final_cookies = await _follow_redirects_collecting_cookies(
        client, "POST", login_url, step_cookies, data=payload, headers=local_headers
    )
    response.raise_for_status()

    # --- DATA FLYWHEEL LOGIC ---
    if not is_login_failed(response):
        try:
            img_hash = hashlib.md5(captcha_response.content).hexdigest()
            filename = f"{captcha_text}_{img_hash}.png"
            timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            
            with sqlite3.connect(DB_PATH, timeout=5.0) as conn:
                conn.execute(
                    "INSERT INTO captchas (image_hash, text_label, filename, created_at) VALUES (?, ?, ?, ?)",
                    (img_hash, captcha_text, filename, timestamp)
                )
            
            with open(os.path.join(IMAGES_DIR, filename), "wb") as f:
                f.write(captcha_response.content)
            logger.info(f"[FLYWHEEL] Saved new verified captcha: {filename}")
        except sqlite3.IntegrityError:
            pass  # Duplicate hash, silently ignore
        except Exception as e:
            logger.error(f"[FLYWHEEL] Error saving dataset: {e}")
    # ---------------------------

    for key in ("kl_erp_device_id", "SERVERID"):
        if key not in final_cookies and key in seed_cookies:
            final_cookies[key] = seed_cookies[key]

    return response, final_cookies

def build_register_url(base_url: str, href: str) -> str | None:
    try:
        full_relative_path = href.split(base_url)[-1]
        register_url_segment = full_relative_path.split('?r=')[-1]
        r_param_end = register_url_segment.find('&')
        if r_param_end != -1:
            r_path = unquote(register_url_segment[:r_param_end])
            params_raw = register_url_segment[r_param_end:]
            return f"{base_url}/index.php?r={r_path}{params_raw}"
        return f"{base_url}/index.php?r={unquote(register_url_segment)}"
    except Exception as e:
        logger.error(f"[REGISTER_URL] Reconstruct error: {e}")
        return None

# ------------------ SINGLE-FLIGHT COALESCING & 30s RAM CACHE ------------------
class SingleFlightCache:
    """Zero-overhead in-memory SingleFlight request coalescing and RAM cache.
    - If identical requests arrive concurrently, only 1 request hits the university ERP;
      all other callers await the leader and share the result.
    - Successful responses are cached for ttl_seconds (default 30s).
    - Cache hits resolve in < 1ms and log sub-millisecond telemetry.
    - Protects the university portal & AWS Gateways from DoS and spam loops without
      affecting legitimate users.
    """
    def __init__(self, ttl_seconds: float = 30.0):
        self.ttl = ttl_seconds
        self._cache: dict = {}      # key -> (expiry_monotonic, data_dict)
        self._inflight: dict = {}   # key -> asyncio.Task
        self._lock = asyncio.Lock()

    def _cleanup_inflight(self, key: tuple):
        self._inflight.pop(key, None)

    async def execute(self, key: tuple, coro_func, route_name: str, start_time: float):
        now = time.monotonic()

        # 1. Fast path: Check RAM cache
        cached = self._cache.get(key)
        if cached is not None:
            expiry, data = cached
            if now < expiry:
                hit_latency = (time.time() - start_time) * 1000.0
                record_latency(route_name, hit_latency, 200)
                student_id = key[1] if len(key) > 1 else "unknown"
                logger.info(f"[{route_name.upper()}] Cache hit for {student_id} ({hit_latency:.2f}ms)")
                return data.copy()

        # 2. Check or create SingleFlight task
        leader = False
        async with self._lock:
            task = self._inflight.get(key)
            if task is None:
                task = asyncio.create_task(coro_func())
                self._inflight[key] = task
                task.add_done_callback(lambda _: self._cleanup_inflight(key))
                leader = True

        student_id = key[1] if len(key) > 1 else "unknown"
        if not leader:
            logger.info(f"[{route_name.upper()}] Coalescing concurrent request for {student_id} via SingleFlight")

        result = await asyncio.shield(task)
        if leader and isinstance(result, dict) and result.get("success"):
            self._cache[key] = (time.monotonic() + self.ttl, result)
            self._prune_expired()
        return result.copy() if isinstance(result, dict) else result

    def _prune_expired(self):
        if len(self._cache) > 2000:
            now = time.monotonic()
            expired = [k for k, (exp, _) in self._cache.items() if now >= exp]
            for k in expired:
                self._cache.pop(k, None)

    def clear(self):
        self._cache.clear()
        self._inflight.clear()

erp_cache = SingleFlightCache(ttl_seconds=30.0)

def single_flight_cached(route_name: str, key_builder, ttl_seconds: float = 30.0):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                cache_key = key_builder(kwargs)
            except Exception:
                return await func(*args, **kwargs)
            return await erp_cache.execute(cache_key, lambda: func(*args, **kwargs), route_name, start_time)
        return wrapper
    return decorator


# ------------------ LOGIN ENDPOINT ------------------
@app.post("/login")
@with_gateway_retries(max_retries=3)
@single_flight_cached(
    route_name="login",
    key_builder=lambda kwargs: (
        "login",
        kwargs.get("username", "").strip(),
        hashlib.sha256(kwargs.get("password", "").encode("utf-8")).hexdigest()[:16]
    )
)
async def login(username: str = Form(...), password: str = Form(...)):
    try:
        async with make_erp_client() as client:
            login_response = None
            fresh_cookies = {}
            for attempt in range(3):
                if attempt > 0:
                    sleep_time = random.uniform(1.0, 2.5)
                    logger.info(f"[LOGIN] Backoff waiting {sleep_time:.2f}s before retry context split...")
                    await asyncio.sleep(sleep_time)
                login_response, fresh_cookies = await auto_login(client, username, password, seed_cookies={})
                if not is_login_failed(login_response):
                    break
                logger.warning(f"[LOGIN] Attempt {attempt+1} rejected. Retrying captcha.")
            else:
                raise HTTPException(status_code=401, detail="Invalid credentials or captcha timeout.")

            fresh_csrf = extract_csrf(login_response.text)
            return {
                "success": True,
                "message": "Cookies generated successfully.",
                "cookies": {
                    "PHPSESSID": fresh_cookies.get("PHPSESSID"),
                    "kl_erp_device_id": fresh_cookies.get("kl_erp_device_id"),
                    "SERVERID": fresh_cookies.get("SERVERID", "erp3"),
                    "_csrf_token": fresh_csrf,
                    "_csrf": fresh_csrf
                }
            }
    except HTTPException:
        raise
    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.NetworkError, httpx.TimeoutException, httpx.ProxyError, httpx.HTTPStatusError, GatewayUnavailableError) as net_err:
        if isinstance(net_err, httpx.HTTPStatusError) and net_err.response.status_code not in (502, 503):
            raise
        logger.error(f"[NETWORK REJECTION] /login - University gateway down: {net_err}")
        raise GatewayRequestError("ERP request failed via API Gateway")
    except Exception as e:
        logger.error(f"[LOGIN_ROUTE] Exception: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal processing fault during authorization sync.")

# ------------------ FETCH ATTENDANCE ------------------
@app.post("/fetch-attendance")
@with_gateway_retries(max_retries=3)
@single_flight_cached(
    route_name="attendance",
    key_builder=lambda kwargs: (
        "attendance",
        kwargs.get("username", "").strip(),
        str(kwargs.get("academic_year_code", "")).strip(),
        str(kwargs.get("semester_id", "")).strip()
    )
)
async def fetch_attendance_summary(
    username: str = Form(...),
    password: str = Form(...),
    php_sess_id: str = Form(default=""),
    csrf_cookie: str = Form(default=""),
    device_id: str = Form(default=""),
    server_id: str = Form(default="erp3"),
    academic_year_code: str = Form(...),
    semester_id: str = Form(...)
):
    start_time = time.time()
    cookie_jar = {
        "_csrf": unquote(csrf_cookie) if csrf_cookie else "",
        "PHPSESSID": php_sess_id,
        "kl_erp_device_id": unquote(device_id) if device_id else "",
        "SERVERID": server_id
    }
    attendance_url = f"{BASE_URL}/index.php?r=studentattendance%2Fstudentdailyattendance%2Fcourselist"

    def _make_payload(csrf: str) -> dict:
        return {
            "_csrf": csrf,
            "DynamicModel[academicyear]": academic_year_code,
            "DynamicModel[semesterid]": semester_id,
        }

    try:
        async with make_erp_client() as client:

            if not php_sess_id or not csrf_cookie:
                logger.info(f"[ATTENDANCE] No session cookies — running cold-start auto-login for {username}")
                for attempt in range(3):
                    if attempt > 0:
                        sleep_time = random.uniform(1.0, 2.5)
                        logger.info(f"[LOGIN] Backoff waiting {sleep_time:.2f}s before retry...")
                        await asyncio.sleep(sleep_time)
                    login_response, cookie_jar = await auto_login(client, username, password, seed_cookies={})
                    if not is_login_failed(login_response):
                        break
                else:
                    raise HTTPException(status_code=401, detail="Cold-start login failed. Check credentials.")
                php_sess_id = cookie_jar.get("PHPSESSID", "")
                page_csrf = extract_csrf(login_response.text)
            else:
                attendance_landing = f"{BASE_URL}/index.php?r=studentattendance%2Fstudentdailyattendance"
                logger.info(f"[ATTENDANCE] GET landing page for fresh CSRF (PHPSESSID={php_sess_id[:6]}...)")
                get_response, cookie_jar = await _follow_redirects_collecting_cookies(
                    client, "GET", attendance_landing, cookie_jar, timeout=15
                )

                if is_login_failed(get_response):
                    logger.warning("[ATTENDANCE] Session expired on GET. Running auto-healer...")
                    for attempt in range(3):
                        if attempt > 0:
                            await asyncio.sleep(random.uniform(1.0, 2.0))
                        login_response, cookie_jar = await auto_login(client, username, password, seed_cookies=cookie_jar)
                        if not is_login_failed(login_response):
                            break
                    else:
                        raise HTTPException(status_code=401, detail="ERP system rejected fallback login.")
                    page_csrf = extract_csrf(login_response.text)
                else:
                    page_csrf = extract_csrf(get_response.text)

            if not page_csrf:
                raise HTTPException(status_code=500, detail="Could not extract CSRF token from attendance page.")

            logger.info(f"[ATTENDANCE] Fresh CSRF extracted. Submitting POST to courselist...")

            post_response, cookie_jar = await _follow_redirects_collecting_cookies(
                client, "POST", attendance_url, cookie_jar, timeout=15,
                data=_make_payload(page_csrf)
            )

            if is_login_failed(post_response):
                logger.warning("[ATTENDANCE] Session expired on POST. Running auto-healer...")
                for attempt in range(3):
                    if attempt > 0:
                        await asyncio.sleep(random.uniform(1.0, 2.0))
                    login_response, cookie_jar = await auto_login(client, username, password, seed_cookies=cookie_jar)
                    if not is_login_failed(login_response):
                        break
                else:
                    raise HTTPException(status_code=401, detail="ERP system rejected fallback login.")

                page_csrf = extract_csrf(login_response.text)
                if not page_csrf:
                    raise HTTPException(status_code=500, detail="Could not reconcile session CSRF signatures.")

                post_response, cookie_jar = await _follow_redirects_collecting_cookies(
                    client, "POST", attendance_url, cookie_jar, timeout=15,
                    data=_make_payload(page_csrf)
                )

            post_response.raise_for_status()
            html_content = post_response.text

        table_match = re.search(r'<table.*?>(.*?)</table>', html_content, re.DOTALL | re.IGNORECASE)
        if not table_match:
            raise ValueError("Attendance table layout structure unverified.")

        table_body = table_match.group(1)
        tbody_match = re.search(r'<tbody.*?>(.*?)</tbody>', table_body, re.DOTALL | re.IGNORECASE)
        if not tbody_match:
            return {"success": True, "attendance": [], "message": "Attendance arrays are empty."}

        raw_rows = re.findall(r'<tr.*?>(.*?)</tr>', tbody_match.group(1), re.DOTALL | re.IGNORECASE)
        attendance_data = []

        for row in raw_rows:
            cells = re.findall(r'<td.*?>(.*?)</td>', row, re.DOTALL | re.IGNORECASE)
            if not cells or len(cells) < 14:
                continue
            href_match = re.search(r'href=["\'](.*?)["\']', cells[13], re.IGNORECASE)
            raw_href = href_match.group(1) if href_match else None
            clean_href = raw_href.replace("&amp;", "&") if raw_href else None

            attendance_data.append({
                "index": re.sub(r'<.*?>', '', cells[0]).strip(),
                "course_code": re.sub(r'<.*?>', '', cells[1]).strip(),
                "course_name": re.sub(r'<.*?>', '', cells[2]).strip(),
                "type": re.sub(r'<.*?>', '', cells[3]).strip(),
                "section": re.sub(r'<.*?>', '', cells[4]).strip(),
                "academic_year": re.sub(r'<.*?>', '', cells[5]).strip(),
                "semester": re.sub(r'<.*?>', '', cells[6]).strip(),
                "conducted": re.sub(r'<.*?>', '', cells[8]).strip(),
                "attended": re.sub(r'<.*?>', '', cells[9]).strip(),
                "absent": re.sub(r'<.*?>', '', cells[10]).strip(),
                "percentage": re.sub(r'<.*?>', '', cells[12]).strip(),
                "register_href": clean_href
            })

        updated_session_id = cookie_jar.get("PHPSESSID")
        has_refreshed = updated_session_id != php_sess_id
        final_csrf = cookie_jar.get("_csrf", page_csrf)

        logger.info(f"[ATTENDANCE] Fetch loop successful. Refreshed Status: {has_refreshed} in {time.time() - start_time:.3f}s")
        record_latency("attendance", (time.time() - start_time) * 1000.0, 200)
        return {
            "success": True,
            "session_refreshed": has_refreshed,
            "cookies": {
                "PHPSESSID": updated_session_id,
                "kl_erp_device_id": cookie_jar.get("kl_erp_device_id", device_id),
                "SERVERID": cookie_jar.get("SERVERID", server_id),
                "_csrf_token": final_csrf,
                "_csrf": final_csrf
            },
            "attendance": attendance_data
        }
    except HTTPException:
        raise
    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.NetworkError, httpx.TimeoutException, httpx.ProxyError, httpx.HTTPStatusError, GatewayUnavailableError) as net_err:
        if isinstance(net_err, httpx.HTTPStatusError) and net_err.response.status_code not in (502, 503):
            raise
        logger.error(f"[NETWORK REJECTION] /fetch-attendance - University gateway down: {net_err}")
        raise GatewayRequestError("ERP request failed via API Gateway")
    except Exception as e:
        logger.error(f"[ATTENDANCE] Crash: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ------------------ FETCH REGISTER DETAILS ------------------
@app.post("/fetch-register-detail")
@with_gateway_retries(max_retries=3)
async def fetch_register_details(
    username: str = Form(...),
    password: str = Form(...),
    php_sess_id: str = Form(default=""),
    csrf_cookie: str = Form(default=""),
    device_id: str = Form(default=""),
    server_id: str = Form(default="erp3"),
    register_href: str = Form(...)
):
    start_time = time.time()
    register_url = build_register_url(BASE_URL, register_href)
    if not register_url:
        raise HTTPException(status_code=400, detail="Target path failure.")

    cookie_jar = {
        "_csrf": unquote(csrf_cookie) if csrf_cookie else "",
        "PHPSESSID": php_sess_id,
        "kl_erp_device_id": unquote(device_id) if device_id else "",
        "SERVERID": server_id
    }

    try:
        # TWO-CLIENT STRATEGY:
        # 1. Login goes through AWS API Gateway → protects host IP from 429 rate limiting
        # 2. Register fetch goes direct → avoids gateway corrupting binary params (%F5%B6...)
        #
        # ERP sessions are NOT IP-bound (confirmed: phone sessions survive network switches).
        # The earlier 302 was caused by SERVERID (sticky-session) not being captured from
        # the gateway login response, causing the load balancer to route the fetch to a
        # different backend server that didn't have the session. We now explicitly default
        # SERVERID to "erp1" if the gateway response didn't return one.

        if not php_sess_id or not csrf_cookie:
            logger.info(f"[LAZY-REGISTER] Cold-start auto-login via gateway for {username}")
            async with make_erp_client() as gw_client:
                for attempt in range(3):
                    if attempt > 0:
                        await asyncio.sleep(random.uniform(1.0, 2.0))
                    login_response, cookie_jar = await auto_login(gw_client, username, password, seed_cookies={})
                    if not is_login_failed(login_response):
                        break
                else:
                    raise HTTPException(status_code=401, detail="Cold-start login failed. Check credentials.")
            active_csrf = extract_csrf(login_response.text)
            php_sess_id = cookie_jar.get("PHPSESSID", "")
            # Ensure SERVERID is set — if gateway didn't return one, default to erp1
            # so the load balancer routes the fetch to the correct sticky backend server.
            if not cookie_jar.get("SERVERID"):
                cookie_jar["SERVERID"] = "erp1"
        else:
            active_csrf = unquote(csrf_cookie)

        # Register fetch goes direct — gateway corrupts binary params in the URL
        async with httpx.AsyncClient(
            verify=False, headers=DEFAULT_HEADERS, http2=True,
            event_hooks={"response": [log_rate_limit]}
        ) as direct_client:
            register_url_with_csrf = f"{register_url}&_csrf={active_csrf}"
            response = await direct_client.get(register_url_with_csrf, cookies=cookie_jar, timeout=15)

            if response.status_code in (301, 302, 303) or response.status_code == 500 or is_login_failed(response):
                logger.warning("[LAZY-REGISTER] Session invalid. Auto-healing via gateway...")
                async with make_erp_client() as gw_client:
                    for attempt in range(3):
                        if attempt > 0:
                            await asyncio.sleep(random.uniform(1.0, 2.0))
                        login_response, cookie_jar = await auto_login(gw_client, username, password, seed_cookies=cookie_jar)
                        if not is_login_failed(login_response):
                            break
                    else:
                        raise HTTPException(status_code=401, detail="Authentication credentials expired.")

                if not cookie_jar.get("SERVERID"):
                    cookie_jar["SERVERID"] = "erp1"
                active_csrf = extract_csrf(login_response.text) or cookie_jar.get("_csrf", "")
                register_url_with_csrf = f"{register_url}&_csrf={active_csrf}"
                response = await direct_client.get(register_url_with_csrf, cookies=cookie_jar, timeout=15)

            response.raise_for_status()
            html_text = response.text




        try:
            soup = BeautifulSoup(html_text, "lxml")
        except Exception:
            soup = BeautifulSoup(html_text, "html.parser")
        table = soup.find("table", class_=lambda c: c and "table-striped" in c and "table-bordered" in c)
        if not table:
            return {"success": False, "message": "Register table missing."}

        headers = [th.get_text(strip=True) for th in table.find_all("th") if th.get_text(strip=True)]

        metadata_count = 14
        metadata_headers = headers[:metadata_count]
        daily_headers = headers[metadata_count:]

        tbody = table.find("tbody")
        if not tbody:
            return {"success": False, "message": "Calendar data rows missing."}

        cells = [td.get_text(strip=True) for td in tbody.find_all("td")]

        if len(cells) < metadata_count:
            logger.warning(f"[LAZY-REGISTER] Only {len(cells)} cells, expected {metadata_count}+")
            return {"success": False, "message": "Truncated layout array returns."}

        metadata = {header: cells[i] for i, header in enumerate(metadata_headers) if i < len(cells)}

        daily_attendance = [
            {"date_slot": header, "status": cells[metadata_count + i]}
            for i, header in enumerate(daily_headers)
            if metadata_count + i < len(cells)
        ]


        updated_session_id = cookie_jar.get("PHPSESSID")
        has_refreshed = updated_session_id != php_sess_id
        final_csrf = cookie_jar.get("_csrf", active_csrf)

        logger.info(f"[LAZY-REGISTER] Register loop complete. Refreshed Status: {has_refreshed} in {time.time() - start_time:.3f}s")
        return {
            "success": True,
            "session_refreshed": has_refreshed,
            "cookies": {
                "PHPSESSID": updated_session_id,
                "kl_erp_device_id": cookie_jar.get("kl_erp_device_id", device_id),
                "SERVERID": cookie_jar.get("SERVERID", server_id),
                "_csrf_token": final_csrf,
                "_csrf": final_csrf
            },
            "metadata": metadata,
            "daily_attendance": daily_attendance
        }
    except HTTPException:
        raise
    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.NetworkError, httpx.TimeoutException, httpx.ProxyError, httpx.HTTPStatusError, GatewayUnavailableError) as net_err:
        if isinstance(net_err, httpx.HTTPStatusError) and net_err.response.status_code not in (502, 503):
            raise
        logger.error(f"[NETWORK REJECTION] /fetch-register-detail - University gateway down: {net_err}")
        raise GatewayRequestError("ERP request failed via API Gateway")
    except Exception as e:
        logger.error(f"[REGISTER] Crash: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ------------------ FETCH SEATING PLAN ------------------
@app.post("/fetch-seating-plan")
@with_gateway_retries(max_retries=3)
async def fetch_seating_plan(
    username: str = Form(...),
    password: str = Form(...),
    php_sess_id: str = Form(default=""),
    csrf_cookie: str = Form(default=""),
    device_id: str = Form(default=""),
    server_id: str = Form(default="erp3")
):
    start_time = time.time()
    cookie_jar = {
        "_csrf": unquote(csrf_cookie) if csrf_cookie else "",
        "PHPSESSID": php_sess_id,
        "kl_erp_device_id": unquote(device_id) if device_id else "",
        "SERVERID": server_id
    }
    seating_plan_url = f"{BASE_URL}/index.php?r=examsection%2Fexam-invigilator-student-room-allotment-info%2Fstud_my_seating_plan"

    try:
        async with make_erp_client() as client:
            if not php_sess_id or not csrf_cookie:
                logger.info(f"[SEATING] Cold-start auto-login for {username}")
                for attempt in range(3):
                    if attempt > 0:
                        await asyncio.sleep(random.uniform(1.0, 2.0))
                    login_response, cookie_jar = await auto_login(client, username, password, seed_cookies={})
                    if not is_login_failed(login_response):
                        break
                else:
                    raise HTTPException(status_code=401, detail="Cold-start login failed. Check credentials.")
                php_sess_id = cookie_jar.get("PHPSESSID", "")

            response = await client.get(seating_plan_url, cookies=cookie_jar, timeout=15)

            if response.status_code in (301, 302, 303) or response.status_code == 500 or is_login_failed(response):
                logger.warning("[SEATING] Session invalid or redirected (302). Executing tracking fallback...")
                for attempt in range(3):
                    if attempt > 0:
                        await asyncio.sleep(random.uniform(1.0, 2.0))
                    login_response, cookie_jar = await auto_login(client, username, password, seed_cookies=cookie_jar)
                    if not is_login_failed(login_response):
                        break
                else:
                    raise HTTPException(status_code=401, detail="ERP Session rejected.")

                response = await client.get(seating_plan_url, cookies=cookie_jar, timeout=15)

            response.raise_for_status()
            html_content = response.text

        table_match = re.search(r'<table.*?>(.*?)</table>', html_content, re.DOTALL | re.IGNORECASE)
        if not table_match:
            raise HTTPException(status_code=404, detail="Seating plan layout missing.")

        table_body = table_match.group(1)
        tbody_match = re.search(r'<tbody.*?>(.*?)</tbody>', table_body, re.DOTALL | re.IGNORECASE)
        if not tbody_match:
            return {"success": True, "seating_plan": [], "message": "No exam schedules mapped."}

        rows = re.findall(r'<tr.*?>(.*?)</tr>', tbody_match.group(1), re.DOTALL | re.IGNORECASE)
        seating_plan_data = []

        for row in rows:
            cells = re.findall(r'<td.*?>(.*?)</td>', row, re.DOTALL | re.IGNORECASE)
            if not cells or len(cells) < 8:
                continue
            seating_plan_data.append({
                "index": re.sub(r'<.*?>', '', cells[0]).strip(),
                "ref_id": re.sub(r'<.*?>', '', cells[1]).strip(),
                "date": re.sub(r'<.*?>', '', cells[2]).strip(),
                "exam_type": re.sub(r'<.*?>', '', cells[3]).strip(),
                "time_slot": re.sub(r'<.*?>', '', cells[4]).strip(),
                "university_id": re.sub(r'<.*?>', '', cells[5]).strip(),
                "course_code": re.sub(r'<.*?>', '', cells[6]).strip(),
                "room_no": re.sub(r'<.*?>', '', cells[7]).strip()
            })

        updated_session_id = cookie_jar.get("PHPSESSID")
        has_refreshed = updated_session_id != php_sess_id
        final_csrf = cookie_jar.get("_csrf", unquote(csrf_cookie) if csrf_cookie else "")

        logger.info(f"[SEATING] Seating plan loop complete. Refreshed Status: {has_refreshed} in {time.time() - start_time:.3f}s")
        return {
            "success": True,
            "session_refreshed": has_refreshed,
            "cookies": {
                "PHPSESSID": updated_session_id,
                "kl_erp_device_id": cookie_jar.get("kl_erp_device_id", device_id),
                "SERVERID": cookie_jar.get("SERVERID", server_id),
                "_csrf_token": final_csrf,
                "_csrf": final_csrf
            },
            "seating_plan": seating_plan_data
        }
    except HTTPException:
        raise
    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.NetworkError, httpx.TimeoutException, httpx.ProxyError, httpx.HTTPStatusError, GatewayUnavailableError) as net_err:
        if isinstance(net_err, httpx.HTTPStatusError) and net_err.response.status_code not in (502, 503):
            raise
        logger.error(f"[NETWORK REJECTION] /fetch-seating-plan - University gateway down: {net_err}")
        raise GatewayRequestError("ERP request failed via API Gateway")
    except Exception as e:
        logger.error(f"[SEATING] Crash: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ------------------ FETCH TIMETABLE ------------------
@app.post("/fetch-timetable")
@with_gateway_retries(max_retries=3)
@single_flight_cached(
    route_name="timetable",
    key_builder=lambda kwargs: (
        "timetable",
        kwargs.get("username", "").strip(),
        str(kwargs.get("academic_year_code", "19")).strip(),
        str(kwargs.get("semester_id", "1")).strip()
    )
)
async def fetch_timetable(
    username: str = Form(...),
    password: str = Form(...),
    php_sess_id: str = Form(default=""),
    csrf_cookie: str = Form(default=""),
    device_id: str = Form(default=""),
    server_id: str = Form(default="erp3"),
    academic_year_code: str = Form(default="19"),
    semester_id: str = Form(default="1")
):
    start_time = time.time()
    cookie_jar = {
        "_csrf": unquote(csrf_cookie) if csrf_cookie else "",
        "PHPSESSID": php_sess_id,
        "kl_erp_device_id": unquote(device_id) if device_id else "",
        "SERVERID": server_id
    }
    tt_url = (
        f"{BASE_URL}/index.php?r=timetables%2Funiversitymasteracademictimetableview%2Findividualstudenttimetableget"
        f"&UniversityMasterAcademicTimetableView%5Bacademicyear%5D={academic_year_code}"
        f"&UniversityMasterAcademicTimetableView%5Bsemesterid%5D={semester_id}"
    )

    try:
        async with make_erp_client() as client:
            if not php_sess_id or not csrf_cookie:
                logger.info(f"[TIMETABLE] Cold-start auto-login for {username}")
                for attempt in range(3):
                    if attempt > 0:
                        await asyncio.sleep(random.uniform(1.0, 2.0))
                    login_response, cookie_jar = await auto_login(client, username, password, seed_cookies={})
                    if not is_login_failed(login_response):
                        break
                else:
                    raise HTTPException(status_code=401, detail="Cold-start login failed. Check credentials.")
                php_sess_id = cookie_jar.get("PHPSESSID", "")

            response = await client.get(tt_url, cookies=cookie_jar, timeout=12)

            if response.status_code in (301, 302, 303) or response.status_code == 500 or is_login_failed(response):
                logger.warning("[TIMETABLE] Session invalid or redirected (302). Executing automated auto-healing...")
                for attempt in range(3):
                    if attempt > 0:
                        await asyncio.sleep(random.uniform(1.0, 2.0))
                    login_response, cookie_jar = await auto_login(client, username, password, seed_cookies=cookie_jar)
                    if not is_login_failed(login_response):
                        break
                else:
                    raise HTTPException(status_code=401, detail="ERP credentials invalid.")

                response = await client.get(tt_url, cookies=cookie_jar, timeout=15)

            response.raise_for_status()
            html_content = response.text

        table_match = re.search(r'<table.*?>(.*?)</table>', html_content, re.DOTALL | re.IGNORECASE)
        if not table_match:
            raise HTTPException(status_code=404, detail="Timetable grid missing.")

        table_body = table_match.group(1)
        thead_match = re.search(r'<thead.*?>(.*?)</thead>', table_body, re.DOTALL | re.IGNORECASE)
        if not thead_match:
            raise HTTPException(status_code=500, detail="Failed to locate timetable header.")

        raw_headers = re.findall(r'<th.*?>(.*?)</th>', thead_match.group(1), re.IGNORECASE)
        headers = [re.sub(r'<.*?>', '', h).strip() for h in raw_headers][1:]

        tbody_match = re.search(r'<tbody.*?>(.*?)</tbody>', table_body, re.DOTALL | re.IGNORECASE)
        if not tbody_match:
            return {"success": True, "timetable": {}, "message": "Timetable schedules are empty."}

        rows = re.findall(r'<tr.*?>(.*?)</tr>', tbody_match.group(1), re.DOTALL | re.IGNORECASE)
        timetable_data = {}

        for row in rows:
            cells = re.findall(r'<td.*?>(.*?)</td>', row, re.DOTALL | re.IGNORECASE)
            if not cells:
                continue
            day_name = re.sub(r'<.*?>', '', cells[0]).strip()
            slot_contents = [re.sub(r'<.*?>', '', cell).strip() for cell in cells[1:]]
            timetable_data[day_name] = dict(zip(headers, slot_contents))

        updated_session_id = cookie_jar.get("PHPSESSID")
        has_refreshed = updated_session_id != php_sess_id
        final_csrf = cookie_jar.get("_csrf", unquote(csrf_cookie) if csrf_cookie else "")

        logger.info(f"[TIMETABLE] Timetable loop complete. Refreshed Status: {has_refreshed} in {time.time() - start_time:.3f}s")
        record_latency("timetable", (time.time() - start_time) * 1000.0, 200)
        return {
            "success": True,
            "session_refreshed": has_refreshed,
            "cookies": {
                "PHPSESSID": updated_session_id,
                "kl_erp_device_id": cookie_jar.get("kl_erp_device_id", device_id),
                "SERVERID": cookie_jar.get("SERVERID", server_id),
                "_csrf_token": final_csrf,
                "_csrf": final_csrf
            },
            "timetable": timetable_data
        }
    except HTTPException:
        raise
    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.NetworkError, httpx.TimeoutException, httpx.ProxyError, httpx.HTTPStatusError, GatewayUnavailableError) as net_err:
        if isinstance(net_err, httpx.HTTPStatusError) and net_err.response.status_code not in (502, 503):
            raise
        logger.error(f"[NETWORK REJECTION] /fetch-timetable - University gateway down: {net_err}")
        raise GatewayRequestError("ERP request failed via API Gateway")
    except Exception as e:
        logger.error(f"[TIMETABLE] Crash: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/fetch-cgpa")
@with_gateway_retries(max_retries=3)
@single_flight_cached(
    route_name="cgpa",
    key_builder=lambda kwargs: (
        "cgpa",
        kwargs.get("username", "").strip()
    )
)
async def fetch_cgpa_summary(
    username: str = Form(...),
    password: str = Form(...),
    php_sess_id: str = Form(default=""),
    csrf_cookie: str = Form(default=""),
    device_id: str = Form(default=""),
    server_id: str = Form(default="erp1")
):
    start_time = time.time()
    cookie_jar = {
        "_csrf": unquote(csrf_cookie) if csrf_cookie else "",
        "PHPSESSID": php_sess_id,
        "kl_erp_device_id": unquote(device_id) if device_id else "",
        "SERVERID": server_id
    }
    cgpa_url = f"{BASE_URL}/index.php?r=studentinfo%2Fstudentendexamresult%2Fsearchgetmycgpa"

    try:
        async with make_erp_client() as client:
            if not php_sess_id or not csrf_cookie:
                logger.info(f"[CGPA] Cold start authentication trigger for {username}")
                for attempt in range(3):
                    if attempt > 0:
                        await asyncio.sleep(random.uniform(1.0, 2.0))
                    res, cookie_jar = await auto_login(client, username, password, seed_cookies={})
                    if not is_login_failed(res):
                        break
                else:
                    raise HTTPException(status_code=401, detail="Authentication initialization failed.")
                php_sess_id = cookie_jar.get("PHPSESSID", "")

            response = await client.get(cgpa_url, cookies=cookie_jar, timeout=15)

            if response.status_code in (301, 302, 303) or response.status_code == 500 or is_login_failed(response):
                logger.warning("[CGPA] Session handshake dropped. Executing auto-heal retry routine...")
                for attempt in range(3):
                    if attempt > 0:
                        await asyncio.sleep(random.uniform(1.0, 2.0))
                    res, cookie_jar = await auto_login(client, username, password, seed_cookies=cookie_jar)
                    if not is_login_failed(res):
                        break
                else:
                    raise HTTPException(status_code=401, detail="Gateway authentication dropped permanently.")

                response = await client.get(cgpa_url, cookies=cookie_jar, timeout=15)

            response.raise_for_status()
            html_content = response.text

        # Locate the core container table via regex
        # --- Corrected Pure Regex Parser Matrix ---
        table_match = re.search(r'<table.*?>(.*?)</table>', html_content, re.DOTALL | re.IGNORECASE)
        if not table_match:
            raise HTTPException(status_code=404, detail="Academic performance summary grid layout missing.")

        table_body = table_match.group(1)
        rows = re.findall(r'<tr.*?>(.*?)</tr>', table_body, re.DOTALL | re.IGNORECASE)

        courses_history_list = []
        for row in rows:
            if "<th" in row.lower():
                continue

            cells = re.findall(r'<td.*?>(.*?)</td>', row, re.DOTALL | re.IGNORECASE)
            # Safe boundary check: we need at least 11 columns to parse the cells array safely
            if len(cells) < 11:
                continue

            # Capture dynamic encrypted validation lookup reference parameters
            link_match = re.search(r'href=["\']([^"\']+)["\']', row, re.IGNORECASE)
            raw_href = link_match.group(1).replace("&amp;", "&") if link_match else ""

            # Explicit column mappings based on actual ERP layout indexes
            courses_history_list.append({
                "course_code": re.sub(r'<.*?>', '', cells[3]).strip(),          # Col 3: 22UC0021
                "course_name": re.sub(r'<.*?>', '', cells[4]).strip(),          # Col 4: SOCIAL IMMERSIVE LEARNING-1
                "grade": re.sub(r'<.*?>', '', cells[5]).strip(),                # Col 5: O
                "grade_point": re.sub(r'<.*?>', '', cells[6]).strip(),          # Col 6: 10
                "credits": re.sub(r'<.*?>', '', cells[7]).strip(),              # Col 7: 1
                "promotion_status": re.sub(r'<.*?>', '', cells[8]).strip(),     # Col 8: P
                "academic_year": re.sub(r'<.*?>', '', cells[9]).strip(),        # Col 9: 2024-2025
                "semester": re.sub(r'<.*?>', '', cells[10]).strip(),            # Col 10: Even Sem
                "target_href": raw_href                                         # Extraction dynamic link path
            })


        updated_session_id = cookie_jar.get("PHPSESSID")
        has_refreshed = updated_session_id != php_sess_id
        final_csrf = cookie_jar.get("_csrf", unquote(csrf_cookie) if csrf_cookie else "")

        logger.info(f"[CGPA] Successfully structured course array layout. Refreshed: {has_refreshed} in {time.time() - start_time:.3f}s")
        record_latency("cgpa", (time.time() - start_time) * 1000.0, 200)
        return {
            "success": True,
            "session_refreshed": has_refreshed,
            "cookies": {
                "PHPSESSID": updated_session_id,
                "kl_erp_device_id": cookie_jar.get("kl_erp_device_id", device_id),
                "SERVERID": cookie_jar.get("SERVERID", server_id),
                "_csrf_token": final_csrf,
                "_csrf": final_csrf
            },
            "data": courses_history_list
        }

    except HTTPException:
        raise
    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.NetworkError, httpx.TimeoutException, httpx.ProxyError, httpx.HTTPStatusError, GatewayUnavailableError) as net_err:
        if isinstance(net_err, httpx.HTTPStatusError) and net_err.response.status_code not in (502, 503):
            raise
        logger.error(f"[NETWORK REJECTION] /fetch-cgpa - University gateway down: {net_err}")
        raise GatewayRequestError("ERP request failed via API Gateway")
    except Exception as e:
        logger.error(f"[CGPA ERROR] Processing sequence crashed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/fetch-marks-detail")
@with_gateway_retries(max_retries=3)
@single_flight_cached(
    route_name="internal_marks",
    key_builder=lambda kwargs: (
        "internal_marks",
        kwargs.get("username", "").strip(),
        str(kwargs.get("target_href", "")).strip()
    )
)
async def fetch_marks_detail(
    target_href: str = Form(...),
    username: str = Form(...),
    password: str = Form(...),
    php_sess_id: str = Form(default=""),
    csrf_cookie: str = Form(default=""),
    device_id: str = Form(default=""),
    server_id: str = Form(default="erp1")
):
    start_time = time.time()
    cookie_jar = {
        "_csrf": unquote(csrf_cookie) if csrf_cookie else "",
        "PHPSESSID": php_sess_id,
        "kl_erp_device_id": unquote(device_id) if device_id else "",
        "SERVERID": server_id
    }

    if target_href.startswith("http"):
        full_detail_url = target_href
    else:
        full_detail_url = f"{BASE_URL}/{target_href.lstrip('/')}"

    try:
        # TWO-CLIENT STRATEGY (same as fetch-register-detail):
        # 1. Login via gateway → protects host IP from 429
        # 2. Fetch via direct client → avoids 400 from binary param corruption
        # SERVERID explicitly defaulted to "erp1" if gateway doesn't return it.
        async with httpx.AsyncClient(
            verify=False, headers=DEFAULT_HEADERS, http2=True,
            event_hooks={"response": [log_rate_limit]}
        ) as direct_client:
            response = await direct_client.get(full_detail_url, cookies=cookie_jar, timeout=15)

            if response.status_code in (301, 302, 303) or response.status_code == 500 or is_login_failed(response):
                logger.warning("[MARKS DETAIL] Token expired. Launching auto-login fallback via gateway...")
                async with make_erp_client() as gw_client:
                    for attempt in range(3):
                        if attempt > 0:
                            await asyncio.sleep(random.uniform(1.0, 2.0))
                        res, cookie_jar = await auto_login(gw_client, username, password, seed_cookies=cookie_jar)
                        if not is_login_failed(res):
                            break
                    else:
                        raise HTTPException(status_code=401, detail="Session verification recovery rejected.")

                if not cookie_jar.get("SERVERID"):
                    cookie_jar["SERVERID"] = "erp1"
                response = await direct_client.get(full_detail_url, cookies=cookie_jar, timeout=15)

            response.raise_for_status()
            html_content = response.text

        # Extract rows using matching id markers.
        # Yii assigns grid widget ids dynamically (w0, w1, ...) depending on
        # how many widgets the page renders — accept any of them, not just w0.
        detail_table_match = re.search(r'<table id="w\d+".*?>(.*?)</table>', html_content, re.DOTALL | re.IGNORECASE)
        if not detail_table_match:
            logger.warning(
                f"[MARKS DETAIL] scorecard table not found. status={response.status_code} "
                f"req_url={full_detail_url} final_url={response.url} "
                f"body_head={html_content[:600]!r}"
            )
            raise HTTPException(status_code=404, detail="Consolidated detailed scorecard layout missing.")

        detail_body = detail_table_match.group(1)
        detail_rows = re.findall(r'<tr.*?>(.*?)</tr>', detail_body, re.DOTALL | re.IGNORECASE)

        marks_map = {}
        for row in detail_rows:
            th_match = re.search(r'<th.*?>(.*?)</th>', row, re.DOTALL | re.IGNORECASE)
            td_match = re.search(r'<td.*?>(.*?)</td>', row, re.DOTALL | re.IGNORECASE)

            if th_match and td_match:
                raw_key = re.sub(r'<.*?>', '', th_match.group(1)).strip()
                field_key = raw_key.lower().replace(" ", "_")
                val_content = re.sub(r'<.*?>', '', td_match.group(1)).strip()
                marks_map[field_key] = val_content

        updated_session_id = cookie_jar.get("PHPSESSID")
        has_refreshed = updated_session_id != php_sess_id
        final_csrf = cookie_jar.get("_csrf", unquote(csrf_cookie) if csrf_cookie else "")

        scorecard = dict(marks_map)
        if "course_desc" in scorecard and "course_name" not in scorecard:
            scorecard["course_name"] = scorecard["course_desc"]

        logger.info(f"[MARKS DETAIL] Scorecard processed in {time.time() - start_time:.3f}s")
        record_latency("internal_marks", (time.time() - start_time) * 1000.0, 200)
        return {
            "success": True,
            "session_refreshed": has_refreshed,
            "cookies": {
                "PHPSESSID": updated_session_id,
                "kl_erp_device_id": cookie_jar.get("kl_erp_device_id", device_id),
                "SERVERID": cookie_jar.get("SERVERID", server_id),
                "_csrf_token": final_csrf,
                "_csrf": final_csrf
            },
            "scorecard": scorecard
        }

    except HTTPException:
        raise
    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.NetworkError, httpx.TimeoutException, httpx.ProxyError, httpx.HTTPStatusError, GatewayUnavailableError) as net_err:
        if isinstance(net_err, httpx.HTTPStatusError) and net_err.response.status_code not in (502, 503):
            raise
        logger.error(f"[NETWORK REJECTION] /fetch-marks-detail - University gateway down: {net_err}")
        raise GatewayRequestError("ERP request failed via API Gateway")
    except Exception as e:
        logger.error(f"[MARKS ERROR] Deep scorecard extraction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# AUTO-FEEDBACK SUBMISSION (personal use only)
# ============================================================

FEEDBACK_ALLOWED_USERS = os.environ.get("FEEDBACK_ALLOWED_USERS", "2400032717").split(",")

FEEDBACK_LISTING_URL = f"{BASE_URL}/index.php?r=feedback%2Ffeedbackstudentresultinfo%2Fcreatestudentfeedback"
FEEDBACK_SUBMIT_URL = f"{BASE_URL}/index.php?r=feedback%2Ffeedbackstudentresultinfo%2Fsavestudentfeedback"


def _parse_feedback_form(form_html: str):
    """
    Parse a Yii2 feedback form page.
    Returns (csrf, payload_tuples, course_desc, faculty_name, question_count)
    where payload_tuples is a list of (name, value) ready to POST.
    """
    # Extract CSRF from the form's hidden input
    form_csrf = ""
    csrf_m = re.search(r'<input[^>]+name=["\']_csrf["\'][^>]+value=["\']([^"\']+)["\']', form_html)
    if csrf_m:
        form_csrf = csrf_m.group(1)
    if not form_csrf:
        csrf_m = re.search(r'<input[^>]+value=["\']([^"\']+)["\'][^>]+name=["\']_csrf["\']', form_html)
        if csrf_m:
            form_csrf = csrf_m.group(1)

    # Find all <input> tags
    all_inputs = re.findall(r'<input[^>]+>', form_html, re.IGNORECASE)

    hidden_fields = []       # (name, value) for all hidden inputs except _csrf and DynamicModel answers
    question_ids = {}        # {index_str: qid}
    course_desc = ""

    for inp in all_inputs:
        is_hidden = ('type="hidden"' in inp or "type='hidden'" in inp)
        if not is_hidden:
            continue

        name_m = re.search(r'name=["\']([^"\']+)["\']', inp)
        if not name_m:
            continue
        name = name_m.group(1)

        value_m = re.search(r'value=["\']([^"\']*)["\']', inp)
        value = value_m.group(1) if value_m else ""

        # Skip _csrf (added separately)
        if name == "_csrf":
            continue

        # Track questionnaire IDs
        qid_m = re.match(r'FeedbackStudentResultInfo\[(\d+)\]\[fsri_student_questionaire_info_id\]', name)
        if qid_m:
            question_ids[qid_m.group(1)] = value

        # Track course description for logging
        if "fsri_course_desc" in name and not course_desc:
            course_desc = value

        # Skip DynamicModel answer hidden (we'll build our own)
        if "DynamicModel" in name and "answer_option" in name:
            continue

        hidden_fields.append((name, value))

    # Build final payload as ordered list of tuples (duplicate keys OK)
    payload = [("_csrf", form_csrf)]
    # We need to interleave the hidden fields and answer options in correct index order
    # Group hidden fields by question index
    max_idx = max((int(k) for k in question_ids), default=-1)

    for idx in range(max_idx + 1):
        idx_str = str(idx)
        qid = question_ids.get(idx_str, "")

        # Add all FeedbackStudentResultInfo[idx][...] hidden fields for this index
        # They come before the answer in the form, so add profile_id first
        for name, value in hidden_fields:
            if f"[{idx_str}]" in name:
                payload.append((name, value))
                # Insert the DynamicModel answer right after the questionaire_info_id
                if "fsri_student_questionaire_info_id" in name and qid:
                    answer_key = f"DynamicModel[{idx_str}][fsri_student_questionaire_answer_option]"
                    payload.append((answer_key, ""))        # Yii hidden (empty)
                    payload.append((answer_key, f"{qid}:::1"))  # Option 1 = best

    return form_csrf, payload, course_desc, len(question_ids)


@app.post("/auto-feedback")
@with_gateway_retries(max_retries=3)
async def auto_feedback(
    username: str = Form(...),
    password: str = Form(...),
    php_sess_id: str = Form(default=""),
    csrf_cookie: str = Form(default=""),
    device_id: str = Form(default=""),
    server_id: str = Form(default="erp3"),
):
    if username not in FEEDBACK_ALLOWED_USERS:
        raise HTTPException(status_code=403, detail="Auto-feedback is not enabled for this account.")

    start_time = time.time()
    cookie_jar = {
        "_csrf": unquote(csrf_cookie) if csrf_cookie else "",
        "PHPSESSID": php_sess_id,
        "kl_erp_device_id": unquote(device_id) if device_id else "",
        "SERVERID": server_id
    }

    try:
        async with make_erp_client() as client:

            # ---------- Step 0: login if no session ----------
            if not php_sess_id or not csrf_cookie:
                for attempt in range(3):
                    if attempt > 0:
                        await asyncio.sleep(random.uniform(1.0, 2.5))
                    login_resp, cookie_jar = await auto_login(client, username, password, seed_cookies={})
                    if not is_login_failed(login_resp):
                        break
                    logger.warning(f"[FEEDBACK] Login attempt {attempt+1} failed.")
                else:
                    raise HTTPException(status_code=401, detail="Login failed.")

            # ---------- Step 1: GET feedback listing page ----------
            listing_resp, cookie_jar = await _follow_redirects_collecting_cookies(
                client, "GET", FEEDBACK_LISTING_URL, cookie_jar, timeout=20
            )

            # Auto-heal if session expired
            if is_login_failed(listing_resp):
                logger.info("[FEEDBACK] Session expired, re-authenticating...")
                for attempt in range(3):
                    if attempt > 0:
                        await asyncio.sleep(random.uniform(1.0, 2.5))
                    login_resp, cookie_jar = await auto_login(client, username, password, seed_cookies=cookie_jar)
                    if not is_login_failed(login_resp):
                        break
                else:
                    raise HTTPException(status_code=401, detail="Session recovery failed.")
                listing_resp, cookie_jar = await _follow_redirects_collecting_cookies(
                    client, "GET", FEEDBACK_LISTING_URL, cookie_jar, timeout=20
                )

            listing_html = listing_resp.text

            # ---------- Step 2: parse all feedback links ----------
            # Links look like: <a class="crudjax2" href="/index.php?r=...&amp;id=...">Faculty Name</a>
            link_matches = re.findall(
                r'<a[^>]*class=["\']crudjax2["\'][^>]*href=["\']([^"\']+)["\'][^>]*>\s*(.*?)\s*</a>',
                listing_html, re.IGNORECASE | re.DOTALL
            )

            if not link_matches:
                elapsed = round(time.time() - start_time, 2)
                return {
                    "success": True,
                    "message": "No pending feedback found.",
                    "submitted": [],
                    "failed": [],
                    "elapsed_seconds": elapsed
                }

            # Deduplicate by href
            seen_hrefs = set()
            unique_entries = []
            for href_raw, faculty_name in link_matches:
                href_clean = href_raw.replace("&amp;", "&")
                if href_clean not in seen_hrefs:
                    seen_hrefs.add(href_clean)
                    faculty_clean = re.sub(r'<[^>]*>', '', faculty_name).strip()
                    faculty_clean = re.sub(r'\s+', ' ', faculty_clean)
                    unique_entries.append((href_clean, faculty_clean))

            logger.info(f"[FEEDBACK] Found {len(unique_entries)} pending feedback(s) to submit.")

            # ---------- Step 3: submit each one ----------
            submitted = []
            failed = []

            for href, faculty in unique_entries:
                form_url = f"{BASE_URL}{href}" if href.startswith("/") else href

                try:
                    # Fresh HTTP/1.1 client per feedback (avoids H2 stream state issues)
                    async with make_erp_client(
                        http2=False, follow_redirects=True, timeout=20
                    ) as fc:
                        # GET the feedback form page
                        form_resp = await fc.get(form_url, cookies=cookie_jar)
                        cookie_jar = collect_cookies(form_resp, cookie_jar)

                        if is_login_failed(form_resp):
                            for attempt in range(3):
                                if attempt > 0:
                                    await asyncio.sleep(random.uniform(1.0, 2.5))
                                login_resp, cookie_jar = await auto_login(fc, username, password, seed_cookies=cookie_jar)
                                if not is_login_failed(login_resp):
                                    break
                            form_resp = await fc.get(form_url, cookies=cookie_jar)
                            cookie_jar = collect_cookies(form_resp, cookie_jar)

                        form_html = form_resp.text

                        # Check if form exists
                        if "studentfeedbackform" not in form_html:
                            failed.append({"faculty": faculty, "error": "Feedback form not found on page (already submitted?)"})
                            continue

                        # Parse the form
                        form_csrf, payload, course_desc, q_count = _parse_feedback_form(form_html)

                        if not form_csrf:
                            failed.append({"faculty": faculty, "course": course_desc, "error": "CSRF token not found"})
                            continue

                        if q_count == 0:
                            failed.append({"faculty": faculty, "course": course_desc, "error": "No questions found in form"})
                            continue

                        # POST the feedback (don't follow redirect, 302 = success)
                        # Manually URL-encode to avoid httpx sync stream bug with data=list_of_tuples
                        from urllib.parse import urlencode as _urlencode
                        encoded_body = _urlencode(payload)
                        submit_resp = await fc.post(
                            FEEDBACK_SUBMIT_URL,
                            cookies=cookie_jar,
                            content=encoded_body.encode("utf-8"),
                            headers={
                                "Content-Type": "application/x-www-form-urlencoded",
                                "Origin": BASE_URL,
                                "Referer": form_url,
                            },
                            follow_redirects=False,
                        )
                        cookie_jar = collect_cookies(submit_resp, cookie_jar)

                        # Yii returns 302 after successful save
                        if submit_resp.status_code in (200, 302):
                            submitted.append({
                                "faculty": faculty,
                                "course": course_desc,
                                "questions": q_count,
                                "status": "submitted"
                            })
                            logger.info(f"[FEEDBACK] ✅ {course_desc} — {faculty} ({q_count} questions)")
                        else:
                            failed.append({
                                "faculty": faculty,
                                "course": course_desc,
                                "error": f"Submit returned status {submit_resp.status_code}"
                            })
                            logger.warning(f"[FEEDBACK] ❌ {course_desc} — {faculty}: HTTP {submit_resp.status_code}")

                    # Small delay between submissions
                    await asyncio.sleep(random.uniform(0.3, 0.8))

                except Exception as e:
                    failed.append({"faculty": faculty, "error": str(e)})
                    logger.error(f"[FEEDBACK] Error for {faculty}: {e}", exc_info=True)

            elapsed = round(time.time() - start_time, 2)
            return {
                "success": True,
                "total_found": len(unique_entries),
                "submitted": submitted,
                "failed": failed,
                "elapsed_seconds": elapsed
            }

    except HTTPException:
        raise
    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.NetworkError, httpx.TimeoutException, httpx.ProxyError, httpx.HTTPStatusError, GatewayUnavailableError) as net_err:
        if isinstance(net_err, httpx.HTTPStatusError) and net_err.response.status_code not in (502, 503):
            raise
        logger.error(f"[NETWORK] /auto-feedback - ERP unreachable: {net_err}")
        raise GatewayRequestError("ERP request failed via API Gateway")
    except Exception as e:
        logger.error(f"[FEEDBACK] Crash: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# OBSERVATORY & LATENCY TELEMETRY (MongoDB + In-Memory Fallback)
# ============================================================

_latency_pings_col = None
_latency_daily_col = None

MONGODB_URI = os.environ.get("MONGODB_URI", "")
if _pymongo_available and MONGODB_URI:
    try:
        _mongo_client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=4000)
        _mongo_client.admin.command("ping")
        _obs_db = _mongo_client["timetablekl"]
        _latency_pings_col = _obs_db["latency_pings"]
        _latency_daily_col = _obs_db["latency_daily_rollups"]
        # 30-day native TTL auto-expiry on raw latency pings (2592000 seconds)
        try:
            _latency_pings_col.create_index("timestamp", expireAfterSeconds=2592000)
            _latency_daily_col.create_index([("date", ASCENDING), ("route", ASCENDING)], unique=True)
            _latency_daily_col.create_index("route")
        except Exception as idx_err:
            logger.debug(f"[LATENCY] Index setup notice: {idx_err}")
        logger.info("✅ MongoDB connected — latency observatory persistent.")
    except Exception as e:
        logger.error(f"[OBSERVATORY] MongoDB init failed, using in-memory fallback: {e}")
        _latency_pings_col = None
        _latency_daily_col = None

_mem_latency_rollups = {}  # (date, route) -> stats dict

def record_latency(route_key: str, latency_ms: float, status_code: int = 200) -> None:
    """Non-blocking background telemetry logger: saves to MongoDB or in-memory fallback."""
    def _bg_worker():
        try:
            now = datetime.datetime.utcnow()
            today_str = now.strftime("%Y-%m-%d")
            lat = round(float(latency_ms), 2)

            if _latency_pings_col is not None and _latency_daily_col is not None:
                try:
                    _latency_pings_col.insert_one({
                        "route": route_key,
                        "latency_ms": lat,
                        "status_code": status_code,
                        "timestamp": now
                    })
                except Exception:
                    pass

                _latency_daily_col.update_one(
                    {"date": today_str, "route": route_key},
                    {
                        "$inc": {"count": 1, "total_latency_ms": lat},
                        "$min": {"min_latency_ms": lat},
                        "$max": {"max_latency_ms": lat},
                        "$set": {"last_updated": now}
                    },
                    upsert=True
                )
            else:
                key = (today_str, route_key)
                if key not in _mem_latency_rollups:
                    _mem_latency_rollups[key] = {
                        "count": 1,
                        "total_latency_ms": lat,
                        "min_latency_ms": lat,
                        "max_latency_ms": lat,
                        "last_updated": now
                    }
                else:
                    item = _mem_latency_rollups[key]
                    item["count"] += 1
                    item["total_latency_ms"] += lat
                    item["min_latency_ms"] = min(item["min_latency_ms"], lat)
                    item["max_latency_ms"] = max(item["max_latency_ms"], lat)
                    item["last_updated"] = now
        except Exception as err:
            logger.debug(f"[LATENCY] Telemetry record error: {err}")

    threading.Thread(target=_bg_worker, daemon=True).start()


# ============================================================
# R1 BENCHMARK OBSERVATORY & STATUS WEBSITE
# ============================================================

_BENCHMARK_ROUTES = [
    {"key": "timetable", "name": "Timetable", "path": "/fetch-timetable"},
    {"key": "attendance", "name": "Attendance", "path": "/fetch-attendance"},
    {"key": "cgpa", "name": "CGPA Ledger", "path": "/fetch-cgpa"},
    {"key": "internal_marks", "name": "Internal Marks", "path": "/fetch-marks-detail"},
]

def get_benchmark_stats(time_range: str = "all") -> dict:
    """Aggregates latency metrics across 4 core routes from MongoDB (or in-memory fallback)."""
    now = datetime.datetime.utcnow()
    today_str = now.strftime("%Y-%m-%d")
    by_route = {}

    if time_range == "24h":
        if _latency_pings_col is not None:
            cutoff = now - datetime.timedelta(hours=24)
            pipeline = [
                {"$match": {"timestamp": {"$gte": cutoff}}},
                {"$group": {
                    "_id": "$route",
                    "count": {"$sum": 1},
                    "total_ms": {"$sum": "$latency_ms"},
                    "min_ms": {"$min": "$latency_ms"},
                    "max_ms": {"$max": "$latency_ms"},
                    "last_updated": {"$max": "$timestamp"}
                }}
            ]
            try:
                results = list(_latency_pings_col.aggregate(pipeline))
                by_route = {r["_id"]: r for r in results}
            except Exception as e:
                logger.debug(f"[LATENCY STATS] 24h ping agg error: {e}")
        if not by_route and _latency_daily_col is not None:
            try:
                results = list(_latency_daily_col.find({"date": today_str}))
                by_route = {
                    r["route"]: {
                        "_id": r["route"],
                        "count": r.get("count", 0),
                        "total_ms": r.get("total_latency_ms", 0.0),
                        "min_ms": r.get("min_latency_ms", 0.0),
                        "max_ms": r.get("max_latency_ms", 0.0),
                        "last_updated": r.get("last_updated")
                    }
                    for r in results
                }
            except Exception:
                pass
    else:
        match_filter = {}
        if time_range == "10d":
            start_date = (now.date() - datetime.timedelta(days=9)).strftime("%Y-%m-%d")
            match_filter = {"date": {"$gte": start_date}}
        elif time_range == "30d":
            start_date = (now.date() - datetime.timedelta(days=29)).strftime("%Y-%m-%d")
            match_filter = {"date": {"$gte": start_date}}

        if _latency_daily_col is not None:
            pipeline = [
                {"$match": match_filter},
                {"$group": {
                    "_id": "$route",
                    "count": {"$sum": "$count"},
                    "total_ms": {"$sum": "$total_latency_ms"},
                    "min_ms": {"$min": "$min_latency_ms"},
                    "max_ms": {"$max": "$max_latency_ms"},
                    "last_updated": {"$max": "$last_updated"}
                }}
            ]
            try:
                results = list(_latency_daily_col.aggregate(pipeline))
                by_route = {r["_id"]: r for r in results}
            except Exception as e:
                logger.debug(f"[LATENCY STATS] daily agg error: {e}")

    # Fallback to in-memory store if MongoDB returned no rows
    if not by_route and _mem_latency_rollups:
        mem_grouped = {}
        for (d_str, r_key), item in _mem_latency_rollups.items():
            if time_range == "24h" and d_str != today_str:
                continue
            if time_range == "10d":
                start_d = (now.date() - datetime.timedelta(days=9)).strftime("%Y-%m-%d")
                if d_str < start_d:
                    continue
            if time_range == "30d":
                start_d = (now.date() - datetime.timedelta(days=29)).strftime("%Y-%m-%d")
                if d_str < start_d:
                    continue

            if r_key not in mem_grouped:
                mem_grouped[r_key] = {
                    "_id": r_key,
                    "count": 0,
                    "total_ms": 0.0,
                    "min_ms": item["min_latency_ms"],
                    "max_ms": item["max_latency_ms"],
                    "last_updated": item["last_updated"]
                }
            g = mem_grouped[r_key]
            g["count"] += item["count"]
            g["total_ms"] += item["total_latency_ms"]
            g["min_ms"] = min(g["min_ms"], item["min_latency_ms"])
            g["max_ms"] = max(g["max_ms"], item["max_latency_ms"])
            g["last_updated"] = item["last_updated"]
        by_route = mem_grouped

    route_stats = []
    total_all_requests = 0
    total_all_ms = 0.0

    for r in _BENCHMARK_ROUTES:
        r_key = r["key"]
        data = by_route.get(r_key, {})
        cnt = data.get("count", 0)
        tot = data.get("total_ms", 0.0)
        min_val = data.get("min_ms", 0.0) if cnt > 0 else 0.0
        max_val = data.get("max_ms", 0.0) if cnt > 0 else 0.0
        avg_val = round(tot / cnt, 1) if cnt > 0 else 0.0
        last_up = data.get("last_updated")
        if isinstance(last_up, datetime.datetime):
            last_up_str = last_up.strftime("%Y-%m-%d %H:%M:%S UTC")
        else:
            last_up_str = "None"

        total_all_requests += cnt
        total_all_ms += tot

        route_stats.append({
            "key": r_key,
            "name": r["name"],
            "path": r["path"],
            "count": cnt,
            "avg_ms": avg_val,
            "min_ms": round(min_val, 1),
            "max_ms": round(max_val, 1),
            "last_updated": last_up_str
        })

    overall_avg = round(total_all_ms / total_all_requests, 1) if total_all_requests > 0 else 0.0

    return {
        "success": True,
        "time_range": time_range,
        "total_requests": total_all_requests,
        "overall_avg_ms": overall_avg,
        "database": "MongoDB Atlas" if _latency_daily_col is not None else "In-Memory",
        "routes": route_stats,
        "server_time": now.strftime("%Y-%m-%d %H:%M:%S UTC")
    }

@app.get("/api/benchmarks/stats")
async def api_benchmarks_stats(range: str = "all"):
    """JSON API endpoint returning live latency statistics and rolling averages."""
    clean_range = range.lower().strip()
    if clean_range not in ("24h", "10d", "30d", "all"):
        clean_range = "all"
    stats = get_benchmark_stats(clean_range)
    return JSONResponse(content=stats)

@app.get("/benchmarks", response_class=HTMLResponse)
@app.get("/benchmarks/", response_class=HTMLResponse)
@app.get("/status", response_class=HTMLResponse)
async def benchmarks_website():
    """Serves the complete retro-terminal benchmark and latency dashboard."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TimeTableKL · R1 Benchmark Observatory</title>
    <style>
        :root {
            --bg-color: #F6F4EE;
            --text-color: #111111;
            --border-color: #D3CEBE;
            --pill-bg: #EAE6D9;
            --bar-color: #0284C7;
            --bar-accent: #E25C27;
            --active-tab-bg: #111111;
            --active-tab-text: #F6F4EE;
            --grid-line: #E2DDD0;
            --card-bg: #FFFFFF;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: ui-monospace, "SF Mono", "Cascadia Code", "JetBrains Mono", Menlo, Consolas, monospace;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-color);
            padding: 24px 20px 60px;
            max-width: 1180px;
            margin: 0 auto;
            -webkit-font-smoothing: antialiased;
        }

        /* TABS HEADER */
        .tabs-nav {
            display: flex;
            flex-wrap: wrap;
            gap: 2px;
            background-color: var(--border-color);
            padding: 1px;
            border: 1px solid var(--border-color);
            margin-bottom: 28px;
        }

        .tab-button {
            background-color: var(--pill-bg);
            color: #444;
            border: none;
            padding: 9px 18px;
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 0.5px;
            cursor: pointer;
            transition: all 0.15s ease;
            text-transform: uppercase;
        }

        .tab-button:hover {
            background-color: #DFDAC9;
            color: #000;
        }

        .tab-button.active {
            background-color: var(--active-tab-bg);
            color: var(--active-tab-text);
        }

        /* HEADER & FILTERS */
        .main-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            flex-wrap: wrap;
            gap: 16px;
            margin-bottom: 12px;
        }

        .title-row {
            display: flex;
            align-items: center;
            gap: 12px;
            flex-wrap: wrap;
        }

        h1 {
            font-size: 20px;
            font-weight: 900;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }

        .metric-badge {
            background: #111;
            color: #fff;
            padding: 4px 10px;
            font-size: 11px;
            font-weight: 700;
            border-radius: 3px;
            display: inline-flex;
            align-items: center;
            gap: 4px;
        }

        .subtitle {
            font-size: 13px;
            color: #555;
            margin-top: 4px;
        }

        /* FILTER CONTROLS */
        .range-controls {
            display: flex;
            gap: 6px;
            align-items: center;
        }

        .range-btn {
            background: transparent;
            border: 1px solid #111;
            color: #111;
            padding: 5px 12px;
            font-size: 11px;
            font-weight: 800;
            cursor: pointer;
            text-transform: uppercase;
            transition: all 0.1s ease;
        }

        .range-btn.active, .range-btn:hover {
            background: #111;
            color: #fff;
        }

        /* VERDICT BANNER */
        .verdict-banner {
            background-color: #111;
            color: #fff;
            padding: 16px 20px;
            border-radius: 4px;
            margin: 20px 0 32px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 12px;
        }

        .verdict-text {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }

        .verdict-tag {
            font-size: 10px;
            color: #e25c27;
            font-weight: 800;
            letter-spacing: 1px;
            text-transform: uppercase;
        }

        .verdict-message {
            font-size: 14px;
            font-weight: 700;
        }

        .verdict-meta {
            font-size: 12px;
            color: #999;
            text-align: right;
        }

        /* CHART CONTAINER */
        .chart-card {
            background: #FAF8F5;
            border: 1px solid var(--border-color);
            padding: 32px 24px 20px;
            margin-bottom: 32px;
            position: relative;
        }

        .legend-row {
            display: flex;
            justify-content: flex-end;
            align-items: center;
            gap: 16px;
            margin-bottom: 24px;
            font-size: 12px;
            font-weight: 700;
        }

        .legend-item {
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .legend-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
        }

        .chart-viewport {
            height: 360px;
            position: relative;
            display: flex;
            margin-left: 70px;
            margin-bottom: 40px;
            border-bottom: 2px solid #333;
        }

        /* Y-AXIS GRID */
        .y-axis {
            position: absolute;
            left: -70px;
            top: 0;
            bottom: 0;
            width: 60px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            pointer-events: none;
        }

        .y-label {
            font-size: 11px;
            color: #666;
            text-align: right;
            line-height: 1;
        }

        .grid-lines {
            position: absolute;
            left: 0;
            right: 0;
            top: 0;
            bottom: 0;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            pointer-events: none;
        }

        .grid-line {
            width: 100%;
            height: 1px;
            border-top: 1px dashed var(--grid-line);
        }

        /* BARS AREA */
        .bars-container {
            position: relative;
            z-index: 2;
            width: 100%;
            height: 100%;
            display: flex;
            justify-content: space-around;
            align-items: flex-end;
        }

        .bar-group {
            display: flex;
            flex-direction: column;
            align-items: center;
            height: 100%;
            justify-content: flex-end;
            width: 120px;
            position: relative;
        }

        .bar-fill {
            width: 44px;
            background-color: var(--bar-color);
            transition: height 0.6s cubic-bezier(0.16, 1, 0.3, 1);
            position: relative;
            cursor: pointer;
            border-top: 2px solid #015f91;
        }

        .bar-fill:hover {
            opacity: 0.9;
            filter: brightness(1.1);
        }

        .bar-value {
            position: absolute;
            top: -24px;
            left: 50%;
            transform: translateX(-50%);
            font-size: 11px;
            font-weight: 800;
            color: #111;
            white-space: nowrap;
        }

        .route-label {
            position: absolute;
            bottom: -32px;
            left: 50%;
            transform: translateX(-50%);
            font-size: 12px;
            font-weight: 700;
            white-space: nowrap;
            color: #222;
        }

        /* CARDS TABLE */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }

        .stat-card {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            padding: 16px;
        }

        .stat-card-title {
            font-size: 11px;
            color: #666;
            text-transform: uppercase;
            font-weight: 700;
            margin-bottom: 6px;
        }

        .stat-card-value {
            font-size: 22px;
            font-weight: 900;
            color: #111;
        }

        .stat-card-sub {
            font-size: 11px;
            color: #777;
            margin-top: 4px;
        }

        /* DATA TABLE */
        .table-card {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            overflow-x: auto;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
            text-align: left;
        }

        th {
            background-color: var(--pill-bg);
            padding: 12px 16px;
            border-bottom: 1px solid var(--border-color);
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        td {
            padding: 12px 16px;
            border-bottom: 1px solid #ECE7DA;
            color: #222;
        }

        tr:hover td {
            background-color: #FAF8F2;
        }

        .status-pill {
            display: inline-block;
            padding: 3px 8px;
            font-size: 10px;
            font-weight: 800;
            border-radius: 2px;
            background: #E6F4EA;
            color: #137333;
        }

        /* FOOTER */
        .footer {
            margin-top: 32px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 11px;
            color: #777;
            border-top: 1px solid var(--border-color);
            padding-top: 16px;
            flex-wrap: wrap;
            gap: 8px;
        }

        .refresh-btn {
            background: #111;
            color: #fff;
            border: none;
            padding: 6px 12px;
            font-size: 11px;
            font-weight: 700;
            cursor: pointer;
        }
    </style>
</head>
<body>

    <!-- NAVIGATION TABS -->
    <div class="tabs-nav">
        <button class="tab-button" onclick="selectTab(1)">[01] TIMETABLE GRAPH</button>
        <button class="tab-button" onclick="selectTab(2)">[02] ATTENDANCE GRAPH</button>
        <button class="tab-button" onclick="selectTab(3)">[03] CGPA GRAPH</button>
        <button class="tab-button active" onclick="selectTab(4)">[04] ALL-ROUTES SPECTRUM</button>
        <button class="tab-button" onclick="selectTab(5)">[05] PEAK CONCURRENCY (RPS)</button>
    </div>

    <!-- MAIN HEADER -->
    <div class="main-header">
        <div>
            <div class="title-row">
                <h1 id="view-title">ALL-ROUTES LATENCY SPECTRUM (SINGLE REQUEST)</h1>
                <span class="metric-badge">&darr; LOWER IS BETTER (MS)</span>
            </div>
            <div class="subtitle" id="view-subtitle">Continuous live latency telemetry · Real student traffic via EC2 & KL ERP</div>
        </div>

        <div class="range-controls">
            <button class="range-btn" onclick="setRange('24h')">LAST 24H</button>
            <button class="range-btn" onclick="setRange('10d')">LAST 10 DAYS</button>
            <button class="range-btn" onclick="setRange('30d')">LAST 30 DAYS</button>
            <button class="range-btn active" onclick="setRange('all')">ALL TIME</button>
        </div>
    </div>

    <!-- VERDICT BANNER -->
    <div class="verdict-banner">
        <div class="verdict-text">
            <div class="verdict-tag">VERDICT // LOWER IS BETTER (ms)</div>
            <div class="verdict-message" id="verdict-msg">R1 Telemetry Engine — Live continuous sample averages over 4 core routes</div>
        </div>
        <div class="verdict-meta" id="verdict-meta">
            Database: Connecting...<br>
            Updated: Just now
        </div>
    </div>

    <!-- SUMMARY CARDS -->
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-card-title">Total Requests Sampled</div>
            <div class="stat-card-value" id="card-total-req">0</div>
            <div class="stat-card-sub" id="card-total-sub">Across 4 monitored routes</div>
        </div>
        <div class="stat-card">
            <div class="stat-card-title">Average Latency (Running Avg)</div>
            <div class="stat-card-value" id="card-avg-latency">0 ms</div>
            <div class="stat-card-sub">Weighted across all sampled pings</div>
        </div>
        <div class="stat-card">
            <div class="stat-card-title">Fastest Recorded Route</div>
            <div class="stat-card-value" id="card-fastest">--</div>
            <div class="stat-card-sub" id="card-fastest-sub">Sub-second execution</div>
        </div>
        <div class="stat-card">
            <div class="stat-card-title">Persistence & TTL Retention</div>
            <div class="stat-card-value" id="card-db-mode">MongoDB Atlas</div>
            <div class="stat-card-sub">30-day raw TTL + daily rollups</div>
        </div>
    </div>

    <!-- CHART CARD -->
    <div class="chart-card">
        <div class="legend-row">
            <div class="legend-item">
                <div class="legend-dot" style="background-color: var(--bar-color);"></div>
                <span>TimeTableKL Live (Real Endpoints)</span>
            </div>
            <div class="legend-item">
                <span id="sample-indicator" style="color: #666;">Sampled: 0 requests</span>
            </div>
        </div>

        <div class="chart-viewport">
            <!-- Y-Axis Scale (0 to 7000 ms) -->
            <div class="y-axis">
                <div class="y-label">7000 ms</div>
                <div class="y-label">5250 ms</div>
                <div class="y-label">3500 ms</div>
                <div class="y-label">1750 ms</div>
                <div class="y-label">0 ms</div>
            </div>

            <!-- Horizontal Dashed Grid Lines -->
            <div class="grid-lines">
                <div class="grid-line"></div>
                <div class="grid-line"></div>
                <div class="grid-line"></div>
                <div class="grid-line"></div>
                <div class="grid-line" style="border-top: none;"></div>
            </div>

            <!-- Interactive Bars -->
            <div class="bars-container" id="bars-container">
                <!-- Injected via JavaScript -->
            </div>
        </div>
    </div>

    <!-- DETAILED ROUTES TABLE -->
    <div class="table-card">
        <table>
            <thead>
                <tr>
                    <th>Route Key</th>
                    <th>Endpoint Path</th>
                    <th>Total Requests (N)</th>
                    <th>Average Latency</th>
                    <th>Min / Max Latency</th>
                    <th>Last Sampled</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody id="routes-tbody">
                <!-- Injected via JavaScript -->
            </tbody>
        </table>
    </div>

    <!-- FOOTER -->
    <div class="footer">
        <div>
            TimeTableKL R1 Observability · AWS API Gateway Rotator Protected · Decoupled Cloud Storage
        </div>
        <div style="display: flex; align-items: center; gap: 12px;">
            <span id="auto-refresh-label">Auto-refresh in 10s</span>
            <button class="refresh-btn" onclick="fetchStats()">SYNC NOW</button>
        </div>
    </div>

    <script>
        let currentRange = 'all';
        let countdown = 10;
        const MAX_Y_MS = 7000;

        function setRange(range) {
            currentRange = range;
            document.querySelectorAll('.range-btn').forEach(btn => {
                btn.classList.toggle('active', btn.textContent.toLowerCase().includes(range));
            });
            fetchStats();
        }

        function selectTab(idx) {
            document.querySelectorAll('.tab-button').forEach((btn, i) => {
                btn.classList.toggle('active', i === idx - 1);
            });
            const titles = {
                1: "TIMETABLE ROUTE LATENCY GRAPH",
                2: "ATTENDANCE ROUTE LATENCY GRAPH",
                3: "CGPA LEDGER ROUTE LATENCY GRAPH",
                4: "ALL-ROUTES LATENCY SPECTRUM (SINGLE REQUEST)",
                5: "PEAK CONCURRENCY & RPS ESTIMATION"
            };
            document.getElementById('view-title').textContent = titles[idx] || titles[4];
            fetchStats();
        }

        async function fetchStats() {
            try {
                countdown = 10;
                const res = await fetch('/api/benchmarks/stats?range=' + currentRange);
                const data = await res.json();
                renderUI(data);
            } catch (err) {
                console.error("Failed to fetch benchmark stats:", err);
            }
        }

        function renderUI(data) {
            // Update Summary Cards
            document.getElementById('card-total-req').textContent = data.total_requests.toLocaleString();
            document.getElementById('card-avg-latency').textContent = (data.overall_avg_ms || 0) + ' ms';
            document.getElementById('card-db-mode').textContent = data.database || 'MongoDB Atlas';
            document.getElementById('sample-indicator').textContent = 'Sampled: ' + data.total_requests.toLocaleString() + ' requests';

            document.getElementById('verdict-meta').innerHTML = 
                'Storage: ' + data.database + '<br>' +
                'Updated: ' + (data.server_time || 'Just now');

            let fastest = null;
            let fastestMs = Infinity;

            const barsContainer = document.getElementById('bars-container');
            const tbody = document.getElementById('routes-tbody');
            barsContainer.innerHTML = '';
            tbody.innerHTML = '';

            data.routes.forEach(r => {
                if (r.count > 0 && r.avg_ms < fastestMs) {
                    fastestMs = r.avg_ms;
                    fastest = r.name;
                }

                // Render Bar
                const group = document.createElement('div');
                group.className = 'bar-group';

                const barFill = document.createElement('div');
                barFill.className = 'bar-fill';
                
                // Height calculation relative to MAX_Y_MS (7000ms)
                const clampedMs = Math.min(r.avg_ms, MAX_Y_MS);
                const heightPercent = r.count > 0 ? Math.max((clampedMs / MAX_Y_MS) * 100, 3) : 2;
                barFill.style.height = heightPercent + '%';

                const barVal = document.createElement('div');
                barVal.className = 'bar-value';
                barVal.textContent = r.count > 0 ? (r.avg_ms + ' ms') : '0 ms';

                const routeLabel = document.createElement('div');
                routeLabel.className = 'route-label';
                routeLabel.textContent = r.name;

                barFill.appendChild(barVal);
                group.appendChild(barFill);
                group.appendChild(routeLabel);
                barsContainer.appendChild(group);

                // Render Table Row
                const tr = document.createElement('tr');
                tr.innerHTML = 
                    '<td><strong>' + r.key + '</strong></td>' +
                    '<td><code>' + r.path + '</code></td>' +
                    '<td>' + r.count.toLocaleString() + '</td>' +
                    '<td><strong>' + (r.count > 0 ? r.avg_ms + ' ms' : '--') + '</strong></td>' +
                    '<td>' + (r.count > 0 ? (r.min_ms + ' ms / ' + r.max_ms + ' ms') : '--') + '</td>' +
                    '<td>' + r.last_updated + '</td>' +
                    '<td><span class="status-pill">' + (r.count > 0 ? 'ACTIVE' : 'IDLE') + '</span></td>';
                tbody.appendChild(tr);
            });

            if (fastest) {
                document.getElementById('card-fastest').textContent = fastest;
                document.getElementById('card-fastest-sub').textContent = fastestMs + ' ms avg latency';
                document.getElementById('verdict-msg').textContent = 
                    'Fastest live route: ' + fastest + ' (' + fastestMs + ' ms) · Total ' + data.total_requests.toLocaleString() + ' requests tracked';
            } else {
                document.getElementById('card-fastest').textContent = 'Awaiting pings';
                document.getElementById('card-fastest-sub').textContent = 'Send requests to generate stats';
            }
        }

        // Auto-refresh ticker
        setInterval(() => {
            countdown--;
            document.getElementById('auto-refresh-label').textContent = 'Auto-refresh in ' + countdown + 's';
            if (countdown <= 0) {
                fetchStats();
            }
        }, 1000);

        fetchStats();
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)



