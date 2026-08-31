from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import httpx
import logging
import os
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
import hmac
import hashlib
import base64
import secrets
import datetime
import functools
from bs4 import BeautifulSoup

from requests_ip_rotator import ApiGateway
from gateway_proxy import ApiGatewayTransport, GatewayUnavailableError


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

# Optional MongoDB (game leaderboard). Falls back to in-memory if unavailable.
try:
    from pymongo import MongoClient, ASCENDING
    from pymongo.errors import DuplicateKeyError
    _pymongo_available = True
except ImportError:
    _pymongo_available = False
    logger = logging.getLogger(__name__)
    logger.warning("pymongo not installed — game leaderboard will use in-memory store.")

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
    global gateway_manager
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

# ------------------ LOGIN ENDPOINT ------------------
@app.post("/login")
@with_gateway_retries(max_retries=3)
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
        # Register detail URLs contain encrypted binary parameters with non-UTF-8 bytes 
        # (e.g. %F5%B6%EA%9F...) which AWS API Gateway REST APIs parse, re-encode and corrupt, 
        # resulting in HTTP 400 Bad Request. We request directly from the host.
        async with httpx.AsyncClient(
            verify=False, headers=DEFAULT_HEADERS, http2=True,
            event_hooks={"response": [log_rate_limit]}
        ) as client:
            if not php_sess_id or not csrf_cookie:
                logger.info(f"[LAZY-REGISTER] Cold-start auto-login for {username}")
                for attempt in range(3):
                    if attempt > 0:
                        await asyncio.sleep(random.uniform(1.0, 2.0))
                    login_response, cookie_jar = await auto_login(client, username, password, seed_cookies={})
                    if not is_login_failed(login_response):
                        break
                else:
                    raise HTTPException(status_code=401, detail="Cold-start login failed. Check credentials.")
                active_csrf = extract_csrf(login_response.text)
                php_sess_id = cookie_jar.get("PHPSESSID", "")
            else:
                active_csrf = unquote(csrf_cookie)

            register_url_with_csrf = f"{register_url}&_csrf={active_csrf}"
            response = await client.get(register_url_with_csrf, cookies=cookie_jar, timeout=15)

            if response.status_code in (301, 302, 303) or response.status_code == 500 or is_login_failed(response):
                logger.warning("[LAZY-REGISTER] Session invalid or redirected (302). Auto-healing context stream...")
                for attempt in range(3):
                    if attempt > 0:
                        await asyncio.sleep(random.uniform(1.0, 2.0))
                    login_response, cookie_jar = await auto_login(client, username, password, seed_cookies=cookie_jar)
                    if not is_login_failed(login_response):
                        break
                else:
                    raise HTTPException(status_code=401, detail="Authentication credentials expired.")

                active_csrf = extract_csrf(login_response.text) or cookie_jar.get("_csrf", "")
                register_url_with_csrf = f"{register_url}&_csrf={active_csrf}"
                response = await client.get(register_url_with_csrf, cookies=cookie_jar, timeout=15)

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
        # Marks detail URLs contain encrypted binary parameters with non-UTF-8 bytes 
        # (e.g. %F5%B6%EA%9F...) which AWS API Gateway REST APIs parse, re-encode and corrupt, 
        # resulting in HTTP 400 Bad Request. We request directly from the host.
        async with httpx.AsyncClient(
            verify=False, headers=DEFAULT_HEADERS, http2=True,
            event_hooks={"response": [log_rate_limit]}
        ) as client:
            response = await client.get(full_detail_url, cookies=cookie_jar, timeout=15)

            if response.status_code in (301, 302, 303) or response.status_code == 500 or is_login_failed(response):
                logger.warning("[MARKS DETAIL] Token expired. Launching auto-login fallback...")
                for attempt in range(3):
                    if attempt > 0:
                        await asyncio.sleep(random.uniform(1.0, 2.0))
                    res, cookie_jar = await auto_login(client, username, password, seed_cookies=cookie_jar)
                    if not is_login_failed(res):
                        break
                else:
                    raise HTTPException(status_code=401, detail="Session verification recovery rejected.")

                response = await client.get(full_detail_url, cookies=cookie_jar, timeout=15)

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
# BREAK-TIME GAMES · LEADERBOARD (JWT anti-cheat + MongoDB)
# ============================================================

GAME_JWT_SECRET = os.environ.get("GAME_JWT_SECRET", "change-me-in-production-kl-games")
GAME_TOKEN_TTL = 600  # 10 minutes

# ------------------ Mongo init (optional, in-memory fallback) ------------------
_scores_col = None
_jti_col = None

MONGODB_URI = os.environ.get("MONGODB_URI", "")
if _pymongo_available and MONGODB_URI:
    try:
        _mongo_client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=4000)
        _mongo_client.admin.command("ping")
        _game_db = _mongo_client["timetablekl"]
        _scores_col = _game_db["game_scores"]
        _jti_col = _game_db["game_jtis"]
        _radio_state_col = _game_db["radio_state"]
        _radio_queue_col = _game_db["radio_queue"]
        _radio_history_col = _game_db["radio_history"]
        _radio_cooldowns_col = _game_db["radio_cooldowns"]
        # Daily leagues: scores are scoped per (gameId, userId, day, device) —
        # a player's phone best and pc best are separate records, so a score
        # made on one device can never leak into the other league's board.
        for legacy in ("gameId_1_userId_1", "gameId_1_userId_1_day_1"):
            try:
                _scores_col.drop_index(legacy)
            except Exception:
                pass
        _scores_col.create_index(
            [("gameId", ASCENDING), ("userId", ASCENDING), ("day", ASCENDING), ("device", ASCENDING)],
            unique=True,
        )
        _jti_col.create_index("expiresAt", expireAfterSeconds=0)
        _radio_history_col.create_index("played_at")
        _radio_queue_col.create_index("queue_id", unique=True)
        _radio_cooldowns_col.create_index("user_id", unique=True)
        logger.info("✅ MongoDB connected — game leaderboard and campus radio persistent.")
    except Exception as e:
        logger.error(f"[GAME/RADIO] MongoDB init failed, using in-memory fallback: {e}")
        _scores_col = None
        _jti_col = None
        _radio_state_col = None
        _radio_queue_col = None
        _radio_history_col = None
        _radio_cooldowns_col = None

_mem_scores = {}  # (gameId, userId, day) -> score int
_mem_jtis = {}    # jti -> expiry epoch

def _game_day() -> str:
    """IST calendar day — the leaderboard resets at midnight IST."""
    ist = datetime.datetime.utcnow() + datetime.timedelta(hours=5, minutes=30)
    return ist.strftime("%Y-%m-%d")

# ------------------ Anti-cheat: gameplay validation ------------------
# KL student IDs are 10-digit numbers (e.g. 2400032717).
KL_ID_RE = re.compile(r"^\d{10}$")

# Per-point playtime bounds, derived from simulating the client's fixed
# spawn rhythm (60fps): fastest possible ~0.61s/pt, slowest legit ~1.48s/pt.
MIN_SEC_PER_POINT = 0.55   # below this = bot instant-submit
MAX_SEC_PER_POINT = 1.60   # above this = started token, idled, then submitted
MAX_TIME_SLACK = 10.0      # covers screen travel time + network latency

# Per-userId throttle (campus shares one NAT IP, so per-IP is useless).
_throttle_buckets = {}  # (action, userId) -> [epoch, ...]
THROTTLE_WINDOW = 60.0
THROTTLE_LIMIT = 12     # max starts+submits combined per user per minute

def _throttle_ok(action: str, user_id: str) -> bool:
    now = time.time()
    key = (action, user_id)
    arr = [t for t in _throttle_buckets.get(key, []) if now - t < THROTTLE_WINDOW]
    if len(arr) >= THROTTLE_LIMIT:
        _throttle_buckets[key] = arr
        return False
    arr.append(now)
    _throttle_buckets[key] = arr
    return True

# ------------------ Minimal HS256 JWT (no external dep) ------------------
def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

def _b64url_decode(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))

def create_game_jwt(payload: dict, ttl: int = GAME_TOKEN_TTL) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    now = int(time.time())
    body = {**payload, "iat": now, "exp": now + ttl}
    h = _b64url_encode(json.dumps(header, separators=(",", ":")).encode())
    p = _b64url_encode(json.dumps(body, separators=(",", ":")).encode())
    sig = _b64url_encode(hmac.new(GAME_JWT_SECRET.encode(), f"{h}.{p}".encode(), hashlib.sha256).digest())
    return f"{h}.{p}.{sig}"

def verify_game_jwt(token: str) -> dict:
    try:
        h, p, sig = token.split(".")
        expected = _b64url_encode(
            hmac.new(GAME_JWT_SECRET.encode(), f"{h}.{p}".encode(), hashlib.sha256).digest()
        )
        if not hmac.compare_digest(sig, expected):
            raise HTTPException(status_code=401, detail="Invalid game token signature.")
        payload = json.loads(_b64url_decode(p))
        if int(time.time()) > payload.get("exp", 0):
            raise HTTPException(status_code=401, detail="Game token expired. Start a new run.")
        return payload
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=401, detail="Malformed game token.")

# ------------------ Replay protection (single-use jti) ------------------
def _consume_jti(jti: str, exp: int) -> bool:
    """True if fresh (now consumed); False if replay."""
    if not jti:
        return False
    if _jti_col is not None:
        try:
            _jti_col.insert_one({
                "jti": jti,
                "expiresAt": datetime.datetime.utcfromtimestamp(exp + 120),
            })
            return True
        except DuplicateKeyError:
            return False
        except Exception as e:
            logger.error(f"[GAME] jti store error: {e}")
            return False
    now = time.time()
    for k in [k for k, v in _mem_jtis.items() if v < now]:
        _mem_jtis.pop(k, None)
    if jti in _mem_jtis:
        return False
    _mem_jtis[jti] = exp + 120
    return True

# ------------------ Leaderboard helpers (daily reset, IST) ------------------
def _save_score(game_id: str, user_id: str, score: int, device: str = "unknown") -> int:
    """Upsert today's personal best *in this device league*. Returns that best."""
    day = _game_day()
    if _scores_col is not None:
        _scores_col.update_one(
            {"gameId": game_id, "userId": user_id, "day": day, "device": device},
            {"$max": {"score": score}, "$set": {"updatedAt": datetime.datetime.utcnow()}},
            upsert=True,
        )
        doc = _scores_col.find_one(
            {"gameId": game_id, "userId": user_id, "day": day, "device": device}
        ) or {}
        return int(doc.get("score", score))
    key = (game_id, user_id, day, device)
    _mem_scores[key] = max(score, _mem_scores.get(key, 0))
    return _mem_scores[key]

def _top_scores(game_id: str, limit: int = 3, device: str | None = None) -> list:
    """Top scores for today, optionally filtered by device class (mobile/desktop)."""
    day = _game_day()
    query = {"gameId": game_id, "day": day}
    if device in ("mobile", "desktop"):
        query["device"] = device
    if _scores_col is not None:
        cursor = _scores_col.find(query).sort("score", -1).limit(limit)
        return [
            {"userId": d["userId"], "score": int(d["score"]), "device": d.get("device", "unknown")}
            for d in cursor
        ]
    ranked = sorted(
        (
            (uid, sc, dev)
            for (gid, uid, d, dev), sc in _mem_scores.items()
            if gid == game_id and d == day
            and (device not in ("mobile", "desktop") or dev == device)
        ),
        key=lambda x: x[1], reverse=True,
    )[:limit]
    return [{"userId": uid, "score": sc, "device": dev} for uid, sc, dev in ranked]

# ------------------ Game endpoints ------------------
@app.post("/api/game/start")
@app.post("/game/start")
async def game_start(
    userId: str = Form(...),
    gameId: str = Form(default="dino"),
    deviceClass: str = Form(default="unknown"),
):
    """Issue a short-lived signed game token. One token = one run."""
    user_id = userId.strip()
    if not KL_ID_RE.match(user_id):
        raise HTTPException(status_code=400, detail="Invalid student ID format.")
    if not _throttle_ok("start", user_id):
        raise HTTPException(status_code=429, detail="Too many game starts. Slow down.")
    device = deviceClass if deviceClass in ("mobile", "desktop") else "unknown"
    token = create_game_jwt({
        "userId": user_id,
        "gameId": gameId,
        "device": device,
        "startTime": int(time.time()),
        "jti": secrets.token_hex(16),
    })
    logger.info(f"[GAME] Token issued for {user_id} ({gameId})")
    return {"success": True, "token": token, "expiresIn": GAME_TOKEN_TTL}

@app.post("/api/game/submit")
@app.post("/game/submit")
async def game_submit(token: str = Form(...), score: int = Form(...)):
    """Verify token, sanity-check elapsed time, persist personal best."""
    payload = verify_game_jwt(token)
    user_id = payload.get("userId", "")
    game_id = payload.get("gameId", "dino")
    jti = payload.get("jti", "")
    start_time = float(payload.get("startTime", 0))
    elapsed = time.time() - start_time

    if not _consume_jti(jti, int(payload.get("exp", 0))):
        raise HTTPException(status_code=409, detail="Game token already used.")

    if not _throttle_ok("submit", user_id):
        raise HTTPException(status_code=429, detail="Too many submissions. Slow down.")

    if score < 0 or score > 10000:
        raise HTTPException(status_code=400, detail="Invalid score.")

    # Time-band check: obstacles spawn on a fixed, unstoppable rhythm, so any
    # score has a minimum AND maximum plausible playtime.
    if elapsed < score * MIN_SEC_PER_POINT - 1.0:
        logger.warning(f"[GAME] Rejected too-fast score: {user_id} score={score} elapsed={elapsed:.1f}s")
        raise HTTPException(status_code=400, detail="Score rejected: impossibly fast run.")
    if elapsed > score * MAX_SEC_PER_POINT + MAX_TIME_SLACK:
        logger.warning(f"[GAME] Rejected idle-then-submit: {user_id} score={score} elapsed={elapsed:.1f}s")
        raise HTTPException(status_code=400, detail="Score rejected: run took impossibly long.")

    best = _save_score(game_id, user_id, score, payload.get("device", "unknown"))
    logger.info(f"[GAME] {user_id} scored {score} on {game_id} (best {best}) in {elapsed:.1f}s")
    return {
        "success": True,
        "best": best,
        # board for the run's own device class — what the player actually races in
        "leaderboard": _top_scores(game_id, device=payload.get("device")),
    }

@app.get("/api/game/leaderboard")
@app.get("/game/leaderboard")
async def game_leaderboard(gameId: str = "dino", limit: int = 3, device: str = "all"):
    dev = device if device in ("mobile", "desktop") else None
    return {"success": True, "gameId": gameId, "leaderboard": _top_scores(gameId, limit, dev)}


# ============================================================
# CAMPUS RADIO · SYNCHRONIZED PLAYBACK & TIME-DECAYED QUEUE
# ============================================================

import math
from urllib.parse import quote_plus

RADIO_JWT_SECRET = os.environ.get("RADIO_JWT_SECRET", "change-me-in-production-kl-radio")
RADIO_TOKEN_TTL = 2 * 86400  # 2 days
RADIO_REPORT_THRESHOLD = int(os.environ.get("RADIO_REPORT_THRESHOLD", "8"))
RADIO_ADD_COOLDOWN_SEC = 600  # 10 minutes between song additions per student
RADIO_MAX_USER_QUEUE = 2      # Max active songs in queue per student
RADIO_ANTI_REPEAT_SEC = 2700  # 45 minutes anti-repeat window

_radio_advance_lock = asyncio.Lock()

# ------------------ Radio In-Memory State & Fallback ------------------
_mem_radio_state = {
    "track": None,         # dict or None
    "started_at": 0,       # epoch ms
    "status": "idle",      # "playing" | "idle"
    "reports": [],         # list of user_ids who reported current track
}
_mem_radio_queue = []      # list of dicts
_mem_radio_history = []    # list of dicts: {"videoId", "title", "artist", "added_by", "played_at"}
_mem_radio_cooldowns = {}  # user_id -> float (last add epoch)
_verified_students_cache = {}  # user_id -> float (last verified epoch)

def create_radio_jwt(payload: dict, ttl: int = RADIO_TOKEN_TTL) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    now = int(time.time())
    body = {**payload, "iat": now, "exp": now + ttl}
    h = _b64url_encode(json.dumps(header, separators=(",", ":")).encode())
    p = _b64url_encode(json.dumps(body, separators=(",", ":")).encode())
    sig = _b64url_encode(hmac.new(RADIO_JWT_SECRET.encode(), f"{h}.{p}".encode(), hashlib.sha256).digest())
    return f"{h}.{p}.{sig}"

def verify_radio_jwt(token: str) -> dict:
    try:
        h, p, sig = token.split(".")
        expected = _b64url_encode(
            hmac.new(RADIO_JWT_SECRET.encode(), f"{h}.{p}".encode(), hashlib.sha256).digest()
        )
        if not hmac.compare_digest(sig, expected):
            raise HTTPException(status_code=401, detail="Invalid radio token signature.")
        payload = json.loads(_b64url_decode(p))
        if int(time.time()) > payload.get("exp", 0):
            raise HTTPException(status_code=401, detail="Radio token expired. Re-authenticate.")
        return payload
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=401, detail="Malformed radio token.")

# ------------------ Radio Student Auth Helper ------------------
async def _extract_radio_user(
    request: Request,
    username: str = Form(default=""),
    password: str = Form(default=""),
    php_sess_id: str = Form(default=""),
    csrf_cookie: str = Form(default=""),
    device_id: str = Form(default=""),
    server_id: str = Form(default="erp3"),
) -> str:
    """Authenticates the student via Bearer JWT or directly via ERP credentials/session."""
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
        payload = verify_radio_jwt(token)
        user_id = payload.get("userId", "")
        if user_id:
            return str(user_id)

    if username:
        clean_user = username.strip()
        now = time.time()
        # Fast path if student verified recently (last 24 hours)
        if _verified_students_cache.get(clean_user, 0) > now - 86400:
            return clean_user

        # Authenticate against ERP
        seed_cookies = {
            "_csrf": unquote(csrf_cookie) if csrf_cookie else "",
            "PHPSESSID": php_sess_id,
            "kl_erp_device_id": unquote(device_id) if device_id else "",
            "SERVERID": server_id,
        }
        async with httpx.AsyncClient(verify=False, limits=limits_pool, headers=DEFAULT_HEADERS, http2=True) as client:
            login_resp, _ = await auto_login(client, clean_user, password, seed_cookies=seed_cookies)
            if is_login_failed(login_resp):
                raise HTTPException(status_code=401, detail="Invalid university credentials for radio access.")

        _verified_students_cache[clean_user] = now
        return clean_user

    raise HTTPException(status_code=401, detail="Authentication required. Provide Authorization Bearer token or student credentials.")

# ------------------ Zero-API-Key YouTube Music Search ------------------
def _parse_yt_duration(dur_str: str) -> int:
    try:
        parts = dur_str.strip().split(":")
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except Exception:
        pass
    return 0

async def search_youtube_music_no_key(query: str, limit: int = 15) -> list[dict]:
    """Scrapes YouTube search results without needing any Google/YouTube API key."""
    clean_query = query.strip()
    if not clean_query:
        return []

    # sp=EgIQAQ%253D%253D filters results to videos only
    search_url = f"https://www.youtube.com/results?search_query={quote_plus(clean_query)}&sp=EgIQAQ%253D%253D"
    headers = {
        "User-Agent": DEFAULT_HEADERS["User-Agent"],
        "Accept-Language": "en-US,en;q=0.9,te;q=0.8,hi;q=0.7",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    try:
        async with httpx.AsyncClient(verify=False, headers=headers, follow_redirects=True, timeout=12) as client:
            resp = await client.get(search_url)
            html = resp.text

        match = re.search(r'var ytInitialData = ({.*?});</script>', html)
        if not match:
            match = re.search(r'ytInitialData\s*=\s*({.+?});', html)
        if not match:
            logger.warning("[RADIO SEARCH] Could not find ytInitialData in response.")
            return []

        data = json.loads(match.group(1))
        videos = []
        sections = (
            data.get("contents", {})
            .get("twoColumnSearchResultsRenderer", {})
            .get("primaryContents", {})
            .get("sectionListRenderer", {})
            .get("contents", [])
        )

        for sec in sections:
            item_sec = sec.get("itemSectionRenderer", {})
            for item in item_sec.get("contents", []):
                if "videoRenderer" in item:
                    vr = item["videoRenderer"]
                    vid = vr.get("videoId")
                    title_runs = vr.get("title", {}).get("runs", [])
                    title = title_runs[0].get("text", "") if title_runs else ""
                    owner_runs = vr.get("ownerText", {}).get("runs", [])
                    channel = owner_runs[0].get("text", "") if owner_runs else "Unknown Artist"
                    dur_text = vr.get("lengthText", {}).get("simpleText", "")
                    dur_sec = _parse_yt_duration(dur_text) if dur_text else 0
                    thumbs = vr.get("thumbnail", {}).get("thumbnails", [])
                    thumb = thumbs[-1].get("url", "") if thumbs else f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg"

                    # Duration filtering: 60s (1 min) to 600s (10 min) bounds per spec
                    if vid and title and 60 <= dur_sec <= 600:
                        videos.append({
                            "videoId": vid,
                            "title": title,
                            "artist": channel,
                            "duration_sec": dur_sec,
                            "duration_text": dur_text,
                            "thumbnail": thumb,
                        })
                        if len(videos) >= limit:
                            break
            if len(videos) >= limit:
                break

        return videos
    except Exception as e:
        logger.error(f"[RADIO SEARCH] Scrape error for query '{query}': {e}", exc_info=True)
        return []

# ------------------ Radio DB & Persistence Helpers ------------------
def _get_radio_state_doc() -> dict:
    if _radio_state_col is not None:
        try:
            doc = _radio_state_col.find_one({"_id": "current_state"})
            if doc:
                return doc
        except Exception as e:
            logger.error(f"[RADIO] Mongo state read error: {e}")
    return _mem_radio_state

def _save_radio_state_doc(state: dict):
    global _mem_radio_state
    _mem_radio_state = state
    if _radio_state_col is not None:
        try:
            _radio_state_col.replace_one(
                {"_id": "current_state"},
                {"_id": "current_state", **state},
                upsert=True
            )
        except Exception as e:
            logger.error(f"[RADIO] Mongo state save error: {e}")

def _get_radio_queue_docs() -> list[dict]:
    if _radio_queue_col is not None:
        try:
            return list(_radio_queue_col.find({}, {"_id": 0}))
        except Exception as e:
            logger.error(f"[RADIO] Mongo queue read error: {e}")
    return list(_mem_radio_queue)

def _add_radio_queue_doc(item: dict):
    if _radio_queue_col is not None:
        try:
            _radio_queue_col.insert_one({**item})
        except Exception as e:
            logger.error(f"[RADIO] Mongo queue insert error: {e}")
    _mem_radio_queue.append(item)

def _remove_radio_queue_doc(queue_id: str):
    global _mem_radio_queue
    if _radio_queue_col is not None:
        try:
            _radio_queue_col.delete_one({"queue_id": queue_id})
        except Exception as e:
            logger.error(f"[RADIO] Mongo queue delete error: {e}")
    _mem_radio_queue = [q for q in _mem_radio_queue if q.get("queue_id") != queue_id]

def _update_radio_queue_votes(queue_id: str, votes: list[str]):
    global _mem_radio_queue
    if _radio_queue_col is not None:
        try:
            _radio_queue_col.update_one({"queue_id": queue_id}, {"$set": {"votes": votes}})
        except Exception as e:
            logger.error(f"[RADIO] Mongo queue vote update error: {e}")
    for q in _mem_radio_queue:
        if q.get("queue_id") == queue_id:
            q["votes"] = votes
            break

def _add_radio_history_doc(item: dict):
    hist_entry = {
        "videoId": item.get("videoId"),
        "title": item.get("title"),
        "artist": item.get("artist"),
        "added_by": item.get("added_by"),
        "played_at": time.time(),
    }
    if _radio_history_col is not None:
        try:
            _radio_history_col.insert_one(hist_entry)
        except Exception as e:
            logger.error(f"[RADIO] Mongo history insert error: {e}")
    _mem_radio_history.append(hist_entry)

def _get_recent_radio_history(since_sec: int = RADIO_ANTI_REPEAT_SEC) -> list[dict]:
    cutoff = time.time() - since_sec
    if _radio_history_col is not None:
        try:
            return list(_radio_history_col.find({"played_at": {"$gte": cutoff}}, {"_id": 0}))
        except Exception as e:
            logger.error(f"[RADIO] Mongo history read error: {e}")
    return [h for h in _mem_radio_history if h.get("played_at", 0) >= cutoff]

def _check_and_update_cooldown(user_id: str) -> tuple[bool, int]:
    """Returns (is_allowed, remaining_cooldown_seconds)."""
    now = time.time()
    last_add = 0.0
    if _radio_cooldowns_col is not None:
        try:
            doc = _radio_cooldowns_col.find_one({"user_id": user_id})
            if doc:
                last_add = doc.get("last_add", 0.0)
        except Exception as e:
            logger.error(f"[RADIO] Mongo cooldown read error: {e}")
    else:
        last_add = _mem_radio_cooldowns.get(user_id, 0.0)

    elapsed = now - last_add
    if elapsed < RADIO_ADD_COOLDOWN_SEC:
        return False, int(RADIO_ADD_COOLDOWN_SEC - elapsed)

    # Update cooldown
    if _radio_cooldowns_col is not None:
        try:
            _radio_cooldowns_col.update_one(
                {"user_id": user_id},
                {"$set": {"user_id": user_id, "last_add": now}},
                upsert=True
            )
        except Exception as e:
            logger.error(f"[RADIO] Mongo cooldown save error: {e}")
    _mem_radio_cooldowns[user_id] = now
    return True, 0

# ------------------ Time-Decayed Queue Scoring & Weighted Lottery ------------------
def _calculate_decayed_score(item: dict, now: float = None) -> float:
    """Calculates time-decayed score: (votes + 1) / (hours_since_added + 1)^1.5"""
    if now is None:
        now = time.time()
    added_at = float(item.get("added_at", now))
    hours_since_added = max(0.0, (now - added_at) / 3600.0)
    votes_count = len(item.get("votes", []))
    score = (votes_count + 1.0) / math.pow(hours_since_added + 1.0, 1.5)
    return round(score, 4)

async def _advance_radio_track(force: bool = False) -> dict:
    """Picks the next track via weighted lottery from top ~10 scored items with anti-repeat rules."""
    async with _radio_advance_lock:
        state = _get_radio_state_doc()
        now_ms = int(time.time() * 1000)
        now_sec = time.time()

        current_track = state.get("track")
        if current_track and not force:
            duration_ms = int(current_track.get("duration_sec", 0)) * 1000
            started_at = int(state.get("started_at", 0))
            if now_ms - started_at < duration_ms:
                # Track is still playing, do not advance
                return state

        queue = _get_radio_queue_docs()
        if not queue:
            # Queue is empty, radio goes idle
            new_state = {
                "track": None,
                "started_at": 0,
                "status": "idle",
                "reports": [],
            }
            _save_radio_state_doc(new_state)
            logger.info("[RADIO] Queue is empty. Radio is now idle.")
            return new_state

        # Compute score for all items in queue
        for item in queue:
            item["score"] = _calculate_decayed_score(item, now_sec)

        # Sort descending by score
        queue.sort(key=lambda x: x["score"], reverse=True)
        top_pool = queue[:10]

        # Anti-repeat check: avoid same artist or submitter in last 45 minutes
        recent_history = _get_recent_radio_history(since_sec=RADIO_ANTI_REPEAT_SEC)
        recent_artists = {h.get("artist", "").strip().lower() for h in recent_history if h.get("artist")}
        recent_submitters = {h.get("added_by", "").strip() for h in recent_history if h.get("added_by")}

        eligible_pool = [
            item for item in top_pool
            if item.get("artist", "").strip().lower() not in recent_artists
            and item.get("added_by", "").strip() not in recent_submitters
        ]

        # If all candidates in top 10 violate anti-repeat, fallback to top_pool
        pool_to_draw = eligible_pool if eligible_pool else top_pool
        weights = [max(0.01, item["score"]) for item in pool_to_draw]

        # Weighted random pick
        chosen = random.choices(pool_to_draw, weights=weights, k=1)[0]

        # Remove chosen from queue
        _remove_radio_queue_doc(chosen["queue_id"])

        # Add to history
        _add_radio_history_doc(chosen)

        # Set as current track
        new_state = {
            "track": {
                "videoId": chosen["videoId"],
                "title": chosen["title"],
                "artist": chosen["artist"],
                "duration_sec": chosen["duration_sec"],
                "duration_text": chosen.get("duration_text", ""),
                "thumbnail": chosen.get("thumbnail", ""),
                "added_by": chosen.get("added_by", "anonymous"),
            },
            "started_at": now_ms,
            "status": "playing",
            "reports": [],
        }
        _save_radio_state_doc(new_state)
        logger.info(f"[RADIO] 🎵 Now playing: '{chosen['title']}' by {chosen['artist']} (queued by {chosen.get('added_by')})")
        return new_state

async def _get_current_radio_state(user_id: str = "") -> dict:
    """Returns the synchronized radio state, automatically advancing if the track finished."""
    state = _get_radio_state_doc()
    now_ms = int(time.time() * 1000)

    # Check if currently playing track has ended
    if state.get("status") == "playing" and state.get("track"):
        duration_ms = int(state["track"].get("duration_sec", 0)) * 1000
        started_at = int(state.get("started_at", 0))
        if now_ms - started_at >= duration_ms:
            state = await _advance_radio_track(force=False)

    # Format queue with score and user vote indicator
    queue = _get_radio_queue_docs()
    now_sec = time.time()
    for item in queue:
        item["score"] = _calculate_decayed_score(item, now_sec)
        item["votes_count"] = len(item.get("votes", []))
        item["user_voted"] = bool(user_id and user_id in item.get("votes", []))

    queue.sort(key=lambda x: x["score"], reverse=True)

    elapsed_ms = 0
    if state.get("status") == "playing" and state.get("started_at"):
        elapsed_ms = max(0, now_ms - int(state["started_at"]))

    return {
        "success": True,
        "server_time": now_ms,
        "status": state.get("status", "idle"),
        "started_at": state.get("started_at", 0),
        "elapsed_ms": elapsed_ms,
        "current_track": state.get("track"),
        "reports_count": len(state.get("reports", [])),
        "queue": queue,
    }

# ============================================================
# RADIO ENDPOINTS
# ============================================================

@app.post("/api/radio/auth")
@app.post("/radio/auth")
async def radio_auth(
    username: str = Form(...),
    password: str = Form(...),
    php_sess_id: str = Form(default=""),
    csrf_cookie: str = Form(default=""),
    device_id: str = Form(default=""),
    server_id: str = Form(default="erp3"),
):
    """Verifies student identity against university ERP and issues a 30-day signed radio JWT."""
    clean_user = username.strip()
    seed_cookies = {
        "_csrf": unquote(csrf_cookie) if csrf_cookie else "",
        "PHPSESSID": php_sess_id,
        "kl_erp_device_id": unquote(device_id) if device_id else "",
        "SERVERID": server_id,
    }

    try:
        async with httpx.AsyncClient(verify=False, limits=limits_pool, headers=DEFAULT_HEADERS, http2=True) as client:
            login_resp, _ = await auto_login(client, clean_user, password, seed_cookies=seed_cookies)
            if is_login_failed(login_resp):
                raise HTTPException(status_code=401, detail="Invalid ERP login credentials.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[RADIO AUTH] ERP connection fault: {e}")
        raise HTTPException(status_code=503, detail="University portal unreachable for verification.")

    _verified_students_cache[clean_user] = time.time()
    token = create_radio_jwt({"userId": clean_user, "isVerified": True})
    logger.info(f"[RADIO] Issued radio JWT for student {clean_user}")
    return {
        "success": True,
        "token": token,
        "userId": clean_user,
        "expiresIn": RADIO_TOKEN_TTL,
    }

@app.get("/api/radio/search")
@app.post("/api/radio/search")
@app.get("/radio/search")
@app.post("/radio/search")
async def radio_search(q: str = ""):
    """Searches YouTube directly for songs (duration 1-10 min) with zero API key required."""
    if not q or len(q.strip()) < 2:
        raise HTTPException(status_code=400, detail="Search query must be at least 2 characters.")
    results = await search_youtube_music_no_key(q, limit=15)
    return {"success": True, "query": q, "results": results}

@app.get("/api/radio/state")
@app.get("/radio/state")
async def radio_state(request: Request):
    """Returns synchronized state (current track, server_time, elapsed_ms, queue)."""
    user_id = ""
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        try:
            payload = verify_radio_jwt(auth_header[7:].strip())
            user_id = payload.get("userId", "")
        except Exception:
            pass
    return await _get_current_radio_state(user_id=user_id)

@app.post("/api/radio/queue")
@app.post("/radio/queue")
async def radio_add_queue(
    request: Request,
    videoId: str = Form(...),
    title: str = Form(...),
    artist: str = Form(...),
    duration_sec: int = Form(...),
    duration_text: str = Form(default=""),
    thumbnail: str = Form(default=""),
    username: str = Form(default=""),
    password: str = Form(default=""),
    php_sess_id: str = Form(default=""),
    csrf_cookie: str = Form(default=""),
    device_id: str = Form(default=""),
    server_id: str = Form(default="erp3"),
):
    """Adds a song to the radio queue (enforces 10-min cooldown, max 2 active songs per student)."""
    user_id = await _extract_radio_user(
        request, username=username, password=password, php_sess_id=php_sess_id,
        csrf_cookie=csrf_cookie, device_id=device_id, server_id=server_id
    )

    # Sanity checks
    if duration_sec < 60 or duration_sec > 600:
        raise HTTPException(status_code=400, detail="Song duration must be between 1 and 10 minutes.")

    # Check cooldown (1 add per 10 minutes per student)
    allowed, remaining = _check_and_update_cooldown(user_id)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Cooldown active: please wait {remaining // 60}m {remaining % 60}s before queuing another song."
        )

    # Check active queue limit per user (max 2)
    current_queue = _get_radio_queue_docs()
    user_queued_count = sum(1 for q in current_queue if q.get("added_by") == user_id)
    if user_queued_count >= RADIO_MAX_USER_QUEUE:
        raise HTTPException(
            status_code=429,
            detail=f"You already have {user_queued_count} songs in the queue. Wait for them to play first."
        )

    # Check if song is already in queue
    if any(q.get("videoId") == videoId for q in current_queue):
        raise HTTPException(status_code=400, detail="This song is already in the queue.")

    queue_item = {
        "queue_id": secrets.token_hex(12),
        "videoId": videoId.strip(),
        "title": title.strip(),
        "artist": artist.strip(),
        "duration_sec": int(duration_sec),
        "duration_text": duration_text.strip() or f"{duration_sec // 60}:{duration_sec % 60:02d}",
        "thumbnail": thumbnail.strip() or f"https://i.ytimg.com/vi/{videoId}/hqdefault.jpg",
        "added_by": user_id,
        "added_at": time.time(),
        "votes": [user_id],  # Submitter automatically upvotes
    }

    _add_radio_queue_doc(queue_item)
    logger.info(f"[RADIO] {user_id} queued '{title}' ({videoId})")

    # If radio is idle, start playing immediately!
    state = _get_radio_state_doc()
    if state.get("status") != "playing" or not state.get("track"):
        await _advance_radio_track(force=True)

    return await _get_current_radio_state(user_id=user_id)

@app.post("/api/radio/vote")
@app.post("/radio/vote")
async def radio_vote(
    request: Request,
    queue_id: str = Form(...),
    username: str = Form(default=""),
    password: str = Form(default=""),
    php_sess_id: str = Form(default=""),
    csrf_cookie: str = Form(default=""),
    device_id: str = Form(default=""),
    server_id: str = Form(default="erp3"),
):
    """Upvotes a song in the queue (1 vote per song per student ID)."""
    user_id = await _extract_radio_user(
        request, username=username, password=password, php_sess_id=php_sess_id,
        csrf_cookie=csrf_cookie, device_id=device_id, server_id=server_id
    )

    queue = _get_radio_queue_docs()
    target_item = next((q for q in queue if q.get("queue_id") == queue_id), None)
    if not target_item:
        raise HTTPException(status_code=404, detail="Song not found in queue.")

    votes = target_item.get("votes", [])
    if user_id in votes:
        raise HTTPException(status_code=400, detail="You have already voted for this song.")

    votes.append(user_id)
    _update_radio_queue_votes(queue_id, votes)
    logger.info(f"[RADIO] {user_id} upvoted {target_item.get('title')} (total votes: {len(votes)})")

    return await _get_current_radio_state(user_id=user_id)

@app.post("/api/radio/report")
@app.post("/radio/report")
async def radio_report(
    request: Request,
    reason: str = Form(default="inappropriate"),
    username: str = Form(default=""),
    password: str = Form(default=""),
    php_sess_id: str = Form(default=""),
    csrf_cookie: str = Form(default=""),
    device_id: str = Form(default=""),
    server_id: str = Form(default="erp3"),
):
    """Reports currently playing track. Force-skips when threshold (>= 3) is reached."""
    user_id = await _extract_radio_user(
        request, username=username, password=password, php_sess_id=php_sess_id,
        csrf_cookie=csrf_cookie, device_id=device_id, server_id=server_id
    )

    state = _get_radio_state_doc()
    if state.get("status") != "playing" or not state.get("track"):
        raise HTTPException(status_code=400, detail="No track is currently playing to report.")

    reports = state.get("reports", [])
    if user_id in reports:
        raise HTTPException(status_code=400, detail="You have already reported this track.")

    reports.append(user_id)
    state["reports"] = reports
    _save_radio_state_doc(state)

    logger.warning(f"[RADIO REPORT] {user_id} reported current track '{state['track']['title']}' (reports: {len(reports)}/{RADIO_REPORT_THRESHOLD})")

    # Threshold reached -> force-skip
    if len(reports) >= RADIO_REPORT_THRESHOLD:
        logger.info(f"[RADIO] Report threshold reached ({len(reports)}). Force-skipping track...")
        await _advance_radio_track(force=True)

    return await _get_current_radio_state(user_id=user_id)

@app.post("/api/radio/advance")
@app.post("/radio/advance")
async def radio_advance():
    """Advances to the next track if the current song has completed."""
    await _advance_radio_track(force=False)
    return await _get_current_radio_state()

