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
from bs4 import BeautifulSoup

# ------------------ LOGGING WITH TIME SEEDS & FILE PERSISTENCE ------------------
os.makedirs("logs", exist_ok=True)

log_format_string = "%(asctime)s.%(munit)s [%(levelname)s] %(message)s"
log_date_format = "%Y-%m-%d %H:%M:%S"

log_formatter = logging.Formatter(fmt=log_format_string, datefmt=log_date_format)

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)

file_handler = logging.FileHandler("logs/production_api.log", mode="a", encoding="utf-8")
file_handler.setFormatter(log_formatter)

logging.basicConfig(
    level=logging.INFO,
    handlers=[console_handler, file_handler]
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

# ------------------ STRUCTURAL STATICS (OPTIMIZED RESIDENTIAL FINGERPRINT) ------------------
BASE_URL = "https://newerp.kluniversity.in"

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

# ------------------ GLOBAL CONNECTION LIFESPAN ------------------
limits_pool = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global limits_pool
    logger.info("✅ FastAPI app starting (Universal Concurrent Auto-Healing Engine)...")
    limits_pool = httpx.Limits(max_keepalive_connections=50, max_connections=200, keepalive_expiry=30.0)
    logger.info("🚀 Global Concurrent Resource Pool initialized with HTTP/2 Overrides.")
    yield
    logger.info("🛑 Global Connection Pool safely terminated.")

app = FastAPI(title="TimeTable & Attendance Backend", version="7.5.0", lifespan=lifespan)

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
    return {"message": "Backend running high-speed concurrent loops ✅", "status": "healthy"}

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
    await asyncio.sleep(random.uniform(0.4, 0.8))

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
    await asyncio.sleep(random.uniform(0.3, 0.6))

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
async def login(username: str = Form(...), password: str = Form(...)):
    try:
        async with httpx.AsyncClient(verify=False, limits=limits_pool, headers=DEFAULT_HEADERS, http2=True) as client:
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
    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.NetworkError, httpx.TimeoutException) as net_err:
        logger.error(f"[NETWORK REJECTION] /login - University gateway down: {net_err}")
        raise HTTPException(
            status_code=503, 
            detail="University ERP portal is down or under maintenance. Please try again later."
        )
    except Exception as e:
        logger.error(f"[LOGIN_ROUTE] Exception: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal processing fault during authorization sync.")

# ------------------ FETCH ATTENDANCE ------------------
@app.post("/fetch-attendance")
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
        async with httpx.AsyncClient(verify=False, limits=limits_pool, headers=DEFAULT_HEADERS, http2=True) as client:

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
    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.NetworkError, httpx.TimeoutException) as net_err:
        logger.error(f"[NETWORK REJECTION] /fetch-attendance - University gateway down: {net_err}")
        raise HTTPException(
            status_code=503, 
            detail="University ERP portal is down or under maintenance. Please try again later."
        )
    except Exception as e:
        logger.error(f"[ATTENDANCE] Crash: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ------------------ FETCH REGISTER DETAIL ------------------
@app.post("/fetch-register-detail")
async def fetch_register_detail(
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
        async with httpx.AsyncClient(verify=False, limits=limits_pool, headers=DEFAULT_HEADERS, http2=True) as client:
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

        table_match = re.search(
            r'<table[^>]*class=["\']table table-striped table-bordered["\'][^>]*>(.*?)</table>',
            html_text, re.DOTALL | re.IGNORECASE
        )
        if not table_match:
            return {"success": False, "message": "Register table missing."}

        table_body = table_match.group(1)
        raw_headers = re.findall(r'<th.*?>(.*?)</th>', table_body, re.IGNORECASE)
        headers = [re.sub(r'<.*?>', '', h).strip() for h in raw_headers if h.strip()]

        metadata_count = 14
        metadata_headers = headers[:metadata_count]
        daily_headers = headers[metadata_count:]

        tbody_match = re.search(r'<tbody.*?>(.*?)</tbody>', table_body, re.DOTALL | re.IGNORECASE)
        if not tbody_match:
            return {"success": False, "message": "Calendar data rows missing."}

        cells = re.findall(r'<td.*?>(.*?)</td>', tbody_match.group(1), re.DOTALL | re.IGNORECASE)
        if len(cells) < metadata_count:
            return {"success": False, "message": "Truncated layout array returns."}

        metadata = {header: re.sub(r'<.*?>', '', cells[i]).strip()
                    for i, header in enumerate(metadata_headers) if i < len(cells)}

        daily_attendance = [
            {"date_slot": header, "status": re.sub(r'<.*?>', '', cells[metadata_count + i]).strip()}
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
    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.NetworkError, httpx.TimeoutException) as net_err:
        logger.error(f"[NETWORK REJECTION] /fetch-register-detail - University gateway down: {net_err}")
        raise HTTPException(
            status_code=503, 
            detail="University ERP portal is down or under maintenance. Please try again later."
        )
    except Exception as e:
        logger.error(f"[REGISTER] Crash: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ------------------ FETCH SEATING PLAN ------------------
@app.post("/fetch-seating-plan")
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
        async with httpx.AsyncClient(verify=False, limits=limits_pool, headers=DEFAULT_HEADERS, http2=True) as client:
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
    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.NetworkError, httpx.TimeoutException) as net_err:
        logger.error(f"[NETWORK REJECTION] /fetch-seating-plan - University gateway down: {net_err}")
        raise HTTPException(
            status_code=503, 
            detail="University ERP portal is down or under maintenance. Please try again later."
        )
    except Exception as e:
        logger.error(f"[SEATING] Crash: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ------------------ FETCH TIMETABLE ------------------
@app.post("/fetch-timetable")
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
        async with httpx.AsyncClient(verify=False, limits=limits_pool, headers=DEFAULT_HEADERS, http2=True) as client:
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
    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.NetworkError, httpx.TimeoutException) as net_err:
        logger.error(f"[NETWORK REJECTION] /fetch-timetable - University gateway down: {net_err}")
        raise HTTPException(
            status_code=503, 
            detail="University ERP portal is down or under maintenance. Please try again later."
        )
    except Exception as e:
        logger.error(f"[TIMETABLE] Crash: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/fetch-cgpa")
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
        async with httpx.AsyncClient(verify=False, limits=limits_pool, headers=DEFAULT_HEADERS, http2=True) as client:
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
    except Exception as e:
        logger.error(f"[CGPA ERROR] Processing sequence crashed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/fetch-marks-detail")
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
        async with httpx.AsyncClient(verify=False, limits=limits_pool, headers=DEFAULT_HEADERS, http2=True) as client:
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

        # Extract rows using matching id markers
        detail_table_match = re.search(r'<table id="w0".*?>(.*?)</table>', html_content, re.DOTALL | re.IGNORECASE)
        if not detail_table_match:
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
    except Exception as e:
        logger.error(f"[MARKS ERROR] Deep scorecard extraction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))