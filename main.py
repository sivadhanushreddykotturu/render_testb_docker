from fastapi import FastAPI, Form, HTTPException
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

# ------------------ CAPTCHA SOLVER INIT ------------------
try:
    with open("model/crnn.json", "r") as f:
        _captcha_meta = json.load(f)
    _captcha_alphabet = _captcha_meta["alphabet"]
    _captcha_img_w = _captcha_meta["img_w"]
    _captcha_img_h = _captcha_meta["img_h"]
    _captcha_session = ort.InferenceSession("model/crnn.onnx")
except Exception as e:
    print(f"Warning: Failed to load captcha model: {e}")

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

# ------------------ LOGGING ------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------ STRUCTURAL STATICS ------------------
BASE_URL = "https://newerp.kluniversity.in"
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1"
}

# ------------------ GLOBAL CONNECTION LIFESPAN ------------------
# We use an httpx.AsyncLimits pool to back individual isolated client frames natively.
limits_pool = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global limits_pool
    logger.info("✅ FastAPI app starting (Universal Concurrent Auto-Healing Engine)...")
    limits_pool = httpx.Limits(max_keepalive_connections=50, max_connections=200, keepalive_expiry=30.0)
    logger.info("🚀 Global Concurrent Resource Pool initialized.")
    yield
    logger.info("🛑 Global Connection Pool safely terminated.")

app = FastAPI(title="TimeTable & Attendance Backend", version="7.3.0", lifespan=lifespan)

# ------------------ CORS ------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Session-ID"],
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
    """
    THREAD-SAFE COURIER: Executes manual tracking utilizing isolated client 
    contexts passed directly from specific invocation runtimes.
    """
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

# ------------------ AUTO LOGIN (THREAD ISOLATED) ------------------
async def auto_login(client: httpx.AsyncClient, username: str, password: str, seed_cookies: dict) -> tuple[httpx.Response, dict]:
    login_url = f"{BASE_URL}/index.php?r=site%2Flogin"
    logger.info(f"[LOGIN] Running thread-isolated ONNX auto-login for user={username}")

    res, step_cookies = await _follow_redirects_collecting_cookies(client, "GET", login_url, {})
    res.raise_for_status()

    csrf = extract_csrf(res.text)
    if not csrf:
        raise Exception("CSRF token not found on login page.")

    dummy_data = {"_csrf": csrf, "LoginForm[username]": "", "LoginForm[password]": ""}
    res_post, step_cookies = await _follow_redirects_collecting_cookies(
        client, "POST", login_url, step_cookies, data=dummy_data
    )
    res_post.raise_for_status()

    captcha_match = re.search(r'src="([^"]*?r=site%2Fcaptcha[^"]*?)"', res_post.text)
    if not captcha_match:
        raise Exception("CAPTCHA image locator missing from layout.")

    captcha_url = BASE_URL + captcha_match.group(1).replace("&amp;", "&")
    captcha_response, step_cookies = await _follow_redirects_collecting_cookies(
        client, "GET", captcha_url, step_cookies
    )
    captcha_response.raise_for_status()

    captcha_text = solve_captcha(captcha_response.content)
    logger.info(f"[LOGIN] Captcha solved: {captcha_text}")

    payload = {
        "_csrf": csrf,
        "LoginForm[username]": username,
        "LoginForm[password]": password,
        "LoginForm[captcha]": captcha_text,
        "LoginForm[rememberMe]": "0",
        "LoginForm[qr_code]": "",
    }
    response, final_cookies = await _follow_redirects_collecting_cookies(
        client, "POST", login_url, step_cookies, data=payload
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
        async with httpx.AsyncClient(verify=False, limits=limits_pool, headers=DEFAULT_HEADERS) as client:
            login_response = None
            fresh_cookies = {}
            for attempt in range(3):
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
                    "_csrf_token": fresh_csrf
                }
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[LOGIN_ROUTE] Exception: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal processing fault during authorization sync.")

# ------------------ FETCH ATTENDANCE ------------------
@app.post("/fetch-attendance")
async def fetch_attendance_summary(
    username: str = Form(...),
    password: str = Form(...),
    php_sess_id: str = Form(...),
    csrf_cookie: str = Form(...),
    device_id: str = Form(...),
    server_id: str = Form(default="erp3"),
    academic_year_code: str = Form(...),
    semester_id: str = Form(...)
):
    start_time = time.time()
    cookie_jar = {
        "_csrf": unquote(csrf_cookie),
        "PHPSESSID": php_sess_id,
        "kl_erp_device_id": unquote(device_id),
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
        async with httpx.AsyncClient(verify=False, limits=limits_pool, headers=DEFAULT_HEADERS) as client:
            logger.info(f"[ATTENDANCE] Isolated POST query initialization (PHPSESSID={php_sess_id[:6]}...)")
            post_response, cookie_jar = await _follow_redirects_collecting_cookies(
                client, "POST", attendance_url, cookie_jar, timeout=15,
                data=_make_payload(unquote(csrf_cookie))
            )

            if is_login_failed(post_response):
                logger.warning("[ATTENDANCE] Session expired. Running automatic fallback healer...")
                for attempt in range(3):
                    login_response, cookie_jar = await auto_login(client, username, password, seed_cookies=cookie_jar)
                    if not is_login_failed(login_response):
                        break
                else:
                    raise HTTPException(status_code=401, detail="ERP system rejected fallback login.")

                fresh_csrf = extract_csrf(login_response.text)
                if not fresh_csrf:
                    raise HTTPException(status_code=500, detail="Could not reconcile session CSRF signatures.")

                post_response, cookie_jar = await _follow_redirects_collecting_cookies(
                    client, "POST", attendance_url, cookie_jar, timeout=15,
                    data=_make_payload(fresh_csrf)
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

        return {
            "success": True,
            "session_refreshed": has_refreshed,
            "cookies": {
                "PHPSESSID": updated_session_id,
                "_csrf_token": cookie_jar.get("_csrf"),
                "_csrf": cookie_jar.get("_csrf"),
                "kl_erp_device_id": cookie_jar.get("kl_erp_device_id", device_id),
                "SERVERID": cookie_jar.get("SERVERID", server_id)
            },
            "attendance": attendance_data
        }
    except Exception as e:
        logger.error(f"[ATTENDANCE] Crash: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ------------------ FETCH REGISTER DETAIL ------------------
@app.post("/fetch-register-detail")
async def fetch_register_detail(
    username: str = Form(...),
    password: str = Form(...),
    php_sess_id: str = Form(...),
    csrf_cookie: str = Form(...),
    device_id: str = Form(...),
    server_id: str = Form(default="erp3"),
    register_href: str = Form(...)
):
    register_url = build_register_url(BASE_URL, register_href)
    if not register_url:
        raise HTTPException(status_code=400, detail="Target path failure.")

    cookie_jar = {
        "_csrf": unquote(csrf_cookie),
        "PHPSESSID": php_sess_id,
        "kl_erp_device_id": unquote(device_id),
        "SERVERID": server_id
    }

    try:
        async with httpx.AsyncClient(verify=False, limits=limits_pool, headers=DEFAULT_HEADERS) as client:
            register_url_with_csrf = f"{register_url}&_csrf={unquote(csrf_cookie)}"
            response = await client.get(register_url_with_csrf, cookies=cookie_jar, timeout=15)

            if response.status_code == 500 or is_login_failed(response):
                logger.warning("[LAZY-REGISTER] Session invalid. Auto-healing context stream...")
                for attempt in range(3):
                    login_response, cookie_jar = await auto_login(client, username, password, seed_cookies=cookie_jar)
                    if not is_login_failed(login_response):
                        break
                else:
                    raise HTTPException(status_code=401, detail="Authentication credentials expired.")

                fresh_csrf = extract_csrf(login_response.text) or cookie_jar.get("_csrf", "")
                register_url_with_csrf = f"{register_url}&_csrf={fresh_csrf}"
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

        return {
            "success": True,
            "session_refreshed": has_refreshed,
            "cookies": {
                "PHPSESSID": updated_session_id,
                "_csrf": cookie_jar.get("_csrf"),
                "kl_erp_device_id": cookie_jar.get("kl_erp_device_id", device_id),
                "SERVERID": cookie_jar.get("SERVERID", server_id)
            } if has_refreshed else {},
            "metadata": metadata,
            "daily_attendance": daily_attendance
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ------------------ FETCH SEATING PLAN ------------------
@app.post("/fetch-seating-plan")
async def fetch_seating_plan(
    username: str = Form(...),
    password: str = Form(...),
    php_sess_id: str = Form(...),
    csrf_cookie: str = Form(...),
    device_id: str = Form(...),
    server_id: str = Form(default="erp3")
):
    cookie_jar = {
        "_csrf": unquote(csrf_cookie),
        "PHPSESSID": php_sess_id,
        "kl_erp_device_id": unquote(device_id),
        "SERVERID": server_id
    }
    seating_plan_url = f"{BASE_URL}/index.php?r=examsection%2Fexam-invigilator-student-room-allotment-info%2Fstud_my_seating_plan"

    try:
        async with httpx.AsyncClient(verify=False, limits=limits_pool, headers=DEFAULT_HEADERS) as client:
            response = await client.get(seating_plan_url, cookies=cookie_jar, timeout=15)

            if response.status_code == 500 or is_login_failed(response):
                logger.warning("[SEATING] Session invalid. Executing tracking fallback...")
                for attempt in range(3):
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

        return {
            "success": True,
            "session_refreshed": has_refreshed,
            "cookies": {
                "PHPSESSID": updated_session_id,
                "_csrf": cookie_jar.get("_csrf"),
                "kl_erp_device_id": cookie_jar.get("kl_erp_device_id", device_id),
                "SERVERID": cookie_jar.get("SERVERID", server_id)
            } if has_refreshed else {},
            "seating_plan": seating_plan_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ------------------ FETCH TIMETABLE ------------------
@app.post("/fetch-timetable")
async def fetch_timetable(
    username: str = Form(...),
    password: str = Form(...),
    php_sess_id: str = Form(...),
    csrf_cookie: str = Form(...),
    device_id: str = Form(...),
    server_id: str = Form(default="erp3"),
    academic_year_code: str = Form(default="19"),
    semester_id: str = Form(default="1")
):
    cookie_jar = {
        "_csrf": unquote(csrf_cookie),
        "PHPSESSID": php_sess_id,
        "kl_erp_device_id": unquote(device_id),
        "SERVERID": server_id
    }
    tt_url = (
        f"{BASE_URL}/index.php?r=timetables%2Funiversitymasteracademictimetableview%2Findividualstudenttimetableget"
        f"&UniversityMasterAcademicTimetableView%5Bacademicyear%5D={academic_year_code}"
        f"&UniversityMasterAcademicTimetableView%5Bsemesterid%5D={semester_id}"
    )

    try:
        async with httpx.AsyncClient(verify=False, limits=limits_pool, headers=DEFAULT_HEADERS) as client:
            response = await client.get(tt_url, cookies=cookie_jar, timeout=12)

            if response.status_code == 500 or is_login_failed(response):
                logger.warning("[TIMETABLE] Session invalid. Executing automated auto-healing...")
                for attempt in range(3):
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

        return {
            "success": True,
            "session_refreshed": has_refreshed,
            "cookies": {
                "PHPSESSID": updated_session_id,
                "_csrf": cookie_jar.get("_csrf"),
                "kl_erp_device_id": cookie_jar.get("kl_erp_device_id", device_id),
                "SERVERID": cookie_jar.get("SERVERID", server_id)
            } if has_refreshed else {},
            "timetable": timetable_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ------------------ GITHUB COMMIT ROUTE ------------------
OWNER = "sivadhanushreddykotturu"
REPO = "TimeTablekl"
_cached_commit = None
_last_fetch_time = 0
CACHE_TTL = 300
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

@app.get("/latest-commit")
async def latest_commit():
    global _cached_commit, _last_fetch_time

    if _cached_commit and (time.time() - _last_fetch_time < CACHE_TTL):
        return {**_cached_commit, "cached": True}

    url = f"https://api.github.com/repos/{OWNER}/{REPO}/commits"
    headers = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            data = resp.json()

        commit = data[0]
        latest = {
            "author": commit["commit"]["author"]["name"],
            "message": commit["commit"]["message"],
            "avatar": commit["author"]["avatar_url"] if commit.get("author") else None,
            "url": commit["html_url"],
            "date": commit["commit"]["author"]["date"],
        }

        _cached_commit = latest
        _last_fetch_time = time.time()
        return {**latest, "cached": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")