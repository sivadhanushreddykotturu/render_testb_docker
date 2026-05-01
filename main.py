from fastapi import FastAPI, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from bs4 import BeautifulSoup
import httpx
import asyncio
from io import BytesIO
import logging
import os
import secrets
from datetime import datetime, timedelta
from urllib.parse import unquote
import json
import numpy as np
from PIL import Image
import onnxruntime as ort
import io
import time

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

app = FastAPI(title="TimeTable & Attendance Backend", version="5.2.0 (Cookie Fix + Debug Logs)")

@app.on_event("startup")
async def startup_event():
    logger.info("✅ FastAPI app starting (Cookie Fix + Debug Logs Enabled)...")
    logger.info(f"Environment: PORT={os.getenv('PORT', '8080')}")

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
    return {"message": "Backend running ✅ (Cookie Fix + Debug Logs)", "status": "healthy"}

# ------------------ CAPTCHA STORE ------------------
# Store: { session_id: { "cookies": httpx.Cookies, "csrf": str, "created_at": datetime } }
captcha_sessions = {}

def cleanup_expired_sessions():
    current_time = datetime.now()
    expired = [
        sid for sid, d in captcha_sessions.items()
        if current_time - d["created_at"] > timedelta(minutes=10)
    ]
    for sid in expired:
        del captcha_sessions[sid]
    if expired:
        logger.info(f"🧹 Cleaned up {len(expired)} expired sessions")

BASE_URL = "https://newerp.kluniversity.in"
DEFAULT_HEADERS = {"User-Agent": "Mozilla/5.0"}

# ------------------ DEBUG HELPERS ------------------
def log_html_snippet(tag: str, html: str, length: int = 800):
    snippet = html[:length].replace("\n", " ").replace("\r", " ")
    logger.info(f"[{tag}] HTML snippet ({length} chars): {snippet}")

def is_login_failed(response: httpx.Response) -> bool:
    url_str = str(response.url)
    if "site%2Flogin" in url_str or "site/login" in url_str:
        return True
    if "LoginForm[username]" in response.text or "LoginForm[password]" in response.text:
        return True
    if "<h4" in response.text and "Login" in response.text:
        return True
    return False

# ------------------ HELPERS ------------------
async def auto_login(client: httpx.AsyncClient, username: str, password: str) -> httpx.Response:
    login_url = f"{BASE_URL}/index.php?r=site%2Flogin"

    logger.info(f"[LOGIN] Attempting auto-login for user={username}")
    
    # 1. Fetch CSRF
    res = await client.get(login_url, headers=DEFAULT_HEADERS, timeout=30)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, "html.parser")
    csrf_meta = soup.find("meta", {"name": "csrf-token"})
    if not csrf_meta:
        raise Exception("CSRF token not found on login page")
    csrf = csrf_meta["content"]
    
    # 2. Trigger CAPTCHA dummy post
    dummy_data = {"_csrf": csrf, "LoginForm[username]": "", "LoginForm[password]": ""}
    res_post = await client.post(login_url, data=dummy_data, headers=DEFAULT_HEADERS, timeout=30)
    res_post.raise_for_status()
    soup_post = BeautifulSoup(res_post.text, "html.parser")
    
    captcha_img = (
        soup_post.find("img", src=lambda x: x and "r=site%2Fcaptcha" in x)
        or soup.find("img", src=lambda x: x and "r=site%2Fcaptcha" in x)
    )
    if not captcha_img:
        raise Exception("CAPTCHA image not found.")
    
    captcha_url = BASE_URL + captcha_img["src"].replace("&amp;", "&")
    captcha_response = await client.get(captcha_url, timeout=30)
    captcha_response.raise_for_status()
    
    # 3. Solve CAPTCHA
    captcha_text = solve_captcha(captcha_response.content)
    logger.info(f"[LOGIN] Auto-solved captcha: {captcha_text}")
    
    # 4. Login POST
    payload = {
        "_csrf": csrf,
        "LoginForm[username]": username,
        "LoginForm[password]": password,
        "LoginForm[captcha]": captcha_text,
        "LoginForm[qr_code]": "",
    }
    
    response = await client.post(login_url, data=payload, headers=DEFAULT_HEADERS, timeout=30)
    logger.info(f"[LOGIN] Status Code: {response.status_code}, Final URL: {response.url}")
    response.raise_for_status()
    return response


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
        logger.error(f"[REGISTER_URL] Failed to build URL. href={href}, error={e}")
        with open("error_log.txt", "a") as f:
            f.write(f"Failed to build URL. href={href}, error={e}\n")
        return None


async def fetch_register_details(client: httpx.AsyncClient, href: str, csrf: str) -> dict:
    register_url = build_register_url(BASE_URL, href)
    if not register_url:
        return {"message": "Could not reconstruct register URL."}

    try:
        register_url_with_csrf = f"{register_url}&_csrf={csrf}"
        logger.info(f"[REGISTER] Fetching register: {register_url_with_csrf}")

        resp = await client.get(register_url_with_csrf, headers=DEFAULT_HEADERS, timeout=30)

        logger.info(f"[REGISTER] Status Code: {resp.status_code}")
        logger.info(f"[REGISTER] Final URL: {resp.url}")

        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        table_container = soup.find("div", class_="card-body") or soup.find("div", id="w0")
        table_register = table_container.find("table", class_="table table-striped table-bordered") if table_container else None

        if not table_register:
            log_html_snippet("REGISTER_PAGE_NO_TABLE", resp.text)
            return {"message": "No register table found on detail page."}

        all_headers = [th.text.strip() for th in table_register.find("thead").find_all("th") if th.text.strip()]
        metadata_count = 14
        metadata_headers = all_headers[:metadata_count]
        daily_headers = all_headers[metadata_count:]

        rows = table_register.find("tbody").find_all("tr")
        if not rows:
            return {"message": "No rows found in register tbody."}

        cells = rows[0].find_all("td")
        if len(cells) < metadata_count:
            return {"message": f"Incomplete data. Expected {metadata_count} cells, got {len(cells)}."}

        metadata = {header: cells[i].text.strip() for i, header in enumerate(metadata_headers) if i < len(cells)}
        daily_attendance = [
            {"date_slot": header, "status": cells[metadata_count + i].text.strip()}
            for i, header in enumerate(daily_headers)
            if metadata_count + i < len(cells)
        ]

        logger.info(f"[REGISTER] Parsed metadata keys: {list(metadata.keys())}")
        logger.info(f"[REGISTER] Parsed daily attendance entries: {len(daily_attendance)}")

        return {"metadata": metadata, "daily_attendance": daily_attendance}

    except httpx.HTTPStatusError as e:
        logger.error(f"[REGISTER] HTTP error: {e}")
        return {"message": "Network/authorization error fetching register."}
    except Exception as e:
        logger.error(f"[REGISTER] Parsing error: {e}", exc_info=True)
        return {"message": "Error parsing register details."}


# ------------------ CAPTCHA ROUTE ------------------
@app.get("/get-captcha")
async def get_captcha():
    cleanup_expired_sessions()
    login_url = f"{BASE_URL}/index.php?r=site%2Flogin"

    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            logger.info("[CAPTCHA] Fetching login page...")

            res = await client.get(login_url, headers=DEFAULT_HEADERS, timeout=30)
            logger.info(f"[CAPTCHA] Login page status: {res.status_code}")
            res.raise_for_status()

            soup = BeautifulSoup(res.text, "html.parser")
            csrf_meta = soup.find("meta", {"name": "csrf-token"})
            if not csrf_meta:
                logger.error("[CAPTCHA] CSRF token not found on login page")
                raise HTTPException(status_code=500, detail="Failed to get CSRF token")

            csrf = csrf_meta["content"]
            logger.info(f"[CAPTCHA] CSRF token extracted: {csrf[:25]}...")

            logger.info("[CAPTCHA] Triggering captcha generation using dummy POST...")
            dummy_data = {"_csrf": csrf, "LoginForm[username]": "", "LoginForm[password]": ""}

            res_post = await client.post(login_url, data=dummy_data, headers=DEFAULT_HEADERS, timeout=30)
            logger.info(f"[CAPTCHA] Dummy POST status: {res_post.status_code}")
            res_post.raise_for_status()

            soup_post = BeautifulSoup(res_post.text, "html.parser")

            captcha_img = (
                soup_post.find("img", src=lambda x: x and "r=site%2Fcaptcha" in x)
                or soup.find("img", src=lambda x: x and "r=site%2Fcaptcha" in x)
            )
            if not captcha_img:
                logger.error("[CAPTCHA] CAPTCHA image not found in response HTML")
                log_html_snippet("CAPTCHA_HTML", res_post.text)
                raise HTTPException(status_code=500, detail="CAPTCHA image not found.")

            captcha_url = BASE_URL + captcha_img["src"].replace("&amp;", "&")
            logger.info(f"[CAPTCHA] Captcha URL: {captcha_url}")

            captcha_response = await client.get(captcha_url, timeout=30)
            logger.info(f"[CAPTCHA] Captcha image status: {captcha_response.status_code}")
            captcha_response.raise_for_status()

            session_id = secrets.token_urlsafe(16)

            # ✅ IMPORTANT FIX: Store cookies as httpx.Cookies object, not dict
            captcha_sessions[session_id] = {
                "cookies": client.cookies,
                "csrf": csrf,
                "created_at": datetime.now(),
            }

            logger.info(f"[CAPTCHA] Session created: {session_id[:8]}...")
            logger.info(f"[CAPTCHA] Cookies stored: {list(client.cookies.keys())}")

            response = StreamingResponse(BytesIO(captcha_response.content), media_type="image/jpeg")
            response.headers["X-Session-ID"] = session_id
            return response

    except httpx.RequestError as e:
        logger.error(f"[CAPTCHA] Network error: {e}")
        raise HTTPException(status_code=500, detail="Network error while fetching CAPTCHA")
    except Exception as e:
        logger.error(f"[CAPTCHA] Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


# ------------------ FETCH TIMETABLE ------------------
@app.post("/fetch-timetable")
async def fetch_timetable(
    username: str = Form(...),
    password: str = Form(...),
    captcha: str = Form(default=""),
    session_id: str = Form(default=""),
    academic_year_code: str = Form(default="19"),
    semester_id: str = Form(default="1")
):
    logger.info(f"[TIMETABLE] Request received for user={username}")

    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            login_response = None
            for attempt in range(3):
                login_response = await auto_login(client, username, password)
                if not is_login_failed(login_response):
                    break
                logger.warning(f"[TIMETABLE] Login failed (attempt {attempt+1}), retrying...")
            else:
                logger.error("[TIMETABLE] Login failed after 3 attempts")
                raise HTTPException(status_code=400, detail="Invalid credentials or unable to auto-solve captcha")

            logger.info("[TIMETABLE] Login successful")

            tt_url = (
                f"{BASE_URL}/index.php?r=timetables%2Funiversitymasteracademictimetableview%2Findividualstudenttimetableget"
                f"&UniversityMasterAcademicTimetableView%5Bacademicyear%5D={academic_year_code}"
                f"&UniversityMasterAcademicTimetableView%5Bsemesterid%5D={semester_id}"
            )

            logger.info(f"[TIMETABLE] Fetching timetable URL: {tt_url}")

            tt_response = await client.get(tt_url, headers=DEFAULT_HEADERS, timeout=30)

            logger.info(f"[TIMETABLE] Status Code: {tt_response.status_code}")
            logger.info(f"[TIMETABLE] Final URL: {tt_response.url}")

            tt_response.raise_for_status()

            soup_tt = BeautifulSoup(tt_response.text, "html.parser")
            log_html_snippet("TIMETABLE_PAGE", tt_response.text)

            table = soup_tt.find("table")
            if not table:
                logger.error("[TIMETABLE] Timetable table not found")
                raise HTTPException(status_code=404, detail="Timetable not found")

            headers = [th.text.strip() for th in table.find("thead").find_all("th")][1:]
            logger.info(f"[TIMETABLE] Headers extracted: {headers}")

            timetable = {}
            for row in table.find("tbody").find_all("tr"):
                cols = row.find_all("td")
                if not cols:
                    continue
                day = cols[0].text.strip()
                slots = [td.text.strip() for td in cols[1:]]
                timetable[day] = dict(zip(headers, slots))

            logger.info(f"[TIMETABLE] Parsed timetable days count: {len(timetable)}")

            return {"success": True, "timetable": timetable}

    except httpx.RequestError as e:
        logger.error(f"[TIMETABLE] Network error: {e}")
        raise HTTPException(status_code=500, detail="Network error while fetching timetable")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[TIMETABLE] Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


# ------------------ FETCH ATTENDANCE ------------------
@app.post("/fetch-attendance")
async def fetch_attendance(
    username: str = Form(...),
    password: str = Form(...),
    captcha: str = Form(default=""),
    session_id: str = Form(default=""),
    academic_year_code: str = Form(...),
    semester_id: str = Form(...)
):
    logger.info(f"[ATTENDANCE] Request received for user={username}")

    try:
        async with httpx.AsyncClient(follow_redirects=True, verify=False) as client:
            login_response = None
            for attempt in range(3):
                login_response = await auto_login(client, username, password)
                if not is_login_failed(login_response):
                    break
                logger.warning(f"[ATTENDANCE] Login failed (attempt {attempt+1}), retrying...")
            else:
                logger.error("[ATTENDANCE] Login failed after 3 attempts")
                raise HTTPException(status_code=400, detail="Invalid credentials or unable to auto-solve captcha")

            logger.info("[ATTENDANCE] Login successful")

            post_login_soup = BeautifulSoup(login_response.text, "html.parser")
            post_login_csrf_meta = post_login_soup.find("meta", {"name": "csrf-token"})
            if not post_login_csrf_meta:
                logger.error("[ATTENDANCE] CSRF token missing after login")
                raise HTTPException(status_code=500, detail="Could not find CSRF token on post-login page.")

            post_login_csrf = post_login_csrf_meta["content"]
            logger.info(f"[ATTENDANCE] Post-login CSRF extracted: {post_login_csrf[:25]}...")

            attendance_url = f"{BASE_URL}/index.php?r=studentattendance%2Fstudentdailyattendance%2Fcourselist"
            attendance_payload = {
                "_csrf": post_login_csrf,
                "DynamicModel[academicyear]": academic_year_code,
                "DynamicModel[semesterid]": semester_id,
            }

            logger.info(f"[ATTENDANCE] Posting course list request: {attendance_url}")
            logger.info(f"[ATTENDANCE] Payload: {attendance_payload}")

            attendance_response = await client.post(
                attendance_url, data=attendance_payload, headers=DEFAULT_HEADERS, timeout=30
            )

            logger.info(f"[ATTENDANCE] Status Code: {attendance_response.status_code}")
            logger.info(f"[ATTENDANCE] Final URL: {attendance_response.url}")

            attendance_response.raise_for_status()

            attendance_soup = BeautifulSoup(attendance_response.text, "html.parser")
            log_html_snippet("ATTENDANCE_PAGE", attendance_response.text)

            container = attendance_soup.find("div", id="w0")
            if not container:
                logger.error("[ATTENDANCE] Container div#w0 not found (HTML changed?)")
                raise HTTPException(status_code=404, detail="Could not find the attendance data container.")

            table = container.find("table", class_="table table-striped table-bordered")
            if not table:
                logger.error("[ATTENDANCE] Attendance table not found inside container")
                raise HTTPException(status_code=404, detail="Could not find the attendance table.")

            table_headers = [th.text.strip() for th in table.find("thead").find_all("th")]
            rows = table.find("tbody").find_all("tr")

            logger.info(f"[ATTENDANCE] Headers extracted: {table_headers}")
            logger.info(f"[ATTENDANCE] Total rows found: {len(rows)}")

            parsed_rows = []
            register_hrefs = []

            for row in rows:
                cells = row.find_all("td")
                if not cells:
                    continue

                row_data = {table_headers[i]: cells[i].text.strip() for i in range(len(table_headers) - 1)}
                register_link = cells[-1].find("a", class_="crudjax")
                href = register_link["href"] if (register_link and "href" in register_link.attrs) else None

                parsed_rows.append(row_data)
                register_hrefs.append(href)

            logger.info(f"[ATTENDANCE] Register links found: {sum(1 for x in register_hrefs if x)} / {len(register_hrefs)}")
            logger.info(f"[ATTENDANCE] Fetching {len(register_hrefs)} registers concurrently...")

            async def _unavailable_register():
                return {"message": "Register link not available"}

            register_tasks = [
                fetch_register_details(client, href, post_login_csrf) if href
                else _unavailable_register()
                for href in register_hrefs
            ]

            register_results = await asyncio.gather(*register_tasks, return_exceptions=False)

            attendance_data = []
            for row_data, register_details in zip(parsed_rows, register_results):
                row_data["register_details"] = register_details
                attendance_data.append(row_data)

            if not attendance_data:
                logger.warning("[ATTENDANCE] No attendance data found")
                return {"success": True, "message": "No attendance data found for the selected period.", "attendance": []}

            logger.info(f"[ATTENDANCE] Attendance data parsed successfully: {len(attendance_data)} courses")

            return {"success": True, "attendance": attendance_data}

    except httpx.RequestError as e:
        logger.error(f"[ATTENDANCE] Network error: {e}")
        raise HTTPException(status_code=500, detail="A network error occurred.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[ATTENDANCE] Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal server error occurred.")
    finally:
        logger.info(f"[ATTENDANCE] Session {session_id[:8]}... cleaned up.")


# ------------------ LATEST COMMIT ------------------
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
        logger.info("[GITHUB] Returning cached commit info")
        return {**_cached_commit, "cached": True}

    url = f"https://api.github.com/repos/{OWNER}/{REPO}/commits"
    headers = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"

    try:
        async with httpx.AsyncClient() as client:
            logger.info("[GITHUB] Fetching latest commit from GitHub API...")
            resp = await client.get(url, headers=headers, timeout=15)

            logger.info(f"[GITHUB] Status Code: {resp.status_code}")
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

        logger.info("[GITHUB] Latest commit fetched successfully")
        return {**latest, "cached": False}

    except httpx.HTTPStatusError as e:
        logger.error(f"[GITHUB] HTTP error: {e}")
        if e.response.status_code == 403:
            raise HTTPException(status_code=429, detail="GitHub API rate limit reached or token missing.")
        raise HTTPException(status_code=500, detail="Failed to fetch commit data from GitHub.")
    except Exception as e:
        logger.error(f"[GITHUB] Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


# ------------------ FETCH SEATING PLAN ------------------
@app.post("/fetch-seating-plan")
async def fetch_seating_plan(
    username: str = Form(...),
    password: str = Form(...),
    captcha: str = Form(default=""),
    session_id: str = Form(default=""),
):
    logger.info(f"[SEATING] Request received for user={username}")

    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            login_response = None
            for attempt in range(3):
                login_response = await auto_login(client, username, password)
                if not is_login_failed(login_response):
                    break
                logger.warning(f"[SEATING] Login failed (attempt {attempt+1}), retrying...")
            else:
                logger.error("[SEATING] Login failed after 3 attempts")
                raise HTTPException(status_code=400, detail="Invalid credentials or unable to auto-solve captcha")

            logger.info("[SEATING] Login successful")

            seating_plan_url = f"{BASE_URL}/index.php?r=examsection%2Fexam-invigilator-student-room-allotment-info%2Fstud_my_seating_plan"
            logger.info(f"[SEATING] Fetching seating plan URL: {seating_plan_url}")

            seating_plan_response = await client.get(seating_plan_url, headers=DEFAULT_HEADERS, timeout=30)

            logger.info(f"[SEATING] Status Code: {seating_plan_response.status_code}")
            logger.info(f"[SEATING] Final URL: {seating_plan_response.url}")

            seating_plan_response.raise_for_status()

            soup_sp = BeautifulSoup(seating_plan_response.text, "html.parser")
            log_html_snippet("SEATING_PLAN_PAGE", seating_plan_response.text)

            container = soup_sp.find("div", id="exam-invigilator-student-room-allotment-info-pjax")
            if not container:
                logger.error("[SEATING] Seating plan container not found")
                raise HTTPException(status_code=404, detail="Could not find the seating plan container.")

            table = container.find("table")
            if not table:
                logger.error("[SEATING] Seating plan table not found")
                raise HTTPException(status_code=404, detail="Could not find the seating plan table.")

            tbody = table.find("tbody")
            seating_plan_data = []

            for row in tbody.find_all("tr"):
                cells = row.find_all("td")
                if not cells or len(cells) < 8:
                    continue

                seating_plan_data.append({
                    "date": cells[2].text.strip(),
                    "exam_type": cells[3].text.strip(),
                    "time_slot": cells[4].text.strip(),
                    "admission_number": cells[5].text.strip(),
                    "course_code": cells[6].text.strip(),
                    "room_no": cells[7].text.strip()
                })

            if not seating_plan_data:
                logger.warning("[SEATING] No seating plan data found")
                raise HTTPException(status_code=404, detail="No seating plan details found.")

            logger.info(f"[SEATING] Seating plan parsed successfully: {len(seating_plan_data)} entries")

            return {"success": True, "seating_plan": seating_plan_data}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[SEATING] Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")
    finally:
        logger.info(f"[SEATING] Session {session_id[:8]}... cleaned up.")
