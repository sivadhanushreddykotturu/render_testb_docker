# KLU TimeTable & Attendance Backend

A production-grade FastAPI backend designed to serve TimeTable and Attendance data to frontend applications.

## Features

- **Concurrent Request Handling**: Efficiently handles multiple student requests using `asyncio` and `httpx`.
- **Automatic Redirect Chasing**: Intelligently follows HTTP redirects while preserving cookies.
- **Login Protection**: Automatically detects and handles login loops using multiple heuristic checks.
- **Production Ready**:
  - **Lifespan Management**: Graceful startup and shutdown of global resources.
  - **CORS Support**: Configured for cross-origin resource sharing.
  - **Error Handling**: Custom exception handlers for `RequestValidationError` and other HTTP errors.
  - **Logging**: Structured logging with millisecond precision.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

1.  **Clone the repository** (if you haven't already).

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## ⚙️ Configuration

The backend uses environment variables for configuration. Ensure the following are set in your environment or a `.env` file:

| Variable | Description | Default |
| :--- | :--- | :--- |
| `PORT` | Port to run the server on. | `8001` |
| `BASE_URL` | Base URL of the KLU Portal. | `https://newerp.kluniversity.in` |

## 🏃 Running the Server

Start the development server using `uvicorn`:

```bash
uvicorn main:app --host [IP_ADDRESS] --port 8001 --reload
```

**Note:** The `--reload` flag enables hot-reload for development purposes. Remove it for production.

### Production Deployment

For production, use a production-grade ASGI server like Gunicorn:

```bash
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker -b [IP_ADDRESS]:8001
```

## 📚 API Documentation

Once the server is running, you can access the interactive API documentation:

- **Swagger UI**: [http://localhost:8001/docs](http://localhost:8001/docs)
- **ReDoc**: [http://localhost:8001/redoc](http://localhost:8001/redoc)

### Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Health check. Returns server status and version. |
| `POST` | `/auth/captcha` | Retrieves a captcha image from the portal. |
| `POST` | `/auth/login` | Exchanges credentials and captcha for a session cookie. |
| `GET` | `/timetable` | Fetches the timetable for a session and semester. |
| `GET` | `/attendance` | Fetches attendance records for a student. |
| `GET` | `/schedule` | Fetches the schedule (combines timetable + timings). |

## 🛡️ Security & Architecture

- **Idempotency Tokens**: All write operations (Login, Timetable, Attendance) utilize idempotent tokens to prevent race conditions and accidental duplicates.
- **Session Management**: Sessions are managed using secure, domain-wide cookies collected from the KLU Portal.
- **Error Handling**: The server returns meaningful HTTP status codes and user-friendly error messages, often including a "Please clear cache and refresh" suggestion to handle frontend-backend mismatches.
- **Resource Management**: Uses `contextlib.asynccontextmanager` for clean lifecycle management of the global `httpx.Limits` pool.

## 🔧 Debugging & Logging

The backend supports detailed logging. You can adjust the log level by setting the `LOG_LEVEL` environment variable:

```bash
# Verbose mode
LOG_LEVEL=DEBUG uvicorn main:app ...

# Normal mode (default)
LOG_LEVEL=INFO uvicorn main:app ...
```

The log format includes timestamps with millisecond precision (e.g., `2023-10-27 14:30:55.123`) to help correlate events across the stack.

## 📦 Dependencies

- **FastAPI**: Web framework.
- **httpx**: Async HTTP client.
- **uvicorn**: ASGI server.
- **numpy, Pillow**: Image processing for captcha.
- **python-multipart**: Form data handling.