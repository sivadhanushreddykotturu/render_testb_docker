FROM python:3.10-slim

# Prevent Python from writing byte-code and buffer logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /code

# Install essential system utilities for building or network checking
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Pre-copy and install dependencies to maximize layer caching speeds
COPY requirements.txt /code/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all your local repository files into the container
COPY . /code/

# Expose the internal port FastAPI binds to
EXPOSE 8000

# Run Uvicorn pointing directly to your app instance inside main.py
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]