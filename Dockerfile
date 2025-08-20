# syntax=docker/dockerfile:1

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps for scientific stack and faiss
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirement files
COPY requirements-app.txt ./

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements-app.txt

# Copy application
COPY app ./app

# Expose Streamlit default port
EXPOSE 8501

# Streamlit configuration to run in container
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_PORT=8501

# Healthcheck: app should serve a page
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
 CMD wget -qO- http://localhost:8501/_stcore/health || exit 1

# Default command: run the Streamlit app
CMD ["streamlit", "run", "app/agent/agent_ui.py"]


