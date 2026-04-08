FROM node:20-bookworm-slim AS frontend-build

WORKDIR /ui
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# Use an official Python 3.12 slim image as base
FROM python:3.12-slim-bookworm

# (Optional) Install system packages needed by any libs (if any).
# For example, if using certain PDF or image libs you might need apt packages.
# PyMuPDF doesn't require external libs, so this can usually be skipped.
# RUN apt-get update && apt-get install -y --no-install-recommends <packages> && rm -rf /var/lib/apt/lists/*

# --- Install uv (fast Python package manager) ---
# Install curl (for the installer script)
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && rm -rf /var/lib/apt/lists/*
# Copy and run uv installer (pin a specific version for reproducibility)
ADD https://astral.sh/uv/0.8.22/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
# Ensure uv (installed to ~/.local/bin) is on PATH
ENV PATH="/root/.local/bin:${PATH}"

# Tell uv to install packages in the global system environment (no venv needed)
ENV UV_SYSTEM_PYTHON=1

# Set working directory for app
WORKDIR /app

# Copy in requirements first (for caching layer)
COPY pyproject.toml ./
# Use uv to install all Python dependencies
RUN uv pip install -r pyproject.toml

# Copy the rest of the application code
COPY . ./
COPY --from=frontend-build /ui/dist /opt/gtm-ui-dist

ENV UI_DIST_DIR=/opt/gtm-ui-dist
ENV WEB_HOST=0.0.0.0
ENV WEB_PORT=8000

# (Optional) Set environment variables for any configuration
# (Better to set sensitive API keys at runtime rather than here)
# ENV LANGCHAIN_API_KEY=<key> ...

# Default command for the take-home app's demo-ready web runtime
CMD ["python", "-m", "gtm_diligence_assistant.web_app"]
