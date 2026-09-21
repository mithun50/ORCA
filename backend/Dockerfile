FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# shapely needs geos at runtime; nothing else here needs a compiler
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgeos-c1v5 curl \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install -r /app/backend/requirements.txt

COPY backend /app/backend
COPY knowledge /app/knowledge
COPY data /app/data
COPY frontend /app/frontend

ENV PYTHONPATH=/app/backend
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "orca.main:app", "--host", "0.0.0.0", "--port", "8000"]
