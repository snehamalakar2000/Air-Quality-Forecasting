# Multi-platform Python 3.11 slim Bookworm index, resolved 2026-10-01.
FROM python:3.11-slim-bookworm@sha256:a36c24f9cbdf4fd0f52d67f0823eeac19c2028c637cecc392d97f980d4fec56b
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
COPY requirements.txt pyproject.toml ./
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY tests ./tests
COPY data ./data
COPY docs ./docs
COPY config.json README.md ./
RUN python -m pip install --no-deps --no-build-isolation -e .
CMD ["python", "-m", "air_quality", "smoke"]
