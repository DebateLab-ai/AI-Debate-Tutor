FROM python:3.11-slim-bookworm

# WeasyPrint needs Pango/Cairo/GLib at runtime for /api/v1/debates/{id}/report.pdf.
# A Dockerfile is more reliable on Railway than nixpacks aptPkgs alone.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpango-1.0-0 \
    libpangoft2-1.0-0 \
    libharfbuzz0b \
    libharfbuzz-subset0 \
    libfontconfig1 \
    libgdk-pixbuf-2.0-0 \
    libglib2.0-0 \
    libcairo2 \
    shared-mime-info \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
