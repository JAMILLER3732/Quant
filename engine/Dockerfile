FROM python:3.12-slim

WORKDIR /app

# Package names below target current python:3.12-slim's Debian base (trixie),
# which renamed several of these from their older bookworm-era names (notably
# libgdk-pixbuf2.0-0 -> libgdk-pixbuf-2.0-0). If this base image moves to a
# different Debian release again, `apt-cache search` inside the image is the
# fastest way to re-find the current names — don't guess from an old error's
# suggested "replacement" package, which can be a different, wrong package
# (e.g. apt suggested libgdk-pixbuf-xlib-2.0-0, an X11 add-on, not the base).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libcairo2 \
    libgdk-pixbuf-2.0-0 \
    libffi-dev \
    shared-mime-info \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
