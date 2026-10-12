FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken

# Tesseract for the scanned (image-only) PDF pages.
RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-dev.txt ./
RUN pip install -r requirements-dev.txt \
 # Bake the cl100k_base vocabulary into the image so token counting never needs the network.
 && python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

COPY app ./app
COPY tests ./tests
COPY tests_corpus ./tests_corpus
COPY Makefile pytest.ini ./

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=60 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"

# One worker: conversation state (sessions, pending confirmations) lives in process memory.
CMD ["uvicorn", "app.main:get_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
