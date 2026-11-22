FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

# build-essential is only needed if a dependency has no prebuilt ARM wheel
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Windows line endings (CRLF) would break the script on Linux
RUN sed -i 's/\r$//' start.sh && chmod +x start.sh

# Only the Streamlit UI is public. Backend (8000) and LiteLLM (4000) stay internal.
EXPOSE 8501

CMD ["./start.sh"]
