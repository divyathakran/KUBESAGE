#!/usr/bin/env bash
# Starts the three KubeSage processes inside one container, so every
# "localhost" URL in the code (localhost:4000, localhost:8000) keeps working.
set -uo pipefail

wait_for_port() {
  # wait_for_port <port> <name>
  for _ in $(seq 1 90); do
    if (echo > /dev/tcp/127.0.0.1/"$1") 2>/dev/null; then
      echo "[start] $2 is up on port $1"
      return 0
    fi
    sleep 2
  done
  echo "[start] ERROR: $2 did not come up on port $1" >&2
  exit 1
}

# Stop everything if the script exits for any reason
trap 'kill $(jobs -p) 2>/dev/null || true' EXIT

# 1. LiteLLM gateway (internal only)
litellm --config app/gateway/litellm_config.yaml --host 127.0.0.1 --port 4000 &
wait_for_port 4000 "LiteLLM gateway"

# 2. FastAPI backend (internal only)
uvicorn app.main:app --host 127.0.0.1 --port 8000 &
wait_for_port 8000 "FastAPI backend"

# 3. Streamlit chat UI (public)
# NOTE: change ui/app.py if your chat UI file has a different name
streamlit run ui/app.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.headless true &
wait_for_port 8501 "Streamlit UI"

# If any of the three processes dies, exit so Docker restarts the container
wait -n
echo "[start] a process exited, shutting down" >&2
exit 1
