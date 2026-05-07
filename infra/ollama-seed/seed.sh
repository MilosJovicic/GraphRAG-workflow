#!/usr/bin/env bash
set -euo pipefail

OLLAMA_URL="${OLLAMA_URL:-http://ollama:11434}"
: "${EMBEDDING_MODEL:?EMBEDDING_MODEL is required}"

echo "[ollama-seed] Waiting for Ollama at ${OLLAMA_URL}"
for _ in $(seq 1 60); do
  if curl -fs "${OLLAMA_URL}/api/tags" > /dev/null; then
    break
  fi
  sleep 2
done

if ! curl -fs "${OLLAMA_URL}/api/tags" > /dev/null; then
  echo "[ollama-seed] Ollama never became reachable" >&2
  exit 1
fi

echo "[ollama-seed] Pulling model: ${EMBEDDING_MODEL}"
curl -fSN "${OLLAMA_URL}/api/pull" -d "{\"name\": \"${EMBEDDING_MODEL}\"}"

echo "[ollama-seed] Done."
