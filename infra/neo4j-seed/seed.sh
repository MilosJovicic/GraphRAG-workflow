#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-/data}"
MARKER="${DATA_DIR}/.seeded"
DUMP_PATH="/tmp/graph.dump"

if [ -f "$MARKER" ]; then
  echo "[neo4j-seed] Volume already seeded at ${MARKER}, skipping."
  exit 0
fi

: "${NEO4J_DUMP_URL:?NEO4J_DUMP_URL is required}"
: "${NEO4J_DUMP_SHA256:?NEO4J_DUMP_SHA256 is required}"

echo "[neo4j-seed] Downloading dump from ${NEO4J_DUMP_URL}"
curl -fL --retry 3 --retry-delay 5 "${NEO4J_DUMP_URL}" -o "${DUMP_PATH}"

echo "[neo4j-seed] Verifying SHA-256"
echo "${NEO4J_DUMP_SHA256}  ${DUMP_PATH}" | sha256sum -c -

echo "[neo4j-seed] Loading dump into Neo4j volume"
neo4j-admin database load --from-path=/tmp --overwrite-destination=true neo4j

touch "${MARKER}"
echo "[neo4j-seed] Done."
