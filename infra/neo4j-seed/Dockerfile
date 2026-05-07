FROM neo4j:2026.04-enterprise

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    coreutils \
    && rm -rf /var/lib/apt/lists/*

COPY seed.sh /usr/local/bin/seed.sh
RUN chmod +x /usr/local/bin/seed.sh

USER neo4j

ENTRYPOINT ["/usr/local/bin/seed.sh"]
