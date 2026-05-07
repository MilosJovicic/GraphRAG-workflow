FROM alpine:3.20

RUN apk add --no-cache curl bash ca-certificates

COPY seed.sh /usr/local/bin/seed.sh
RUN chmod +x /usr/local/bin/seed.sh

ENTRYPOINT ["/usr/local/bin/seed.sh"]
