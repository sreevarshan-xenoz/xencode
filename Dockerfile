# Multi-stage Dockerfile for Xencode (Rust): builds the `xencode` binary,
# then ships it in a slim runtime running the collaboration server.
FROM rust:1-bookworm as builder

WORKDIR /build

# Copy the Rust workspace and build the release binary
COPY rust/ ./
RUN cargo build --release -p xencode-cli

# Production stage
FROM debian:bookworm-slim as production

ENV XENCODE_ENV=production

# Runtime dependencies (curl for the health check, CA certs for HTTPS providers)
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r xencode && useradd -r -g xencode xencode

# Create directories
RUN mkdir -p /app /app/logs /app/data && \
    chown -R xencode:xencode /app

WORKDIR /app

# Copy the release binary from the builder stage
COPY --from=builder --chown=xencode:xencode /build/target/release/xencode /usr/local/bin/xencode

# Switch to non-root user
USER xencode

# Expose the collaboration server port
EXPOSE 8765

# Health check against the server status endpoint
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8765/api/status || exit 1

# Default command: collaboration server (override for CLI use)
CMD ["xencode", "server", "--port", "8765"]
