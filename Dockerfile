# Builds from the working tree and uv.lock, not from a git tag.
#   docker build -t bootstrapper:1.0.0 .
#   docker run --rm bootstrapper:1.0.0 --help
FROM python:3.12-slim-bookworm@sha256:a116514e19457bcb7af7efe9c3dd0b9b71e85b317694e7882a1c52aa15a78134

COPY --from=ghcr.io/astral-sh/uv:0.11.29@sha256:eb2843a1e56fd9e30c7276ce1a52cba86e64c7b385f5e3279a0e08e02dd058fc /uv /bin/uv

# git fetches the commit-pinned source dependencies; the waterz extra compiles
# C++ against the boost headers.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        g++ \
        libboost-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY bootstrapper ./bootstrapper
RUN uv sync --locked --all-extras --no-dev

ENV PATH="/app/.venv/bin:$PATH"
ENTRYPOINT ["bs"]
