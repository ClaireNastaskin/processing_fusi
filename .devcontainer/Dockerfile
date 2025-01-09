FROM mcr.microsoft.com/devcontainers/python:3.12

# Install system dependencies
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    git \
    cifs-utils \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install uv to manage dependencies
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/


# We'll mount the project directory here when running the container