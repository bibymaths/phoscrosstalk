# syntax=docker/dockerfile:1.7

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_SYSTEM_PYTHON=1 \
    JAX_PLATFORMS=cpu \
    MPLBACKEND=Agg

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    graphviz \
    libgraphviz-dev \
    pkg-config \
    libxml2 \
    libxslt1.1 \
    libcairo2 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf-2.0-0 \
    libffi-dev \
    shared-mime-info \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --upgrade pip && pip install uv

COPY pyproject.toml README.md LICENSE ./

# Copy package and script code before editable install.
# This matters because hatchling needs the actual package present.
COPY phoscrosstalk ./phoscrosstalk
COPY scripts ./scripts
COPY config.toml ./config.toml

RUN uv pip install -e .

ENTRYPOINT ["phoscrosstalk"]
CMD ["--config", "config.toml"]