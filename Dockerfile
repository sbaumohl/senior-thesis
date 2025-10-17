FROM ghcr.io/astral-sh/uv:python3.13-trixie-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src/ ./src/
COPY configs/ ./configs/
COPY utils/ ./utils/

# Install dependencies using uv
RUN uv sync --frozen --no-dev & mkdir -p /outputs /logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Volume for outputs (mount this when running the container)
VOLUME ["/outputs", "/logs"]

# Default command
CMD ["uv", "run", "src/train.py", "--config", "configs/dpo_example.yaml"]
