# Multi-platform Dockerfile for diabetes prediction batch processor
# Supports both M3 (ARM64) and x86_64 architectures

FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install uv for dependency management
RUN pip install --no-cache-dir uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies using uv (locked versions for reproducibility)
RUN uv sync --locked --no-dev

# Copy source code and configurations
COPY src/ ./src/
COPY configs/ ./configs/

# Create necessary directories
RUN mkdir -p /data/incoming /data/predictions /data/rejected /logs/predictions /logs/batch

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Default command runs batch prediction
# Override input file via volume mount and command args
ENTRYPOINT ["uv", "run", "python", "-m", "src.inference.batch_predict"]
CMD ["--help"]
