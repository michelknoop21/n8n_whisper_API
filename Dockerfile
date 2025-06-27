FROM python:3.10-slim

ARG WHISPER_MODEL=openai/whisper-small
ARG ADMIN_KEY=default-admin-key-change-me

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    build-essential \
    gcc \
    python3-dev \
    libsndfile1 \
    libavcodec-extra \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .

# Install PyTorch with CPU-only version first (lighter and more compatible)
RUN pip install --no-cache-dir torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cpu

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir numpy==1.24.3

# Create necessary directories
RUN mkdir -p /app/static

# Copy the application files
COPY . /app/

# Copy static files
COPY static/ /app/static/

# Expose port
EXPOSE 5000

# Set environment variables
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED="1"

# Set variables that can be overridden at runtime
ENV ADMIN_KEY=${ADMIN_KEY}
ENV WHISPER_MODEL=${WHISPER_MODEL}

# Create a non-root user and switch to it
RUN useradd -m appuser && \
    chown -R appuser:appuser /app

USER appuser

# Start the application
CMD ["python", "-m", "uvicorn", "src.insanely_fast_whisper_api.app.app:app", "--host", "0.0.0.0", "--port", "5000"]
