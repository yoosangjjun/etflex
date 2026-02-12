FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for matplotlib
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        libffi-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create directories for persistent data
RUN mkdir -p /app/logs /app/ml/models

CMD ["python", "main.py", "serve"]
