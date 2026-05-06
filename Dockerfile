FROM python:3.10-slim

WORKDIR /app

# 1. System dependencies (added gcc and build tools for insightface)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    curl \
    gcc \
    g++ \
    cmake \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Upgrade pip first
RUN pip install --no-cache-dir --upgrade pip

# 3. Create models directory
RUN mkdir -p /app/models

# 3. Download models from GitHub Releases (RELIABLE)
RUN curl -L https://github.com/RidaFatima9/models/releases/download/onnx/w600k_r50.onnx \
    -o /app/models/w600k_r50.onnx && \
    curl -L https://github.com/RidaFatima9/models/releases/download/pklfile/arcface_db.pkl \
    -o /app/models/arcface_db.pkl

# 5. Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy application code
COPY handler.py .

CMD ["python", "handler.py"]