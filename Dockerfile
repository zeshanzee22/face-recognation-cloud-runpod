FROM python:3.10-slim

WORKDIR /app

# 1. Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. Install gdown to reliably download from Google Drive
RUN pip install gdown

# 3. Create models directory
RUN mkdir -p /app/models

# 4. Download ONNX model from Google Drive
RUN gdown --fuzzy "https://drive.google.com/file/d/1m2e698UzYx6Va3YsV5Sx5gXrV61VJLyw/view?usp=sharing" \
    -O /app/models/w600k_r50.onnx

# 5. Download PKL database from Google Drive
RUN gdown --fuzzy "https://drive.google.com/file/d/1q1mwy9X1BasdA3_kGhmCKuujofy0gi2j/view?usp=sharing" \
    -O /app/models/arcface_db.pkl

# 6. Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 7. Copy handler
COPY handler.py .

CMD ["python", "handler.py"]