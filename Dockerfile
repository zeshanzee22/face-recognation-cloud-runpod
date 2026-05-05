FROM python:3.10-slim

WORKDIR /app

# 1. Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. Create the target directory (standardized to lowercase)
RUN mkdir -p /app/models

# 3. Download ONNX Model (Direct Link format)
# Note: Google Drive "view" links are changed to "uc?export=download"
RUN wget --no-check-certificate 'https://drive.google.com/file/d/1m2e698UzYx6Va3YsV5Sx5gXrV61VJLyw/view?usp=sharing' -O /app/models/w600k_r50.onnx

# 4. Download PKL Database (Direct Link format)
RUN wget --no-check-certificate 'https://drive.google.com/file/d/1q1mwy9X1BasdA3_kGhmCKuujofy0gi2j/view?usp=sharing' -O /app/models/arcface_db.pkl

# 5. Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the handler script
COPY handler.py .

CMD ["python", "handler.py"]