FROM pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --break-system-packages -r requirements.txt
RUN pip install --no-cache-dir --break-system-packages tensorrt

COPY . .

RUN mkdir -p /app/data /app/output /app/weights

CMD ["python", "main.py", "--source", "data/crowd.mp4", "--backend", "tensorrt", "--output", "output/result.mp4"]
