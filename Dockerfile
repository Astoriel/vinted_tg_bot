FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download MobileNetV3 weights during build
RUN python -c "from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights; mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)"

# Create storage directories
RUN mkdir -p storage/weights storage/memory_bank storage/dream_logs

EXPOSE 7860

CMD ["python", "app.py"]
