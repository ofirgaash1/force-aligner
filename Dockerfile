FROM python:3.11-slim

# Environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    ALIGN_MODEL=imvladikon/wav2vec2-xls-r-300m-hebrew \
    ALIGN_DEVICE=cpu \
    MAX_AUDIO_SEC=60

# System deps (audio + ssl roots)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ffmpeg libsndfile1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (leverage Docker layer cache)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY app ./app

# Optional: pre-download the model into /models during build
ARG PRELOAD_MODEL=false
RUN mkdir -p /models \
    && if [ "$PRELOAD_MODEL" = "true" ]; then \
         python - <<'PY'; \
from app.model import get_model
from app.config import get_settings
s = get_settings()
print('Preloading model:', s.align_model)
get_model(s.align_model, s.device)
print('Model cached under HF_HOME:', s.hf_home)
PY
       fi

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

