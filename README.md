Hebrew Forced Aligner (FastAPI)
================================

Word-level forced alignment for Hebrew using `imvladikon/wav2vec2-xls-r-300m-hebrew`. Ships with a Docker setup that caches the model into a host volume so it only downloads once.

Quick Start (Docker)
--------------------

Clone and start with Compose:

```
git clone git@github.com:ofirgaash1/force-aligner.git
cd force-aligner
docker compose build
docker compose up -d
```

This launches FastAPI on port 8000 and mounts `./models` → `/models` inside the container to persist the Hugging Face cache across runs.

Send a Request
--------------

- local

```
curl -X POST -F "audio=@sample.wav" -F "transcript=שלום עולם" http://localhost:8000/align
```

- Remote example=

```
curl -X POST -F "audio=@sample.wav" -F "transcript=שלום עולם" http://silence-remover.com:8000/align
```

Example Response:

```
{
  "duration_sec": 10.69,
  "transcript": "שלום עולם",
  "words": [
    { "word": "שלום", "start": 5.647, "end": 5.88, "score": 0.72 },
    { "word": "עולם", "start": 7.02,  "end": 7.36, "score": 0.69 }
  ]
}
```

Performance
-----------

- On an AWS t2.micro: ~20 seconds to align a 10-second (~1 MB) WAV.
- First call downloads ~1.2 GB of model weights to `./models`. With the volume mapping, subsequent runs reuse the cache.

API
---

- `GET /healthz` → `{ "status": "ok" }`
- `POST /align`
  - multipart form fields:
    - `audio`: audio file (wav/mp3/webm; wav is fastest)
    - `transcript`: Hebrew transcript (spaces separate words)
  - returns: `{ duration_sec, transcript, words: [{word,start,end,score}] }`

Configuration
-------------

Environment variables (defined in `docker-compose.yml`):

- `ALIGN_MODEL` (default: `imvladikon/wav2vec2-xls-r-300m-hebrew`)
- `ALIGN_DEVICE` (default: `cpu`)
- `MAX_AUDIO_SEC` (default: `60`)
- `HF_HOME` (default: `/models`, persisted via volume)

Development (without Docker)
----------------------------

Requirements: Python 3.10+, ffmpeg, libsndfile

```
python -m venv .venv
source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Notes & Tips
------------

- The container uses CPU-only PyTorch wheels to keep the image lean and avoid CUDA downloads.
- WAV inputs work without ffmpeg; MP3/WEBM decoding needs ffmpeg (already installed in the image).
- Current alignment uses a greedy CTC path heuristic (fast, good baseline). If you need higher precision later, we can add an optional DP-based aligner.

