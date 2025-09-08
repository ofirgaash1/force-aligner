Hebrew Forced Aligner (FastAPI)
================================

Word-level forced alignment for Hebrew using `imvladikon/wav2vec2-xls-r-300m-hebrew`.

Quickstart
----------

- Prereqs: Python 3.10+, `ffmpeg` (for MP3/WEBM decoding via torchaudio)

Setup:

1) Create venv and install deps

   - PowerShell
     - `python -m venv .venv`
     - `.\.venv\Scripts\Activate.ps1`
     - `pip install -U pip`
     - `pip install -r requirements.txt`

2) Optional: cache models to a local directory

   - `setx HF_HOME "%CD%\models"`

3) Run the API

   - `uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload`

Environment
-----------

- `ALIGN_MODEL` (default: `imvladikon/wav2vec2-xls-r-300m-hebrew`)
- `HF_HOME` (optional, to cache models under `/models`)
- `ALIGN_DEVICE` (default: `cpu`)
- `MAX_AUDIO_SEC` (default: `60`)

Endpoints
---------

- `GET /healthz` → `{ "status": "ok" }`
- `POST /align` (multipart form)
  - `audio`: audio file (wav/mp3/webm)
  - `transcript`: Hebrew transcript (spaces separate words)
  - Response: `{ duration_sec, transcript, words: [{word,start,end,score}] }`

Example (PowerShell):

```
curl -Method Post \
  -Uri http://localhost:8000/align \
  -Form @{ audio = Get-Item .\sample.wav; transcript = "שלום עולם" }
```

Notes
-----

- First run downloads the model weights (~1.2GB). Set `HF_HOME` to persist cache.
- MP3/WEBM support requires `ffmpeg` for `torchaudio.load`.
- Alignment uses CTC segmentation; scores are averaged character confidences.

