from __future__ import annotations

import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from .config import get_settings


app = FastAPI(title="Hebrew Forced Aligner")


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/align")
async def align(
    audio: UploadFile = File(...),
    transcript: str = Form(...),
):
    if not transcript or not transcript.strip():
        raise HTTPException(status_code=400, detail="Transcript is required")

    # Defer heavy imports to runtime to keep startup fast during scaffolding.
    from .config import get_settings
    from .model import get_model
    from .audio import load_audio_to_16k
    from .alignment import logits_from_audio, align_words

    settings = get_settings()

    # Read bytes
    audio_bytes = await audio.read()

    # Load and resample
    waveform, sr, duration = load_audio_to_16k(audio_bytes, 16000, audio.filename)

    if duration > settings.max_audio_sec:
        raise HTTPException(status_code=413, detail="Audio too long")

    # Load model
    model_bundle = get_model(settings.align_model, settings.device)

    # Get emissions and align
    emissions = logits_from_audio(model_bundle, waveform, sr)
    try:
        words = align_words(
            emissions, transcript, model_bundle.processor, duration
        )
    except NotImplementedError:
        raise HTTPException(
            status_code=501,
            detail="Alignment not implemented yet; foundation scaffold complete",
        )

    return JSONResponse(
        content={
            "duration_sec": duration,
            "transcript": transcript,
            "words": words,
        }
    )
