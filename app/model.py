from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class AlignerModel:
    processor: object
    model: object
    sample_rate: int = 16000
    device: str = "cpu"


_CACHED: Optional[AlignerModel] = None


def get_model(model_id: str, device: str = "cpu") -> AlignerModel:
    """
    Lazy-load and cache the HF processor + CTC model.
    Imports are local to avoid hard dependency at import time.
    """
    global _CACHED
    if _CACHED is not None and _CACHED.device == device:
        return _CACHED

    # Local imports to keep module import light without deps installed yet.
    from transformers import AutoProcessor, AutoModelForCTC  # type: ignore
    import torch  # type: ignore

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForCTC.from_pretrained(model_id)
    model.to(device)
    model.eval()

    _CACHED = AlignerModel(processor=processor, model=model, device=device)
    return _CACHED

