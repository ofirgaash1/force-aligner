from __future__ import annotations

from io import BytesIO
from typing import Tuple, Union, Optional
import os
import tempfile


def _to_tensor(waveform, target_sr: int):
    import torch  # type: ignore
    import torchaudio  # type: ignore

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    # Convert to mono
    if waveform.size(0) > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    return waveform


def load_audio_to_16k(
    source: Union[str, bytes, BytesIO],
    target_sr: int = 16000,
    hint_filename: Optional[str] = None,
) -> Tuple["torch.Tensor", int, float]:
    """
    Load audio from path or bytes-like to mono tensor at 16 kHz.
    Returns (waveform[1, T], sample_rate, duration_sec)
    """
    import torch  # type: ignore
    import torchaudio  # type: ignore

    def _load_with_torchaudio(path: str):
        return torchaudio.load(path)

    def _load_with_soundfile(path: str):
        import soundfile as sf  # type: ignore
        import numpy as np  # type: ignore

        data, sr_local = sf.read(path, dtype="float32", always_2d=True)  # [T, C]
        data = data.T  # [C, T]
        return torch.from_numpy(np.ascontiguousarray(data)), int(sr_local)

    if isinstance(source, (bytes, BytesIO)):
        # On some platforms, torchaudio cannot load from file-like buffers.
        # Fallback: write to a temporary file with an appropriate suffix.
        if isinstance(source, bytes):
            buffer = BytesIO(source)
        else:
            buffer = source
        buffer.seek(0)
        suffix = ""
        if hint_filename:
            _, ext = os.path.splitext(hint_filename)
            if ext:
                suffix = ext
        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(suffix=suffix or ".wav")
            with os.fdopen(fd, "wb") as f:
                f.write(buffer.read())
            try:
                waveform, sr = _load_with_torchaudio(tmp_path)
            except Exception:
                waveform, sr = _load_with_soundfile(tmp_path)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
    else:
        try:
            waveform, sr = _load_with_torchaudio(source)
        except Exception:
            waveform, sr = _load_with_soundfile(source)

    waveform = _to_tensor(waveform, target_sr)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        sr = target_sr

    duration_sec = float(waveform.size(1)) / float(sr)
    return waveform, sr, duration_sec
