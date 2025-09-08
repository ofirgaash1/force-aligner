import os
from functools import lru_cache


class Settings:
    align_model: str = os.getenv(
        "ALIGN_MODEL", "imvladikon/wav2vec2-xls-r-300m-hebrew"
    )
    hf_home: str | None = os.getenv("HF_HOME")
    device: str = os.getenv("ALIGN_DEVICE", "cpu")
    max_audio_sec: float = float(os.getenv("MAX_AUDIO_SEC", "60"))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

