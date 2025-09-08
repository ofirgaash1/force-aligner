from typing import List, Optional
from pydantic import BaseModel, Field


class WordAlignment(BaseModel):
    word: str = Field(..., description="The word text as in transcript")
    start: float = Field(..., ge=0, description="Start time in seconds")
    end: float = Field(..., ge=0, description="End time in seconds")
    score: float = Field(..., ge=0, le=1, description="Confidence score 0..1")


class AlignmentResult(BaseModel):
    duration_sec: float
    words: List[WordAlignment]
    transcript: Optional[str] = None

