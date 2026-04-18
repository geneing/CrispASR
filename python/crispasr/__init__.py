"""CrispASR — lightweight speech recognition via ggml."""

from ._binding import (
    CrispASR,
    DiarizeMethod,
    DiarizeSegment,
    LidMethod,
    LidResult,
    Segment,
    Session,
    SessionSegment,
    SessionWord,
    detect_language_pcm,
    diarize_segments,
)

__all__ = [
    "CrispASR",
    "DiarizeMethod",
    "DiarizeSegment",
    "LidMethod",
    "LidResult",
    "Segment",
    "Session",
    "SessionSegment",
    "SessionWord",
    "detect_language_pcm",
    "diarize_segments",
]
__version__ = "0.4.6"
