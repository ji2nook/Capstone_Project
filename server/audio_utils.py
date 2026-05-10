import wave
from pathlib import Path


def save_pcm_as_wav(
    pcm_bytes: bytes,
    output_path: str | Path,
    sample_rate: int = 24000,
    channels: int = 1,
    sampwidth: int = 2,
) -> None:
    """PCM raw bytes에 WAV 헤더를 붙여 파일로 저장한다."""
    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
