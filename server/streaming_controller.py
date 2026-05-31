# 데이터 스트림 가공/관리

import logging
import re
import struct
from typing import AsyncGenerator, Callable

from server.experiment_config import AUDIO_CHUNK_SIZE, TEXT_CHUNK_SIZE

logger = logging.getLogger(__name__)


# ========================================
# 상수 정의
# ========================================

# 문장부호 뒤 공백(\s+), 스트림 끝($), 독립 개행(\n+)을 경계로 인식
SENTENCE_END = re.compile(r'(?<=[.?!。])\s+|(?<=[.?!。])$|\n+')

# tts_audio_chunker 전용 PCM 포맷 상수 (api_manager TTS_OUTPUT_FORMAT="pcm_24000")
_PCM_SAMPLE_RATE: int = 24_000
_PCM_SAMPLE_WIDTH: int = 2  # 16-bit signed



# ========================================
# LLM 텍스트 청킹 (실험 2)
# ========================================

async def llm_text_chunker(
    token_stream: AsyncGenerator[str, None],
    max_chars: int | Callable[[], int] = TEXT_CHUNK_SIZE,
    use_sentence_end: bool = True,
) -> AsyncGenerator[str, None]:
    # 문장 경계 감지와 고정 문장 길이 감지 방식을 결합한 하이브리드 방식으로 청크 분할
    # max_chars에 callable을 넘기면 매 토큰마다 호출해 동적으로 크기를 재평가한다
    if callable(max_chars):
        _get_max = max_chars
    else:
        _limit = max_chars
        _get_max = lambda: _limit

    buffer = ""
    async for token in token_stream:
        buffer += token
        if use_sentence_end:
            # 하나의 토큰 안에 경계가 여러 개 있을 수 있으므로 루프로 처리
            while True:
                match = SENTENCE_END.search(buffer)
                if not match:
                    break
                chunk = buffer[: match.end()].strip()
                buffer = buffer[match.end():]
                if chunk:
                    logger.debug("llm_text_chunker: 문장 경계 %d자", len(chunk))
                    yield chunk
        # 경계 처리 후 남은 버퍼가 현재 max_chars 이상이면 강제 flush
        if len(buffer) >= _get_max():
            logger.debug("llm_text_chunker: max_chars 초과 %d자", len(buffer))
            yield buffer.strip()
            buffer = ""
    if buffer.strip():
        logger.debug("llm_text_chunker: 잔여 %d자 flush", len(buffer.strip()))
        yield buffer.strip()



# ========================================
# TTS 오디오 청킹 (실험 2)
# ========================================

def _find_silence_boundary(
    pcm_bytes: bytes,
    threshold: int,
    min_silence_bytes: int,
    search_offset_bytes: int,
) -> int | None:
    # 무음 구간이 min_silence_bytes 이상 지속되는 시작 위치(byte index)를 반환
    # search_offset_bytes 이전 구간은 탐색에서 제외하여 최소 청크 크기 보장
    if len(pcm_bytes) <= search_offset_bytes:
        return None

    sample_offset = search_offset_bytes // _PCM_SAMPLE_WIDTH
    sample_count = len(pcm_bytes) // _PCM_SAMPLE_WIDTH
    samples = struct.unpack_from(f"<{sample_count}h", pcm_bytes)

    silence_start: int | None = None
    for i in range(sample_offset, sample_count):
        if abs(samples[i]) < threshold:
            if silence_start is None:
                silence_start = i
            if (i - silence_start + 1) * _PCM_SAMPLE_WIDTH >= min_silence_bytes:
                return silence_start * _PCM_SAMPLE_WIDTH
        else:
            silence_start = None
    return None


async def tts_audio_chunker(
    audio_stream: AsyncGenerator[bytes, None],
    silence_threshold: int = 300,
    min_silence_ms: int = 50,
    max_chunk_bytes: int = AUDIO_CHUNK_SIZE * 4,
) -> AsyncGenerator[bytes, None]:
    # 무음 구간 감지와 고정 바이트 길이 감지 방식을 결합한 하이브리드 방식으로 청크 분할
    min_silence_bytes = (
        int(_PCM_SAMPLE_RATE * min_silence_ms / 1000) * _PCM_SAMPLE_WIDTH
    )
    buffer = b""
    async for raw_chunk in audio_stream:
        if not raw_chunk:
            continue
        buffer += raw_chunk
        boundary = _find_silence_boundary(
            buffer, silence_threshold, min_silence_bytes, AUDIO_CHUNK_SIZE
        )
        if boundary is not None:
            logger.debug("tts_audio_chunker: 무음 경계 %d bytes", boundary)
            yield buffer[:boundary]
            buffer = buffer[boundary:]
        elif len(buffer) >= max_chunk_bytes:
            logger.debug("tts_audio_chunker: max_chunk_bytes 초과 강제 flush")
            yield buffer[:max_chunk_bytes]
            buffer = buffer[max_chunk_bytes:]
    if buffer:
        logger.debug("tts_audio_chunker: 잔여 %d bytes flush", len(buffer))
        yield buffer
