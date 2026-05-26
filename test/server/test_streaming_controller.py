import asyncio
import logging
import struct
import sys

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

sys.path.append(".")

from server.api_manager import generate_streaming_response
from server.streaming_controller import (
    AUDIO_CHUNK_SIZE,
    TEXT_CHUNK_SIZE,
    llm_text_chunker,
    tts_audio_chunker,
)

TEST_PROMPT = "5문장으로 너에 대해서 자기소개해줘."


# ===== LLM 텍스트 청킹 =====

async def test_llm_hybrid() -> None:
    # 하이브리드 모드: 문장 경계 우선, max_chars 초과 시 강제 flush
    print("=== LLM — 하이브리드 (문장 경계 + max_chars 안전망) ===")
    n = 0
    async for chunk in llm_text_chunker(
        generate_streaming_response(TEST_PROMPT),
        max_chars=TEXT_CHUNK_SIZE,
        use_sentence_end=True,
    ):
        n += 1
        print(f"[청크 {n}] ({len(chunk)}자) {chunk}")
    print(f"총 {n}개 청크\n")


async def test_llm_fixed_only() -> None:
    # 고정 길이 단독 모드: 문장 경계 무시, max_chars마다 청크 전송
    print("=== LLM — 고정 길이 전용 (50자) ===")
    n = 0
    async for chunk in llm_text_chunker(
        generate_streaming_response(TEST_PROMPT),
        max_chars=50,
        use_sentence_end=False,
    ):
        n += 1
        print(f"[청크 {n}] ({len(chunk)}자) {chunk}")
    print(f"총 {n}개 청크\n")


# ===== TTS 오디오 청킹 =====

def _pcm(amplitude: int, n_samples: int) -> bytes:
    return struct.pack(f"<{n_samples}h", *([amplitude] * n_samples))


async def _dummy_audio_stream():
    # 무음 구간이 포함된 더미 PCM 스트림 (실제 TTS 대용)
    # 구조: 음성 8192B + 음성 16384B + 무음 2400B (~50ms) + 음성 8192B
    yield _pcm(400, AUDIO_CHUNK_SIZE // 2)   # 8192 bytes 음성
    yield _pcm(400, 16384 // 2)              # 16384 bytes 음성
    yield _pcm(0, 2400 // 2)                 # 2400 bytes 무음 (경계 감지 대상)
    yield _pcm(400, AUDIO_CHUNK_SIZE // 2)   # 8192 bytes 음성


async def test_tts_hybrid() -> None:
    # 하이브리드 모드: 무음 경계 우선, max_chunk_bytes 초과 시 강제 flush
    print("=== TTS — 하이브리드 (무음 경계 + max_chunk_bytes 안전망) ===")
    n = 0
    total = 0
    async for chunk in tts_audio_chunker(
        _dummy_audio_stream(),
        silence_threshold=300,
        min_silence_ms=50,
        max_chunk_bytes=AUDIO_CHUNK_SIZE * 4,
    ):
        n += 1
        total += len(chunk)
        print(f"[청크 {n}] {len(chunk):,} bytes")
    print(f"총 {n}개 청크, {total:,} bytes\n")


async def test_tts_fixed_only() -> None:
    # 고정 크기 전용 모드: 무음 경계 무시, max_chunk_bytes마다 강제 flush
    print("=== TTS — 고정 크기 전용 (8192 bytes) ===")
    n = 0
    total = 0
    async for chunk in tts_audio_chunker(
        _dummy_audio_stream(),
        silence_threshold=0,    # abs(x) < 0 는 항상 false -> 무음 미감지
        min_silence_ms=9999,
        max_chunk_bytes=AUDIO_CHUNK_SIZE,
    ):
        n += 1
        total += len(chunk)
        print(f"[청크 {n}] {len(chunk):,} bytes")
    print(f"총 {n}개 청크, {total:,} bytes\n")


asyncio.run(test_llm_hybrid())
asyncio.run(test_llm_fixed_only())
asyncio.run(test_tts_hybrid())
asyncio.run(test_tts_fixed_only())
