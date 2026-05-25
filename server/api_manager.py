# API_Manager : LLM, TTS API 관리

import os
import logging
from typing import AsyncGenerator

from openai import AsyncOpenAI
from elevenlabs.client import AsyncElevenLabs

logger = logging.getLogger(__name__)

# ========================================
# 상수 정의
# ========================================

# API 키 로드
LLM_API_KEY: str = os.environ.get("LLM_API_KEY", "")
TTS_API_KEY: str = os.environ.get("TTS_API_KEY", "")

# LLM 모델 설정
LLM_MODEL: str = "openai/gpt-4o-mini-2024-07-18"

# TTS 모델 설정
TTS_VOICE_ID: str = "JBFqnCBsd6RMkjVDRZzb"
TTS_MODEL_ID: str = "eleven_flash_v2_5"
TTS_OUTPUT_FORMAT: str = "pcm_24000"



# ========================================
# 클라이언트 초기화 (모듈 로드 시 한 번만 생성)
# ========================================

# OpenAI 비동기 클라이언트
llm_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=LLM_API_KEY
)

# ElevenLabs 비동기 클라이언트
tts_client = AsyncElevenLabs(
    api_key=TTS_API_KEY
)



# ========================================
# 순차적 처리 함수 (실험 1)
# ========================================

# LLM 전체 응답 생성 함수
# 스트리밍 없이 완성된 응답 텍스트 전체를 한 번에 받는다.
async def generate_full_response(prompt: str) -> str:
    logger.debug("LLM 요청 시작: %s", prompt)

    response = await llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    text: str = response.choices[0].message.content

    logger.debug("LLM 응답 완료: %d자", len(text))
    return text

# TTS 전체 오디오 생성 함수
# 스트리밍 없이 전체 오디오를 메모리에 수집한 뒤 한 번에 반환한다.
async def synthesize_full_audio(text: str) -> bytes:
    logger.debug("TTS 요청 시작: %s", text)

    audio_stream = tts_client.text_to_speech.convert(
        text=text,
        voice_id=TTS_VOICE_ID,
        model_id=TTS_MODEL_ID,
        output_format=TTS_OUTPUT_FORMAT
    )

    audio_buffer = bytearray()

    async for chunk in audio_stream:
        if chunk:
            audio_buffer.extend(chunk)

    audio_bytes: bytes = bytes(audio_buffer)
    logger.debug("TTS 생성 완료: %d bytes", len(audio_bytes))
    return audio_bytes



# ========================================
# 스트리밍 처리 함수 (실험 2, 실험 3)
# ========================================

# LLM 스트리밍 응답 생성 함수
# stream=True 옵션으로 토큰을 수신하는 즉시 yield한다.
# OpenAI SDK에서 stream=True일 때 create()는 스트림 객체를 반환하는 coroutine이므로 await 필요
async def generate_streaming_response(prompt: str) -> AsyncGenerator[str, None]:
    #LLM 토큰 스트림을 AsyncGenerator로 yield
    logger.debug("LLM 스트리밍 요청 시작: %s", prompt)

    stream = await llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    token_count = 0
    total_chars = 0
    async for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            token_count += 1
            total_chars += len(token)
            yield token

    logger.debug("LLM 스트리밍 완료: %d 토큰, %d자", token_count, total_chars)


# TTS 스트리밍 오디오 합성 함수
# ElevenLabs SDK가 전달하는 raw 바이트를 수신 단위 그대로 yield한다.
# 고정 크기 정규화 및 적응형 분할은 streaming_controller.py에서 담당한다.
# convert()는 AsyncGenerator를 직접 반환하므로 await 없이 사용한다.
async def synthesize_streaming_audio(text: str) -> AsyncGenerator[bytes, None]:
    """ElevenLabs 스트리밍 TTS. 수신된 raw 바이트를 그대로 yield."""
    logger.debug("TTS 스트리밍 요청 시작: %s", text)

    audio_stream = tts_client.text_to_speech.convert(
        voice_id=TTS_VOICE_ID,
        text=text,
        model_id=TTS_MODEL_ID,
        output_format=TTS_OUTPUT_FORMAT,
    )

    chunk_count = 0
    total_bytes = 0
    async for raw_chunk in audio_stream:
        if raw_chunk:
            yield raw_chunk
            chunk_count += 1
            total_bytes += len(raw_chunk)

    logger.debug("TTS 스트리밍 완료: %d 청크, %d bytes", chunk_count, total_bytes)



# ========================================
# 워밍업 함수
# ========================================

async def warmup() -> None:
    # 서버 시작 시 LLM/TTS API와 미리 연결하여 초기 연결의 긴 대기시간 제거
    logger.info("API 워밍업 시작")
    text = await generate_full_response("안녕")
    await synthesize_full_audio(text)
    logger.info("API 워밍업 완료")

