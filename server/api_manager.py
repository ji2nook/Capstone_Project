# API_Manager : LLM, TTS API 관리

import os
import logging

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
# 스트리밍 처리 함수 (실험 2)
# ========================================




# ========================================
# 적응형 스트리밍 처리 함수 (실험 3)
# ========================================