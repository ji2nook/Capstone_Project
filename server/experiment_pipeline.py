# 실험 1, 2, 3 파이프라인 관리

# 실험 1 : 순차적 처리 방식 (Sequential)
# 실험 2 : 스트리밍 처리 방식 (Streaming)
# 실험 3 : 적응형 스트리밍 처리 방식 (Adaptive Streaming)

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator

from server import api_manager, streaming_controller
from server.audio_utils import save_pcm_as_wav

_AUDIO_DIR = Path("results") / "audio"

logger = logging.getLogger(__name__)

# ========================================
# 성능 지표 데이터 클래스
# ========================================

# 모든 성능 지표 단위는 밀리초(ms)
# 실제 사용자 체감 E2E Latency는 logger.py에서 클라이언트 지표와 합산해 산출한다

@dataclass
class BaseMetrics:
    # 모든 실험 파이프라인의 공통 성능 지표
    llm_latency: float   # LLM API 요청 ~ LLM 응답 완료까지 소요 시간
    tts_latency: float   # TTS API 요청 ~ TTS 응답 완료까지 소요 시간
    e2e_latency: float   # 파이프라인 시작 ~ 오디오 전송 준비 완료 (서버 측)


@dataclass
class SequentialMetrics(BaseMetrics):
    # 실험 1 파이프라인 성능 지표
    ttft: float   # 파이프라인 시작 ~ 첫 토큰 생성 완료 (실험 1: = llm_latency)
    ttfa: float   # 파이프라인 시작 ~ 첫 음성 생성 완료 (실험 1: = e2e_latency)


@dataclass
class StreamingMetrics(BaseMetrics):
    # 실험 2 파이프라인 성능 지표
    ttft: float               # 파이프라인 시작 ~ 첫 텍스트 청크 확정 시점 (ms)
    ttfa: float               # 파이프라인 시작 ~ 첫 오디오 청크 수신 시점 (ms)
    text_chunk_count: int     # llm_text_chunker가 생성한 텍스트 청크 수
    audio_chunk_count: int    # tts_audio_chunker가 생성한 오디오 청크 수


# ========================================
# 파이프라인 실행 결과
# ========================================

@dataclass
class PipelineResult:
    # 파이프라인 실행 결과: 클라이언트로 전송할 오디오 데이터와 성능 지표
    audio_bytes: bytes   # 최종 PCM 오디오 바이트 (클라이언트에 그대로 전송)
    metrics: BaseMetrics # 이 실행에서 측정한 성능 지표 (실험별 서브클래스로 전달됨)
    prompt: str          # 사용자 입력 프롬프트
    llm_response: str    # LLM이 생성한 텍스트 응답
    audio_path: str      # MOS 평가용 WAV 파일 경로


def _save_audio(audio_bytes: bytes, experiment: str) -> str:
    _AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    path = _AUDIO_DIR / f"{experiment}_{ts}.wav"
    save_pcm_as_wav(audio_bytes, path)
    return str(path)


# ========================================
# 실험 1: 순차적 처리 파이프라인
# ========================================

# LLM 전체 응답 생성 완료 후 TTS 전체 음성 생성

async def run_sequential(prompt: str) -> PipelineResult:

    # -------------------------------
    # 시작 시간 기록
    # -------------------------------
    # perf_counter: sleep·I/O 대기 포함 실제 경과 시간을 나노초 단위로 측정
    t_start = time.perf_counter()

    logger.info("[Sequential] 파이프라인 시작")


    # -------------------------------
    # LLM 단계
    # -------------------------------
    # await: 비동기 I/O가 완료될 때까지 대기
    text = await api_manager.generate_full_response(prompt)

    # LLM API 호출이 반환된 시각 기록
    t_llm_done = time.perf_counter()

    llm_latency = (t_llm_done - t_start) * 1000

    # TTFT(실험 1): 응답 전체가 한 번에 도착하므로 TTFT == LLM Latency
    ttft = llm_latency

    logger.info("[Sequential] LLM 완료 — %.1fms, %d자", llm_latency, len(text))


    # ------------------------------
    # TTS 단계
    # ------------------------------
    audio_bytes = await api_manager.synthesize_full_audio(text)

    # TTS 응답 완료 시각 기록
    t_tts_done = time.perf_counter()

    tts_latency = (t_tts_done - t_llm_done) * 1000

    # E2E Latency(server): 클라이언트 재생 시간은 별도 측정
    e2e_latency = (t_tts_done - t_start) * 1000

    # TTFA(실험 1): TTS가 응답을 완료해야 음성이 출력되기 시작하므로 TTFA == E2E Lantency(server)
    ttfa = e2e_latency

    logger.info("[Sequential] TTS 완료 — %.1fms, %d bytes", tts_latency, len(audio_bytes))
    logger.info("[Sequential] 서버 측 E2E Latency: %.1fms", e2e_latency)

    # 레이턴시 측정 완료 후 WAV 저장 (측정값에 영향 없음)
    audio_path = _save_audio(audio_bytes, "sequential")
    logger.info("[Sequential] 오디오 저장: %s", audio_path)


    # ------------------------------
    # 성능 지표 객체 생성
    # ------------------------------
    # 측정한 값들을 SequentialMetrics 데이터 클래스에 담아 반환
    metrics = SequentialMetrics(
        llm_latency=llm_latency,
        tts_latency=tts_latency,
        ttft=ttft,
        ttfa=ttfa,
        e2e_latency=e2e_latency,
    )


    # ------------------------------
    # 결과 반환
    # ------------------------------
    # 오디오 바이트와 성능 지표를 PipelineResult에 묶어 반환
    # 오디오 바이트는 connection_manager를 통해 WebSocket으로 전송
    # 성능 지표는 connection_manager를 통해 metrics_server로 전달
    return PipelineResult(
        audio_bytes=audio_bytes,
        metrics=metrics,
        prompt=prompt,
        llm_response=text,
        audio_path=audio_path,
    )


# ========================================
# 실험 2: 스트리밍 처리 파이프라인
# ========================================

# LLM 토큰 스트림을 고정된 청크 단위 / 문장 경계 감지하여 분할 후 전송 및 TTS 청크 합성 후 오디오 청크 스트리밍

async def run_streaming(
    prompt: str,
    max_text_chars: int = streaming_controller.TEXT_CHUNK_SIZE,
) -> AsyncGenerator[tuple[bytes, StreamingMetrics | None], None]:

    t_start = time.perf_counter()
    ttft: float | None = None
    ttfa: float | None = None
    all_audio = bytearray()
    text_chunk_count = 0
    audio_chunk_count = 0

    # 마지막 텍스트 청크의 TTS 시작 시점 추적 (tts_latency 근사치 산출용)
    t_last_tts_start: float = t_start

    logger.info("[Streaming] 파이프라인 시작")

    # ------------------------------
    # LLM 스트리밍 및 텍스트 청크 분할
    # ------------------------------
    # 문장 경계 우선 flush, max_text_chars 초과 시 강제 flush (하이브리드)
    token_stream = api_manager.generate_streaming_response(prompt)
    text_chunks = streaming_controller.llm_text_chunker(
        token_stream,
        max_chars=max_text_chars,
        use_sentence_end=True,
    )

    async for text_chunk in text_chunks:

        # 첫 텍스트 청크 확정 시점 = TTFT
        if ttft is None:
            ttft = (time.perf_counter() - t_start) * 1000
            logger.info("[Streaming] 첫 텍스트 청크 확정 — TTFT=%.1fms, %d자", ttft, len(text_chunk))

        text_chunk_count += 1

        # ------------------------------
        # TTS 스트리밍 및 오디오 청크 분할
        # ------------------------------
        # synthesize_streaming_audio: ElevenLabs raw 바이트를 불규칙한 크기로 yield
        # 무음 경계 우선 flush, max_chunk_bytes 초과 시 강제 flush (하이브리드)
        t_last_tts_start = time.perf_counter()
        raw_audio_stream = api_manager.synthesize_streaming_audio(text_chunk)

        async for audio_chunk in streaming_controller.tts_audio_chunker(raw_audio_stream):

            # 첫 오디오 청크 수신 시점 = TTFA
            if ttfa is None:
                ttfa = (time.perf_counter() - t_start) * 1000
                logger.info("[Streaming] 첫 오디오 청크 수신 — TTFA=%.1fms", ttfa)

            audio_chunk_count += 1
            all_audio.extend(audio_chunk)
            yield audio_chunk, None

    t_end = time.perf_counter()
    e2e_latency = (t_end - t_start) * 1000

    # tts_latency: 마지막 텍스트 청크의 TTS 처리 소요 시간 (근사)
    # 여러 청크에 걸쳐 TTS가 순차 호출되므로 전체 합산이 아닌 마지막 기준으로 기록
    tts_latency = (t_end - t_last_tts_start) * 1000

    # 레이턴시 측정 완료 후 WAV 저장
    audio_path = _save_audio(bytes(all_audio), "streaming")

    logger.info(
        "[Streaming] 파이프라인 완료 — E2E=%.1fms, 텍스트 청크=%d, 오디오 청크=%d, %d bytes, 저장=%s",
        e2e_latency, text_chunk_count, audio_chunk_count, len(all_audio), audio_path,
    )

    # ------------------------------
    # 종료 신호 + 성능 지표 반환
    # ------------------------------
    # 빈 바이트와 함께 StreamingMetrics를 yield해 호출 측이 종료를 감지하도록 함
    # ttft / ttfa가 None이면 오디오가 전혀 생성되지 않은 경우이므로 0.0으로 기록
    metrics = StreamingMetrics(
        llm_latency=ttft if ttft is not None else 0.0,
        tts_latency=tts_latency,
        e2e_latency=e2e_latency,
        ttft=ttft if ttft is not None else 0.0,
        ttfa=ttfa if ttfa is not None else 0.0,
        text_chunk_count=text_chunk_count,
        audio_chunk_count=audio_chunk_count,
    )
    yield b"", metrics
