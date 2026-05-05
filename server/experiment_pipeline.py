# 실험 1, 2, 3 파이프라인 관리

# 실험 1 : 순차적 처리 방식 (Sequential)
# 실험 2 : 스트리밍 처리 방식 (Streaming)
# 실험 3 : 적응형 스트리밍 처리 방식 (Adaptive Streaming)

import logging
import time

from dataclasses import dataclass

from server import api_manager

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


# ========================================
# 파이프라인 실행 결과
# ========================================

@dataclass
class PipelineResult:
    # 파이프라인 실행 결과: 클라이언트로 전송할 오디오 데이터와 성능 지표
    audio_bytes: bytes   # 최종 PCM 오디오 바이트 (클라이언트에 그대로 전송)
    metrics: BaseMetrics # 이 실행에서 측정한 성능 지표 (실험별 서브클래스로 전달됨)


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
    return PipelineResult(audio_bytes=audio_bytes, metrics=metrics)
