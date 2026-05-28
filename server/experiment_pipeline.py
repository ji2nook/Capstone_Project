# 실험 1, 2, 3 파이프라인 관리

# 실험 1 : 순차적 처리 방식 (Sequential)
# 실험 2 : 스트리밍 처리 방식 (Streaming)
# 실험 3 : 적응형 스트리밍 처리 방식 (Adaptive Streaming)

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator

from server import api_manager, streaming_controller
from server.adaptive_controller import AdaptiveController
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
    llm_response: str = ""    # LLM이 생성한 전체 텍스트 (텍스트 청크 누적)
    audio_path: str = ""      # MOS 평가용 WAV 파일 경로


@dataclass
class AdaptiveMetrics(BaseMetrics):
    # 실험 3 파이프라인 성능 지표
    ttft: float                     # 파이프라인 시작 ~ 첫 텍스트 청크 확정 시점 (ms)
    ttfa: float                     # 파이프라인 시작 ~ 첫 오디오 청크 수신 시점 (ms)
    text_chunk_count: int           # llm_text_chunker가 생성한 텍스트 청크 수
    audio_chunk_count: int          # tts_audio_chunker가 생성한 오디오 청크 수
    llm_response: str = ""          # LLM이 생성한 전체 텍스트 (텍스트 청크 누적)
    audio_path: str = ""            # MOS 평가용 WAV 파일 경로

    # 적응형 제어 전용 필드
    # control_snapshots: 텍스트 청크 확정 시점마다 저장한 AdaptiveState 스냅샷 리스트
    # Step 8에서 지표 기록에 활용되며, 청크별 제어 결정 히스토리를 제공한다
    control_snapshots: list[dict] = field(default_factory=list)
    avg_text_chunk_size: float = 0.0    # 세션 전체 결정된 텍스트 청크 크기 평균
    avg_audio_chunk_size: float = 0.0   # 세션 전체 결정된 오디오 청크 크기 평균
    # Step 6에서 버퍼 고갈 긴급 축소 로직 구현 후 실제 값 반영
    starvation_adjustments: int = 0


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

    # 큐 종료를 알리는 sentinel — bytes·Queue와 타입이 달라 isinstance 없이 구별 가능
    _DONE: object = object()

    t_start = time.perf_counter()
    ttft: float | None = None
    ttfa: float | None = None
    all_audio = bytearray()
    text_chunk_count = 0
    audio_chunk_count = 0
    t_last_tts_start: float = t_start
    full_llm_text: str = ""

    # pipeline: 텍스트 청크 순서대로 등록되는 "오디오 큐의 큐"
    # 원소: asyncio.Queue(오디오 청크 담당) | _DONE(텍스트 스트림 종료 신호)
    pipeline: asyncio.Queue = asyncio.Queue()

    logger.info("[Streaming] 파이프라인 시작 (max_text_chars=%d)", max_text_chars)

    # ------------------------------
    # TTS worker
    # ------------------------------
    # 텍스트 청크 1개를 전담하는 독립 태스크
    # 이전 청크의 오디오가 클라이언트로 전송되는 동안 다음 청크의 TTS가 병렬로 실행됨
    async def _tts_worker(text_chunk: str, audio_q: asyncio.Queue) -> None:
        try:
            raw_audio_stream = api_manager.synthesize_streaming_audio(text_chunk)
            async for audio_chunk in streaming_controller.tts_audio_chunker(raw_audio_stream):
                await audio_q.put(audio_chunk)
        except Exception:
            logger.error("[Streaming] TTS worker 오류 (%d자)", len(text_chunk), exc_info=True)
        finally:
            # 성공·실패 모두 sentinel을 넣어 소비자가 무한 대기하지 않도록 보장
            await audio_q.put(_DONE)

    # ------------------------------
    # 텍스트 프로듀서
    # ------------------------------
    # LLM 토큰 스트림 -> 텍스트 청크 -> TTS 태스크 즉시 시작
    async def _text_producer() -> None:
        nonlocal ttft, text_chunk_count, t_last_tts_start, full_llm_text

        token_stream = api_manager.generate_streaming_response(prompt)
        text_chunks = streaming_controller.llm_text_chunker(
            token_stream,
            max_chars=max_text_chars,
            use_sentence_end=True,
        )
        async for text_chunk in text_chunks:
            if ttft is None:
                ttft = (time.perf_counter() - t_start) * 1000
                logger.info("[Streaming] 첫 텍스트 청크 확정 — TTFT=%.1fms, %d자", ttft, len(text_chunk))

            full_llm_text += text_chunk
            text_chunk_count += 1
            t_last_tts_start = time.perf_counter()

            # pipeline에 audio_q를 먼저 등록해 소비자가 순서를 유지할 수 있게 함
            # TTS 태스크는 등록 직후 즉시 백그라운드에서 시작
            audio_q: asyncio.Queue = asyncio.Queue()
            await pipeline.put(audio_q)
            asyncio.create_task(_tts_worker(text_chunk, audio_q))

        await pipeline.put(_DONE)

    producer = asyncio.create_task(_text_producer())

    # ------------------------------
    # 소비자
    # ------------------------------
    # pipeline에서 audio_q를 순서대로 꺼내 오디오 청크를 yield
    
    # audio_q N을 소비하는 동안 audio_q N+1의 TTS는 이미 백그라운드에서 실행 중
    # → TTS API 왕복 대기 시간이 이전 청크 재생 시간과 겹쳐 전체 E2E가 단축됨
    while True:
        item = await pipeline.get()
        if item is _DONE:
            break

        audio_q: asyncio.Queue = item
        while True:
            audio_chunk = await audio_q.get()
            if audio_chunk is _DONE:
                break

            if ttfa is None:
                ttfa = (time.perf_counter() - t_start) * 1000
                logger.info("[Streaming] 첫 오디오 청크 수신 — TTFA=%.1fms", ttfa)

            audio_chunk_count += 1
            all_audio.extend(audio_chunk)
            yield audio_chunk, None

    await producer

    t_end = time.perf_counter()
    e2e_latency = (t_end - t_start) * 1000

    # tts_latency: 마지막 TTS 시작 ~ 파이프라인 완료까지의 시간
    # 병렬 실행으로 인해 단일 TTS 소요 시간의 근사치임
    tts_latency = (t_end - t_last_tts_start) * 1000

    audio_path = _save_audio(bytes(all_audio), "streaming")

    logger.info(
        "[Streaming] 파이프라인 완료 — E2E=%.1fms, 텍스트 청크=%d, 오디오 청크=%d, %d bytes, 저장=%s",
        e2e_latency, text_chunk_count, audio_chunk_count, len(all_audio), audio_path,
    )

    # ------------------------------
    # 성능 지표 객체 생성
    # ------------------------------
    metrics = StreamingMetrics(
        llm_latency=ttft if ttft is not None else 0.0,
        tts_latency=tts_latency,
        e2e_latency=e2e_latency,
        ttft=ttft if ttft is not None else 0.0,
        ttfa=ttfa if ttfa is not None else 0.0,
        text_chunk_count=text_chunk_count,
        audio_chunk_count=audio_chunk_count,
        llm_response=full_llm_text,
        audio_path=audio_path,
    )
    yield b"", metrics


# ========================================
# 실험 3: 적응형 스트리밍 처리 파이프라인
# ========================================

# LLM 토큰 속도, TTS 합성 속도, 버퍼 상태, 네트워크 지표를 실시간으로 모니터링
# 텍스트 청크 크기와 오디오 청크 크기를 동적으로 조정하는 적응형 파이프라인

async def run_adaptive(
    prompt: str,
) -> AsyncGenerator[tuple[bytes, AdaptiveMetrics | None], None]:

    _DONE: object = object()

    t_start = time.perf_counter()
    ttft: float | None = None
    ttfa: float | None = None
    all_audio = bytearray()
    text_chunk_count = 0
    audio_chunk_count = 0
    t_last_tts_start: float = t_start
    full_llm_text: str = ""

    # 파이프라인 시작 시 1회 생성, 이후 각 콜백 지점에 전달
    controller = AdaptiveController()

    # 텍스트/오디오 청크 크기 결정값 누적 (세션 평균 계산에 사용)
    decided_text_chunk_sizes: list[int] = []
    decided_audio_chunk_sizes: list[int] = []

    # 텍스트 청크 확정 시점마다 저장하는 AdaptiveState 스냅샷 리스트
    control_snapshots: list[dict] = []

    pipeline: asyncio.Queue = asyncio.Queue()

    logger.info("[Adaptive] 파이프라인 시작")

    # ------------------------------
    # 토큰 인터셉터
    # ------------------------------
    # LLM 토큰 스트림을 그대로 통과시키면서 on_llm_token() 콜백 삽입
    
    # llm_text_chunker는 토큰 스트림을 소비하는 구조이므로, 청커에 토큰을 넘기기 전에
    # 래퍼 제너레이터를 거쳐 컨트롤러 상태를 먼저 갱신
    async def _token_interceptor(
        token_stream: AsyncGenerator[str, None],
    ) -> AsyncGenerator[str, None]:
        async for token in token_stream:
            controller.on_llm_token(token)
            yield token

    # ------------------------------
    # TTS worker
    # ------------------------------
        
    # run_streaming()과 달리 controller를 클로저로 참조하여 오디오 청크 크기를 결정하고
    # TTS 오디오 수신 콜백을 호출
    async def _tts_worker(text_chunk: str, audio_q: asyncio.Queue) -> None:
        # 이 텍스트 청크를 처리할 시점의 오디오 청크 크기 결정
        audio_chunk_size = controller.decide_audio_chunk_size()
        decided_audio_chunk_sizes.append(audio_chunk_size)
        try:
            raw_audio_stream = api_manager.synthesize_streaming_audio(text_chunk)
            async for audio_chunk in streaming_controller.tts_audio_chunker(
                raw_audio_stream,
                max_chunk_bytes=audio_chunk_size,
            ):
                # TTS 오디오 수신 콜백: Step 5에서 tts_synthesis_rate 갱신에 활용
                controller.on_tts_audio_received(len(audio_chunk))
                await audio_q.put(audio_chunk)
        except Exception:
            logger.error("[Adaptive] TTS worker 오류 (%d자)", len(text_chunk), exc_info=True)
        finally:
            await audio_q.put(_DONE)

    # ------------------------------
    # 텍스트 프로듀서
    # ------------------------------    
    async def _text_producer() -> None:
        nonlocal ttft, text_chunk_count, t_last_tts_start, full_llm_text

        raw_token_stream = api_manager.generate_streaming_response(prompt)
        # 토큰 인터셉터를 통해 각 토큰을 컨트롤러에 기록한 뒤 청커로 전달
        token_stream = _token_interceptor(raw_token_stream)

        # controller.decide_text_chunk_size를 callable로 전달 — 매 토큰마다 현재 속도 기반으로 재평가
        text_chunks = streaming_controller.llm_text_chunker(
            token_stream,
            max_chars=controller.decide_text_chunk_size,
            use_sentence_end=True,
        )

        async for text_chunk in text_chunks:
            if ttft is None:
                ttft = (time.perf_counter() - t_start) * 1000
                logger.info("[Adaptive] 첫 텍스트 청크 확정 — TTFT=%.1fms, %d자", ttft, len(text_chunk))

            full_llm_text += text_chunk
            text_chunk_count += 1
            t_last_tts_start = time.perf_counter()

            # 텍스트 청크 확정 시점에 컨트롤러 상태 기록
            decided_text_chunk_sizes.append(controller.decide_text_chunk_size())
            # TTS 요청 직전 콜백: tts_last_send_ts, tts_last_chunk_size 갱신
            controller.on_tts_chunk_sent(len(text_chunk))
            # 현재 AdaptiveState 스냅샷 저장 (제어 결정 히스토리)
            control_snapshots.append(controller.snapshot())

            audio_q: asyncio.Queue = asyncio.Queue()
            await pipeline.put(audio_q)
            asyncio.create_task(_tts_worker(text_chunk, audio_q))

        await pipeline.put(_DONE)

    producer = asyncio.create_task(_text_producer())

    # ------------------------------
    # 소비자
    # ------------------------------
        
    # run_streaming()과 동일한 순서 보장 구조이나, 각 청크 전송 후 on_ws_chunk_sent() 호출
    while True:
        item = await pipeline.get()
        if item is _DONE:
            break

        audio_q: asyncio.Queue = item
        while True:
            audio_chunk = await audio_q.get()
            if audio_chunk is _DONE:
                break

            if ttfa is None:
                ttfa = (time.perf_counter() - t_start) * 1000
                logger.info("[Adaptive] 첫 오디오 청크 수신 — TTFA=%.1fms", ttfa)

            # WebSocket 전송 시점 기록: throughput 및 전송 간격 갱신
            send_ts = time.perf_counter()
            controller.on_ws_chunk_sent(len(audio_chunk), send_ts)

            audio_chunk_count += 1
            all_audio.extend(audio_chunk)
            yield audio_chunk, None

    await producer

    t_end = time.perf_counter()
    e2e_latency = (t_end - t_start) * 1000
    tts_latency = (t_end - t_last_tts_start) * 1000

    audio_path = _save_audio(bytes(all_audio), "adaptive")

    avg_text = (
        sum(decided_text_chunk_sizes) / len(decided_text_chunk_sizes)
        if decided_text_chunk_sizes
        else 0.0
    )
    avg_audio = (
        sum(decided_audio_chunk_sizes) / len(decided_audio_chunk_sizes)
        if decided_audio_chunk_sizes
        else 0.0
    )

    logger.info(
        "[Adaptive] 파이프라인 완료 — E2E=%.1fms, 텍스트 청크=%d, 오디오 청크=%d, %d bytes, 저장=%s",
        e2e_latency, text_chunk_count, audio_chunk_count, len(all_audio), audio_path,
    )

    # ------------------------------
    # 성능 지표 객체 생성
    # ------------------------------
    metrics = AdaptiveMetrics(
        llm_latency=ttft if ttft is not None else 0.0,
        tts_latency=tts_latency,
        e2e_latency=e2e_latency,
        ttft=ttft if ttft is not None else 0.0,
        ttfa=ttfa if ttfa is not None else 0.0,
        text_chunk_count=text_chunk_count,
        audio_chunk_count=audio_chunk_count,
        llm_response=full_llm_text,
        audio_path=audio_path,
        control_snapshots=control_snapshots,
        avg_text_chunk_size=avg_text,
        avg_audio_chunk_size=avg_audio,
        starvation_adjustments=0,
    )
    yield b"", metrics
