# 실험 3 적응형 스트리밍 파이프라인 — 런타임 모니터링 및 청크 크기 제어

import logging
import time
from dataclasses import asdict, dataclass

logger = logging.getLogger(__name__)



# ========================================
# 기본 제어 파라미터
# ========================================

# 실험 2 고정값과 동일하게 시작
DEFAULT_TEXT_CHUNK_SIZE: int = 20
DEFAULT_AUDIO_CHUNK_SIZE: int = 8192
MIN_TEXT_CHUNK: int = 10
MAX_TEXT_CHUNK: int = 120
MIN_AUDIO_CHUNK: int = 4096
MAX_AUDIO_CHUNK: int = 32768


def _clamp(value: float, lo: float, hi: float) -> float:
    """value를 [lo, hi] 범위로 클리핑."""
    return max(lo, min(hi, value))


# 텍스트 청크 간격 기반 rate 계산 시 chars → tokens 환산 상수 (한국어 기준)
_CHARS_PER_TOKEN_KO: float = 2.5


# ========================================
# 모니터링 변수 상태
# ========================================

@dataclass
class AdaptiveState:
    # ----------------------------------------
    # LLM
    # ----------------------------------------
    llm_token_rate: float = 0.0           # 텍스트 청크 간격 기반 추정 tokens/s (on_tts_chunk_sent 갱신)
    llm_tokens_generated: int = 0      # 누적 수신 토큰 수
    llm_last_token_ts: float = 0.0     # 직전 토큰 수신 시각 (perf_counter)

    # ----------------------------------------
    # TTS
    # ----------------------------------------
    # tts_synthesis_rate: 슬라이딩 윈도우 계산은 Step 5에서 구현, 여기서는 0.0 유지
    tts_synthesis_rate: float = 0.0
    tts_last_send_ts: float = 0.0      # 직전 TTS 요청 전송 시각
    tts_last_chunk_size: int = 0       # 직전 TTS 요청 텍스트 문자 수

    # ----------------------------------------
    # 버퍼 (클라이언트 역수신)
    # ----------------------------------------
    buffer_level: int = 0              # JitterBuffer 현재 청크 수
    buffer_starvation_count: int = 0   # 누적 버퍼 고갈 횟수

    # ----------------------------------------
    # 네트워크
    # ----------------------------------------
    # latency_ms: NetworkSnapshot 초기값 주입은 Step 7에서 구현
    latency_ms: float = 0.0
    throughput_bps: float = 0.0        # 최근 전송 처리량 (bytes/s)
    last_send_interval_ms: float = 0.0 # 직전 오디오 청크 전송 간격 (ms)

    # ----------------------------------------
    # 현재 제어 파라미터 (Step 4~7에서 동적 계산값으로 교체)
    # ----------------------------------------
    text_chunk_size: int = DEFAULT_TEXT_CHUNK_SIZE    # decide_text_chunk_size() 호출 시 갱신
    audio_chunk_size: int = DEFAULT_AUDIO_CHUNK_SIZE  # Step 5에서 동적 계산으로 교체



# ========================================
# 적응형 컨트롤러
# ========================================

class AdaptiveController:
    """실험 3 런타임 모니터링 및 청크 크기 제어.

    파이프라인 각 지점에서 on_*() 콜백을 호출하여 AdaptiveState를 갱신하고,
    decide_*() 메서드로 다음 청크 크기를 결정한다.
    Step 1에서는 decide_*()가 고정값을 반환하며, Step 4~7에서 실제 제어 로직으로 교체된다.
    """

    def __init__(self) -> None:
        self._state = AdaptiveState()
        self._session_start: float = time.perf_counter()
        self._last_ws_send_ts: float = 0.0



    # ========================================
    # 상태 업데이트 콜백
    # ========================================

    def on_llm_token(self, token: str) -> None:
        """LLM 토큰 1개 수신 시 호출. 누적 카운트 및 수신 시각 갱신.
        rate는 on_tts_chunk_sent()에서 청크 확정 간격 기반으로 계산된다."""
        self._state.llm_tokens_generated += 1
        self._state.llm_last_token_ts = time.perf_counter()
        logger.debug("on_llm_token: 누적=%d", self._state.llm_tokens_generated)

    def on_tts_chunk_sent(self, text_len: int) -> None:
        """TTS API 요청 직전 호출. 청크 확정 간격으로 LLM 문자 생성 속도를 추정한다.

        asyncio 배치 처리 영향을 피하기 위해 토큰 단위가 아닌 텍스트 청크 단위로 측정한다.
        첫 번째 청크는 이전 간격이 없으므로 rate = 0.0 유지 (DEFAULT 청크 크기 사용).
        """
        now = time.perf_counter()
        if self._state.tts_last_send_ts > 0:
            interval_s = now - self._state.tts_last_send_ts
            if interval_s > 0:
                chars_per_s = text_len / interval_s
                self._state.llm_token_rate = chars_per_s / _CHARS_PER_TOKEN_KO
        self._state.tts_last_send_ts = now
        self._state.tts_last_chunk_size = text_len
        logger.debug(
            "on_tts_chunk_sent: %d자, rate=%.1f t/s",
            text_len,
            self._state.llm_token_rate,
        )

    def on_tts_audio_received(self, audio_bytes: int) -> None:
        """TTS 오디오 청크 수신 시 호출. 합성 속도 계산은 Step 5에서 구현."""
        logger.debug("on_tts_audio_received: %d bytes", audio_bytes)

    def on_ws_chunk_sent(self, chunk_bytes: int, send_ts: float) -> None:
        """WebSocket 청크 전송 후 호출. 전송 간격 및 처리량 갱신."""
        if self._last_ws_send_ts > 0:
            interval_ms = (send_ts - self._last_ws_send_ts) * 1000
            self._state.last_send_interval_ms = interval_ms
            if interval_ms > 0:
                self._state.throughput_bps = chunk_bytes / (interval_ms / 1000)
        self._last_ws_send_ts = send_ts
        logger.debug(
            "on_ws_chunk_sent: %d bytes, 간격=%.1fms, throughput=%.0f bps",
            chunk_bytes,
            self._state.last_send_interval_ms,
            self._state.throughput_bps,
        )

    def on_client_feedback(self, buffer_level: int, starvation_count: int) -> None:
        """클라이언트 버퍼 피드백 수신 후 호출. buffer_level 및 starvation_count 갱신."""
        self._state.buffer_level = buffer_level
        self._state.buffer_starvation_count = starvation_count
        logger.debug(
            "on_client_feedback: buffer=%d, starvation=%d",
            buffer_level,
            starvation_count,
        )



    # ========================================
    # 제어 결정 (Step 1: 고정값 반환)
    # ========================================

    def decide_text_chunk_size(self) -> int:
        """LLM 토큰 속도(tokens/s) 기반으로 텍스트 청크 최대 문자 수를 동적 계산.

        토큰이 빠를수록 작은 청크로 자주 전송해 TTS 응답을 앞당기고,
        느릴수록 큰 청크로 TTS 큐 버퍼를 확보한다.
        rate_factor = clamp(rate / 30.0, 0.5, 2.0) — 30 t/s를 기준(factor=1.0)으로 정규화.
        """
        rate = self._state.llm_token_rate
        if rate <= 0.0:
            # 초기 단계(토큰 2개 미만)는 기본값 유지
            size = DEFAULT_TEXT_CHUNK_SIZE
        else:
            rate_factor = _clamp(rate / 30.0, 0.5, 2.0)
            dynamic_size = DEFAULT_TEXT_CHUNK_SIZE / rate_factor
            size = int(_clamp(dynamic_size, MIN_TEXT_CHUNK, MAX_TEXT_CHUNK))
        self._state.text_chunk_size = size
        return size

    def decide_audio_chunk_size(self) -> int:
        """현재 상태 기반 오디오 청크 최대 bytes 반환.
        Step 5에서 TTS 합성 속도 기반 동적 계산으로 교체된다."""
        return self._state.audio_chunk_size



    # ========================================
    # 스냅샷
    # ========================================

    def snapshot(self) -> dict:
        """현재 AdaptiveState를 dict로 직렬화. 지표 기록 및 제어 히스토리에 사용."""
        return asdict(self._state)
