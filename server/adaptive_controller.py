# 실험 3 적응형 스트리밍 파이프라인 — 런타임 모니터링 및 청크 크기 제어

import logging
import time
from collections import deque
from dataclasses import asdict, dataclass

from server import experiment_config as cfg
from server.network_probe import NetworkSnapshot

logger = logging.getLogger(__name__)


def _clamp(value: float, lo: float, hi: float) -> float:
    """value를 [lo, hi] 범위로 클리핑."""
    return max(lo, min(hi, value))


# 텍스트 청크 간격 기반 rate 계산 시 chars → tokens 환산 상수 (한국어 기준)
_CHARS_PER_TOKEN_KO: float = 2.5


# ========================================
# TTS 속도 추적기
# ========================================

class _TtsRateTracker:
    """1초 슬라이딩 윈도우로 TTS 오디오 출력 속도(bytes/s)를 계산.

    각 수신 샘플을 (타임스탬프, bytes) 형태로 저장하고,
    WINDOW_S 이전 샘플은 자동으로 제거하여 최근 1초의 평균 속도를 반환한다.
    """

    WINDOW_S: float = 1.0

    def __init__(self) -> None:
        self._samples: deque[tuple[float, int]] = deque()

    def record(self, nbytes: int) -> float:
        """nbytes 수신 기록 후 현재 추정 속도(bytes/s) 반환."""
        now = time.perf_counter()
        self._samples.append((now, nbytes))
        # 윈도우 밖 오래된 샘플 제거
        cutoff = now - self.WINDOW_S
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()
        if len(self._samples) < 2:
            return 0.0
        elapsed = self._samples[-1][0] - self._samples[0][0]
        total_bytes = sum(b for _, b in self._samples)
        return total_bytes / elapsed if elapsed > 0 else 0.0


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
    tts_synthesis_rate: float = 0.0         # 슬라이딩 윈도우 기반 bytes/s (on_tts_audio_received 갱신)
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
    # latency_ms: ping_llm_ms 기반 — 오디오 클라이언트 딜리버리 지연 proxy (AUDIO_WEIGHTS)
    latency_ms: float = 0.0
    # tts_latency_ms: ping_tts_ms 기반 — TTS API 호출 효율성 기준 (TEXT_WEIGHTS)
    tts_latency_ms: float = 0.0
    throughput_bps: float = 0.0        # 최근 전송 처리량 (bytes/s)
    last_send_interval_ms: float = 0.0 # 직전 오디오 청크 전송 간격 (ms)

    # ----------------------------------------
    # 현재 제어 파라미터 (Step 4~7에서 동적 계산값으로 교체)
    # ----------------------------------------
    text_chunk_size: int = cfg.DEFAULT_TEXT_CHUNK_SIZE    # decide_text_chunk_size() 호출 시 갱신
    audio_chunk_size: int = cfg.DEFAULT_AUDIO_CHUNK_SIZE  # decide_audio_chunk_size() 호출 시 갱신



# ========================================
# 적응형 컨트롤러
# ========================================

class AdaptiveController:
    """실험 3 런타임 모니터링 및 청크 크기 제어.

    파이프라인 각 지점에서 on_*() 콜백을 호출하여 AdaptiveState를 갱신하고,
    decide_*() 메서드로 다음 청크 크기를 결정한다.
    """

    def __init__(self, network_snapshot: NetworkSnapshot | None = None) -> None:
        self._state = AdaptiveState()
        self._session_start: float = time.perf_counter()
        self._last_ws_send_ts: float = 0.0
        self._tts_tracker = _TtsRateTracker()

        # 마지막으로 처리한 starvation_count: 새로운 고갈 이벤트 감지
        self._prev_starvation: int = 0
        # 버퍼 고갈로 cfg.MIN_TEXT_CHUNK를 반환한 누적 횟수
        self._starvation_adjustments: int = 0

        # Step 7: 서버 시작 시 측정한 NetworkSnapshot으로 초기값 주입.
        # latency_ms, throughput_bps가 0.0이면 _norm_*()이 중립값(0.5)을 반환하므로
        # 첫 번째 청크 결정이 네트워크 환경을 반영하지 못한다(콜드 스타트 편향).
        # 초기값을 주입하면 세션 시작 직후부터 실제 네트워크 조건이 제어에 반영된다.
        if network_snapshot is not None:
            self._state.latency_ms = network_snapshot.ping_llm_ms or 0.0
            self._state.tts_latency_ms = network_snapshot.ping_tts_ms or 0.0
            # download_mbps(Mbps) → bytes/s 변환: Mbps * 1e6 / 8
            self._state.throughput_bps = (network_snapshot.download_mbps or 0.0) * 1e6 / 8
            logger.debug(
                "AdaptiveController 초기값 주입 — llm_latency=%.1fms, tts_latency=%.1fms, throughput=%.0f bps",
                self._state.latency_ms,
                self._state.tts_latency_ms,
                self._state.throughput_bps,
            )



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
        """TTS 오디오 청크 수신 시 호출. 슬라이딩 윈도우로 합성 속도(bytes/s) 갱신."""
        rate = self._tts_tracker.record(audio_bytes)
        self._state.tts_synthesis_rate = rate
        logger.debug("on_tts_audio_received: %d bytes, rate=%.0f B/s", audio_bytes, rate)

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
    # 정규화 함수 (Step 6)
    # ========================================
    # 각 모니터링 변수를 [0.0, 1.0]으로 정규화.
    # score가 높을수록 "버퍼/네트워크 여유 있음" -> 큰 청크를 의미한다.

    def _norm_llm_rate(self, rate: float) -> float:
        """LLM 토큰 속도 → [0, 1]. 느릴수록 1.0 (큰 청크 선호, 역관계).
        rate=0(초기) → 0.5 중립, rate=100 t/s → 0.0."""
        if rate <= 0.0:
            return 0.5
        return 1.0 - _clamp(rate / 100.0, 0.0, 1.0)

    def _norm_tts_rate(self, rate: float) -> float:
        """TTS 합성 속도 → [0, 1]. 빠를수록 1.0 (큰 청크 가능, 정관계).
        rate=0(초기) → 0.5 중립, rate=500000 B/s → 1.0.
        실측 범위(370K~1.3M B/s)를 반영해 60K에서 상향 조정."""
        if rate <= 0.0:
            return 0.5
        return _clamp(rate / 500_000.0, 0.0, 1.0)

    def _norm_buffer(self, level: int) -> float:
        """버퍼 청크 수 → [0, 1]. 많을수록 1.0 (여유 있음, 정관계).
        level=0 && starvation=0: 초기 상태로 보고 0.5 중립 반환.
        level=0 && starvation>0: 0.0 (실제 고갈 위험)."""
        if level == 0 and self._state.buffer_starvation_count == 0:
            return 0.5
        return _clamp(level / 10.0, 0.0, 1.0)

    def _norm_latency_tts(self, latency_ms: float) -> float:
        """TTS API 지연 → [0, 1]. 높을수록 1.0 (API 호출 배치 효율 선호, 정관계).
        latency가 높을수록 TTS API 호출 횟수를 줄이기 위해 큰 텍스트 청크를 선호한다.
        0ms → 0.0, 200ms 이상 → 1.0."""
        if latency_ms <= 0.0:
            return 0.5
        return _clamp(latency_ms / 200.0, 0.0, 1.0)

    def _norm_latency(self, latency_ms: float) -> float:
        """클라이언트 딜리버리 지연 (proxy) → [0, 1]. 낮을수록 1.0 (역관계).
        latency가 높을수록 청크가 늦게 도착하므로 작은 오디오 청크를 자주 보내야 한다.
        0ms → 1.0, 200ms 이상 → 0.0."""
        if latency_ms <= 0.0:
            return 0.5
        return max(0.0, 1.0 - latency_ms / 200.0)

    def _norm_throughput(self, throughput_bps: float) -> float:
        """처리량 → [0, 1]. 높을수록 1.0 (전송 용량 충분, 정관계).
        throughput=0(초기) → 0.5 중립, 8 MB/s 이상 → 1.0.
        실측 네트워크(80 Mbps = 10 MB/s) 기준으로 100K에서 상향 조정."""
        if throughput_bps <= 0.0:
            return 0.5
        return _clamp(throughput_bps / 8_000_000.0, 0.0, 1.0)



    # ========================================
    # 통합 제어 계산 (Step 6)
    # ========================================

    def _compute_text_chunk_size(self) -> int:
        """가중치 기반 통합 텍스트 청크 크기 계산.

        버퍼 고갈 신규 발생 시 cfg.MIN_TEXT_CHUNK 즉시 반환 (긴급 처리).
        그 외에는 5개 요인을 정규화 후 가중합하여 [MIN, MAX] 선형 매핑한다.
        """
        s = self._state

        # 신규 starvation 이벤트 감지 → 긴급 축소
        if s.buffer_starvation_count > self._prev_starvation:
            self._prev_starvation = s.buffer_starvation_count
            self._starvation_adjustments += 1
            logger.info(
                "_compute_text_chunk_size: 버퍼 고갈 감지 (count=%d), cfg.MIN_TEXT_CHUNK 반환",
                s.buffer_starvation_count,
            )
            return cfg.MIN_TEXT_CHUNK

        score = (
            cfg.TEXT_WEIGHTS["llm_token_rate"]     * self._norm_llm_rate(s.llm_token_rate)      +
            cfg.TEXT_WEIGHTS["tts_synthesis_rate"] * self._norm_tts_rate(s.tts_synthesis_rate)  +
            cfg.TEXT_WEIGHTS["buffer_level"]       * self._norm_buffer(s.buffer_level)          +
            cfg.TEXT_WEIGHTS["latency"]            * self._norm_latency_tts(s.tts_latency_ms)  +
            cfg.TEXT_WEIGHTS["throughput"]         * self._norm_throughput(s.throughput_bps)
        )
        size = cfg.MIN_TEXT_CHUNK + int(score * (cfg.MAX_TEXT_CHUNK - cfg.MIN_TEXT_CHUNK))
        return int(_clamp(size, cfg.MIN_TEXT_CHUNK, cfg.MAX_TEXT_CHUNK))

    def _compute_audio_chunk_size(self) -> int:
        """가중치 기반 통합 오디오 청크 크기 계산.

        현재 버퍼가 비어있는 위험 상태면 cfg.MIN_AUDIO_CHUNK로 긴급 축소.
        그 외에는 TTS 속도(주), 버퍼/네트워크(보조)를 가중합하여 [MIN, MAX] 선형 매핑한다.
        PCM 16-bit(2바이트) 경계로 정렬하여 샘플 잘림을 방지한다.
        """
        s = self._state

        # 현재 버퍼 고갈 위험 상태 -> 긴급 축소 (starvation 이벤트와 독립적으로 실시간 확인)
        if s.buffer_level == 0 and s.buffer_starvation_count > 0:
            logger.info(
                "_compute_audio_chunk_size: 버퍼 고갈 위험 (level=0), cfg.MIN_AUDIO_CHUNK 반환",
            )
            return (cfg.MIN_AUDIO_CHUNK // 2) * 2

        score = (
            cfg.AUDIO_WEIGHTS["tts_synthesis_rate"] * self._norm_tts_rate(s.tts_synthesis_rate) +
            cfg.AUDIO_WEIGHTS["buffer_level"]       * self._norm_buffer(s.buffer_level)        +
            cfg.AUDIO_WEIGHTS["latency"]            * self._norm_latency(s.latency_ms)         +
            cfg.AUDIO_WEIGHTS["throughput"]         * self._norm_throughput(s.throughput_bps)
        )
        dynamic_size = cfg.MIN_AUDIO_CHUNK + int(score * (cfg.MAX_AUDIO_CHUNK - cfg.MIN_AUDIO_CHUNK))
        size = int(_clamp(dynamic_size, cfg.MIN_AUDIO_CHUNK, cfg.MAX_AUDIO_CHUNK))
        return (size // 2) * 2  # PCM 16-bit 경계 정렬



    # ========================================
    # 제어 결정
    # ========================================

    def decide_text_chunk_size(self) -> int:
        """통합 제어 로직으로 텍스트 청크 최대 문자 수를 결정하여 반환."""
        size = self._compute_text_chunk_size()
        self._state.text_chunk_size = size
        logger.debug("decide_text_chunk_size: %d자", size)
        return size

    def decide_audio_chunk_size(self) -> int:
        """통합 제어 로직으로 오디오 청크 최대 bytes를 결정하여 반환."""
        size = self._compute_audio_chunk_size()
        self._state.audio_chunk_size = size
        logger.debug("decide_audio_chunk_size: %d bytes", size)
        return size



    # ========================================
    # 스냅샷 / 집계
    # ========================================

    @property
    def starvation_adjustments(self) -> int:
        """버퍼 고갈로 인해 cfg.MIN_TEXT_CHUNK를 긴급 반환한 누적 횟수."""
        return self._starvation_adjustments

    def snapshot(self) -> dict:
        """현재 AdaptiveState를 dict로 직렬화. 지표 기록 및 제어 히스토리에 사용."""
        return asdict(self._state)
