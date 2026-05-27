# 오디오 청크 버퍼링

import logging
import threading
from collections import deque

logger = logging.getLogger(__name__)

__all__ = ["JitterBuffer"]


# ========================================
# Jitter Buffer
# ========================================

class JitterBuffer:
    # 실험 1: 전체 오디오를 그대로 통과시키는 pass-through 구조
    # 실험 2: feed() / drain()으로 청크 단위 FIFO 버퍼링
    # 실험 3: 네트워크 상태에 따라 버퍼 크기를 동적으로 조절하는 로직 추가 (예정)

    def __init__(self, min_buffer_chunks: int = 3) -> None:
        self._buffer: deque[bytes] = deque()
        self._starvation_count: int = 0
        self._total_bytes_fed: int = 0   # 누적 공급 바이트 수
        self._chunks_fed: int = 0        # 누적 공급 청크 수
        # 실험 3에서 동적 버퍼 크기 결정에 활용할 하한값
        self._min_buffer: int = min_buffer_chunks
        self._lock = threading.Lock()

    # ----------------------------------------
    # 전체 오디오 재생 (실험 1)
    # ----------------------------------------

    def pass_through(self, audio_bytes: bytes) -> bytes:
        # 전체 오디오를 버퍼링 없이 그대로 반환
        logger.debug("패스스루 — %d bytes", len(audio_bytes))
        return audio_bytes

    # ----------------------------------------
    # 스트리밍 오디오 재생 (실험 2·3)
    # ----------------------------------------

    def feed(self, chunk: bytes) -> None:
        # 수신한 청크를 버퍼 끝에 추가하고 크기 통계를 갱신
        with self._lock:
            self._buffer.append(chunk)
            self._total_bytes_fed += len(chunk)
            self._chunks_fed += 1
        logger.debug("청크 추가 — %d bytes (버퍼: %d개)", len(chunk), len(self._buffer))

    def drain(self) -> bytes | None:
        # 버퍼 앞에서 청크 1개를 꺼내 반환. 버퍼가 비어 있으면 starvation 카운트를 증가시키고 None 반환
        with self._lock:
            if self._buffer:
                return self._buffer.popleft()
            self._starvation_count += 1
        logger.debug("버퍼 고갈 — starvation=%d", self._starvation_count)
        return None

    @property
    def starvation_count(self) -> int:
        # 버퍼 고갈 횟수 — metrics_client.record()에 전달
        return self._starvation_count

    @property
    def stutter_duration_ms(self) -> float:
        # 고갈 횟수 × 실제 평균 청크 재생 시간(ms) — feed()로 수집한 실측 크기 기반
        # 청크가 하나도 공급되지 않았거나 고갈이 없으면 0.0 반환
        if self._chunks_fed == 0 or self._starvation_count == 0:
            return 0.0
        avg_chunk_ms = (self._total_bytes_fed / self._chunks_fed / 2) / 24_000 * 1000
        return self._starvation_count * avg_chunk_ms

    def reset(self) -> None:
        # 버퍼·starvation 카운터·크기 통계를 초기화 — 각 요청 사이클 시작 전 호출
        with self._lock:
            self._buffer.clear()
            self._starvation_count = 0
            self._total_bytes_fed = 0
            self._chunks_fed = 0
        logger.debug("JitterBuffer 초기화")

    @property
    def buffered_chunks(self) -> int:
        # 현재 버퍼에 쌓인 청크 수
        with self._lock:
            return len(self._buffer)
