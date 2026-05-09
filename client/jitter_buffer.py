# 오디오 청크 버퍼링

import logging
from collections import deque

logger = logging.getLogger(__name__)

__all__ = ["JitterBuffer"]


# ========================================
# Jitter Buffer
# ========================================

class JitterBuffer:
    # 실험 1: 전체 오디오를 그대로 통과시키는 Path-through 구조
    # 실험 2: feed() / drain()으로 청크 단위 버퍼링 구현 (예정)
    # 실험 3: 네트워크 상태에 따라 버퍼 크기를 동적으로 조절하는 로직 추가 (예정)

    def __init__(self) -> None:
        self._buffer: deque[bytes] = deque()  # 실험 2, 3에서 청크 누적에 사용
        self._starvation_count: int = 0       # 실험 2, 3에서 버퍼 고갈 횟수 집계

    # ----------------------------------------
    # 전체 오디오 재생 (실험 1)
    # ----------------------------------------

    def pass_through(self, audio_bytes: bytes) -> bytes:
        # 전체 오디오를 버퍼링 없이 그대로 반환
        logger.debug("패스스루 — %d bytes", len(audio_bytes))
        return audio_bytes


    # ----------------------------------------
    # 스트리밍 오디오 재생 (실험 2·3 추가 예정)
    # ----------------------------------------

    def feed(self, chunk: bytes) -> None:
        # 수신한 청크를 버퍼에 추가
        raise NotImplementedError("실험 2에서 구현 예정")

    def drain(self) -> bytes:
        # 버퍼에 쌓인 청크를 합쳐 반환하고 버퍼 비움
        raise NotImplementedError("실험 2에서 구현 예정")

    @property
    def starvation_count(self) -> int:
        # 버퍼 고갈 횟수 (metrics_client.record()에 전달)
        raise NotImplementedError("실험 2에서 구현 예정")
