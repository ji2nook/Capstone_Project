# 클라이언트 측 성능 지표 수집

import logging

from client.input_manager import SessionResult
from server.logger import ClientMetrics

logger = logging.getLogger(__name__)

__all__ = ["MetricsClient"]


# ========================================
# 클라이언트 지표 수집
# ========================================

class MetricsClient:
    # 클라이언트 측 성능 지표 수집 및 보관

    def __init__(self) -> None:
        self._records: list[ClientMetrics] = []

    def record(
        self,
        result: SessionResult,
        play_duration: float,
        buffer_starvation: int = 0,      # 실험 2·3: jitter_buffer가 측정한 값을 전달
        stutter_duration: float = 0.0,   # 실험 2·3: audio_player가 측정한 값을 전달
    ) -> ClientMetrics:

        # 한 사이클의 지표를 계산하여 저장
        ttfa_client = (result.t_recv - result.t_send) * 1000

        entry = ClientMetrics(
            ttfa_client=ttfa_client,
            play_duration=play_duration,
            buffer_starvation=buffer_starvation,
            stutter_duration=stutter_duration,
        )
        self._records.append(entry)

        logger.info(
            "클라이언트 지표 기록 — TTFA: %.1fms | 재생: %.1fms | 버퍼 고갈: %d회 | 끊김: %.1fms",
            ttfa_client,
            play_duration,
            buffer_starvation,
            stutter_duration,
        )
        return entry

    def get_records(self) -> list[ClientMetrics]:
        # 저장된 모든 지표의 복사본 반환
        return list(self._records)

    def clear(self) -> None:
        # 저장된 지표 초기화
        self._records.clear()
        logger.debug("클라이언트 지표 초기화")
