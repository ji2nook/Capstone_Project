# 서버 측 성능 지표 수집

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from server.experiment_pipeline import BaseMetrics

logger = logging.getLogger(__name__)


# ========================================
# 저장 레코드 데이터 클래스
# ========================================

@dataclass
class MetricRecord:
    # 단일 파이프라인 실행 시의 기록 시각 및 성능 지표
    # logger.py에서 이 객체를 JSON/CSV로 직렬화
    timestamp: str      # 기록 시각
    experiment: str     # 실험 파이프라인 식별자 ("sequential" / "streaming" / "adaptive")
    llm_latency: float  # LLM API 소요 시간 (ms)
    tts_latency: float  # TTS API 소요 시간 (ms)
    e2e_latency: float  # 서버 측 E2E Latency (ms)
    ttft: float         # Time To First Token (ms)
    ttfa: float         # Time To First Audio (ms)


# ========================================
# 모듈 수준 in-memory 저장소
# ========================================

# 프로세스가 살아있는 동안 모든 실행 기록을 누적
# logger.py가 서버 종료 시점 또는 주기적인 시점에 이 목록을 파일로 내보냄
_records: list[MetricRecord] = []


# ========================================
# 공개 인터페이스
# ========================================

# 메모리에 성능 지표 추가
def record(metrics: BaseMetrics, experiment: str = "sequential") -> None:
    entry = MetricRecord(
        timestamp=datetime.now(tz=timezone.utc).isoformat(),
        experiment=experiment,
        llm_latency=metrics.llm_latency,
        tts_latency=metrics.tts_latency,
        e2e_latency=metrics.e2e_latency,
        ttft=metrics.ttft,
        ttfa=metrics.ttfa,
    )

    _records.append(entry)

    logger.info(
        "지표 기록 [%s] LLM=%.1fms TTS=%.1fms E2E=%.1fms TTFT=%.1fms TTFA=%.1fms",
        experiment,
        entry.llm_latency,
        entry.tts_latency,
        entry.e2e_latency,
        entry.ttft,
        entry.ttfa,
    )


# 기록된 모든 MetricRecord 복사본을 반환 (원본 기록 노출 X)
def get_records() -> list[MetricRecord]:
    return list(_records)


# 메모리에 쌓인 모든 기록 초기화
def clear() -> None:
    _records.clear()
    logger.debug("성능 지표 초기화 완료")
