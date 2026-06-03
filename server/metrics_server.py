# 서버 측 성능 지표 수집

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from server.experiment_pipeline import (
    BaseMetrics,
    SequentialMetrics,
    StreamingMetrics,
    AdaptiveMetrics,
)

logger = logging.getLogger(__name__)


# ========================================
# 저장 레코드 데이터 클래스
# ========================================

@dataclass
class MetricRecord:
    # 단일 파이프라인 실행 시의 기록 시각 및 성능 지표
    # logger.py에서 이 객체를 JSON으로 직렬화
    timestamp: str           # 기록 시각
    experiment: str          # 실험 파이프라인 식별자 ("sequential" / "streaming" / "adaptive")
    prompt: str              # 사용자 입력 프롬프트
    llm_response: str        # LLM이 생성한 텍스트 응답
    llm_latency: float       # LLM API 소요 시간 (ms)
    tts_latency: float       # TTS API 소요 시간 (ms)
    e2e_latency: float       # 서버 측 E2E Latency (ms)
    ttft: float              # Time To First Token (ms)
    ttfa: float              # Time To First Audio (ms)
    audio_path: str          # 서버 측 TTS WAV 경로 (음질 평가용)
    as_heard_path: str = "" # 클라이언트 루프백 as-heard WAV 경로 (체감 품질 평가용)
    # 실험 2·3 전용 (실험 1에서는 None)
    text_chunk_count: int | None = None      # llm_text_chunker가 생성한 텍스트 청크 수
    audio_chunk_count: int | None = None     # tts_audio_chunker가 생성한 오디오 청크 수
    # 실험 3 전용 (실험 1·2에서는 None)
    avg_text_chunk_size: float | None = None         # 세션 평균 텍스트 청크 크기 (자)
    avg_audio_chunk_size: float | None = None        # 세션 평균 오디오 청크 크기 (bytes)
    starvation_adjustments: int | None = None        # 버퍼 고갈로 인한 긴급 축소 횟수
    control_snapshots: list[dict] | None = None      # 텍스트 청크 확정 시점별 AdaptiveState 스냅샷
    # MOS 평가 결과 (오프라인 평가 시 채워짐)
    utmos: float | None = None               # UTMOS22: TTS 음성 자연스러움 (1~5)
    nisqa_mos: float | None = None           # NISQA: 종합 음질 점수 (1~5)
    nisqa_noisiness: float | None = None     # NISQA NOI: 배경 잡음·왜곡 정도 (1~5, 높을수록 깨끗)
    nisqa_coloration: float | None = None    # NISQA COL: 주파수 특성 열화 (1~5, 높을수록 원음에 가까움)
    nisqa_discontinuity: float | None = None # NISQA DIS: 음성 단절·드롭아웃 (1~5, 높을수록 끊김 없음)
    nisqa_loudness: float | None = None      # NISQA LOUD: 음량 일관성 (1~5, 높을수록 일정한 음량)


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
def record(
    metrics: BaseMetrics,
    experiment: str = "sequential",
    prompt: str = "",
    llm_response: str = "",
    audio_path: str = "",
) -> None:
    # 실험 타입별로 ttft·ttfa 및 전용 필드를 isinstance로 명시적 추출
    text_chunk_count: int | None = None
    audio_chunk_count: int | None = None
    avg_text_chunk_size: float | None = None
    avg_audio_chunk_size: float | None = None
    starvation_adjustments: int | None = None
    control_snapshots: list[dict] | None = None

    if isinstance(metrics, SequentialMetrics):
        ttft = metrics.ttft
        ttfa = metrics.ttfa
    elif isinstance(metrics, StreamingMetrics):
        ttft = metrics.ttft
        ttfa = metrics.ttfa
        text_chunk_count = metrics.text_chunk_count
        audio_chunk_count = metrics.audio_chunk_count
    elif isinstance(metrics, AdaptiveMetrics):
        ttft = metrics.ttft
        ttfa = metrics.ttfa
        text_chunk_count = metrics.text_chunk_count
        audio_chunk_count = metrics.audio_chunk_count
        avg_text_chunk_size = metrics.avg_text_chunk_size
        avg_audio_chunk_size = metrics.avg_audio_chunk_size
        starvation_adjustments = metrics.starvation_adjustments
        control_snapshots = metrics.control_snapshots if metrics.control_snapshots else None
    else:
        ttft = 0.0
        ttfa = 0.0

    entry = MetricRecord(
        timestamp=datetime.now(tz=timezone.utc).isoformat(),
        experiment=experiment,
        prompt=prompt,
        llm_response=llm_response,
        llm_latency=metrics.llm_latency,
        tts_latency=metrics.tts_latency,
        e2e_latency=metrics.e2e_latency,
        ttft=ttft,
        ttfa=ttfa,
        audio_path=audio_path,
        text_chunk_count=text_chunk_count,
        audio_chunk_count=audio_chunk_count,
        avg_text_chunk_size=avg_text_chunk_size,
        avg_audio_chunk_size=avg_audio_chunk_size,
        starvation_adjustments=starvation_adjustments,
        control_snapshots=control_snapshots,
    )

    _records.append(entry)

    chunk_info = (
        f" | 텍스트 청크={text_chunk_count} 오디오 청크={audio_chunk_count}"
        if text_chunk_count is not None else ""
    )
    adaptive_info = (
        f" | 평균 텍스트 청크={avg_text_chunk_size:.1f}자"
        f" 평균 오디오 청크={avg_audio_chunk_size:.0f}bytes"
        f" 긴급 축소={starvation_adjustments}회"
        if avg_text_chunk_size is not None else ""
    )
    logger.info(
        "지표 기록 [%s] LLM=%.1fms TTS=%.1fms E2E=%.1fms TTFT=%.1fms TTFA=%.1fms%s%s",
        experiment,
        entry.llm_latency,
        entry.tts_latency,
        entry.e2e_latency,
        entry.ttft,
        entry.ttfa,
        chunk_info,
        adaptive_info,
    )


# 기록된 모든 MetricRecord 복사본을 반환 (원본 기록 노출 X)
def get_records() -> list[MetricRecord]:
    return list(_records)


# 메모리에 쌓인 모든 기록 초기화
def clear() -> None:
    _records.clear()
    logger.debug("성능 지표 초기화 완료")


def set_last_as_heard_path(path: str) -> None:
    """가장 최근 레코드의 as_heard_path를 갱신한다."""
    if _records:
        _records[-1].as_heard_path = path


# 앞에서 n개만 제거 — export() 완료 후 스냅샷에 포함된 레코드만 삭제할 때 사용
# clear()를 쓰면 export() 실행 중에 추가된 다음 실험의 레코드까지 삭제되는 문제를 방지
def clear_first(n: int) -> None:
    del _records[:n]
    logger.debug("성능 지표 앞 %d건 제거 완료 (잔여 %d건)", n, len(_records))
