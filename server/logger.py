# 실험 결과 통합 기록 및 파일 출력

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from server import metrics_server
from server.metrics_server import MetricRecord
from server.network_probe import NetworkSnapshot

# torch/soundfile이 없는 환경(VM)에서도 서버가 기동될 수 있도록 조건부 import
try:
    from server.mos_evaluator import MOSEvaluator
    _MOS_AVAILABLE = True
except ImportError:
    _MOS_AVAILABLE = False

logger = logging.getLogger(__name__)

# 결과 파일이 저장될 디렉토리 (프로젝트 루트 기준)
RESULTS_DIR = Path("results")

# 실험 타입 매핑
_EXPERIMENT_PREFIX: dict[str, str] = {
    "sequential": "exp1",
    "streaming": "exp2",
    "adaptive": "exp3",
}


# ========================================
# 클라이언트 측 성능 지표
# ========================================

@dataclass
class ClientMetrics:
    # client/metrics_client.py에서 측정한 성능 지표를 담는 컨테이너
    # 실험 완료 후 클라이언트가 서버로 전송하면 export()에서 병합
    ttfa_client: float       # 서버로부터 첫 오디오 수신까지 (ms)
    play_duration: float     # 오디오 재생 시작 ~ 재생 완료까지 (ms)
    buffer_starvation: int   # 재생 중 버퍼 고갈 횟수 (실험 1: 항상 0)
    stutter_duration: float  # 음성 끊김 누적 시간 (ms) (실험 1: 항상 0.0)


# ========================================
# 내부 유틸리티
# ========================================

def _build_output_path(experiment: str, label: str, timestamp: str) -> Path:
    # 실험 타입·레이블·타임스탬프를 조합한 결과 파일 경로 생성
    prefix = _EXPERIMENT_PREFIX.get(experiment, experiment)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / f"{prefix}_{label}_{timestamp}.json"


def _build_key_metrics(row: dict) -> dict:
    # 병합된 지표 dict에서 핵심 성능 지표 6개를 산출
    ttfa_client = row["ttfa_client"]
    play_duration = row["play_duration"]
    stutter_duration = row["stutter_duration"]

    e2e_latency_ms = (
        round(ttfa_client + play_duration, 3)
        if ttfa_client is not None and play_duration is not None
        else None
    )
    buffer_starvation_rate_pct = (
        round(stutter_duration / play_duration * 100, 3)
        if play_duration is not None and play_duration > 0 and stutter_duration is not None
        else None
    )
    return {
        "ttft_ms": round(row["ttft"], 3),
        "ttfa_ms": round(ttfa_client, 3) if ttfa_client is not None else None,
        "e2e_latency_ms": e2e_latency_ms,
        "buffer_starvation_count": row["buffer_starvation"],
        "buffer_starvation_rate_pct": buffer_starvation_rate_pct,
        "buffer_starvation_duration_ms": round(stutter_duration, 3) if stutter_duration is not None else None,
    }


def _build_streaming_info(row: dict) -> dict | None:
    # 실험 2·3 전용 청크 정보 섹션 — 실험 1에서는 None 반환
    text_chunk_count = row.get("text_chunk_count")
    audio_chunk_count = row.get("audio_chunk_count")
    if text_chunk_count is None and audio_chunk_count is None:
        return None
    return {
        "text_chunk_count": text_chunk_count,
        "audio_chunk_count": audio_chunk_count,
    }


def _to_summary_row(row: dict) -> dict:
    # 전체 지표 dict에서 핵심 필드만 추출해 요약 레코드 생성
    return {
        "timestamp": row["timestamp"],
        "experiment": row["experiment"],
        "prompt": row["prompt"],
        "llm_response": row["llm_response"],
        "key_metrics": _build_key_metrics(row),
        "streaming_info": _build_streaming_info(row),
        "mos": {
            "utmos": round(row["utmos"], 3) if row["utmos"] is not None else None,
            "nisqa_mos": round(row["nisqa_mos"], 3) if row["nisqa_mos"] is not None else None,
            "nisqa_noisiness": round(row["nisqa_noisiness"], 3) if row["nisqa_noisiness"] is not None else None,
            "nisqa_coloration": round(row["nisqa_coloration"], 3) if row["nisqa_coloration"] is not None else None,
            "nisqa_discontinuity": round(row["nisqa_discontinuity"], 3) if row["nisqa_discontinuity"] is not None else None,
            "nisqa_loudness": round(row["nisqa_loudness"], 3) if row["nisqa_loudness"] is not None else None,
        },
    }


def _merge(record: MetricRecord, client: ClientMetrics | None) -> dict:
    # 서버 측 지표와 클라이언트 측 지표를 하나의 dict로 병합
    # 실험 1에서는 client 없이 호출 가능 (실험 2, 3에서 해당 지표 추가)

    ttfa_client = client.ttfa_client if client is not None else None
    play_duration = client.play_duration if client is not None else None
    buffer_starvation = client.buffer_starvation if client is not None else None
    stutter_duration = client.stutter_duration if client is not None else None

    return {
        "timestamp": record.timestamp,
        "experiment": record.experiment,
        "prompt": record.prompt,
        "llm_response": record.llm_response,
        "llm_latency": record.llm_latency,
        "tts_latency": record.tts_latency,
        "e2e_latency": record.e2e_latency,
        "ttft": record.ttft,
        "ttfa": record.ttfa,
        "ttfa_client": ttfa_client,
        "play_duration": play_duration,
        "buffer_starvation": buffer_starvation,
        "stutter_duration": stutter_duration,
        "audio_path": record.audio_path,
        # 실험 2·3 전용 — 실험 1에서는 None
        "text_chunk_count": record.text_chunk_count,
        "audio_chunk_count": record.audio_chunk_count,
        "utmos": round(record.utmos, 3) if record.utmos is not None else None,
        "nisqa_mos": round(record.nisqa_mos, 3) if record.nisqa_mos is not None else None,
        "nisqa_noisiness": round(record.nisqa_noisiness, 3) if record.nisqa_noisiness is not None else None,
        "nisqa_coloration": round(record.nisqa_coloration, 3) if record.nisqa_coloration is not None else None,
        "nisqa_discontinuity": round(record.nisqa_discontinuity, 3) if record.nisqa_discontinuity is not None else None,
        "nisqa_loudness": round(record.nisqa_loudness, 3) if record.nisqa_loudness is not None else None,
    }


# ========================================
# 공개 인터페이스
# ========================================

def export(
    experiment: str = "sequential",
    client_metrics: list[ClientMetrics] | None = None,
    network_snapshot: NetworkSnapshot | None = None,
    clear_after: bool = True,
) -> Path:
    #metrics_server에 누적된 기록을 JSON 파일로 내보냄
    # 실험 1에서는 client_metrics 없이 호출
    # 실험 2, 3 구현할 때 클라이언트 측 지표도 함께 전달
    # clear_after=True(기본값)이라면 내보낸 뒤 metrics_server 메모리 초기화
    
    records = metrics_server.get_records()

    if not records:
        logger.warning("내보낼 지표가 없습니다. metrics_server가 비어 있습니다.")
        return Path()

    # 클라이언트 지표 리스트를 서버 기록 수에 맞춰 정렬
    # 클라이언트 지표가 적으면 None으로 패딩해 zip이 짧은 쪽에서 끊기는 것을 방지
    clients: list[ClientMetrics | None]
    if client_metrics:
        clients = list(client_metrics)
        if len(clients) < len(records):
            clients += [None] * (len(records) - len(clients))
    else:
        clients = [None] * len(records)

    # MOS 평가 — torch/soundfile이 설치된 환경(로컬)에서만 실행
    # VM에서는 _MOS_AVAILABLE=False이므로 스킵, MOS 필드는 None으로 유지
    if _MOS_AVAILABLE:
        evaluator = MOSEvaluator()
        for rec in records:
            if not rec.audio_path:
                continue
            try:
                mos = evaluator.evaluate(rec.audio_path)
                rec.utmos = mos.utmos
                rec.nisqa_mos = mos.nisqa_mos
                rec.nisqa_noisiness = mos.nisqa_noisiness
                rec.nisqa_coloration = mos.nisqa_coloration
                rec.nisqa_discontinuity = mos.nisqa_discontinuity
                rec.nisqa_loudness = mos.nisqa_loudness
            except Exception:
                logger.warning("MOS 평가 실패 (스킵): %s", rec.audio_path, exc_info=True)
    else:
        logger.info("MOS 패키지 미설치 — MOS 평가 스킵 (tools/evaluate_mos.py로 별도 실행)")

    # 서버 + 클라이언트 지표를 1:1로 병합
    rows = [_merge(rec, cli) for rec, cli in zip(records, clients)]

    # 서버 파일과 요약 파일의 타임스탬프를 동일하게 고정
    ts = datetime.now(tz=timezone.utc)
    ts_str = ts.strftime("%Y%m%d_%H%M%S")
    exported_at = ts.isoformat()

    # ------------------------------
    # 전체 지표 파일 (server)
    # ------------------------------
    network_data = asdict(network_snapshot) if network_snapshot is not None else None

    output_path = _build_output_path(experiment, "server", ts_str)
    output_path.write_text(
        json.dumps(
            {
                "experiment": experiment,
                "exported_at": exported_at,
                "network_environment": network_data,
                "count": len(rows),
                "results": rows,
            },
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    logger.info("전체 결과 저장 완료: %s (%d건)", output_path, len(rows))

    # ------------------------------
    # 핵심 지표 요약 파일 (summary)
    # ------------------------------
    summary_rows = [_to_summary_row(r) for r in rows]
    summary_path = _build_output_path(experiment, "summary", ts_str)
    summary_path.write_text(
        json.dumps(
            {
                "experiment": experiment,
                "exported_at": exported_at,
                "network_environment": network_data,
                "count": len(summary_rows),
                "results": summary_rows,
            },
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    logger.info("요약 결과 저장 완료: %s (%d건)", summary_path, len(summary_rows))

    if clear_after:
        # 이미 파일로 내보냈으므로 메모리 누적 기록을 비움
        metrics_server.clear()

    return output_path