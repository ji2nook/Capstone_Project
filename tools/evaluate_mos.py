"""
로컬 MOS 일괄 평가 스크립트

실험 종료 후 서버에서 내려받은 results/ JSON 파일에 대해
UTMOS + NISQA 점수를 계산하고 결과를 JSON에 병합한다.

Usage:
    # 단일 파일
    python -m tools.evaluate_mos results/exp1_server_20240101_120000.json

    # results/ 디렉토리 전체 (utmos가 null인 파일만 처리)
    python -m tools.evaluate_mos results/
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from server.mos_evaluator import MOSEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _needs_evaluation(path: Path) -> bool:
    """JSON 파일 내 records 중 utmos가 null인 항목이 하나라도 있으면 True."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return any(r.get("utmos") is None for r in data.get("results", []))
    except Exception:
        return False


def _collect_targets(target: Path) -> list[Path]:
    """대상 경로에서 평가가 필요한 JSON 파일 목록을 수집한다."""
    if target.is_file():
        return [target]

    # 디렉토리인 경우 summary 파일은 제외 (server 파일만 처리)
    candidates = sorted(target.glob("*_server_*.json"))
    return [p for p in candidates if _needs_evaluation(p)]


def _round(v: float | None) -> float | None:
    return round(v, 3) if v is not None else None


def _evaluate_file(path: Path, evaluator: MOSEvaluator) -> None:
    """server JSON에 MOS 점수를 채우고, 대응하는 summary JSON도 동기화한다."""
    data = json.loads(path.read_text(encoding="utf-8"))
    records: list[dict] = data.get("results", [])

    pending = [r for r in records if r.get("utmos") is None and r.get("audio_path")]
    if not pending:
        logger.info("평가 대상 없음: %s", path.name)
        return

    logger.info("MOS 평가 시작: %s (%d건)", path.name, len(pending))

    for i, rec in enumerate(pending, 1):
        audio_path = rec["audio_path"]
        logger.info("평가 중 (%d/%d): %s", i, len(pending), audio_path)
        try:
            score = evaluator.evaluate(audio_path)
            rec["utmos"] = _round(score.utmos)
            rec["nisqa_mos"] = _round(score.nisqa_mos)
            rec["nisqa_noisiness"] = _round(score.nisqa_noisiness)
            rec["nisqa_coloration"] = _round(score.nisqa_coloration)
            rec["nisqa_discontinuity"] = _round(score.nisqa_discontinuity)
            rec["nisqa_loudness"] = _round(score.nisqa_loudness)
        except Exception:
            logger.exception("MOS 평가 실패: %s", audio_path)

    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("server 저장 완료: %s", path)

    # summary 파일도 동기화 — server JSON과 타임스탬프가 같은 파일을 찾아 MOS 갱신
    summary_path = Path(str(path).replace("_server_", "_summary_"))
    if not summary_path.exists():
        logger.warning("summary 파일 없음 (스킵): %s", summary_path.name)
        return

    summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
    # timestamp 기준으로 server 레코드와 1:1 매핑
    mos_by_ts = {r["timestamp"]: r for r in records}
    for row in summary_data.get("results", []):
        src = mos_by_ts.get(row["timestamp"])
        if src is None:
            continue
        row["mos"] = {
            "utmos": _round(src.get("utmos")),
            "nisqa_mos": _round(src.get("nisqa_mos")),
            "nisqa_noisiness": _round(src.get("nisqa_noisiness")),
            "nisqa_coloration": _round(src.get("nisqa_coloration")),
            "nisqa_discontinuity": _round(src.get("nisqa_discontinuity")),
            "nisqa_loudness": _round(src.get("nisqa_loudness")),
        }

    summary_path.write_text(json.dumps(summary_data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("summary 저장 완료: %s", summary_path.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="로컬 MOS 일괄 평가")
    parser.add_argument(
        "target",
        type=Path,
        help="평가할 JSON 파일 또는 results/ 디렉토리 경로",
    )
    parser.add_argument(
        "--nisqa-weights",
        type=Path,
        default=Path("weights/nisqa.tar"),
        help="NISQA 가중치 파일 경로 (기본값: weights/nisqa.tar)",
    )
    args = parser.parse_args()

    targets = _collect_targets(args.target)
    if not targets:
        logger.info("평가할 파일이 없습니다.")
        sys.exit(0)

    logger.info("평가 대상 파일: %d개", len(targets))
    evaluator = MOSEvaluator(nisqa_weights_path=args.nisqa_weights)

    for path in targets:
        _evaluate_file(path, evaluator)

    logger.info("전체 MOS 평가 완료")


if __name__ == "__main__":
    main()
