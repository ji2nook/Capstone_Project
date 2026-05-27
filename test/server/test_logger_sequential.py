# 서버 실행 없이 logger.export()의 동작을 검증한다

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

from server import metrics_server
from server.experiment_pipeline import SequentialMetrics
from server import logger
from server.logger import ClientMetrics


# ----------------------------------------
# 가상의 서버 지표를 metrics_server에 직접 입력
# ----------------------------------------
print("[1] 서버 지표 입력")

fake_metrics = SequentialMetrics(
    llm_latency=1889.6,
    tts_latency=614.0,
    e2e_latency=2503.6,
    ttft=1889.6,
    ttfa=2503.6,
)
metrics_server.record(fake_metrics, experiment="sequential")

records = metrics_server.get_records()
print(f"    metrics_server 기록 수: {len(records)}건")


# ----------------------------------------
# 클라이언트 지표 없이 export (실험 1 기본 시나리오)
# ----------------------------------------
print("\n[2] 클라이언트 지표 없이 export")

output_path = logger.export(experiment="sequential", clear_after=False)

print(f"    저장 경로: {output_path}")
print(f"    파일 존재 여부: {output_path.exists()}")

payload = json.loads(output_path.read_text(encoding="utf-8"))
print(f"    JSON 구조 확인:")
print(json.dumps(payload, indent=4, ensure_ascii=False))


# ----------------------------------------
# 클라이언트 지표 포함하여 export
# ----------------------------------------
print("\n[3] 클라이언트 지표 포함하여 export")

client = ClientMetrics(
    ttfa_client=2550.3,
    play_duration=1950.0,
    buffer_starvation=0,
    stutter_duration=0.0,
)

output_path2 = logger.export(
    experiment="sequential",
    client_metrics=[client],
    clear_after=True,
)

print(f"    저장 경로: {output_path2}")
payload2 = json.loads(output_path2.read_text(encoding="utf-8"))
print(f"    JSON 구조 확인:")
print(json.dumps(payload2, indent=4, ensure_ascii=False))


# ----------------------------------------
# clear_after 동작 확인
# ----------------------------------------
print("\n[4] clear_after=True 후 metrics_server 상태 확인")
remaining = metrics_server.get_records()
print(f"    남은 기록 수: {len(remaining)}건  (0이어야 정상)")