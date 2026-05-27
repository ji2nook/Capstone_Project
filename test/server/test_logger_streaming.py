# 서버 실행 없이 실험 2 지표가 logger.export()에 올바르게 반영되는지 검증한다
# 확인 포인트:
#   1. MetricRecord에 text_chunk_count / audio_chunk_count가 저장되는지
#   2. exp2_server_*.json에 해당 필드가 포함되는지
#   3. exp2_summary_*.json의 streaming_info 섹션이 채워지는지
#   4. ClientMetrics가 서버 지표와 올바르게 병합되는지

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
from server.experiment_pipeline import StreamingMetrics
from server import logger
from server.logger import ClientMetrics


# ----------------------------------------
# 가상의 실험 2 서버 지표를 metrics_server에 입력
# ----------------------------------------
print("[1] 실험 2 서버 지표 입력")

fake_metrics = StreamingMetrics(
    llm_latency=312.5,
    tts_latency=198.4,
    e2e_latency=1850.0,
    ttft=312.5,
    ttfa=510.9,
    text_chunk_count=4,
    audio_chunk_count=7,
)
metrics_server.record(
    fake_metrics,
    experiment="streaming",
    prompt="테스트 프롬프트입니다.",
    llm_response="테스트 LLM 응답입니다.",
    audio_path="results/audio/streaming_test.wav",
)

records = metrics_server.get_records()
print(f"    metrics_server 기록 수: {len(records)}건")

saved = records[0]
assert saved.text_chunk_count == 4, f"text_chunk_count 불일치: {saved.text_chunk_count}"
assert saved.audio_chunk_count == 7, f"audio_chunk_count 불일치: {saved.audio_chunk_count}"
print(f"    text_chunk_count  : {saved.text_chunk_count}  (기대: 4)")
print(f"    audio_chunk_count : {saved.audio_chunk_count}  (기대: 7)")
print("    MetricRecord 저장 검증 통과")


# ----------------------------------------
# 클라이언트 지표와 함께 export
# ----------------------------------------
print("\n[2] 클라이언트 지표 포함하여 export")

client = ClientMetrics(
    ttfa_client=620.0,
    play_duration=2150.0,
    buffer_starvation=2,
    stutter_duration=85.0,
)

output_path = logger.export(
    experiment="streaming",
    client_metrics=[client],
    clear_after=True,
)

print(f"    저장 경로: {output_path}")
assert output_path.name.startswith("exp2_server_"), (
    f"파일명이 exp2_server_로 시작해야 한다: {output_path.name}"
)
print(f"    파일명 검증 통과: {output_path.name}")


# ----------------------------------------
# exp2_server_*.json — 전체 필드 확인
# ----------------------------------------
print("\n[3] exp2_server_*.json 구조 확인")

server_payload = json.loads(output_path.read_text(encoding="utf-8"))
result = server_payload["results"][0]

assert result["text_chunk_count"] == 4
assert result["audio_chunk_count"] == 7
assert result["ttfa_client"] == 620.0
assert result["buffer_starvation"] == 2
assert result["stutter_duration"] == 85.0

print(f"    text_chunk_count  : {result['text_chunk_count']}")
print(f"    audio_chunk_count : {result['audio_chunk_count']}")
print(f"    ttfa_client       : {result['ttfa_client']}ms")
print(f"    buffer_starvation : {result['buffer_starvation']}회")
print(f"    stutter_duration  : {result['stutter_duration']}ms")
print("    전체 필드 검증 통과")


# ----------------------------------------
# exp2_summary_*.json — streaming_info 섹션 확인
# ----------------------------------------
print("\n[4] exp2_summary_*.json streaming_info 섹션 확인")

summary_path = Path(str(output_path).replace("_server_", "_summary_"))
assert summary_path.exists(), f"요약 파일이 존재하지 않는다: {summary_path}"

summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
summary_result = summary_payload["results"][0]

streaming_info = summary_result.get("streaming_info")
assert streaming_info is not None, "streaming_info 섹션이 None이다"
assert streaming_info["text_chunk_count"] == 4
assert streaming_info["audio_chunk_count"] == 7

print(f"    streaming_info: {streaming_info}")
print("    streaming_info 섹션 검증 통과")


# ----------------------------------------
# clear_after 동작 확인
# ----------------------------------------
print("\n[5] clear_after=True 후 metrics_server 상태 확인")

remaining = metrics_server.get_records()
assert len(remaining) == 0, f"기록이 남아있다: {len(remaining)}건"
print(f"    남은 기록 수: {len(remaining)}건  (0이어야 정상)")

print("\n모든 검증 통과")
