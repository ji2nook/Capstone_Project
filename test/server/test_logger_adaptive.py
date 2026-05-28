# 서버 실행 없이 실험 3 지표가 metrics_server와 logger.export()에 올바르게 반영되는지 검증한다
# 확인 포인트:
#   1. MetricRecord에 적응형 전용 4개 필드가 저장되는지
#   2. text_chunk_count / audio_chunk_count도 정상 저장되는지
#   3. control_snapshots가 빈 리스트일 때 None으로 정규화되는지
#   4. 실험 2 MetricRecord에서 적응형 필드가 None으로 격리되는지
#   5. exp3_server_*.json에 avg_text_chunk_size 등 적응형 필드가 포함되는지
#   6. exp3_summary_*.json의 adaptive_control_info 섹션이 채워지는지
#   7. exp3_control_history_*.json이 생성되고 snapshots가 담겨 있는지
#   8. clear_after=True 후 metrics_server가 초기화되는지

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
from server.experiment_pipeline import AdaptiveMetrics, StreamingMetrics
from server import logger
from server.logger import ClientMetrics


# ----------------------------------------
# 가상의 실험 3 서버 지표를 metrics_server에 입력
# ----------------------------------------
print("[1] 실험 3 서버 지표 입력")

fake_snapshots = [
    {"text_chunk_size": 15, "audio_chunk_size": 8192, "llm_token_rate": 42.3, "buffer_level": 3},
    {"text_chunk_size": 20, "audio_chunk_size": 9216, "llm_token_rate": 35.1, "buffer_level": 5},
    {"text_chunk_size": 25, "audio_chunk_size": 10240, "llm_token_rate": 28.7, "buffer_level": 7},
]

fake_metrics = AdaptiveMetrics(
    llm_latency=312.5,
    tts_latency=198.4,
    e2e_latency=1850.0,
    ttft=312.5,
    ttfa=510.9,
    text_chunk_count=3,
    audio_chunk_count=9,
    llm_response="테스트 LLM 응답입니다.",
    audio_path="results/audio/adaptive_test.wav",
    control_snapshots=fake_snapshots,
    avg_text_chunk_size=20.0,
    avg_audio_chunk_size=9216.0,
    starvation_adjustments=1,
)

metrics_server.record(
    fake_metrics,
    experiment="adaptive",
    prompt="테스트 프롬프트입니다.",
    llm_response="테스트 LLM 응답입니다.",
    audio_path="results/audio/adaptive_test.wav",
)

records = metrics_server.get_records()
print(f"    metrics_server 기록 수: {len(records)}건")

saved = records[0]

assert saved.text_chunk_count == 3, f"text_chunk_count 불일치: {saved.text_chunk_count}"
assert saved.audio_chunk_count == 9, f"audio_chunk_count 불일치: {saved.audio_chunk_count}"
print(f"    text_chunk_count  : {saved.text_chunk_count}  (기대: 3)")
print(f"    audio_chunk_count : {saved.audio_chunk_count}  (기대: 9)")
print("    공통 청크 필드 저장 검증 통과")


# ----------------------------------------
# 적응형 전용 4개 필드 검증
# ----------------------------------------
print("\n[2] 적응형 전용 필드 검증")

assert saved.avg_text_chunk_size == 20.0, f"avg_text_chunk_size 불일치: {saved.avg_text_chunk_size}"
assert saved.avg_audio_chunk_size == 9216.0, f"avg_audio_chunk_size 불일치: {saved.avg_audio_chunk_size}"
assert saved.starvation_adjustments == 1, f"starvation_adjustments 불일치: {saved.starvation_adjustments}"
assert saved.control_snapshots == fake_snapshots, "control_snapshots 불일치"

print(f"    avg_text_chunk_size   : {saved.avg_text_chunk_size}자  (기대: 20.0)")
print(f"    avg_audio_chunk_size  : {saved.avg_audio_chunk_size}bytes  (기대: 9216.0)")
print(f"    starvation_adjustments: {saved.starvation_adjustments}회  (기대: 1)")
print(f"    control_snapshots 수  : {len(saved.control_snapshots)}개  (기대: 3)")
print("    적응형 전용 필드 검증 통과")


# ----------------------------------------
# control_snapshots 빈 리스트 → None 정규화 검증
# ----------------------------------------
print("\n[3] control_snapshots 빈 리스트 → None 정규화 검증")

metrics_server.clear()

empty_metrics = AdaptiveMetrics(
    llm_latency=100.0,
    tts_latency=200.0,
    e2e_latency=300.0,
    ttft=100.0,
    ttfa=300.0,
    text_chunk_count=1,
    audio_chunk_count=2,
    control_snapshots=[],
)
metrics_server.record(empty_metrics, experiment="adaptive")
saved_empty = metrics_server.get_records()[0]

assert saved_empty.control_snapshots is None, (
    f"빈 control_snapshots는 None으로 정규화되어야 한다: {saved_empty.control_snapshots}"
)
print(f"    control_snapshots (빈 리스트 입력): {saved_empty.control_snapshots}  (기대: None)")
print("    빈 리스트 → None 정규화 검증 통과")

metrics_server.clear()


# ----------------------------------------
# 실험 2 MetricRecord에서 적응형 필드 격리 검증
# ----------------------------------------
print("\n[4] 실험 2 MetricRecord에서 적응형 필드 격리 검증")

streaming_metrics = StreamingMetrics(
    llm_latency=200.0,
    tts_latency=150.0,
    e2e_latency=800.0,
    ttft=200.0,
    ttfa=350.0,
    text_chunk_count=4,
    audio_chunk_count=8,
)
metrics_server.record(streaming_metrics, experiment="streaming")
saved_streaming = metrics_server.get_records()[0]

assert saved_streaming.avg_text_chunk_size is None, (
    f"실험 2 avg_text_chunk_size는 None이어야 한다: {saved_streaming.avg_text_chunk_size}"
)
assert saved_streaming.avg_audio_chunk_size is None, (
    f"실험 2 avg_audio_chunk_size는 None이어야 한다: {saved_streaming.avg_audio_chunk_size}"
)
assert saved_streaming.starvation_adjustments is None, (
    f"실험 2 starvation_adjustments는 None이어야 한다: {saved_streaming.starvation_adjustments}"
)
assert saved_streaming.control_snapshots is None, (
    f"실험 2 control_snapshots는 None이어야 한다: {saved_streaming.control_snapshots}"
)
print(f"    avg_text_chunk_size   : {saved_streaming.avg_text_chunk_size}  (기대: None)")
print(f"    avg_audio_chunk_size  : {saved_streaming.avg_audio_chunk_size}  (기대: None)")
print(f"    starvation_adjustments: {saved_streaming.starvation_adjustments}  (기대: None)")
print(f"    control_snapshots     : {saved_streaming.control_snapshots}  (기대: None)")
print("    실험 2 필드 격리 검증 통과")

metrics_server.clear()


# ----------------------------------------
# 클라이언트 지표와 함께 export
# ----------------------------------------
print("\n[5] 클라이언트 지표 포함하여 export")

metrics_server.record(
    fake_metrics,
    experiment="adaptive",
    prompt="테스트 프롬프트입니다.",
    llm_response="테스트 LLM 응답입니다.",
    audio_path="results/audio/adaptive_test.wav",
)

client = ClientMetrics(
    ttfa_client=620.0,
    play_duration=2150.0,
    buffer_starvation=2,
    stutter_duration=85.0,
)

output_path = logger.export(
    experiment="adaptive",
    client_metrics=[client],
    clear_after=True,
)

print(f"    저장 경로: {output_path}")
assert output_path.name.startswith("exp3_server_"), (
    f"파일명이 exp3_server_로 시작해야 한다: {output_path.name}"
)
print(f"    파일명 검증 통과: {output_path.name}")


# ----------------------------------------
# exp3_server_*.json — 적응형 전용 필드 확인
# ----------------------------------------
print("\n[6] exp3_server_*.json 구조 확인")

server_payload = json.loads(output_path.read_text(encoding="utf-8"))
result = server_payload["results"][0]

assert result["text_chunk_count"] == 3
assert result["audio_chunk_count"] == 9
assert result["avg_text_chunk_size"] == 20.0
assert result["avg_audio_chunk_size"] == 9216.0
assert result["starvation_adjustments"] == 1
assert result["ttfa_client"] == 620.0
assert result["buffer_starvation"] == 2

print(f"    text_chunk_count      : {result['text_chunk_count']}")
print(f"    audio_chunk_count     : {result['audio_chunk_count']}")
print(f"    avg_text_chunk_size   : {result['avg_text_chunk_size']}자")
print(f"    avg_audio_chunk_size  : {result['avg_audio_chunk_size']}bytes")
print(f"    starvation_adjustments: {result['starvation_adjustments']}회")
print(f"    ttfa_client           : {result['ttfa_client']}ms")
print(f"    buffer_starvation     : {result['buffer_starvation']}회")
print("    exp3_server_*.json 전체 필드 검증 통과")


# ----------------------------------------
# exp3_summary_*.json — adaptive_control_info 섹션 확인
# ----------------------------------------
print("\n[7] exp3_summary_*.json adaptive_control_info 섹션 확인")

summary_path = Path(str(output_path).replace("_server_", "_summary_"))
assert summary_path.exists(), f"요약 파일이 존재하지 않는다: {summary_path}"

summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
summary_result = summary_payload["results"][0]

streaming_info = summary_result.get("streaming_info")
assert streaming_info is not None, "streaming_info 섹션이 None이다"
assert streaming_info["text_chunk_count"] == 3
assert streaming_info["audio_chunk_count"] == 9
print(f"    streaming_info: {streaming_info}")

adaptive_info = summary_result.get("adaptive_control_info")
assert adaptive_info is not None, "adaptive_control_info 섹션이 None이다"
assert adaptive_info["avg_text_chunk_size"] == 20.0
assert adaptive_info["avg_audio_chunk_size"] == 9216.0
assert adaptive_info["starvation_adjustments"] == 1
print(f"    adaptive_control_info: {adaptive_info}")
print("    adaptive_control_info 섹션 검증 통과")


# ----------------------------------------
# exp3_control_history_*.json — 별도 파일 생성 및 snapshots 확인
# ----------------------------------------
print("\n[8] exp3_control_history_*.json 생성 및 snapshots 확인")

history_path = Path(str(output_path).replace("_server_", "_control_history_"))
assert history_path.exists(), f"제어 히스토리 파일이 존재하지 않는다: {history_path}"

history_payload = json.loads(history_path.read_text(encoding="utf-8"))
history_result = history_payload["results"][0]

assert len(history_result["snapshots"]) == 3, (
    f"snapshots 수가 3이어야 한다: {len(history_result['snapshots'])}"
)
assert history_result["snapshots"][0]["text_chunk_size"] == 15
assert history_result["snapshots"][1]["text_chunk_size"] == 20
assert history_result["snapshots"][2]["text_chunk_size"] == 25

print(f"    히스토리 파일: {history_path.name}")
print(f"    snapshots 수: {len(history_result['snapshots'])}개  (기대: 3)")
for i, snap in enumerate(history_result["snapshots"]):
    print(f"      [{i}] text_chunk_size={snap['text_chunk_size']} audio_chunk_size={snap['audio_chunk_size']} llm_token_rate={snap['llm_token_rate']}")
print("    exp3_control_history_*.json 검증 통과")


# ----------------------------------------
# clear_after 동작 확인
# ----------------------------------------
print("\n[9] clear_after=True 후 metrics_server 상태 확인")

remaining = metrics_server.get_records()
assert len(remaining) == 0, f"기록이 남아있다: {len(remaining)}건"
print(f"    남은 기록 수: {len(remaining)}건  (0이어야 정상)")

print("\n모든 검증 통과")
