import asyncio
import logging

import pyaudio
from dotenv import load_dotenv

load_dotenv()

# 터미널에 로그 출력 (DEBUG 레벨로 api_manager 내부 로그까지 확인)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

# 프로젝트 루트를 경로에 추가 (server 패키지를 찾기 위함)
import sys
sys.path.append(".")

from server.experiment_pipeline import PipelineResult, SequentialMetrics, run_sequential
from server import metrics_server


async def main() -> None:
    prompt = "너는 누구야? 간단한 자기소개 부탁해."

    print(f"프롬프트: {prompt}\n")

    # ------------------------------
    # 파이프라인 실행
    # ------------------------------
    result: PipelineResult = await run_sequential(prompt)
    assert isinstance(result.metrics, SequentialMetrics)
    m: SequentialMetrics = result.metrics

    # ------------------------------
    # 성능 지표 출력
    # ------------------------------
    print("===== 성능 지표 =====")
    print(f"  LLM 소요 시간  : {m.llm_latency:.1f}ms")
    print(f"  TTS 소요 시간  : {m.tts_latency:.1f}ms")
    print(f"  TTFT           : {m.ttft:.1f}ms")
    print(f"  TTFA           : {m.ttfa:.1f}ms")
    print(f"  E2E Latency    : {m.e2e_latency:.1f}ms")
    print()

    # ------------------------------
    # 결과 검증
    # ------------------------------
    EPSILON = 1.0   #perf_counter 연산 과정에서의 지연 감안

    assert abs(m.ttft - m.llm_latency) < EPSILON, (
        f"TTFT({m.ttft:.2f}ms)는 llm_latency({m.llm_latency:.2f}ms)와 같아야 한다"
    )
    assert abs(m.ttfa - m.e2e_latency) < EPSILON, (
        f"TTFA({m.ttfa:.2f}ms)는 e2e_latency({m.e2e_latency:.2f}ms)와 같아야 한다"
    )
    # 허용 오차를 10ms로 넉넉하게 설정 (await 전환 비용 등 극소 오버헤드 포함)
    assert abs(m.llm_latency + m.tts_latency - m.e2e_latency) < 10 * EPSILON, (
        f"E2E({m.e2e_latency:.2f}ms)는 LLM({m.llm_latency:.2f}ms) + TTS({m.tts_latency:.2f}ms)와 같아야 한다"
    )

    print("검증 통과")
    print("  TTFT   = llm_latency")
    print("  TTFA   = e2e_latency")
    print("  E2E    = LLM + TTS")
    print()


    # ------------------------------
    # metrics_server 기록 및 검증
    # ------------------------------
    # metrics.server.record()와 get_records()의 동작 여부 확인
    # 실제 서버에서는 connection_manager가 담당
    
    metrics_server.record(m)

    records = metrics_server.get_records()

    assert len(records) == 1, f"기록 수가 1이어야 한다 (실제: {len(records)})"

    saved = records[0]

    # record()가 metrics의 값을 정확히 복사했는지 검증
    assert saved.experiment == "sequential"
    assert abs(saved.llm_latency - m.llm_latency) < EPSILON
    assert abs(saved.e2e_latency - m.e2e_latency) < EPSILON
    assert abs(saved.ttft - m.ttft) < EPSILON
    assert abs(saved.ttfa - m.ttfa) < EPSILON

    print("===== metrics_server 검증 =====")
    print(f"  기록 수        : {len(records)}")
    print(f"  timestamp      : {saved.timestamp}")
    print(f"  experiment     : {saved.experiment}")
    print(f"  llm_latency    : {saved.llm_latency:.1f}ms")
    print(f"  tts_latency    : {saved.tts_latency:.1f}ms")
    print(f"  ttft           : {saved.ttft:.1f}ms")
    print(f"  ttfa           : {saved.ttfa:.1f}ms")
    print(f"  e2e_latency    : {saved.e2e_latency:.1f}ms")
    print("  저장 검증 통과")
    print()


    # ------------------------------
    # 오디오 바이트 확인
    # ------------------------------
    print("===== 오디오 결과 =====")
    print(f"  크기: {len(result.audio_bytes):,} bytes")
    assert len(result.audio_bytes) > 0, "오디오 바이트가 비어 있으면 안 된다"
    print("  오디오 정상 생성 확인")
    print()


    # ------------------------------
    # 오디오 재생
    # ------------------------------
    print("===== 오디오 재생 =====")
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=24000,
        output=True,
    )
    stream.write(result.audio_bytes)
    stream.stop_stream()
    stream.close()
    p.terminate()
    print("재생 완료")


asyncio.run(main())
