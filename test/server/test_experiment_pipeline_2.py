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

from server.experiment_pipeline import StreamingMetrics, run_streaming
from server import metrics_server


async def main() -> None:
    prompt = "너는 누구야? 5문장 이상의 자기소개 부탁해. 5문장 중 한 문장 이상은 80자 이상이어야 해."

    print(f"프롬프트: {prompt}\n")

    # ------------------------------
    # 파이프라인 실행 — 청크 수집 + 스트리밍 재생
    # ------------------------------
    # pyaudio 스트림을 미리 열어두고 오디오 청크가 올 때마다 즉시 write
    # 실제 클라이언트에서 feed_chunk()를 호출하는 방식과 동일한 흐름

    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=24000,
        output=True,
    )

    all_audio = bytearray()
    audio_chunk_sizes: list[int] = []
    m: StreamingMetrics | None = None

    async for audio_chunk, metrics in run_streaming(prompt):
        if metrics is not None:
            m = metrics
        elif audio_chunk:
            audio_chunk_sizes.append(len(audio_chunk))
            all_audio.extend(audio_chunk)
            stream.write(audio_chunk)

    stream.stop_stream()
    stream.close()
    p.terminate()
    print("재생 완료\n")

    assert m is not None, "StreamingMetrics가 반환되지 않았다"

    # ------------------------------
    # 성능 지표 출력
    # ------------------------------
    print("===== 성능 지표 =====")
    print(f"  LLM 소요 시간    : {m.llm_latency:.1f}ms")
    print(f"  TTS 소요 시간    : {m.tts_latency:.1f}ms  (마지막 텍스트 청크 기준 근사)")
    print(f"  TTFT             : {m.ttft:.1f}ms")
    print(f"  TTFA             : {m.ttfa:.1f}ms")
    print(f"  E2E Latency      : {m.e2e_latency:.1f}ms")
    print(f"  텍스트 청크 수   : {m.text_chunk_count}")
    print(f"  오디오 청크 수   : {m.audio_chunk_count}")
    print(f"  오디오 총 크기   : {len(all_audio):,} bytes")
    print()

    # ------------------------------
    # 오디오 청크 크기 분포 출력
    # ------------------------------
    if audio_chunk_sizes:
        print("===== 오디오 청크 크기 분포 =====")
        for i, size in enumerate(audio_chunk_sizes, 1):
            print(f"  청크 {i:>2}: {size:,} bytes")
        print()

    # ------------------------------
    # 결과 검증
    # ------------------------------
    EPSILON = 1.0   #perf_counter 연산 과정에서의 지연 감안

    # llm_latency는 ttft와 동일한 값으로 기록
    assert abs(m.llm_latency - m.ttft) < EPSILON, (
        f"llm_latency({m.llm_latency:.2f}ms)는 ttft({m.ttft:.2f}ms)와 같아야 한다"
    )

    # 텍스트 청크가 확정된 후 TTS 처리가 시작되므로 TTFT < TTFA
    assert m.ttft < m.ttfa, (
        f"TTFT({m.ttft:.2f}ms)는 TTFA({m.ttfa:.2f}ms)보다 작아야 한다"
    )

    # 첫 오디오 청크는 전체 파이프라인 완료 전에 도착해야 함
    assert m.ttfa <= m.e2e_latency, (
        f"TTFA({m.ttfa:.2f}ms)는 E2E({m.e2e_latency:.2f}ms)보다 크면 안 된다"
    )

    # 텍스트 청크가 최소 1개 이상 생성되어야 함
    assert m.text_chunk_count >= 1, (
        f"텍스트 청크 수가 0이다 — LLM 응답이 비어 있을 가능성이 있다"
    )

    # 각 텍스트 청크에서 오디오가 최소 1개 이상 생성되므로 audio >= text
    assert m.audio_chunk_count >= m.text_chunk_count, (
        f"오디오 청크 수({m.audio_chunk_count})가 텍스트 청크 수({m.text_chunk_count})보다 적으면 안 된다"
    )

    # 오디오 바이트가 실제로 생성됐는지 확인
    assert len(all_audio) > 0, "오디오 바이트가 비어 있으면 안 된다"

    print("검증 통과")
    print("  llm_latency      = ttft")
    print("  ttft             < ttfa")
    print("  ttfa             ≤ e2e_latency")
    print("  text_chunk_count ≥ 1")
    print("  audio_chunk_count ≥ text_chunk_count")
    print()

    # ------------------------------
    # metrics_server 기록 및 검증
    # ------------------------------
    # 실제 서버에서는 connection_manager가 담당
    metrics_server.record(
        m,
        experiment="streaming",
        prompt=prompt,
    )

    records = metrics_server.get_records()

    assert len(records) == 1, f"기록 수가 1이어야 한다 (실제: {len(records)})"

    saved = records[0]

    assert saved.experiment == "streaming"
    assert abs(saved.llm_latency - m.llm_latency) < EPSILON
    assert abs(saved.tts_latency - m.tts_latency) < EPSILON
    assert abs(saved.e2e_latency - m.e2e_latency) < EPSILON
    assert abs(saved.ttft - m.ttft) < EPSILON
    assert abs(saved.ttfa - m.ttfa) < EPSILON

    print("===== metrics_server 검증 =====")
    print(f"  기록 수          : {len(records)}")
    print(f"  timestamp        : {saved.timestamp}")
    print(f"  experiment       : {saved.experiment}")
    print(f"  llm_latency      : {saved.llm_latency:.1f}ms")
    print(f"  tts_latency      : {saved.tts_latency:.1f}ms")
    print(f"  ttft             : {saved.ttft:.1f}ms")
    print(f"  ttfa             : {saved.ttfa:.1f}ms")
    print(f"  e2e_latency      : {saved.e2e_latency:.1f}ms")
    print("  저장 검증 통과")


asyncio.run(main())
