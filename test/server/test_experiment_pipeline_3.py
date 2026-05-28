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

from server.experiment_pipeline import AdaptiveMetrics, run_adaptive
from server.adaptive_controller import DEFAULT_AUDIO_CHUNK_SIZE, DEFAULT_TEXT_CHUNK_SIZE
from server import metrics_server


def _print_snapshot(snap: dict) -> None:
    """AdaptiveState 스냅샷 dict를 읽기 쉬운 형태로 출력."""
    print(f"  llm_tokens_generated  : {snap.get('llm_tokens_generated')}")
    print(f"  llm_token_rate        : {snap.get('llm_token_rate', 0.0):.3f} tokens/s")
    print(f"  tts_last_chunk_size   : {snap.get('tts_last_chunk_size')} 자")
    print(f"  tts_synthesis_rate    : {snap.get('tts_synthesis_rate', 0.0):.1f} bytes/s")
    print(f"  buffer_level          : {snap.get('buffer_level')}")
    print(f"  buffer_starvation_cnt : {snap.get('buffer_starvation_count')}")
    print(f"  throughput_bps        : {snap.get('throughput_bps', 0.0):.1f}")
    print(f"  last_send_interval_ms : {snap.get('last_send_interval_ms', 0.0):.1f}ms")
    print(f"  text_chunk_size       : {snap.get('text_chunk_size')} 자  (결정값)")
    print(f"  audio_chunk_size      : {snap.get('audio_chunk_size')} bytes  (결정값)")


async def main() -> None:
    # 텍스트 청크가 여러 개 생성되도록 긴 응답을 유도하는 프롬프트
    prompt = "너는 누구야? 5문장 이상의 자기소개 부탁해. 5문장 중 한 문장 이상은 80자 이상이어야 해."

    print(f"프롬프트: {prompt}\n")

    # ------------------------------
    # 파이프라인 실행 — 청크 수집 + 스트리밍 재생
    # ------------------------------
    # pyaudio 스트림을 미리 열어두고 오디오 청크가 올 때마다 즉시 write
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=24000,
        output=True,
    )

    all_audio = bytearray()
    audio_chunk_sizes: list[int] = []
    m: AdaptiveMetrics | None = None

    async for audio_chunk, metrics in run_adaptive(prompt):
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

    assert m is not None, "AdaptiveMetrics가 반환되지 않았다"

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
    # 적응형 제어 지표 출력
    # ------------------------------
    print("===== 적응형 제어 지표 =====")
    print(f"  평균 텍스트 청크 크기 : {m.avg_text_chunk_size:.1f} 자")
    print(f"  평균 오디오 청크 크기 : {m.avg_audio_chunk_size:.1f} bytes")
    print(f"  긴급 축소 횟수        : {m.starvation_adjustments}")
    print(f"  제어 스냅샷 수        : {len(m.control_snapshots)}")
    print()

    # 텍스트 청크별 컨트롤러 상태 변화 확인 (Step 4~7에서 값이 바뀌기 시작함)
    if m.control_snapshots:
        print("===== 제어 스냅샷 (첫 번째) =====")
        _print_snapshot(m.control_snapshots[0])
        if len(m.control_snapshots) > 1:
            print(f"===== 제어 스냅샷 (마지막 — {len(m.control_snapshots)}번째) =====")
            _print_snapshot(m.control_snapshots[-1])
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
    EPSILON = 1.0   # perf_counter 연산 과정에서의 지연 감안

    # llm_latency는 ttft와 동일한 값으로 기록 (run_streaming()과 동일한 규칙)
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

    # 스냅샷은 텍스트 청크 확정 시점마다 1회 저장되므로 수가 일치해야 함
    assert len(m.control_snapshots) == m.text_chunk_count, (
        f"스냅샷 수({len(m.control_snapshots)})는 텍스트 청크 수({m.text_chunk_count})와 같아야 한다"
    )

    # 각 스냅샷은 AdaptiveState 전체 필드를 포함한 dict여야 함
    if m.control_snapshots:
        snap = m.control_snapshots[0]
        assert "text_chunk_size" in snap, "스냅샷에 text_chunk_size 필드가 없다"
        assert "audio_chunk_size" in snap, "스냅샷에 audio_chunk_size 필드가 없다"
        assert "llm_tokens_generated" in snap, "스냅샷에 llm_tokens_generated 필드가 없다"
        assert "buffer_level" in snap, "스냅샷에 buffer_level 필드가 없다"

    # Step 2: decide_*()가 고정값을 반환하므로 평균 = DEFAULT 값과 같아야 함
    assert m.avg_text_chunk_size == float(DEFAULT_TEXT_CHUNK_SIZE), (
        f"Step 2 기준: avg_text_chunk_size({m.avg_text_chunk_size})는 DEFAULT({DEFAULT_TEXT_CHUNK_SIZE})와 같아야 한다"
    )
    assert m.avg_audio_chunk_size == float(DEFAULT_AUDIO_CHUNK_SIZE), (
        f"Step 2 기준: avg_audio_chunk_size({m.avg_audio_chunk_size})는 DEFAULT({DEFAULT_AUDIO_CHUNK_SIZE})와 같아야 한다"
    )

    # Step 2: 버퍼 고갈 긴급 축소 로직 미구현 — starvation_adjustments는 항상 0
    assert m.starvation_adjustments == 0, (
        f"Step 2 기준: starvation_adjustments는 0이어야 한다 (실제: {m.starvation_adjustments})"
    )

    print("검증 통과")
    print("  llm_latency            = ttft")
    print("  ttft                   < ttfa")
    print("  ttfa                   ≤ e2e_latency")
    print("  text_chunk_count       ≥ 1")
    print("  audio_chunk_count      ≥ text_chunk_count")
    print("  len(control_snapshots) = text_chunk_count")
    print(f"  avg_text_chunk_size    = {DEFAULT_TEXT_CHUNK_SIZE}  (Step 2 고정)")
    print(f"  avg_audio_chunk_size   = {DEFAULT_AUDIO_CHUNK_SIZE}  (Step 2 고정)")
    print("  starvation_adjustments = 0  (Step 2 미구현)")
    print()

    # ------------------------------
    # metrics_server 기록 및 검증
    # ------------------------------
    # 실제 서버에서는 connection_manager가 담당
    metrics_server.record(
        m,
        experiment="adaptive",
        prompt=prompt,
        llm_response=m.llm_response,
        audio_path=m.audio_path,
    )

    records = metrics_server.get_records()

    assert len(records) == 1, f"기록 수가 1이어야 한다 (실제: {len(records)})"

    saved = records[0]

    assert saved.experiment == "adaptive"
    assert abs(saved.llm_latency - m.llm_latency) < EPSILON
    assert abs(saved.tts_latency - m.tts_latency) < EPSILON
    assert abs(saved.e2e_latency - m.e2e_latency) < EPSILON

    # Step 8 전까지 metrics_server.record()는 AdaptiveMetrics를 else 분기로 처리하므로
    # ttft·ttfa가 0.0으로 기록된다. Step 8에서 AdaptiveMetrics 분기 추가 후 이 검증을 수정할 것.
    assert saved.ttft == 0.0, (
        f"Step 8 전: ttft는 0.0으로 기록되어야 한다 (실제: {saved.ttft})"
    )
    assert saved.ttfa == 0.0, (
        f"Step 8 전: ttfa는 0.0으로 기록되어야 한다 (실제: {saved.ttfa})"
    )

    print("===== metrics_server 검증 =====")
    print(f"  기록 수          : {len(records)}")
    print(f"  timestamp        : {saved.timestamp}")
    print(f"  experiment       : {saved.experiment}")
    print(f"  llm_latency      : {saved.llm_latency:.1f}ms")
    print(f"  tts_latency      : {saved.tts_latency:.1f}ms")
    print(f"  ttft             : {saved.ttft:.1f}ms  ← Step 8 전: else 분기로 0.0 기록")
    print(f"  ttfa             : {saved.ttfa:.1f}ms  ← Step 8 전: else 분기로 0.0 기록")
    print(f"  e2e_latency      : {saved.e2e_latency:.1f}ms")
    print("  저장 검증 통과")


asyncio.run(main())
