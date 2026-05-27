# MetricsClient + JitterBuffer 지표 연동 디버깅
#
# 사용법:
#   python test/client/test_metrics_client.py

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from client.input_manager import SessionResult
from client.jitter_buffer import JitterBuffer
from client.metrics_client import MetricsClient
from server.logger import ClientMetrics

_CHUNK_SIZE = 8192  # tts_audio_chunker 기본 청크 크기


# ========================================
# 테스트 1: stutter_duration_ms 계산 검증
# ========================================

def test_stutter_duration() -> None:
    print("=" * 50)
    print("[테스트 1] stutter_duration_ms 계산")
    print("=" * 50)

    buf = JitterBuffer()

    # 3개 청크 공급
    for i in range(3):
        buf.feed(b"\x00" * _CHUNK_SIZE)
    print(f"  feed 3개 (각 {_CHUNK_SIZE} bytes)")

    # 5회 drain — 마지막 2회 starvation
    for _ in range(5):
        buf.drain()
    print(f"  drain 5회 → starvation={buf.starvation_count}")

    expected_avg_ms = (_CHUNK_SIZE / 2) / 24_000 * 1000
    expected_stutter = 2 * expected_avg_ms
    actual_stutter = buf.stutter_duration_ms

    print(f"  예상 평균 청크 재생시간  : {expected_avg_ms:.2f}ms")
    print(f"  예상 stutter_duration_ms : {expected_stutter:.2f}ms")
    print(f"  실제 stutter_duration_ms : {actual_stutter:.2f}ms")

    assert buf.starvation_count == 2
    assert abs(actual_stutter - expected_stutter) < 0.01, \
        f"stutter_duration_ms 불일치: {actual_stutter:.4f} != {expected_stutter:.4f}"
    print("  검증 통과\n")


# ========================================
# 테스트 2: MetricsClient.record() 반환값 및 값 연동 확인
# ========================================

def test_metrics_record() -> None:
    print("=" * 50)
    print("[테스트 2] MetricsClient.record() 반환값 및 연동")
    print("=" * 50)

    buf = JitterBuffer()
    for _ in range(3):
        buf.feed(b"\x00" * _CHUNK_SIZE)
    for _ in range(5):
        buf.drain()

    # 가짜 SessionResult: 왕복 500ms 가정
    result = SessionResult(
        prompt="테스트",
        audio_bytes=b"\x00" * 100,
        t_send=0.0,
        t_recv=0.5,
    )
    play_duration = 1200.0
    buffer_starvation = buf.starvation_count
    stutter_duration = buf.stutter_duration_ms

    metrics = MetricsClient()
    entry = metrics.record(result, play_duration, buffer_starvation, stutter_duration)

    print(f"  반환 타입              : {type(entry).__name__}")
    print(f"  ttfa_client            : {entry.ttfa_client:.1f}ms  (기대: 500.0ms)")
    print(f"  play_duration          : {entry.play_duration:.1f}ms")
    print(f"  buffer_starvation      : {entry.buffer_starvation}회")
    print(f"  stutter_duration       : {entry.stutter_duration:.2f}ms")

    assert isinstance(entry, ClientMetrics), "record()가 ClientMetrics를 반환해야 한다"
    assert abs(entry.ttfa_client - 500.0) < 0.1
    assert entry.play_duration == play_duration
    assert entry.buffer_starvation == 2
    assert abs(entry.stutter_duration - buf.stutter_duration_ms) < 0.01

    records = metrics.get_records()
    assert len(records) == 1, f"저장된 레코드 수가 1이어야 한다 (실제: {len(records)})"
    print(f"  내부 레코드 수         : {len(records)}")
    print("  검증 통과\n")


# ========================================
# 테스트 3: reset() 후 stutter_duration_ms 초기화 확인
# ========================================

def test_reset() -> None:
    print("=" * 50)
    print("[테스트 3] reset() 후 초기화 확인")
    print("=" * 50)

    buf = JitterBuffer()
    buf.feed(b"\x00" * _CHUNK_SIZE)
    buf.drain()
    buf.drain()  # starvation 1회 발생

    print(f"  reset 전 starvation    : {buf.starvation_count}")
    print(f"  reset 전 stutter_ms    : {buf.stutter_duration_ms:.2f}ms")

    buf.reset()

    print(f"  reset 후 starvation    : {buf.starvation_count}")
    print(f"  reset 후 stutter_ms    : {buf.stutter_duration_ms:.2f}ms")
    print(f"  reset 후 buffered      : {buf.buffered_chunks}개")

    assert buf.starvation_count == 0
    assert buf.stutter_duration_ms == 0.0
    assert buf.buffered_chunks == 0
    print("  검증 통과\n")


# ========================================
# 진입점
# ========================================

def main() -> None:
    test_stutter_duration()
    test_metrics_record()
    test_reset()

    print("=" * 50)
    print("모든 테스트 완료")
    print("=" * 50)


main()
