# AudioPlayer 스트리밍 재생 + JitterBuffer 순서 보장 디버깅
#
# 사용법:
#   python test/client/test_audio_play.py

import glob
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from client.audio_player import AudioPlayer
from client.jitter_buffer import JitterBuffer

_AUDIO_CHUNK_SIZE = 8192  # 서버 AUDIO_CHUNK_SIZE와 동일


# ========================================
# 테스트 1: JitterBuffer FIFO 순서 확인
# ========================================

def test_jitter_buffer() -> None:
    print("=" * 50)
    print("[테스트 1] JitterBuffer — FIFO 순서 및 starvation")
    print("=" * 50)

    buf = JitterBuffer()

    # 5개 청크 공급
    chunks = [f"chunk_{i}".encode() for i in range(5)]
    for c in chunks:
        buf.feed(c)
        print(f"  feed: {c}  (버퍼: {buf.buffered_chunks}개)")

    print()

    # 7개 소비 — 마지막 2개는 starvation
    for i in range(7):
        result = buf.drain()
        print(f"  drain {i + 1}: {result!r:15}  starvation={buf.starvation_count}")

    assert buf.starvation_count == 2, f"starvation 횟수 불일치: {buf.starvation_count}"
    print("\n  순서 정상, starvation=2 확인 완료")


# ========================================
# 테스트 2: JitterBuffer → AudioPlayer 스트리밍 재생
# ========================================

def test_streaming_playback() -> None:
    print()
    print("=" * 50)
    print("[테스트 2] JitterBuffer → AudioPlayer 스트리밍 재생")
    print("=" * 50)

    wav_files = sorted(glob.glob("results/audio/sequential_*.wav"))
    if not wav_files:
        print("  WAV 파일 없음 — 테스트 건너뜀 (실험 1을 먼저 실행하세요)")
        return

    wav_path = wav_files[-1]
    print(f"  WAV 소스: {wav_path}")

    with wave.open(wav_path, "rb") as wf:
        pcm_data = wf.readframes(wf.getnframes())

    print(f"  PCM 크기: {len(pcm_data):,} bytes\n")

    buf = JitterBuffer()
    chunk_count = 0

    # WAV를 청크 단위로 분할해서 버퍼에 공급
    for offset in range(0, len(pcm_data), _AUDIO_CHUNK_SIZE):
        buf.feed(pcm_data[offset : offset + _AUDIO_CHUNK_SIZE])
        chunk_count += 1

    print(f"  총 {chunk_count}개 청크를 버퍼에 적재 완료")
    print("  재생 시작...\n")

    with AudioPlayer() as player:
        player.open_stream()

        drained = 0
        while buf.buffered_chunks > 0:
            chunk = buf.drain()
            if chunk:
                player.feed_chunk(chunk)
                drained += 1

        play_duration = player.close_stream()

    print(f"  재생 완료: {drained}개 청크, {play_duration:.0f}ms")
    assert buf.starvation_count == 0, f"예상치 못한 starvation 발생: {buf.starvation_count}회"
    print("  starvation=0 확인 완료")


# ========================================
# 진입점
# ========================================

def main() -> None:
    test_jitter_buffer()
    test_streaming_playback()

    print()
    print("=" * 50)
    print("모든 테스트 완료")
    print("=" * 50)


main()
