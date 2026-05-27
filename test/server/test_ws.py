# 실험 1 (sequential) / 실험 2 (streaming) WebSocket 통합 디버깅 클라이언트
# 서버(python -m server.main)가 먼저 실행 중이어야 한다
#
# 사용법:
#   python test/server/test_ws.py               → 실험 1, 2 순서대로 실행
#   python test/server/test_ws.py sequential    → 실험 1만 실행
#   python test/server/test_ws.py streaming     → 실험 2만 실행

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import asyncio

import pyaudio
import websockets
from dotenv import load_dotenv

load_dotenv()

_TOKEN = os.environ.get("WS_TOKEN", "")
BASE_URL = "ws://localhost:8765/ws"

PROMPT_SEQUENTIAL = "너는 누구야? 5문장 이상의 자기소개 부탁해. 5문장 중 한 문장 이상은 80자 이상이어야 해."
PROMPT_STREAMING = "너는 누구야? 5문장 이상의 자기소개 부탁해. 5문장 중 한 문장 이상은 80자 이상이어야 해."


# ========================================
# 실험 1: 순차적 파이프라인
# ========================================

async def test_sequential() -> None:
    url = f"{BASE_URL}?experiment=sequential&token={_TOKEN}"

    print("=" * 55)
    print("[실험 1 — sequential]")
    print(f"  서버 URL : {url}")
    print(f"  프롬프트 : {PROMPT_SEQUENTIAL!r}")
    print("=" * 55)

    async with websockets.connect(url, max_size=16 * 1024 * 1024) as ws:
        print("[1] 연결 성공")

        # 프롬프트 전송 직전 시각 기록 (TTFA_client 기준점)
        t_send = time.perf_counter()
        await ws.send(PROMPT_SEQUENTIAL)
        print("[2] 프롬프트 전송 완료 — LLM + TTS 처리 대기 중...")

        # 전체 오디오를 단일 바이너리 프레임으로 수신
        data = await ws.recv()
        t_recv = time.perf_counter()

        assert isinstance(data, bytes), f"바이너리 응답이 아닌 텍스트 수신: {data}"
        assert len(data) > 0, "수신된 오디오 바이트가 비어 있다"
        print(f"[3] 오디오 수신 완료: {len(data):,} bytes")

        # 성능 지표 계산
        ttfa_client = (t_recv - t_send) * 1000

        # 오디오 재생
        print("[4] 오디오 재생 중...")
        t_play_start = time.perf_counter()
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)
        stream.write(data)
        stream.stop_stream()
        stream.close()
        p.terminate()
        t_play_end = time.perf_counter()
        play_duration = (t_play_end - t_play_start) * 1000
        print("[4] 재생 완료")

        # 성능 지표 출력
        print()
        print("===== 성능 지표 =====")
        print(f"  TTFA (client)  : {ttfa_client:.1f}ms  (프롬프트 전송 ~ 오디오 수신)")
        print(f"  재생 시간      : {play_duration:.1f}ms")
        print(f"  오디오 크기    : {len(data):,} bytes")
        print()

        # 검증
        assert ttfa_client > 0, "TTFA_client가 0 이하다"
        assert play_duration > 0, "재생 시간이 0 이하다"
        print("검증 통과")
        print("  TTFA_client    > 0")
        print("  play_duration  > 0")
        print()

        # 클라이언트 지표를 서버로 역전송 (서버의 receive_text() 대기 해소)
        client_metrics = {
            "ttfa_client": ttfa_client,
            "play_duration": play_duration,
            "buffer_starvation": 0,
            "stutter_duration": 0.0,
        }
        await ws.send(json.dumps(client_metrics))
        print("[5] 클라이언트 지표 전송 완료")

    print("[6] 연결 종료\n")


# ========================================
# 실험 2: 스트리밍 파이프라인
# ========================================

async def test_streaming() -> None:
    url = f"{BASE_URL}?experiment=streaming&token={_TOKEN}"

    print("=" * 55)
    print("[실험 2 — streaming]")
    print(f"  서버 URL : {url}")
    print(f"  프롬프트 : {PROMPT_STREAMING!r}")
    print("=" * 55)

    async with websockets.connect(url, max_size=16 * 1024 * 1024) as ws:
        print("[1] 연결 성공")

        # 프롬프트 전송 직전 시각 기록 (TTFA_client 기준점)
        t_send = time.perf_counter()
        await ws.send(PROMPT_STREAMING)
        print("[2] 프롬프트 전송 완료 — 청크 수신 대기 중...")

        # pyaudio 스트림을 미리 열어 청크가 올 때마다 즉시 재생
        p = pyaudio.PyAudio()
        pa_stream = p.open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)

        chunk_count = 0
        total_bytes = 0
        chunk_sizes: list[int] = []
        t_first_chunk: float | None = None
        t_play_start: float | None = None

        # 종료 신호(b"")를 받을 때까지 청크를 반복 수신
        while True:
            data = await ws.recv()

            # send_end_signal()이 보낸 빈 프레임 — 스트리밍 종료
            if isinstance(data, bytes) and len(data) == 0:
                print("[3] 종료 신호(b\"\") 수신 — 스트리밍 완료")
                break

            if not isinstance(data, bytes):
                print(f"    예상치 못한 텍스트 수신: {data!r}")
                continue

            now = time.perf_counter()
            if t_first_chunk is None:
                t_first_chunk = now
                t_play_start = now
                ttfa_at_first = (now - t_send) * 1000
                print(f"    첫 청크 수신 — TTFA_client: {ttfa_at_first:.1f}ms, {len(data):,} bytes")

            chunk_count += 1
            total_bytes += len(data)
            chunk_sizes.append(len(data))
            pa_stream.write(data)

        t_play_end = time.perf_counter()

        pa_stream.stop_stream()
        pa_stream.close()
        p.terminate()
        print("[4] 재생 완료")

        # 성능 지표 계산
        ttfa_client = (t_first_chunk - t_send) * 1000 if t_first_chunk is not None else 0.0
        play_duration = (t_play_end - t_play_start) * 1000 if t_play_start is not None else 0.0

        # 성능 지표 출력
        print()
        print("===== 성능 지표 =====")
        print(f"  TTFA (client)  : {ttfa_client:.1f}ms  (프롬프트 전송 ~ 첫 청크 수신)")
        print(f"  재생 시간      : {play_duration:.1f}ms")
        print(f"  수신 청크 수   : {chunk_count}")
        print(f"  총 수신 크기   : {total_bytes:,} bytes")
        print()

        # 청크 크기 분포 출력
        if chunk_sizes:
            print("===== 청크 크기 분포 =====")
            for i, sz in enumerate(chunk_sizes, 1):
                print(f"  청크 {i:>2}: {sz:,} bytes")
            print()

        # 검증
        assert chunk_count >= 1, "청크가 1개도 수신되지 않았다"
        assert total_bytes > 0, "수신된 오디오 바이트가 비어 있다"
        assert t_first_chunk is not None, "첫 청크 수신 시각이 기록되지 않았다"
        assert ttfa_client > 0, "TTFA_client가 0 이하다"
        print("검증 통과")
        print("  chunk_count    >= 1")
        print("  total_bytes    > 0")
        print("  TTFA_client    > 0")
        print()

        # 클라이언트 지표를 서버로 역전송 (서버의 receive_text() 대기 해소)
        client_metrics = {
            "ttfa_client": ttfa_client,
            "play_duration": play_duration,
            "buffer_starvation": 0,
            "stutter_duration": 0.0,
        }
        await ws.send(json.dumps(client_metrics))
        print("[5] 클라이언트 지표 전송 완료")

    print("[6] 연결 종료\n")


# ========================================
# 진입점
# ========================================

async def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    if mode not in ("sequential", "streaming", "all"):
        print(f"알 수 없는 모드: {mode!r}")
        print("사용법: python test/server/test_ws.py [sequential | streaming | all]")
        return

    if mode in ("sequential", "all"):
        await test_sequential()

    if mode in ("streaming", "all"):
        await test_streaming()


asyncio.run(main())
