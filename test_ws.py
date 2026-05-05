# 디버깅용 WebSocket 테스트 클라이언트
# 사용법: python test_ws.py
# 서버(python -m server.main)가 먼저 실행 중이어야 한다

import asyncio
import websockets

SERVER_URL = "ws://localhost:8765/ws?experiment=sequential"
TEST_PROMPT = "Say hello in one sentence."


async def main() -> None:
    print(f"[1] 서버 연결 시도: {SERVER_URL}")

    async with websockets.connect(SERVER_URL) as ws:
        print(f"[2] 연결 성공")

        print(f"[3] 프롬프트 전송: {TEST_PROMPT!r}")
        await ws.send(TEST_PROMPT)

        print("[4] 서버 응답 대기 중... (LLM + TTS 처리 시간 소요)")
        data = await ws.recv()

        if isinstance(data, bytes):
            print(f"[5] 오디오 수신 완료: {len(data):,} bytes")
            print("    ✓ 파이프라인 정상 동작 확인")
        else:
            print(f"[5] 텍스트 응답 수신 (오류 가능성): {data}")

    print("[6] 연결 종료")


asyncio.run(main())