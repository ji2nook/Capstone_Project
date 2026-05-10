# 사용자 입력 수집 및 WebSocket 전송

import asyncio
import json
import logging
import os
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import websockets
from dotenv import load_dotenv
from rich.console import Console
from rich.prompt import Prompt

load_dotenv()

logger = logging.getLogger(__name__)
console = Console()

# 종료 키워드
_EXIT_COMMANDS: frozenset[str] = frozenset({"q", "quit", "exit"})

__all__ = ["InputManager", "SessionResult"]


# ========================================
# 세션 결과
# ========================================

@dataclass
class SessionResult:
    # 프롬프트 전송부터 오디오 수신까지 한 사이클의 결과

    prompt: str        # 사용자가 입력한 텍스트 프롬프트
    audio_bytes: bytes # 서버로부터 수신한 raw PCM 24000Hz 오디오 (WAV 헤더 없음)
    t_send: float      # 프롬프트 전송 순간의 perf_counter 값 (E2E 시작점)
    t_recv: float      # 오디오 수신 순간의 perf_counter 값 (클라이언트 기준 TTFA)


# ========================================
# InputManager
# ========================================

class InputManager:
    # CLI 입력 수집과 WebSocket 세션 관리

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        experiment: str = "sequential",
    ) -> None:
        resolved_host = host or os.environ.get("HOST", "localhost")
        resolved_port = port or int(os.environ.get("PORT", "8765"))
        ws_token = os.environ.get("WS_TOKEN", "")

        self._ws_url = f"ws://{resolved_host}:{resolved_port}/ws?experiment={experiment}"
        if ws_token:
            self._ws_url += f"&token={ws_token}"
        self._experiment = experiment
        self._ws: Any = None  # run() 실행 중에만 WebSocket 인스턴스가 할당됨

    async def send_client_metrics(self, payload: dict) -> None:
        # 재생 완료 후 클라이언트 지표를 JSON으로 서버에 역전송
        if self._ws is None:
            return
        await self._ws.send(json.dumps(payload, ensure_ascii=False))

    @staticmethod
    async def _read_line() -> str:
        # Prompt.ask()는 블로킹 함수이므로 스레드 풀에 위임해 이벤트 루프가 멈추지 않도록 함
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: Prompt.ask("[bold green]Prompt[/bold green]")
        )

    async def run(self) -> AsyncIterator[SessionResult]:
        # 서버 연결 및 프롬프트 루프를 실행하는 비동기 제너레이터
        logger.info("[%s] 서버 연결: %s", self._experiment, self._ws_url)

        async with websockets.connect(self._ws_url, max_size=None) as ws:
            self._ws = ws
            console.print(f"[green]✓[/green] 서버 연결 성공  [dim]({self._ws_url})[/dim]")
            console.print("[dim]종료: q / quit / exit 또는 Ctrl+C[/dim]\n")

            while True:
                # -------------------------------
                # 입력 수집
                # -------------------------------
                try:
                    text = await self._read_line()
                except EOFError:
                    logger.info("[%s] stdin 닫힘 — 세션 종료", self._experiment)
                    break

                if not text or text.lower() in _EXIT_COMMANDS:
                    logger.info("[%s] 종료 명령 — 세션 종료", self._experiment)
                    break

                # -------------------------------
                # 서버 전송
                # -------------------------------
                t_send = time.perf_counter()
                await ws.send(text)

                # -------------------------------
                # 오디오 수신 (스피너 표시)
                # -------------------------------
                # console.status()는 백그라운드 스레드에서 스피너를 렌더링하므로
                # await ws.recv()가 블로킹되는 동안에도 애니메이션이 계속 동작함
                with console.status(
                    "[yellow]서버 처리 중… (LLM → TTS)[/yellow]", spinner="dots"
                ):
                    data = await ws.recv()

                t_recv = time.perf_counter()

                if isinstance(data, bytes):
                    logger.info(
                        "[%s] 오디오 수신 완료 — %d bytes | 왕복: %.1fms",
                        self._experiment,
                        len(data),
                        (t_recv - t_send) * 1000,
                    )
                    yield SessionResult(
                        prompt=text,
                        audio_bytes=data,
                        t_send=t_send,
                        t_recv=t_recv,
                    )
                else:
                    # 오디오 대신 텍스트 오류 메시지를 수신한 경우
                    console.print(f"[red]오류 응답: {data}[/red]")
                    logger.warning("[%s] 텍스트 응답 수신 (파이프라인 오류): %s", self._experiment, data)
