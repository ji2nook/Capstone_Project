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
from websockets.exceptions import ConnectionClosed
from dotenv import load_dotenv
from rich.console import Console
from rich.prompt import Prompt

load_dotenv()

logger = logging.getLogger(__name__)
console = Console()

# 종료 키워드
_EXIT_COMMANDS: frozenset[str] = frozenset({"q", "quit", "exit"})

# 실험 전환 키워드 — Prompt 입력 창에서 입력하면 파이프라인 재선택 메뉴로 복귀
_SWITCH_COMMANDS: frozenset[str] = frozenset({"/switch", "switch"})

__all__ = [
    "InputManager",
    "SessionResult",
    "StreamingCycleStart",
    "StreamingCycleChunk",
    "StreamingCycleEnd",
    "StreamingEvent",
    "ExperimentSwitchRequested",
]


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
# 스트리밍 사이클 이벤트 (실험 2)
# ========================================

@dataclass
class StreamingCycleStart:
    # 스트리밍 사이클 시작 — 프롬프트 전송 직후 yield
    prompt: str
    t_send: float


@dataclass
class StreamingCycleChunk:
    # 수신한 오디오 청크 — 이 사이클의 모든 청크에 동일한 t_first_recv 값 포함
    data: bytes
    t_first_recv: float


@dataclass
class StreamingCycleEnd:
    # 스트리밍 사이클 종료 — 서버 종료 신호(b"") 수신 후 yield
    t_send: float
    t_first_recv: float  # ttfa_client = (t_first_recv - t_send) * 1000


StreamingEvent = StreamingCycleStart | StreamingCycleChunk | StreamingCycleEnd


# ========================================
# 예외
# ========================================

class ExperimentSwitchRequested(Exception):
    """사용자가 /switch를 입력해 파이프라인 전환을 요청함"""


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
        try:
            await self._ws.send(json.dumps(payload, ensure_ascii=False))
        except ConnectionClosed:
            # 서버가 이미 연결을 끊은 경우 — 지표 역전송을 건너뜀
            logger.warning("클라이언트 지표 역전송 실패 — 서버 연결이 이미 닫혔습니다")

    async def send_buffer_feedback(
        self, buffer_level: int, starvation_count: int
    ) -> None:
        """adaptive 모드: 오디오 청크 수신 직후 JitterBuffer 상태를 서버에 즉시 역전송.

        서버는 5ms 타임아웃으로 논블로킹 수신하므로, 이 메서드는 청크 처리 직후 호출해야
        다음 청크 크기 결정에 반영될 가능성이 높아진다.
        """
        if self._ws is None:
            return
        try:
            payload = {
                "type": "buffer_feedback",
                "buffer_level": buffer_level,
                "starvation_count": starvation_count,
            }
            await self._ws.send(json.dumps(payload))
        except ConnectionClosed:
            logger.warning("버퍼 피드백 역전송 실패 — 서버 연결이 이미 닫혔습니다")

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
            console.print("[dim]종료: q / quit / exit  또는 Ctrl+C[/dim]  |  전환: /switch\n")

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

                if text.lower() in _SWITCH_COMMANDS:
                    logger.info("[%s] 파이프라인 전환 요청", self._experiment)
                    raise ExperimentSwitchRequested()

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

    async def run_streaming(self) -> AsyncIterator[StreamingEvent]:
        # 스트리밍 실험용 비동기 제너레이터
        # 사이클마다 StreamingCycleStart → N×StreamingCycleChunk → StreamingCycleEnd 순서로 yield
        logger.info("[streaming] 서버 연결: %s", self._ws_url)

        async with websockets.connect(self._ws_url, max_size=None) as ws:
            self._ws = ws
            console.print(f"[green]✓[/green] 서버 연결 성공  [dim]({self._ws_url})[/dim]")
            console.print("[dim]종료: q / quit / exit 또는 Ctrl+C[/dim]  |  전환: /switch\n")

            while True:
                try:
                    text = await self._read_line()
                except EOFError:
                    logger.info("[streaming] stdin 닫힘 — 세션 종료")
                    break

                if not text or text.lower() in _EXIT_COMMANDS:
                    logger.info("[streaming] 종료 명령 — 세션 종료")
                    break

                if text.lower() in _SWITCH_COMMANDS:
                    logger.info("[streaming] 파이프라인 전환 요청")
                    raise ExperimentSwitchRequested()

                t_send = time.perf_counter()
                try:
                    await ws.send(text)
                except ConnectionClosed as exc:
                    logger.warning("[streaming] 프롬프트 전송 실패 — 서버 연결 끊김: %s", exc)
                    break
                logger.info("[streaming] 프롬프트 전송 (%d자)", len(text))

                # 사이클 시작 신호: main.py가 jitter_buffer.reset() / audio_player.open_stream() 호출
                yield StreamingCycleStart(prompt=text, t_send=t_send)

                t_first_recv: float | None = None
                chunk_count = 0
                _conn_alive = True

                while True:
                    try:
                        data = await ws.recv()
                    except ConnectionClosed as exc:
                        # 서버 파이프라인 오류(1011) 또는 예기치 않은 종료
                        logger.warning("[streaming] 서버 연결 끊김 — %s", exc)
                        console.print(f"[red]서버 연결이 끊겼습니다: {exc}[/red]")
                        _conn_alive = False
                        break

                    if not isinstance(data, bytes):
                        console.print(f"[red]오류 응답: {data}[/red]")
                        logger.warning("[streaming] 텍스트 응답 수신 (파이프라인 오류): %s", data)
                        break

                    if len(data) == 0:
                        # server의 send_end_signal()이 보낸 종료 프레임
                        logger.info("[streaming] 종료 신호 수신 — %d개 청크", chunk_count)
                        break

                    if t_first_recv is None:
                        t_first_recv = time.perf_counter()
                        logger.info(
                            "[streaming] 첫 청크 수신 — TTFA: %.1fms | %d bytes",
                            (t_first_recv - t_send) * 1000,
                            len(data),
                        )

                    chunk_count += 1
                    yield StreamingCycleChunk(data=data, t_first_recv=t_first_recv)

                # 사이클 종료 신호: main.py가 close_stream() / 지표 기록 / 역전송 수행
                yield StreamingCycleEnd(
                    t_send=t_send,
                    t_first_recv=t_first_recv if t_first_recv is not None else t_send,
                )

                if not _conn_alive:
                    break  # 외부 프롬프트 루프도 종료
