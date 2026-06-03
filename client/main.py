# 클라이언트 진입점

import asyncio
import json
import logging
import wave
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from rich import box
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.prompt import IntPrompt
from rich.table import Table

from client.audio_player import AudioPlayer
from client.input_manager import (
    ExperimentSwitchRequested,
    InputManager,
    SessionResult,
    StreamingCycleChunk,
    StreamingCycleEnd,
    StreamingCycleStart,
)
from client.jitter_buffer import JitterBuffer
from client.metrics_client import MetricsClient
from server.logger import ClientMetrics

load_dotenv()

console = Console()


# ========================================
# 로깅 설정
# ========================================

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, show_path=False, rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)


# ========================================
# 실험 파이프라인 선택
# ========================================

_EXPERIMENT_OPTIONS: list[tuple[str, str]] = [
    ("sequential", "순차적 파이프라인   (실험 1)"),
    ("streaming",  "스트리밍 파이프라인  (실험 2)"),
    ("adaptive",   "적응형 파이프라인   (실험 3)"),
]

_IMPLEMENTED: frozenset[str] = frozenset({"sequential", "streaming", "adaptive"})

_EXPERIMENT_PREFIX: dict[str, str] = {
    "sequential": "exp1",
    "streaming":  "exp2",
    "adaptive":   "exp3",
}


def _select_experiment() -> str:
    # 실험 파이프라인 선택 메뉴 출력 및 선택값 반환
    console.print(Panel("[bold white]실험 파이프라인 선택[/bold white]", expand=False))

    while True:
        for i, (exp, label) in enumerate(_EXPERIMENT_OPTIONS, 1):
            if exp in _IMPLEMENTED:
                console.print(f"  [bold cyan]{i}.[/bold cyan] {label}")
            else:
                console.print(f"  [dim]{i}. {label}  ← 미구현[/dim]")

        choice = IntPrompt.ask("\n번호 입력", console=console)

        if choice < 1 or choice > len(_EXPERIMENT_OPTIONS):
            console.print("[red]1~3 사이의 번호를 입력하세요.[/red]\n")
            continue

        experiment, label = _EXPERIMENT_OPTIONS[choice - 1]

        if experiment not in _IMPLEMENTED:
            console.print("[yellow]아직 구현되지 않은 파이프라인입니다. 다시 선택하세요.[/yellow]\n")
            continue

        console.print(f"\n[bold]{label}[/bold]을 선택했습니다.\n")
        return experiment


# ========================================
# 결과 출력
# ========================================

def _print_result(
    result: SessionResult,
    play_duration: float,
    total_bytes: int | None = None,
    chunk_count: int | None = None,
    starvation_count: int | None = None,
) -> None:
    # 한 사이클 완료 후 지표를 테이블로 출력
    # total_bytes / chunk_count 는 스트리밍 실험에서만 전달됨
    # starvation_count 는 adaptive 실험에서만 전달됨
    ttfa_ms = (result.t_recv - result.t_send) * 1000

    table = Table(box=box.ROUNDED, show_header=False, padding=(0, 2))
    table.add_column(style="dim", min_width=22)
    table.add_column(justify="right")
    table.add_row("TTFA (클라이언트)", f"[cyan]{ttfa_ms:.0f} ms[/cyan]")
    table.add_row("재생 시간",         f"[cyan]{play_duration:.0f} ms[/cyan]")

    if total_bytes is not None:
        table.add_row("오디오 크기",    f"[cyan]{total_bytes:,} bytes[/cyan]")
        if chunk_count is not None:
            table.add_row("청크 수",    f"[cyan]{chunk_count:,}[/cyan]")
    else:
        table.add_row("오디오 크기",    f"[cyan]{len(result.audio_bytes):,} bytes[/cyan]")

    if starvation_count is not None:
        color = "red" if starvation_count > 0 else "green"
        table.add_row("버퍼 고갈 횟수", f"[{color}]{starvation_count}[/{color}]")

    console.print(table)
    console.print()


# ========================================
# 클라이언트 지표 저장
# ========================================

def _build_client_record(r: ClientMetrics) -> dict:
    return {
        "ttfa_client": r.ttfa_client,
        "play_duration": r.play_duration,
        "buffer_starvation": r.buffer_starvation,
        "stutter_duration": r.stutter_duration,
    }


def _save_as_heard_wav(pcm_bytes: bytes, experiment: str) -> Path:
    # WASAPI 루프백으로 캡처한 PCM을 WAV 파일로 저장하고 경로를 반환
    prefix = _EXPERIMENT_PREFIX.get(experiment, experiment)
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    path = Path("results") / f"{prefix}_as_heard_{timestamp}.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(pcm_bytes)
    return path


def _save_client_metrics(metrics: MetricsClient, experiment: str) -> None:
    # 클라이언트 지표를 results/ 디렉토리에 JSON으로 저장
    records = metrics.get_records()
    if not records:
        return

    prefix = _EXPERIMENT_PREFIX.get(experiment, experiment)
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = Path("results") / f"{prefix}_client_{timestamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "experiment": experiment,
        "exported_at": datetime.now(tz=timezone.utc).isoformat(),
        "count": len(records),
        "results": [_build_client_record(r) for r in records],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"[dim]클라이언트 지표 저장: {path} ({len(records)}건)[/dim]")


# ========================================
# 메인 루프
# ========================================

async def main() -> None:
    loop = asyncio.get_running_loop()

    # /switch 입력 시 이 루프 처음으로 돌아와 파이프라인을 재선택
    while True:
        experiment = _select_experiment()

        input_manager = InputManager(experiment=experiment)
        metrics = MetricsClient()
        jitter_buffer = JitterBuffer()
        should_switch = False

        with AudioPlayer() as audio_player:
            try:
                # ----------------------------------------
                # 실험 1: 순차적 파이프라인
                # ----------------------------------------
                if experiment == "sequential":
                    async for result in input_manager.run():
                        # 매 사이클 시작 시 초기화하여 이전 사이클 starvation이 누적되지 않도록 함
                        jitter_buffer.reset()

                        # 실험 1: 패스스루 (버퍼링 없이 전체 오디오를 그대로 통과)
                        audio = jitter_buffer.pass_through(result.audio_bytes)

                        # AudioPlayer.play()는 블로킹 함수이므로 스레드 풀에 위임
                        play_duration = await loop.run_in_executor(None, audio_player.play, audio)

                        # 실험 1: pass_through는 drain()을 호출하지 않으므로 항상 0
                        buffer_starvation = jitter_buffer.starvation_count
                        stutter_duration = jitter_buffer.stutter_duration_ms

                        metrics.record(result, play_duration, buffer_starvation, stutter_duration)

                        # 클라이언트 지표를 서버에 역전송 → 서버 결과 파일에 통합
                        await input_manager.send_client_metrics({
                            "ttfa_client": (result.t_recv - result.t_send) * 1000,
                            "play_duration": play_duration,
                            "buffer_starvation": buffer_starvation,
                            "stutter_duration": stutter_duration,
                        })

                        _print_result(result, play_duration)

                # ----------------------------------------
                # 실험 2: 스트리밍 파이프라인
                # ----------------------------------------
                elif experiment == "streaming":
                    # 사이클 간 공유 상태 — 각 사이클 시작 시 StreamingCycleStart에서 갱신됨
                    current_prompt = ""
                    current_t_send = 0.0
                    current_t_first_recv = 0.0
                    total_bytes = 0
                    chunk_count = 0

                    async for event in input_manager.run_streaming():
                        if isinstance(event, StreamingCycleStart):
                            # 새 사이클 시작: 버퍼 초기화 및 오디오 스트림 오픈
                            jitter_buffer.reset()
                            audio_player.open_stream()
                            current_prompt = event.prompt
                            current_t_send = event.t_send
                            current_t_first_recv = event.t_send  # 첫 청크 수신 전 기본값
                            total_bytes = 0
                            chunk_count = 0

                        elif isinstance(event, StreamingCycleChunk):
                            # 청크 수신: JitterBuffer에 공급 → 즉시 소비 → PyAudio에 write
                            # t_first_recv 는 사이클 내 첫 청크 수신 시각으로 모든 청크에 동일
                            current_t_first_recv = event.t_first_recv
                            total_bytes += len(event.data)
                            chunk_count += 1

                            jitter_buffer.feed(event.data)
                            drained = jitter_buffer.drain()
                            if drained:
                                # feed_chunk()는 PyAudio stream.write() — 블로킹이므로 스레드 풀에 위임
                                await loop.run_in_executor(None, audio_player.feed_chunk, drained)

                        elif isinstance(event, StreamingCycleEnd):
                            # 사이클 종료: 잔여 버퍼 소진 후 스트림 닫기 및 지표 기록
                            # buffered_chunks 선확인: walrus 패턴은 빈 버퍼에서 drain()을 한 번 더 호출해 starvation false positive 유발
                            while jitter_buffer.buffered_chunks > 0:
                                remaining = jitter_buffer.drain()
                                if remaining is not None:
                                    await loop.run_in_executor(None, audio_player.feed_chunk, remaining)

                            buffer_starvation = audio_player.starvation_count
                            play_duration, stutter_duration = audio_player.close_stream()
                            # close_stream() 후 루프백 녹음 스레드가 종료되므로 여기서 수집
                            as_heard = audio_player.get_as_heard_bytes()
                            if as_heard:
                                _save_as_heard_wav(as_heard, experiment)

                            # 스트리밍의 경우 audio_bytes는 청크 단위로 재생됐으므로 b""로 표기
                            # t_recv = 첫 청크 수신 시각 → TTFA 계산에 사용
                            result = SessionResult(
                                prompt=current_prompt,
                                audio_bytes=b"",
                                t_send=current_t_send,
                                t_recv=current_t_first_recv,
                            )

                            metrics.record(result, play_duration, buffer_starvation, stutter_duration)

                            # as-heard PCM을 서버로 전송 (서버에서 WAV 저장 후 MOS 평가)
                            await input_manager.send_as_heard_pcm(as_heard)

                            await input_manager.send_client_metrics({
                                "ttfa_client": (result.t_recv - result.t_send) * 1000,
                                "play_duration": play_duration,
                                "buffer_starvation": buffer_starvation,
                                "stutter_duration": stutter_duration,
                            })

                            _print_result(
                                result,
                                play_duration,
                                total_bytes=total_bytes,
                                chunk_count=chunk_count,
                            )

                # ----------------------------------------
                # 실험 3: 적응형 스트리밍 파이프라인
                # ----------------------------------------
                elif experiment == "adaptive":
                    # streaming과 동일한 사이클 구조이나,
                    # StreamingCycleChunk 수신 직후 JitterBuffer 상태를 서버에 즉시 역전송한다.
                    # 서버는 5ms 타임아웃으로 이 피드백을 수신해 AdaptiveController를 갱신하고
                    # 다음 청크 크기 결정에 반영한다.
                    current_prompt = ""
                    current_t_send = 0.0
                    current_t_first_recv = 0.0
                    total_bytes = 0
                    chunk_count = 0

                    async for event in input_manager.run_streaming():
                        if isinstance(event, StreamingCycleStart):
                            jitter_buffer.reset()
                            audio_player.open_stream()
                            current_prompt = event.prompt
                            current_t_send = event.t_send
                            current_t_first_recv = event.t_send
                            total_bytes = 0
                            chunk_count = 0

                        elif isinstance(event, StreamingCycleChunk):
                            current_t_first_recv = event.t_first_recv
                            total_bytes += len(event.data)
                            chunk_count += 1

                            jitter_buffer.feed(event.data)
                            drained = jitter_buffer.drain()
                            if drained:
                                await loop.run_in_executor(None, audio_player.feed_chunk, drained)

                            # 청크 수신 직후 버퍼 상태 피드백 — 서버가 다음 청크 크기 조정에 활용
                            await input_manager.send_buffer_feedback(
                                buffer_level=jitter_buffer.buffered_chunks,
                                starvation_count=jitter_buffer.starvation_count,
                            )

                        elif isinstance(event, StreamingCycleEnd):
                            while jitter_buffer.buffered_chunks > 0:
                                remaining = jitter_buffer.drain()
                                if remaining is not None:
                                    await loop.run_in_executor(None, audio_player.feed_chunk, remaining)

                            buffer_starvation = audio_player.starvation_count
                            play_duration, stutter_duration = audio_player.close_stream()
                            # close_stream() 후 루프백 녹음 스레드가 종료되므로 여기서 수집
                            as_heard = audio_player.get_as_heard_bytes()
                            if as_heard:
                                _save_as_heard_wav(as_heard, experiment)

                            result = SessionResult(
                                prompt=current_prompt,
                                audio_bytes=b"",
                                t_send=current_t_send,
                                t_recv=current_t_first_recv,
                            )

                            metrics.record(result, play_duration, buffer_starvation, stutter_duration)

                            # as-heard PCM을 서버로 전송 (서버에서 WAV 저장 후 MOS 평가)
                            await input_manager.send_as_heard_pcm(as_heard)

                            await input_manager.send_client_metrics({
                                "ttfa_client": (result.t_recv - result.t_send) * 1000,
                                "play_duration": play_duration,
                                "buffer_starvation": buffer_starvation,
                                "stutter_duration": stutter_duration,
                            })

                            _print_result(
                                result,
                                play_duration,
                                total_bytes=total_bytes,
                                chunk_count=chunk_count,
                                starvation_count=buffer_starvation,
                            )

            except ExperimentSwitchRequested:
                # /switch 입력: 지표 저장 후 파이프라인 선택 메뉴로 복귀
                should_switch = True
                console.print("[yellow]파이프라인 전환 — 실험을 다시 선택해주세요.[/yellow]\n")

            finally:
                # 정상 종료 / 전환 / 예외 / Ctrl+C 어떤 경우든 지표를 저장
                _save_client_metrics(metrics, experiment=experiment)

        if not should_switch:
            break


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[dim]사용자 중단 — 클라이언트 종료[/dim]")
