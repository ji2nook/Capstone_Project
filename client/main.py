# 클라이언트 진입점

import asyncio
import json
import logging
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
from client.input_manager import InputManager, SessionResult
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

# 현재 구현된 실험만 포함 — 실험 2·3 완성 시 추가
_IMPLEMENTED: frozenset[str] = frozenset({"sequential"})

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

def _print_result(result: SessionResult, play_duration: float) -> None:
    # 한 사이클 완료 후 지표를 테이블로 출력
    ttfa_ms = (result.t_recv - result.t_send) * 1000

    table = Table(box=box.ROUNDED, show_header=False, padding=(0, 2))
    table.add_column(style="dim", min_width=22)
    table.add_column(justify="right")
    table.add_row("TTFA (클라이언트)", f"[cyan]{ttfa_ms:.0f} ms[/cyan]")
    table.add_row("재생 시간",         f"[cyan]{play_duration:.0f} ms[/cyan]")
    table.add_row("오디오 크기",        f"[cyan]{len(result.audio_bytes):,} bytes[/cyan]")

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
    experiment = _select_experiment()

    input_manager = InputManager(experiment=experiment)
    metrics = MetricsClient()
    jitter_buffer = JitterBuffer()
    loop = asyncio.get_running_loop()

    with AudioPlayer() as audio_player:
        try:
            async for result in input_manager.run():

                # 실험 1: 패스스루 | 실험 2·3: jitter_buffer.feed() / drain()으로 교체
                audio = jitter_buffer.pass_through(result.audio_bytes)

                # AudioPlayer.play()는 블로킹 함수이므로 스레드 풀에 위임
                play_duration = await loop.run_in_executor(None, audio_player.play, audio)

                # 실험 1: buffer_starvation, stutter_duration은 기본값 0
                # 실험 2·3: jitter_buffer.starvation_count, audio_player 측정값 전달
                metrics.record(result, play_duration)

                # 클라이언트 지표를 서버에 역전송 → 서버 결과 파일에 통합
                await input_manager.send_client_metrics({
                    "ttfa_client": (result.t_recv - result.t_send) * 1000,
                    "play_duration": play_duration,
                    "buffer_starvation": 0,
                    "stutter_duration": 0.0,
                })

                _print_result(result, play_duration)

        finally:
            # 정상 종료 / 예외 / Ctrl+C 어떤 경우든 지표를 저장
            _save_client_metrics(metrics, experiment=experiment)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[dim]사용자 중단 — 클라이언트 종료[/dim]")
