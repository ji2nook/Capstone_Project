# 실험 시작 전 네트워크 환경 스냅샷 수집

import asyncio
import logging
import subprocess
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_LLM_API_HOST = "openrouter.ai"
_TTS_API_HOST = "api.elevenlabs.io"
_PING_COUNT = 10


@dataclass
class NetworkSnapshot:
    timestamp: float
    download_mbps: float | None
    upload_mbps: float | None
    ping_llm_ms: float | None   # openrouter.ai 평균 RTT
    ping_tts_ms: float | None   # api.elevenlabs.io 평균 RTT


def _ping_host(host: str, count: int = _PING_COUNT) -> float | None:
    # Windows ping으로 대상 호스트까지의 평균 RTT(ms) 측정
    try:
        result = subprocess.run(
            ["ping", "-n", str(count), host],
            capture_output=True,
            text=True,
            timeout=count + 10,
        )
        for line in result.stdout.splitlines():
            # Windows 출력: "평균 = 23ms" 또는 "Average = 23ms"
            if "Average" in line or "평균" in line:
                rtt_str = line.split("=")[-1].strip().replace("ms", "").strip()
                return float(rtt_str)
    except Exception:
        logger.warning("ping 측정 실패: %s", host)
    return None


def _measure_bandwidth() -> tuple[float | None, float | None]:
    # speedtest-cli로 다운로드/업로드 속도(Mbps) 측정
    try:
        import speedtest
        # secure=True: HTTPS로 연결 — HTTP를 차단하는 네트워크 환경에서 필요
        st = speedtest.Speedtest(secure=True)
        st.get_best_server()
        download = round(st.download() / 1e6, 2)
        upload = round(st.upload() / 1e6, 2)
        return download, upload
    except ImportError:
        logger.warning("speedtest-cli 패키지가 없습니다. `pip install speedtest-cli` 로 설치하세요.")
    except Exception as e:
        logger.warning("speedtest 측정 실패: %s", e)
    return None, None


async def snapshot() -> NetworkSnapshot:
    # 네트워크 스냅샷 전체 수집
    # 세 가지 측정 동시 실행
    logger.info("네트워크 환경 측정 시작 (ping × 2 + speedtest 병렬 실행)...")
    ts = time.time()

    loop = asyncio.get_running_loop()
    ping_llm_fut = loop.run_in_executor(None, _ping_host, _LLM_API_HOST)
    ping_tts_fut = loop.run_in_executor(None, _ping_host, _TTS_API_HOST)
    bw_fut = loop.run_in_executor(None, _measure_bandwidth)

    ping_llm, ping_tts, bw = await asyncio.gather(ping_llm_fut, ping_tts_fut, bw_fut)
    download, upload = bw

    snap = NetworkSnapshot(
        timestamp=ts,
        download_mbps=download,
        upload_mbps=upload,
        ping_llm_ms=ping_llm,
        ping_tts_ms=ping_tts,
    )
    logger.info(
        "네트워크 측정 완료 — 다운: %s Mbps, 업: %s Mbps, LLM RTT: %s ms, TTS RTT: %s ms",
        download,
        upload,
        ping_llm,
        ping_tts,
    )
    return snap
