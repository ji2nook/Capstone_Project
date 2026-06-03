# 오디오 재생

import logging
import time

import pyaudio

logger = logging.getLogger(__name__)

# ElevenLabs pcm_24000 포맷 스펙
_SAMPLE_RATE = 24000
_CHANNELS = 1
_CHUNK_SIZE = 1024  # 한 번에 write할 샘플 수 (~42ms)

__all__ = ["AudioPlayer"]


# ========================================
# 오디오 재생
# ========================================

class AudioPlayer:
    # PCM 오디오 재생
    # 컨텍스트 매니저로 사용 권장
    
    def __init__(self) -> None:
        self._pa = pyaudio.PyAudio()
        # 스트리밍 재생 상태 (open_stream / feed_chunk / close_stream 주기 동안 유지)
        self._stream: pyaudio.Stream | None = None
        self._play_start: float = 0.0
        self._total_frames: int = 0

    def play(self, audio_bytes: bytes) -> float:
        # 전체 오디오를 재생하고 재생 시간(ms)을 반환
        # 블로킹 함수 - asyncio 환경에서는 run_in_executor로 호출
        stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=_CHANNELS,
            rate=_SAMPLE_RATE,
            output=True,
        )

        logger.info("재생 시작 — %d bytes", len(audio_bytes))
        t_start = time.perf_counter()

        # _CHUNK_SIZE씩 나눠 write
        # 한 번에 전체를 넘기면 버퍼 오버플로 가능성 존재
        for i in range(0, len(audio_bytes), _CHUNK_SIZE):
            stream.write(audio_bytes[i : i + _CHUNK_SIZE])

        play_duration = (time.perf_counter() - t_start) * 1000

        stream.stop_stream()
        stream.close()
        logger.info("재생 완료 — %.1fms", play_duration)

        return play_duration
    

    # ----------------------------------------
    # 스트리밍 재생 (실험 2·3)
    # ----------------------------------------

    def open_stream(self) -> None:
        # 스트리밍 재생용 PyAudio 스트림을 열고 내부 카운터를 초기화
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=_CHANNELS,
            rate=_SAMPLE_RATE,
            output=True,
            frames_per_buffer=_CHUNK_SIZE,
        )
        self._play_start = time.perf_counter()
        self._total_frames = 0
        logger.info("스트리밍 스트림 오픈")

    def feed_chunk(self, chunk: bytes) -> None:
        # 오디오 청크를 열린 스트림에 write하고 누적 프레임 수를 갱신
        if self._stream is None:
            logger.warning("feed_chunk 호출: 스트림이 열려 있지 않음")
            return
        self._stream.write(chunk)
        # 16-bit PCM = 2 bytes per sample, 채널 1개
        self._total_frames += len(chunk) // 2

    def close_stream(self) -> tuple[float, float]:
        """스트림을 종료하고 (오디오_재생_시간ms, 끊김_시간ms) 튜플을 반환한다.

        끊김 시간 = 벽시계 경과 시간 - 오디오 프레임 기반 재생 시간.
        네트워크 대기로 PyAudio 하드웨어 버퍼가 고갈된 시간을 반영한다.
        """
        wall_ms = (time.perf_counter() - self._play_start) * 1000
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        audio_ms = (self._total_frames / _SAMPLE_RATE) * 1000
        stutter_ms = max(0.0, wall_ms - audio_ms)
        logger.info(
            "스트리밍 스트림 종료 — 오디오: %.1fms, 벽시계: %.1fms, 끊김: %.1fms",
            audio_ms, wall_ms, stutter_ms,
        )
        return audio_ms, stutter_ms

    def close(self) -> None:
        # PyAudio 인스턴스 해체
        self._pa.terminate()

    def __enter__(self) -> "AudioPlayer":
        return self

    def __exit__(self, *_) -> None:
        self.close()
