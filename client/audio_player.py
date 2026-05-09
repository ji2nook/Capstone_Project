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
    

    # --- 실험 2·3 추가 예정 ---
    # 스트림을 열어두고 청크가 들어올 때마다 feed_chunk()로 점진적 재생
    # def open_stream(self) -> None: ...
    # def feed_chunk(self, chunk: bytes) -> None: ...
    # def close_stream(self) -> float: ...  # 재생 완료 후 play_duration 반환

    def close(self) -> None:
        # PyAudio 인스턴스 해체
        self._pa.terminate()

    def __enter__(self) -> "AudioPlayer":
        return self

    def __exit__(self, *_) -> None:
        self.close()
