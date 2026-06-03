# 오디오 재생

import logging
import threading
import time

import pyaudio

try:
    import numpy as np
    import soundcard as sc
    _LOOPBACK_AVAILABLE = True
except ImportError:
    _LOOPBACK_AVAILABLE = False

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
        # WASAPI 루프백 녹음 상태
        self._loopback_frames: list = []
        self._loopback_stop: threading.Event = threading.Event()
        self._loopback_thread: threading.Thread | None = None

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
        # 스트리밍 재생용 PyAudio 스트림을 열고 WASAPI 루프백 녹음을 동시에 시작
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=_CHANNELS,
            rate=_SAMPLE_RATE,
            output=True,
            frames_per_buffer=_CHUNK_SIZE,
        )
        self._play_start = 0.0  # 첫 feed_chunk() 호출 시 기록 — TTFA 대기 시간 제외
        self._total_frames = 0
        self._starvation_count = 0
        self._last_feed_time = 0.0      # 직전 feed_chunk() 호출 시각
        self._last_chunk_dur_s = 0.0    # 직전 청크의 재생 길이(초)
        self._loopback_frames = []
        self._loopback_stop = threading.Event()
        if _LOOPBACK_AVAILABLE:
            self._loopback_thread = threading.Thread(
                target=self._record_loopback, daemon=True
            )
            self._loopback_thread.start()
        else:
            self._loopback_thread = None
            logger.warning("soundcard 미설치 — 루프백 녹음 불가 (as-heard MOS 평가 불가)")
        logger.info("스트리밍 스트림 오픈")

    def _record_loopback(self) -> None:
        # WASAPI 루프백으로 시스템 오디오 출력을 캡처 — PyAudio 재생과 병렬로 실행
        # 마이크 불사용: 스피커로 출력되는 디지털 신호를 소프트웨어 수준에서 가로챔
        try:
            speaker = sc.default_speaker()
            loopback_mic = sc.get_microphone(speaker.name, include_loopback=True)
            with loopback_mic.recorder(samplerate=_SAMPLE_RATE, channels=_CHANNELS) as rec:
                while not self._loopback_stop.is_set():
                    data = rec.record(numframes=_CHUNK_SIZE)  # float32 ndarray (N, 1)
                    self._loopback_frames.append(data)
        except Exception as exc:
            logger.warning("루프백 녹음 실패: %s", exc)

    def feed_chunk(self, chunk: bytes) -> None:
        # 오디오 청크를 열린 스트림에 write
        if self._stream is None:
            logger.warning("feed_chunk 호출: 스트림이 열려 있지 않음")
            return
        now = time.perf_counter()
        if self._play_start == 0.0:
            self._play_start = now
        else:
            # 직전 청크 재생 시간보다 실제 갭이 길면 하드웨어 버퍼가 고갈된 것
            if now - self._last_feed_time > self._last_chunk_dur_s:
                self._starvation_count += 1
                logger.debug(
                    "버퍼 고갈 감지 — gap=%.1fms, 청크재생=%.1fms, count=%d",
                    (now - self._last_feed_time) * 1000,
                    self._last_chunk_dur_s * 1000,
                    self._starvation_count,
                )
        self._last_feed_time = now
        # 16-bit PCM = 2 bytes per sample, 채널 1개
        self._last_chunk_dur_s = (len(chunk) // 2) / _SAMPLE_RATE
        self._stream.write(chunk)
        self._total_frames += len(chunk) // 2

    @property
    def starvation_count(self) -> int:
        # feed_chunk() 간 갭 기반으로 측정한 버퍼 고갈 횟수
        return self._starvation_count

    def get_as_heard_bytes(self) -> bytes:
        # 루프백으로 캡처한 실제 재생 음성을 int16 PCM 바이트로 반환
        # close_stream() 호출 후에 사용해야 모든 녹음 데이터가 수집됨
        if not _LOOPBACK_AVAILABLE or not self._loopback_frames:
            return b""
        audio = np.concatenate(self._loopback_frames, axis=0)
        if audio.ndim > 1:
            audio = audio[:, 0]  # 첫 번째 채널만 사용 (mono)
        audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        return audio_int16.tobytes()

    def close_stream(self) -> tuple[float, float]:
        """스트림을 종료하고 (오디오_재생_시간ms, 끊김_시간ms) 튜플을 반환한다.

        끊김 시간 = 벽시계 경과 시간 - 오디오 프레임 기반 재생 시간.
        네트워크 대기로 PyAudio 하드웨어 버퍼가 고갈된 시간을 반영한다.
        """
        if self._stream is not None:
            self._stream.stop_stream()  # 하드웨어 버퍼 drain 완료까지 블로킹
            self._stream.close()
            self._stream = None
        # stop_stream() 이후에 측정해야 버퍼 잔량이 wall_ms에 포함됨
        wall_ms = (time.perf_counter() - self._play_start) * 1000 if self._play_start > 0.0 else 0.0
        # PyAudio 버퍼 소진 후 루프백 녹음 중단 및 스레드 회수
        self._loopback_stop.set()
        if self._loopback_thread is not None:
            self._loopback_thread.join(timeout=2.0)
            self._loopback_thread = None
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
