import logging
import math
import struct
import sys
import tempfile
import wave
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from server.mos_evaluator import MOSEvaluator, MOSScore, save_pcm_as_wav


def _make_sine_wav(path: str, freq: int = 440, duration: float = 2.0, sample_rate: int = 24000) -> None:
    # 테스트용 사인파 WAV 파일 생성 (실제 TTS 음성 아님)
    n = int(sample_rate * duration)
    amplitude = int(32767 * 0.5)
    frames = struct.pack(f"<{n}h", *(
        int(amplitude * math.sin(2 * math.pi * freq * i / sample_rate))
        for i in range(n)
    ))
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(frames)


def test_save_pcm_as_wav() -> None:
    print("\n[1] save_pcm_as_wav 테스트")
    pcm = struct.pack("<10h", *range(10))
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name
    save_pcm_as_wav(pcm, path, sample_rate=24000)
    with wave.open(path, "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 24000
    print("  save_pcm_as_wav OK")
    Path(path).unlink()


def test_utmos(wav_path: str) -> float:
    print("\n[2] UTMOS 평가 테스트")
    evaluator = MOSEvaluator()
    score = evaluator._score_utmos(wav_path)
    print(f"  UTMOS 점수: {score:.3f}")
    assert 1.0 <= score <= 5.0, f"범위 초과: {score}"
    return score


def test_nisqa(wav_path: str) -> dict[str, float]:
    print("\n[3] NISQA 평가 테스트")
    weights = PROJECT_ROOT / "weights" / "nisqa.tar"
    if not weights.exists():
        print(f"  SKIP: 가중치 파일 없음 ({weights})")
        return {}
    evaluator = MOSEvaluator(nisqa_weights_path=weights)
    result = evaluator._score_nisqa(wav_path)
    print(f"  NISQA MOS:           {result['nisqa_mos']:.3f}")
    print(f"  NISQA Noisiness:     {result['nisqa_noisiness']:.3f}")
    print(f"  NISQA Coloration:    {result['nisqa_coloration']:.3f}")
    print(f"  NISQA Discontinuity: {result['nisqa_discontinuity']:.3f}")
    print(f"  NISQA Loudness:      {result['nisqa_loudness']:.3f}")
    return result


def test_evaluate_full(wav_path: str) -> None:
    weights = PROJECT_ROOT / "weights" / "nisqa.tar"
    if not weights.exists():
        print("\n[4] evaluate() 통합 테스트: SKIP (NISQA 가중치 없음)")
        return
    print("\n[4] evaluate() 통합 테스트")
    evaluator = MOSEvaluator(nisqa_weights_path=weights)
    score: MOSScore = evaluator.evaluate(wav_path)
    print(f"  MOSScore: {score}")


if __name__ == "__main__":
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name

    try:
        print("테스트용 사인파 WAV 생성 중...")
        _make_sine_wav(wav_path)
        print(f"생성 완료: {wav_path}")

        test_save_pcm_as_wav()
        test_utmos(wav_path)
        test_nisqa(wav_path)
        test_evaluate_full(wav_path)

        print("\n모든 테스트 완료")
    finally:
        Path(wav_path).unlink(missing_ok=True)
