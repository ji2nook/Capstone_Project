# MOS 자동 평가 모듈
# UTMOS22: TTS 음성 자연스러움 (사람 음성과의 유사도)
# NISQA  : 음질 다차원 평가 (잡음 / 색조 / 불연속 / 음량)
# 두 모델을 함께 사용해 레이턴시 지표만으로는 포착하지 못하는 음질 차이를 정량화함

import logging
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf
import torch
import torchaudio.functional as F_audio

logger = logging.getLogger(__name__)

# UTMOS는 16kHz 단일 채널 입력을 요구
# ElevenLabs pcm_24000 포맷은 24kHz이므로 평가 전 리샘플링이 필요
_UTMOS_SR = 16000
_PCM_SR = 24000  # ElevenLabs pcm_24000


# ========================================
# 평가 결과 데이터 클래스
# ========================================

@dataclass
class MOSScore:
    # 단일 오디오 파일에 대한 UTMOS + NISQA 평가 결과
    # 모든 점수 범위: 1~5 (높을수록 좋음), 평가 실패 시 None
    utmos: float | None               # UTMOS22: TTS 자연스러움 — 사람 음성과의 유사도
    nisqa_mos: float | None           # NISQA: 종합 음질
    nisqa_noisiness: float | None     # NISQA NOI: 배경 잡음·왜곡 정도
    nisqa_coloration: float | None    # NISQA COL: 주파수 특성 열화 (코덱 압축 등)
    nisqa_discontinuity: float | None # NISQA DIS: 음성 단절·드롭아웃 — 스트리밍 실험의 핵심 지표
    nisqa_loudness: float | None      # NISQA LOUD: 음량 일관성


# ========================================
# MOS 평가기
# ========================================

class MOSEvaluator:
    # UTMOS는 최초 호출 시 가중치를 다운로드·캐시하고 이후 재사용
    # NISQA는 nisqaModel이 __init__에서 가중치와 데이터를 함께 로드하는 구조라
    # 파일마다 인스턴스를 새로 생성한다 (오프라인 배치 평가이므로 성능 허용 범위)
    _UTMOS_REPO = "tarepan/SpeechMOS:v1.2.0"
    _UTMOS_MODEL = "utmos22_strong"

    def __init__(self, nisqa_weights_path: str | Path = "weights/nisqa.tar") -> None:
        self._utmos: torch.nn.Module | None = None
        self._nisqa_weights = Path(nisqa_weights_path)

    # ----------------------------------------
    # UTMOS
    # ----------------------------------------

    def _load_utmos(self) -> None:
        if self._utmos is not None:
            return
        logger.info("UTMOS 모델 로딩 중...")
        self._utmos = torch.hub.load(
            self._UTMOS_REPO, self._UTMOS_MODEL, trust_repo=True
        )
        self._utmos.eval()
        logger.info("UTMOS 로딩 완료")

    def _score_utmos(self, wav_path: str) -> float:
        self._load_utmos()
        # torchaudio 2.11+은 Windows에서 torchcodec만 지원하므로 soundfile로 직접 로드
        wav_np, sr = sf.read(wav_path, dtype="float32")  # [-1, 1] 정규화 float32
        wav = torch.FloatTensor(wav_np).unsqueeze(0)      # [batch=1, time]
        if sr != _UTMOS_SR:
            wav = F_audio.resample(wav, sr, _UTMOS_SR)
        with torch.no_grad():
            score: torch.Tensor = self._utmos(wav, _UTMOS_SR)
        return float(score.mean().item())

    # ----------------------------------------
    # NISQA
    # ----------------------------------------

    def _score_nisqa(self, wav_path: str) -> dict[str, float]:
        if not self._nisqa_weights.exists():
            raise FileNotFoundError(
                f"NISQA 가중치 없음: {self._nisqa_weights.resolve()}\n"
                "NISQA 레포 weights/nisqa.tar 를 weights/ 폴더에 복사하세요"
            )
        from nisqa.NISQA_model import nisqaModel  # noqa: PLC0415

        logger.info("NISQA 평가 중: %s", wav_path)
        model = nisqaModel({
            "mode": "predict_file",
            "pretrained_model": str(self._nisqa_weights),
            "deg": wav_path,   # 평가 대상 파일 경로
            "output_dir": None,
            "ms_channel": None,
        })
        df = model.predict()
        logger.debug("NISQA 출력 컬럼: %s", list(df.columns))
        row = df.iloc[0]
        return {
            "nisqa_mos": float(row["mos_pred"]),
            "nisqa_noisiness": float(row["noi_pred"]),
            "nisqa_coloration": float(row["col_pred"]),
            "nisqa_discontinuity": float(row["dis_pred"]),
            "nisqa_loudness": float(row["loud_pred"]),
        }

    # ----------------------------------------
    # 공개 인터페이스
    # ----------------------------------------

    def evaluate(self, wav_path: str | Path) -> MOSScore:
        # UTMOS / NISQA를 독립적으로 평가 — 한쪽이 실패해도 나머지 결과는 보존
        wav_path = str(wav_path)

        utmos: float | None = None
        try:
            utmos = self._score_utmos(wav_path)
            logger.info("UTMOS: %.3f", utmos)
        except Exception:
            logger.warning("UTMOS 평가 실패: %s", wav_path, exc_info=True)

        nisqa: dict[str, float] | None = None
        try:
            nisqa = self._score_nisqa(wav_path)
            logger.info(
                "NISQA MOS=%.3f NOI=%.3f COL=%.3f DIS=%.3f LOUD=%.3f",
                nisqa["nisqa_mos"], nisqa["nisqa_noisiness"],
                nisqa["nisqa_coloration"], nisqa["nisqa_discontinuity"], nisqa["nisqa_loudness"],
            )
        except Exception:
            logger.warning("NISQA 평가 실패: %s", wav_path, exc_info=True)

        return MOSScore(
            utmos=utmos,
            nisqa_mos=nisqa["nisqa_mos"] if nisqa else None,
            nisqa_noisiness=nisqa["nisqa_noisiness"] if nisqa else None,
            nisqa_coloration=nisqa["nisqa_coloration"] if nisqa else None,
            nisqa_discontinuity=nisqa["nisqa_discontinuity"] if nisqa else None,
            nisqa_loudness=nisqa["nisqa_loudness"] if nisqa else None,
        )
