# 실험 2 (스트리밍 처리) 파이프라인과 실험 3 (적응형 스트리밍 처리) 파이프라인의 파라미터 조정

# ========================================
# 실험 2 — 고정 청크 크기
# ========================================

TEXT_CHUNK_SIZE: int = 80       # LLM 텍스트 청크 크기 (문자 수, 세션 내 고정)
AUDIO_CHUNK_SIZE: int = 12000    # TTS 오디오 청크 크기 (bytes, 세션 내 고정)


# ========================================
# 실험 3 — 적응형 청크 범위 및 초기값
# ========================================

# 초기값을 실험 2 고정값과 동일하게 맞춰 두 실험의 베이스라인을 통일
DEFAULT_TEXT_CHUNK_SIZE: int = 80
DEFAULT_AUDIO_CHUNK_SIZE: int = 12000

MIN_TEXT_CHUNK: int = 40
MAX_TEXT_CHUNK: int = 200
MIN_AUDIO_CHUNK: int = 12000
MAX_AUDIO_CHUNK: int = 24000


# ========================================
# 실험 3 — ABR 버퍼 임계값
# ========================================

# 버퍼 수위(청크 수)에 따른 구간 경계
# 긴급 구간 (이하): 텍스트 최대 청크 → TTS 호출당 오디오 최대 확보
#                   오디오 최소 청크 → 클라이언트에 빠르게 전달
# 안정 구간 (이상): 텍스트 최소 청크 → 반응성 우선
#                   오디오 최대 청크 → 전송 효율 우선
BUFFER_CRITICAL: int = 2
BUFFER_STABLE:   int = 6


# ========================================
# 실험 3 — ABR 가중치 기반 제어 파라미터
# ========================================
#
# score = Σ(가중치 × 정규화된 지표값),  score ∈ [0, 1]
# size  = MIN + score × (MAX - MIN)
#
# 정비례 지표: 값이 클수록 score ↑ → 청크 크기 ↑
# 반비례 지표: 값이 클수록 score ↓ → 청크 크기 ↓  (정규화 시 1 - norm 적용)

# ── 정규화 기준값 ─────────────────────────────────────────────────────────────
# 각 지표를 [0, 1]로 정규화할 때 사용하는 상한값.
# 이 값 이상이면 정규화 결과 = 1.0으로 클리핑된다.

# 버퍼 수위: BUFFER_STABLE 이상이면 최대 기여
BUFFER_LEVEL_MAX_NORM: int = BUFFER_STABLE           # = 6 청크
# TTS 합성 속도: 16kHz·16-bit·mono 실시간 재생 속도를 기준으로 설정
TTS_SYNTHESIS_RATE_MAX_BPS: float = 32_000.0         # bytes/s
# LLM 토큰 생성 속도
LLM_RATE_MAX: float = 30.0                           # tokens/s
# 네트워크 지연 (반비례 — 낮을수록 청크 ↑)
LATENCY_MAX_MS: float = 300.0                        # ms
# 전송 처리량
THROUGHPUT_MAX_BPS: float = 128_000.0                # bytes/s ≈ 1 Mbps

# ── 텍스트 청크 결정 가중치 (합 = 1.0) ───────────────────────────────────────
TEXT_W_BUFFER:     float = 0.30   # 버퍼 레벨          (정비례)
TEXT_W_TTS_RATE:   float = 0.25   # TTS 음성 생성 속도  (정비례)
TEXT_W_LLM_RATE:   float = 0.20   # LLM 토큰 생성 속도  (정비례)
TEXT_W_LATENCY:    float = 0.15   # Latency            (반비례)
TEXT_W_THROUGHPUT: float = 0.10   # Throughput         (정비례)

# ── 오디오 청크 결정 가중치 (합 = 1.0) ───────────────────────────────────────
AUDIO_W_BUFFER:     float = 0.40  # 버퍼 레벨          (정비례)
AUDIO_W_TTS_RATE:   float = 0.30  # TTS 음성 생성 속도  (정비례)
AUDIO_W_LATENCY:    float = 0.20  # Latency            (반비례)
AUDIO_W_THROUGHPUT: float = 0.10  # Throughput         (정비례)
