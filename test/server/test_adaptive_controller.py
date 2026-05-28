import sys
import time

sys.path.append(".")

from server.adaptive_controller import (
    _AUDIO_WEIGHTS,
    _TEXT_WEIGHTS,
    MIN_AUDIO_CHUNK,
    MAX_AUDIO_CHUNK,
    MIN_TEXT_CHUNK,
    MAX_TEXT_CHUNK,
    AdaptiveController,
)


def section(title: str) -> None:
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print("=" * 50)


def check(label: str, actual, expected) -> None:
    status = "OK" if actual == expected else "FAIL"
    print(f"  [{status}] {label}: {actual!r}  (expected {expected!r})")



# ========================================
# 1. 초기 상태
# ========================================
section("1. 초기 상태")

ctrl = AdaptiveController()
snap = ctrl.snapshot()

check("decide_text_chunk_size  in [MIN, MAX]", MIN_TEXT_CHUNK  <= ctrl.decide_text_chunk_size()  <= MAX_TEXT_CHUNK,  True)
check("decide_audio_chunk_size in [MIN, MAX]", MIN_AUDIO_CHUNK <= ctrl.decide_audio_chunk_size() <= MAX_AUDIO_CHUNK, True)
check("llm_tokens_generated", snap["llm_tokens_generated"], 0)
check("buffer_level", snap["buffer_level"], 0)
check("snapshot이 dict인지", isinstance(snap, dict), True)



# ========================================
# 2. on_llm_token
# ========================================
section("2. on_llm_token")

ctrl = AdaptiveController()
ctrl.on_llm_token("안녕")
ctrl.on_llm_token("하세요")
ctrl.on_llm_token(".")
snap = ctrl.snapshot()

check("llm_tokens_generated == 3", snap["llm_tokens_generated"], 3)
check("llm_last_token_ts > 0", snap["llm_last_token_ts"] > 0, True)
# 방법 A: rate는 on_tts_chunk_sent()에서 갱신 -> on_llm_token()만으로는 0.0 유지
check("on_llm_token만으로는 rate = 0.0", snap["llm_token_rate"], 0.0)
check("rate=0 → 범위 내 반환", MIN_TEXT_CHUNK <= ctrl.decide_text_chunk_size() <= MAX_TEXT_CHUNK, True)



# ========================================
# 3. on_tts_chunk_sent
# ========================================
section("3. on_tts_chunk_sent")

ctrl = AdaptiveController()
ctrl.on_tts_chunk_sent(text_len=25)
snap = ctrl.snapshot()

check("tts_last_chunk_size == 25", snap["tts_last_chunk_size"], 25)
check("tts_last_send_ts > 0", snap["tts_last_send_ts"] > 0, True)


# ========================================
# 4. on_tts_audio_received
# ========================================
section("4. on_tts_audio_received (Step 5: 슬라이딩 윈도우 rate 갱신)")

# 샘플 1개 → 워밍업 구간이므로 rate = 0.0
ctrl = AdaptiveController()
ctrl.on_tts_audio_received(8192)
snap = ctrl.snapshot()
check("샘플 1개 → rate = 0.0 (워밍업)", snap["tts_synthesis_rate"], 0.0)

# 샘플 2개 이상 → rate > 0
time.sleep(0.02)
ctrl.on_tts_audio_received(8192)
snap = ctrl.snapshot()
check("샘플 2개 이상 → tts_synthesis_rate > 0", snap["tts_synthesis_rate"] > 0, True)
print(f"  tts_synthesis_rate: {snap['tts_synthesis_rate']:.0f} B/s")



# ========================================
# 5. on_ws_chunk_sent 
# ========================================
section("5. on_ws_chunk_sent")

ctrl = AdaptiveController()
t1 = time.perf_counter()
ctrl.on_ws_chunk_sent(chunk_bytes=8192, send_ts=t1)

time.sleep(0.05)  # 50ms 간격

t2 = time.perf_counter()
ctrl.on_ws_chunk_sent(chunk_bytes=8192, send_ts=t2)
snap = ctrl.snapshot()

print(f"  last_send_interval_ms: {snap['last_send_interval_ms']:.1f} ms  (약 50ms 기대)")
print(f"  throughput_bps:        {snap['throughput_bps']:.0f} bps")
check("last_send_interval_ms > 0", snap["last_send_interval_ms"] > 0, True)
check("throughput_bps > 0", snap["throughput_bps"] > 0, True)



# ========================================
# 6. on_client_feedback
# ========================================
section("6. on_client_feedback")

ctrl = AdaptiveController()
ctrl.on_client_feedback(buffer_level=5, starvation_count=2)
snap = ctrl.snapshot()

check("buffer_level == 5", snap["buffer_level"], 5)
check("buffer_starvation_count == 2", snap["buffer_starvation_count"], 2)



# ========================================
# 7. snapshot 필드 완전성
# ========================================
section("7. snapshot 필드 완전성")

ctrl = AdaptiveController()
snap = ctrl.snapshot()

expected_keys = {
    "llm_token_rate", "llm_tokens_generated", "llm_last_token_ts",
    "tts_synthesis_rate", "tts_last_send_ts", "tts_last_chunk_size",
    "buffer_level", "buffer_starvation_count",
    "latency_ms", "throughput_bps", "last_send_interval_ms",
    "text_chunk_size", "audio_chunk_size",
}
missing = expected_keys - snap.keys()
check("누락 필드 없음", missing, set())

# ========================================
# 8. LLM 토큰 속도 → 텍스트 청크 크기 제어 (Step 4)
# ========================================
section("8. LLM 토큰 속도 → 텍스트 청크 크기 제어 (Step 6 통합 후)")

# rate → chunk size 결정 테이블 (state 직접 조작 -> sleep 없음)
# Step 6: llm rate는 역관계 — 빠를수록 norm 낮음 → score 낮음 → 작은 청크
print("\n  [rate → chunk size 결정 테이블]")
print(f"  {'rate (t/s)':>12}  {'chunk size':>10}  비고")
print(f"  {'-'*55}")
ctrl_table = AdaptiveController()
for rate, note in [
    (0.0,  "초기 상태 (norm=0.5 중립)"),
    (10.0, "느림 (norm=0.83, 큰 청크 선호)"),
    (20.0, "느림 (norm=0.67)"),
    (30.0, "중간 (norm=0.5, 중립)"),
    (40.0, "보통 (norm=0.33)"),
    (60.0, "빠름 (norm=0.0, 작은 청크 선호)"),
    (90.0, "빠름 (norm=0.0, 상한 clamp)"),
]:
    ctrl_table._state.llm_token_rate = rate
    size = ctrl_table.decide_text_chunk_size()
    print(f"  {rate:>10.1f} t/s  {size:>8d}자  {note}")

# 방향성 검증
ctrl_table._state.llm_token_rate = 60.0
size_fast = ctrl_table.decide_text_chunk_size()
ctrl_table._state.llm_token_rate = 10.0
size_slow = ctrl_table.decide_text_chunk_size()
ctrl_table._state.llm_token_rate = 0.0
size_zero = ctrl_table.decide_text_chunk_size()

check("rate=0 → 범위 내 반환", MIN_TEXT_CHUNK <= size_zero <= MAX_TEXT_CHUNK, True)
check("빠른 rate(60) → 느린 rate(10) 보다 작은 청크", size_fast < size_slow, True)
check("size_fast in [MIN, MAX]", MIN_TEXT_CHUNK <= size_fast <= MAX_TEXT_CHUNK, True)
check("size_slow in [MIN, MAX]", MIN_TEXT_CHUNK <= size_slow <= MAX_TEXT_CHUNK, True)

# 청크 간격 기반 rate 측정 시뮬레이션
# 250ms 간격 × 5청크: 각 청크 10자 → chars/s ~ 40, tokens/s ~ 16 (40 / 2.5)
print("\n  [청크 간격 시뮬레이션 (250ms 간격 × 5 청크, 각 10자 → ~ 16 t/s)]")
ctrl_sim = AdaptiveController()
for _ in range(5):
    ctrl_sim.on_tts_chunk_sent(text_len=10)
    time.sleep(0.25)
snap_sim = ctrl_sim.snapshot()
size_sim = ctrl_sim.decide_text_chunk_size()
print(f"  측정된 rate: {snap_sim['llm_token_rate']:.1f} t/s → chunk size: {size_sim}자")
check("시뮬레이션 rate > 0", snap_sim["llm_token_rate"] > 0, True)
check("시뮬레이션 chunk in [MIN, MAX]", MIN_TEXT_CHUNK <= size_sim <= MAX_TEXT_CHUNK, True)


# ========================================
# 9. TTS 합성 속도 → 오디오 청크 크기 제어 (Step 5)
# ========================================
section("9. TTS 합성 속도 → 오디오 청크 크기 제어 (Step 6 통합 후)")

# rate → chunk size 결정 테이블 (state 직접 조작 -> sleep 없음)
# Step 6: tts rate는 정관계 — 빠를수록 norm 높음 → score 높음 → 큰 청크
print("\n  [rate → chunk size 결정 테이블]")
print(f"  {'rate (B/s)':>14}  {'chunk size':>10}  비고")
print(f"  {'-'*60}")
ctrl_table = AdaptiveController()
for rate, note in [
    (0.0,    "초기 상태 (norm=0.5 중립)"),
    (10000,  "느림 (norm=0.17, 작은 청크)"),
    (20000,  "느림 (norm=0.33)"),
    (30000,  "중간 (norm=0.5, 중립)"),
    (45000,  "보통 (norm=0.75)"),
    (60000,  "빠름 (norm=1.0, 큰 청크 선호)"),
    (90000,  "빠름 (norm=1.0, 상한 clamp)"),
]:
    ctrl_table._state.tts_synthesis_rate = rate
    size = ctrl_table.decide_audio_chunk_size()
    print(f"  {rate:>12.0f} B/s  {size:>8d}  {note}")

# 방향성 검증
ctrl_table._state.tts_synthesis_rate = 60000.0
size_fast = ctrl_table.decide_audio_chunk_size()

ctrl_table._state.tts_synthesis_rate = 10000.0
size_slow = ctrl_table.decide_audio_chunk_size()

ctrl_table._state.tts_synthesis_rate = 0.0
size_zero = ctrl_table.decide_audio_chunk_size()

check("rate=0 → 범위 내 반환", MIN_AUDIO_CHUNK <= size_zero <= MAX_AUDIO_CHUNK, True)
check("빠른 rate(60000) → 느린 rate(10000) 보다 큰 청크", size_fast > size_slow, True)
check("size_fast in [MIN, MAX]", MIN_AUDIO_CHUNK <= size_fast <= MAX_AUDIO_CHUNK, True)
check("size_slow in [MIN, MAX]", MIN_AUDIO_CHUNK <= size_slow <= MAX_AUDIO_CHUNK, True)

# PCM 16-bit(2바이트) 정렬 검증 -> 모든 rate 값에서 짝수여야 함
print("\n  [PCM 2바이트 정렬 검증]")
all_even = True
for rate in [0.0, 10000, 30000, 60000, 90000]:
    ctrl_table._state.tts_synthesis_rate = rate
    s = ctrl_table.decide_audio_chunk_size()
    if s % 2 != 0:
        all_even = False
        print(f"  FAIL: rate={rate} → size={s} (홀수)")
check("모든 rate에서 chunk size가 짝수 (PCM 정렬)", all_even, True)

# 슬라이딩 윈도우 시뮬레이션 -> 실제 타이밍 기반 rate 측정
# 20ms 간격 × 6회, 각 8192 bytes → ~ 8192 / 0.02 ~ 40960 B/s (슬라이딩 윈도우 특성상 근사)
print("\n  [슬라이딩 윈도우 시뮬레이션 (20ms 간격 × 6회, 각 8192 bytes → ~ 40960 B/s)]")
ctrl_sim = AdaptiveController()
for _ in range(6):
    ctrl_sim.on_tts_audio_received(8192)
    time.sleep(0.02)
snap_sim = ctrl_sim.snapshot()
size_sim = ctrl_sim.decide_audio_chunk_size()
print(f"  측정된 rate: {snap_sim['tts_synthesis_rate']:.0f} B/s → chunk size: {size_sim} bytes")
check("시뮬레이션 rate > 0", snap_sim["tts_synthesis_rate"] > 0, True)
check("시뮬레이션 chunk in [MIN, MAX]", MIN_AUDIO_CHUNK <= size_sim <= MAX_AUDIO_CHUNK, True)
check("시뮬레이션 chunk size 짝수", size_sim % 2 == 0, True)

# 윈도우 만료 검증 -> 1초 이상 경과 후 rate가 감소 또는 0으로 수렴하는지 확인
print("\n  [윈도우 만료 검증 (1.1초 대기 후 rate 재측정)]")
rate_before = ctrl_sim._state.tts_synthesis_rate
time.sleep(1.1)
# 새 샘플 없이 rate를 재계산하려면 record()를 호출해야 함 -> 1바이트 더미 수신
ctrl_sim.on_tts_audio_received(1)
rate_after = ctrl_sim._state.tts_synthesis_rate
print(f"  대기 전: {rate_before:.0f} B/s  →  대기 후: {rate_after:.0f} B/s")
check("1초 경과 후 rate가 감소 (윈도우 만료)", rate_after < rate_before, True)


# ========================================
# 10. 정규화 함수 방향성 검증 (Step 6)
# ========================================
section("10. 정규화 함수 방향성 검증 (Step 6)")

ctrl = AdaptiveController()

# 각 함수의 경계값 및 방향성만 확인
check("norm_llm: rate=0   → 0.5 중립",        ctrl._norm_llm_rate(0.0),      0.5)
check("norm_llm: rate=60  → 0.0 (최대속도)",  ctrl._norm_llm_rate(60.0),     0.0)
check("norm_llm: 느릴수록 높음 (역관계)",      ctrl._norm_llm_rate(10.0) > ctrl._norm_llm_rate(50.0), True)

check("norm_tts: rate=0      → 0.5 중립",     ctrl._norm_tts_rate(0.0),      0.5)
check("norm_tts: rate=60000  → 1.0",          ctrl._norm_tts_rate(60_000.0), 1.0)
check("norm_tts: 빠를수록 높음 (정관계)",      ctrl._norm_tts_rate(60_000.0) > ctrl._norm_tts_rate(15_000.0), True)

check("norm_buf: level=0  → 0.0 (고갈)",      ctrl._norm_buffer(0),          0.0)
check("norm_buf: level=10 → 1.0 (여유)",       ctrl._norm_buffer(10),         1.0)

check("norm_lat: latency=0   → 0.5 중립",     ctrl._norm_latency(0.0),       0.5)
check("norm_lat: latency=200 → 0.0 (최대지연)", ctrl._norm_latency(200.0),   0.0)
check("norm_lat: 낮을수록 높음 (역관계)",       ctrl._norm_latency(20.0) > ctrl._norm_latency(180.0), True)

check("norm_tput: tput=0      → 0.5 중립",    ctrl._norm_throughput(0.0),    0.5)
check("norm_tput: tput=100000 → 1.0",         ctrl._norm_throughput(100_000.0), 1.0)


# ========================================
# 11. 버퍼 고갈 긴급 처리 (Step 6)
# ========================================
section("11. 버퍼 고갈 긴급 처리 (Step 6)")

# starvation 신규 발생 → text MIN 반환
ctrl = AdaptiveController()
ctrl.on_client_feedback(buffer_level=0, starvation_count=1)
check("starvation count=1 → MIN_TEXT_CHUNK",         ctrl.decide_text_chunk_size(), MIN_TEXT_CHUNK)
# 동일 count 재호출 → 이미 처리됨 → MIN 아님
check("동일 count 재호출 → MIN 아님 (중복 처리 방지)", ctrl.decide_text_chunk_size() > MIN_TEXT_CHUNK, True)
# count 추가 증가 → 다시 MIN 반환
ctrl.on_client_feedback(buffer_level=0, starvation_count=2)
check("starvation count=2 → 다시 MIN_TEXT_CHUNK",    ctrl.decide_text_chunk_size(), MIN_TEXT_CHUNK)

# buffer_level==0 + starvation>0 → audio MIN 반환
ctrl_a = AdaptiveController()
ctrl_a.on_client_feedback(buffer_level=0, starvation_count=1)
check("level=0 + starvation=1 → MIN_AUDIO_CHUNK",    ctrl_a.decide_audio_chunk_size(), MIN_AUDIO_CHUNK)
# buffer_level>0 이면 정상 계산
ctrl_a.on_client_feedback(buffer_level=4, starvation_count=1)
check("level=4 + starvation=1 → MIN_AUDIO 아님",     ctrl_a.decide_audio_chunk_size() > MIN_AUDIO_CHUNK, True)


# ========================================
# 12. 가중치 통합 시나리오 (Step 6)
# ========================================
section("12. 가중치 통합 시나리오 (Step 6)")

def _make(**overrides) -> AdaptiveController:
    c = AdaptiveController()
    for k, v in overrides.items():
        setattr(c._state, k, v)
    return c

scenarios = [
    ("콜드 스타트",         {}),
    ("빠른LLM + 느린TTS",   dict(llm_token_rate=50.0, tts_synthesis_rate=10_000, buffer_level=5)),
    ("느린LLM + 빠른TTS",   dict(llm_token_rate=10.0, tts_synthesis_rate=55_000, buffer_level=9)),
    ("높은 Latency",        dict(llm_token_rate=30.0, tts_synthesis_rate=30_000,
                                 buffer_level=5, latency_ms=180.0, throughput_bps=15_000)),
    ("이상적인 환경",        dict(llm_token_rate=8.0, tts_synthesis_rate=60_000,
                                 buffer_level=10, latency_ms=10.0, throughput_bps=100_000)),
]

print(f"\n  {'시나리오':<20} {'text(자)':>8} {'audio(B)':>9}")
print(f"  {'-'*40}")
results = []
for label, overrides in scenarios:
    c = _make(**overrides)
    tsz = c.decide_text_chunk_size()
    asz = c.decide_audio_chunk_size()
    results.append((label, tsz, asz))
    print(f"  {label:<20} {tsz:>8d} {asz:>9d}")

ideal   = next(r for r in results if r[0] == "이상적인 환경")
bad_env = next(r for r in results if r[0] == "빠른LLM + 느린TTS")
check("이상적 환경 text_chunk  > 열악한 환경", ideal[1] > bad_env[1], True)
check("이상적 환경 audio_chunk > 열악한 환경", ideal[2] > bad_env[2], True)
check("모든 text  chunk가 [MIN, MAX] 내",
      all(MIN_TEXT_CHUNK  <= r[1] <= MAX_TEXT_CHUNK  for r in results), True)
check("모든 audio chunk가 [MIN, MAX] 내",
      all(MIN_AUDIO_CHUNK <= r[2] <= MAX_AUDIO_CHUNK for r in results), True)


print("\n완료")
