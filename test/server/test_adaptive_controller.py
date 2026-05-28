import sys
import time

sys.path.append(".")

from server.adaptive_controller import (
    DEFAULT_AUDIO_CHUNK_SIZE,
    DEFAULT_TEXT_CHUNK_SIZE,
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

check("decide_text_chunk_size", ctrl.decide_text_chunk_size(), DEFAULT_TEXT_CHUNK_SIZE)
check("decide_audio_chunk_size", ctrl.decide_audio_chunk_size(), DEFAULT_AUDIO_CHUNK_SIZE)
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
# 방법 A: rate는 on_tts_chunk_sent()에서 갱신 — on_llm_token()만으로는 0.0 유지
check("on_llm_token만으로는 rate = 0.0", snap["llm_token_rate"], 0.0)
check("rate=0 → DEFAULT 반환", ctrl.decide_text_chunk_size(), DEFAULT_TEXT_CHUNK_SIZE)



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
section("4. on_tts_audio_received (Step 1: 로깅만)")

ctrl = AdaptiveController()
ctrl.on_tts_audio_received(8192)
ctrl.on_tts_audio_received(4096)
snap = ctrl.snapshot()

check("tts_synthesis_rate == 0.0 (Step 1 미구현)", snap["tts_synthesis_rate"], 0.0)
check("decide_audio_chunk_size 변화 없음 (더미)", ctrl.decide_audio_chunk_size(), DEFAULT_AUDIO_CHUNK_SIZE)



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
section("8. LLM 토큰 속도 → 텍스트 청크 크기 제어 (Step 4)")

# rate → chunk size 결정 테이블 (state 직접 조작 — sleep 없음)
print("\n  [rate → chunk size 결정 테이블]")
print(f"  {'rate (t/s)':>12}  {'chunk size':>10}  비고")
print(f"  {'-'*55}")
ctrl_table = AdaptiveController()
for rate, note in [
    (0.0,  "초기 상태 — DEFAULT 반환"),
    (10.0, "느림 (rate_factor=0.5, clamp 하한)"),
    (20.0, "느림"),
    (30.0, "기준 (rate_factor=1.0)"),
    (40.0, "보통"),
    (60.0, "빠름"),
    (90.0, "빠름 (rate_factor=2.0, clamp 상한)"),
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

check("rate=0 → DEFAULT 반환", size_zero, DEFAULT_TEXT_CHUNK_SIZE)
check("빠른 rate(60) → 느린 rate(10) 보다 작은 청크", size_fast < size_slow, True)
check("size_fast in [MIN, MAX]", MIN_TEXT_CHUNK <= size_fast <= MAX_TEXT_CHUNK, True)
check("size_slow in [MIN, MAX]", MIN_TEXT_CHUNK <= size_slow <= MAX_TEXT_CHUNK, True)

# 청크 간격 기반 rate 측정 시뮬레이션
# 250ms 간격 × 5청크: 각 청크 10자 → chars/s ≈ 40, tokens/s ≈ 16 (40 / 2.5)
print("\n  [청크 간격 시뮬레이션 (250ms 간격 × 5 청크, 각 10자 → ≈ 16 t/s)]")
ctrl_sim = AdaptiveController()
for _ in range(5):
    ctrl_sim.on_tts_chunk_sent(text_len=10)
    time.sleep(0.25)
snap_sim = ctrl_sim.snapshot()
size_sim = ctrl_sim.decide_text_chunk_size()
print(f"  측정된 rate: {snap_sim['llm_token_rate']:.1f} t/s → chunk size: {size_sim}자")
check("시뮬레이션 rate > 0", snap_sim["llm_token_rate"] > 0, True)
check("시뮬레이션 chunk in [MIN, MAX]", MIN_TEXT_CHUNK <= size_sim <= MAX_TEXT_CHUNK, True)


print("\n완료")
