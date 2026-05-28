import sys
import time

sys.path.append(".")

from server.adaptive_controller import (
    DEFAULT_AUDIO_CHUNK_SIZE,
    DEFAULT_TEXT_CHUNK_SIZE,
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
check("decide_text_chunk_size 변화 없음 (더미)", ctrl.decide_text_chunk_size(), DEFAULT_TEXT_CHUNK_SIZE)



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

print("\n완료")
