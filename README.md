# 실시간 게임 몰입감 향상을 위한 LLM-TTS 기반 적응형 스트리밍 파이프라인 설계 및 성능 분석

**LLM-TTS 연동 환경에서 순차적 처리 · 스트리밍 · 적응형(ABR) 세 가지 파이프라인을 구현하고 지연 및 음질을 정량 비교한 프로젝트**

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.135-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)

---

## 📋 목차

- [프로젝트 소개](#-프로젝트-소개)
- [시스템 아키텍처](#️-시스템-아키텍처)
- [실험 설계](#-실험-설계)
- [핵심 알고리즘](#-핵심-알고리즘)
- [실험 결과](#-실험-결과)
- [저장소 구조](#-저장소-구조)
- [실행 방법](#️-실행-방법)

---

## 🚀 프로젝트 소개

NVIDIA ACE, Ubisoft NEO NPC 등 게임 산업에서 LLM 기반 NPC 대화 시스템이 주목받고 있습니다. 이러한 시스템에서 LLM-TTS 파이프라인의 첫 음성 도달 시간(TTFA)의 지연은 플레이어의 몰입감을 직접 좌우합니다. 고정 청크 스트리밍은 순차적 처리 방식에 비해 지연 시간을 단축하지만, 청크가 과도하게 작으면 버퍼 고갈로 인해 음성이 끊기고 과도하게 크면 TTFA가 늘어난다는 트레이드오프가 존재합니다.

이 연구는 LLM 토큰 생성 속도, TTS 합성 속도, 버퍼 레벨, 네트워크 RTT 등 런타임 지표를 **ABR(Adaptive Bitrate) 가중치 합산 방식**으로 통합하여 텍스트 및 오디오 청크 크기를 실시간으로 결정하는 적응형 스트리밍 알고리즘을 설계 및 구현하고, 순차적 처리 및 고정 청크 스트리밍 방식과 성능을 정량적으로 비교했습니다.

### 프로젝트 정보

- **과목**: 정보통신 종합 설게 및 실습
- **기간**: 2026.03.01. ~ 2026.10.02.

---

## 🏗️ 시스템 아키텍처

![시스템 아키텍처 - 서버](assets/architecture_server.png)
![시스템 아키텍처 - 클라이언트](assets/architecture_client.png)

| 단계 | 실험 Ⅰ (Sequential) | 실험 Ⅱ (Streaming) | 실험 Ⅲ (Adaptive) |
|------|--------------------|--------------------|-------------------|
| LLM | 전체 대기 | 토큰 스트림 | 토큰 스트림 + 속도 추적 |
| 텍스트 청크 | — | 고정 80자 | 동적 **40~200자** |
| 오디오 청크 | 1회 전송 | 고정 12,000 bytes | 동적 **12,000~24,000 bytes** |
| 피드백 루프 | 없음 | 없음 | buffer_level + starvation_count |

---

## 🧪 실험 설계

### 실험 환경

| 구분 | 서버 | 클라이언트 |
|------|------|------------|
| **위치** | Azure VM (Ubuntu 24.04 LTS) | 로컬 머신 (Windows 11 Home) |
| **하드웨어** | Standard B2als v2, RAM 4 GB | Intel Core i5-1135G7, RAM 8 GB |
| **역할** | FastAPI, LLM/TTS API, tc netem 대역폭 제한 | 오디오 재생 (WASAPI), 루프백 녹음, MOS 평가 |

- **LLM**: OpenAI GPT-4o-mini (`openai/gpt-4o-mini-2024-07-18`, OpenRouter 경유)  
- **TTS**: ElevenLabs Flash v2.5 (`pcm-24000`, PCM 24 kHz · 16-bit signed · mono)  
- **네트워크 조건**: tc netem으로 6조건 (제한 없음/1,000/800/600/400/200 kbps)  
- **반복 횟수**: 3종 × 6조건 × 2길이(100자/500자) × 5회 = 180회

### 실험 Ⅰ — 순차적 처리 (Baseline)

```
사용자 입력 → [await] LLM 전체 완료 → [await] TTS 전체 완료 → 단일 전송
```

### 실험 Ⅱ — 고정 청크 스트리밍

```
LLM 토큰 스트림 → llm_text_chunker (최대 80자)
    → TTS 병렬 태스크 → tts_audio_chunker (무음 경계, 버퍼 상한 48,000 bytes)
        → 청크 반복 전송 → JitterBuffer (pre-buffer 5청크) → 재생
```

### 실험 Ⅲ — 적응형 스트리밍 (ABR)

실험 Ⅱ 파이프라인에 `AdaptiveController` 추가한 형태. 매 청크마다 아래 공식으로 크기를 결정합니다.

```
score = Σ(가중치 × 정규화 지표)  ∈ [0, 1]
chunk_size = MIN + score × (MAX − MIN)
← 클라이언트 buffer_feedback {"buffer_level": N, "starvation_count": M}
```

| 지표 | 텍스트 가중치 | 오디오 가중치 | 방향 |
|------|:---:|:---:|:---:|
| 버퍼 잔량 | 30 % | 40 % | 정비례 |
| TTS 합성 속도 | 25 % | 30 % | 정비례 |
| LLM 토큰 속도 | 20 % | — | 정비례 |
| 네트워크 RTT | 15 % | 20 % | 반비례 |
| 전송 처리량 | 10 % | 10 % | 정비례 |

---

## 🔬 핵심 알고리즘

### 하이브리드 텍스트 청킹 (`llm_text_chunker`)

① 문장부호 뒤 공백·줄바꿈 감지 → 즉시 yield  
② 버퍼 ≥ `max_chars` → overflow flush  
Callable 주입으로 실험 Ⅲ의 동적 크기 적용 가능

### 무음 경계 오디오 청킹 (`tts_audio_chunker`)

① PCM 진폭 절댓값 < 300인 샘플이 50 ms(2,400 bytes) 이상 연속 → 무음 경계에서 yield  
② 탐색 구간: 버퍼 앞 25 % 건너뜀 (최소 청크 길이 보장)  
③ 버퍼 ≥ `max_chunk_bytes` → overflow flush

### Producer-Consumer 순서 보장

TTS 태스크들이 병렬 실행되어도 `asyncio.Queue` 파이프라인이 오디오 전송 순서를 보장

---

## 📊 실험 결과

### TTFA 비교 (ms, 평균 ± 표준편차)

![실험 결과 - TTFA](assets/result_ttfa.png)

### 버퍼 고갈 지속 시간 비교 (ms, 평균 ± 표준편차)

![실험 결과 - 버퍼 고갈 지속 시간](assets/result_buffer_starve_time.png)

### NISQA 불연속성 비교 (점, 평균 ± 표준편차)

![실험 결과 - NISQA](assets/result_nisqa.png)

### 주요 발견

- **TTFA 단축**: 800 kbps 긴 답변에서 스트리밍/적응형이 순차적 처리 대비 **약 95 % 이상** 감소 (29,027 ms → 1,364/1,472 ms)
- **안정성**: 1,000 kbps 긴 답변에서 고정 스트리밍의 TTFA 표준편차 ±3,911 ms, 적응형은 **±333 ms**로 변동성 억제
- **음질 보존**: 400 kbps 이상 전 조건에서 세 방식 모두 NISQA 4.6점 이상 유지. 기본 조건 적응형: 종합 4.92, 불연속성 4.69 (고정 스트리밍 4.70, 4.26 대비 순차 처리 수준에 근접)
- **저대역폭 한계**: 400 kbps 긴 답변에서 적응형 고갈 지속 시간 756 ms (고정 스트리밍 182 ms 대비 역전). 버퍼를 채우기 위해 청크 크기를 늘릴 때 단일 청크 전송 자체가 지연되는 현상이 원인으로, 버퍼 고갈 감지 시 즉각적 강제 축소 로직 도입이 향후 과제

---

## 📂 저장소 구조

```
📁 Capstone_Project/
├── 📁 server/
│   ├── main.py                  # 서버 진입점 (uvicorn + tc 대역폭 설정)
│   ├── connection_manager.py    # FastAPI 앱 + WebSocket 엔드포인트
│   ├── api_manager.py           # LLM/TTS 클라이언트 싱글턴
│   ├── experiment_pipeline.py   # run_sequential / run_streaming / run_adaptive
│   ├── streaming_controller.py  # llm_text_chunker · tts_audio_chunker
│   ├── adaptive_controller.py   # ABR 제어 엔진 (AdaptiveController)
│   ├── experiment_config.py     # 청크 크기·가중치·임계값 상수
│   ├── metrics_server.py        # in-memory 지표 저장소
│   ├── logger.py                # JSON 출력 + MOS 평가 통합
│   ├── network_probe.py         # ping + tc netem 적용
│   ├── audio_utils.py           # PCM → WAV 변환
│   └── mos_evaluator.py         # UTMOS22 + NISQA 래퍼
│
├── 📁 client/
│   ├── main.py                  # 클라이언트 진입점 (실험 선택)
│   ├── input_manager.py         # CLI 입력 + 버퍼 피드백 전송
│   ├── audio_player.py          # PCM 재생 + WASAPI 루프백 녹음
│   ├── jitter_buffer.py         # FIFO 버퍼 + starvation 감지
│   └── metrics_client.py        # TTFA / 끊김 측정
│
├── 📁 tools/
│   └── evaluate_mos.py          # 오프라인 MOS 배치 평가
│
├── 📁 nisqa/                       # NISQA 모델 소스
├── 📁 weights/                     # 사전 학습 모델 가중치
├── 📁 results/                     # 실험 결과 JSON + WAV
└── 📁 test/                        # pytest 테스트
```

---

## 🛠️ 실행 방법

```bash
git clone https://github.com/ji2nook/Adaptive_Streaming_LLM_TTS_for_Games.git
cd Adaptive_Streaming_LLM_TTS_for_Games

python -m venv venv
venv\Scripts\activate        # Windows

pip install -r requirements.txt
cp .env.example .env         # LLM_API_KEY, TTS_API_KEY 입력
```

```bash
# 터미널 1 — 서버 (API warmup + 네트워크 측정 자동 수행)
python -m server.main

# 터미널 2 — 클라이언트 (1: 순차 / 2: 스트리밍 / 3: 적응형)
python -m client.main

# MOS 오프라인 평가 (results/audio/ 의 WAV 파일 평가)
python tools/evaluate_mos.py
```
