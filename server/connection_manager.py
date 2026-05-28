# WebSocket 연결 관리 및 FastAPI 앱 설정

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query

from server import api_manager, experiment_pipeline, metrics_server
from server import logger as result_logger
from server.logger import ClientMetrics
from server.network_probe import NetworkSnapshot, snapshot as network_snapshot

logger = logging.getLogger(__name__)


# ========================================
# FastAPI 앱 인스턴스
# ========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _net_snapshot
    try:
        await api_manager.warmup()
    except Exception:
        logger.warning("API 워밍업 실패 — 서버는 계속 시작됩니다 (첫 요청 시 지연 발생 가능)")
    _net_snapshot = await network_snapshot()
    try:
        yield
    except asyncio.CancelledError:
        # Ctrl+C 종료 시 Windows asyncio에서 발생하는 CancelledError 정상 처리
        pass

# main.py가 이 객체를 import해서 uvicorn으로 전송
app = FastAPI(title="Capstone Pipeline Server", lifespan=lifespan)


# ========================================
# ConnectionManager 클래스
# ========================================

class ConnectionManager:
    #WebSocket 연결 생명주기와 데이터 전송 관리

    def __init__(self) -> None:
        # 현재 활성화된 WebSocket 집합
        # 실험 2, 3에서 여러 개의 클라이언트 브로드캐스트가 필요할 때 이 집합을 순회
        self._active: set[WebSocket] = set()

    async def connect(self, websocket: WebSocket) -> None:
        # HTTP -> WebSocket 업그레이드 수락 및 활성화된 집합에 등록
        await websocket.accept()         # 핸드셰이크 완료 (수락 전에는 데이터 송수신 불가)
        self._active.add(websocket)
        logger.info("클라이언트 연결: 현재 %d개", len(self._active))

    def disconnect(self, websocket: WebSocket) -> None:
        # 활성화된 집합에서 WebSocket을 제거
        self._active.discard(websocket)  # 이미 없는 경우에도 예외 없이 무시
        logger.info("클라이언트 해제: 현재 %d개", len(self._active))

    # ------------------------------
    # 실험 1 전용 전송 메서드
    # ------------------------------
    async def send_bytes(self, websocket: WebSocket, data: bytes) -> None:
        
        # 전체 오디오를 한 번에 전송
        await websocket.send_bytes(data)
        logger.info("오디오 전송 완료: %d bytes", len(data))


    # ------------------------------
    # 실험 2, 3 전용 전송 메서드
    # ------------------------------

    async def send_chunk(self, websocket: WebSocket, chunk: bytes) -> None:
        # 오디오 청크 1개를 WebSocket 바이너리 프레임으로 전송
        # run_streaming()이 yield하는 각 audio_chunk에 대해 반복 호출됨
        await websocket.send_bytes(chunk)
        logger.debug("오디오 청크 전송: %d bytes", len(chunk))

    async def send_end_signal(self, websocket: WebSocket) -> None:
        # 스트리밍 종료를 클라이언트에 알리는 빈 바이트 프레임 전송
        # 클라이언트는 len(data) == 0 조건으로 종료를 감지
        await websocket.send_bytes(b"")
        logger.info("스트리밍 종료 신호 전송 완료")

# 모듈 수준 싱글턴: main.py와 엔드포인트 핸들러가 동일 인스턴스 공유
manager = ConnectionManager()

# 서버 시작 시 한 번 수집한 네트워크 스냅샷 — 모든 export()에 공유
_net_snapshot: NetworkSnapshot | None = None


# ========================================
# WebSocket 엔드포인트 (/ws)
# ========================================

@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    experiment: str = Query(
        default="sequential",
        description="실험 타입: sequential | streaming | adaptive",
    ),
    token: str = Query(default=""),
) -> None:
    # 토큰 검증 — WS_TOKEN 환경변수가 설정된 경우에만 강제
    expected_token = os.environ.get("WS_TOKEN", "")
    if expected_token and token != expected_token:
        await websocket.close(code=1008)  # 1008: Policy Violation
        logger.warning("WebSocket 연결 거부: 유효하지 않은 토큰 (experiment=%s)", experiment)
        return

    # 클라이언트 연결을 수락하고 선택된 파이프라인 반복 실행
    await manager.connect(websocket)

    # 이 연결에서 수신한 클라이언트 지표를 누적 (export 시 서버 지표와 1:1 병합)
    client_metrics_list: list[ClientMetrics] = []

    try:
        # 하나의 연결에서 여러 프롬프트를 연속 처리할 수 있도록 루프로 구성
        while True:
            # 클라이언트가 전송한 텍스트 프롬프트를 수신
            # 수신 전까지 코루틴은 이벤트 루프를 블로킹하지 않고 대기
            prompt: str = await websocket.receive_text()
            logger.info("[%s] 프롬프트 수신 (%d자): %s…", experiment, len(prompt), prompt[:60])

            # ------------------------------
            # 실험 1: 순차적 파이프라인
            # ------------------------------
            if experiment == "sequential":
                result = await experiment_pipeline.run_sequential(prompt)
                
                metrics_server.record(
                    result.metrics,
                    experiment="sequential",
                    prompt=result.prompt,
                    llm_response=result.llm_response,
                    audio_path=result.audio_path,
                )
                
                await manager.send_bytes(websocket, result.audio_bytes)

                # 오디오 전송 후 클라이언트가 역전송한 지표 JSON을 수신
                try:
                    raw = await websocket.receive_text()
                    data = json.loads(raw)
                    client_metrics_list.append(ClientMetrics(**data))
                    logger.info("[sequential] 클라이언트 지표 수신 완료")
                except Exception:
                    logger.warning("[sequential] 클라이언트 지표 수신 실패 — null로 기록됩니다")

            # ------------------------------
            # 실험 2: 스트리밍 파이프라인
            # LLM 토큰 스트림을 실시간으로 TTS에 넘기고 청크 단위로 전송
            # ------------------------------
            elif experiment == "streaming":
                streaming_metrics = None
                audio_chunk_count = 0

                async for audio_chunk, metrics in experiment_pipeline.run_streaming(prompt):
                    if metrics is not None:
                        # 종료 신호: run_streaming()의 마지막 yield (b"", StreamingMetrics)
                        streaming_metrics = metrics
                    elif audio_chunk:
                        await manager.send_chunk(websocket, audio_chunk)
                        audio_chunk_count += 1

                # 클라이언트에 스트리밍 종료 알림
                await manager.send_end_signal(websocket)
                logger.info("[streaming] 오디오 청크 %d개 전송 완료", audio_chunk_count)

                # 성능 지표 기록
                if streaming_metrics is not None:
                    metrics_server.record(
                        streaming_metrics,
                        experiment="streaming",
                        prompt=prompt,
                        llm_response=streaming_metrics.llm_response,
                        audio_path=streaming_metrics.audio_path,
                    )

                # 오디오 전송 후 클라이언트가 역전송한 지표 JSON을 수신
                try:
                    raw = await websocket.receive_text()
                    data = json.loads(raw)
                    client_metrics_list.append(ClientMetrics(**data))
                    logger.info("[streaming] 클라이언트 지표 수신 완료")
                except Exception:
                    logger.warning("[streaming] 클라이언트 지표 수신 실패 — null로 기록됩니다")

            # ------------------------------
            # 실험 3: 적응형 스트리밍 파이프라인
            # run_adaptive()가 청크 크기를 동적으로 결정하며, 실험 2와 동일한 청크 전송 구조를 유지
            # Step 6에서 청크 전송 루프 내 논블로킹 버퍼 피드백 수신 로직이 추가될 예정
            # ------------------------------
            elif experiment == "adaptive":
                adaptive_metrics = None
                audio_chunk_count = 0

                async for audio_chunk, metrics in experiment_pipeline.run_adaptive(prompt):
                    if metrics is not None:
                        # 종료 신호: run_adaptive()의 마지막 yield (b"", AdaptiveMetrics)
                        adaptive_metrics = metrics
                    elif audio_chunk:
                        await manager.send_chunk(websocket, audio_chunk)
                        audio_chunk_count += 1

                await manager.send_end_signal(websocket)
                logger.info("[adaptive] 오디오 청크 %d개 전송 완료", audio_chunk_count)

                if adaptive_metrics is not None:
                    metrics_server.record(
                        adaptive_metrics,
                        experiment="adaptive",
                        prompt=prompt,
                        llm_response=adaptive_metrics.llm_response,
                        audio_path=adaptive_metrics.audio_path,
                    )

                # 오디오 전송 후 클라이언트가 역전송한 지표 JSON 수신
                try:
                    raw = await websocket.receive_text()
                    data = json.loads(raw)
                    client_metrics_list.append(ClientMetrics(**data))
                    logger.info("[adaptive] 클라이언트 지표 수신 완료")
                except Exception:
                    logger.warning("[adaptive] 클라이언트 지표 수신 실패 — null로 기록됩니다")

            else:
                logger.warning("알 수 없는 실험 타입: %s", experiment)
                await websocket.send_text(f"알 수 없는 실험 타입: {experiment}")

    except WebSocketDisconnect:
        # 클라이언트가 정상적으로 연결을 종료했을 때 (RFC 6455 close handshake)
        logger.info("[%s] 클라이언트 정상 종료", experiment)

    except Exception:
        # 파이프라인 오류 등 예기치 못한 예외: 스택 트레이스를 남기고 연결 정리
        logger.exception("[%s] WebSocket 처리 중 오류 발생", experiment)

    finally:
        # 어떤 형태의 종료든 반드시 활성화 집합에서 제거 (메모리 누수 방지)
        manager.disconnect(websocket)

        # export()는 MOS 평가(torch 추론)를 포함하는 동기 함수이므로
        # 이벤트 루프 블로킹을 막기 위해 스레드 풀에서 실행
        await asyncio.to_thread(
            result_logger.export,
            experiment=experiment,
            client_metrics=client_metrics_list or None,
            network_snapshot=_net_snapshot,
        )