# WebSocket 연결 관리 및 FastAPI 앱 설정

import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query

from server import experiment_pipeline, metrics_server
from server import logger as result_logger

logger = logging.getLogger(__name__)


# ========================================
# FastAPI 앱 인스턴스
# ========================================

# main.py가 이 객체를 import해서 uvicorn으로 전송
app = FastAPI(title="Capstone Pipeline Server")


# ========================================
# ConnectionManager 클래스
# ========================================

class ConnectionManager:
    """WebSocket 연결 생명주기와 데이터 전송을 관리한다."""

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
    # 현재 미구현
    # sent_chunk()를 반복 호출하는 식으로 구현
    # ------------------------------
    async def send_chunk(self, websocket: WebSocket, chunk: bytes) -> None:
        
        # 오디오 청크를 스트리밍 방식으로 전송
        # 실험 2에서 LLM 토큰 스트림 -> TTS 청크 파이프라인 구현 완료하면 오디오 청크를 클라이언트에 순차 전달
        raise NotImplementedError("실험 2, 3에서 구현 예정")

    async def broadcast_bytes(self, data: bytes) -> None:
        
        # 모든 활성 클라이언트에 동일 데이터를 동시에 전송
        # 실험 3에서 여러 개의 클라이언트 시나리오가 필요할 경우에 사용
        raise NotImplementedError("실험 3에서 구현 예정")


# 모듈 수준 싱글턴: main.py와 엔드포인트 핸들러가 동일 인스턴스 공유
manager = ConnectionManager()


# ========================================
# WebSocket 엔드포인트 (/ws)
# ========================================

@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    # 실험 파라미터 선택 (ws://host/ws?experiment=sequential)
    # 동일한 서버 바이너리로 실험 파이프라인을 전환하여 비교할 수 있도록 설계
    experiment: str = Query(
        default="sequential",
        description="실험 타입: sequential | streaming | adaptive",
    ),
) -> None:
    
    # 클라이언트 연결을 수락하고 선택된 파이프라인 반복 실행
    await manager.connect(websocket)

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
                
                metrics_server.record(result.metrics, experiment="sequential")
                
                await manager.send_bytes(websocket, result.audio_bytes)

            # ------------------------------
            # 실험 2: 스트리밍 파이프라인 (미구현)
            # LLM 토큰 스트림을 실시간으로 TTS에 넘기고 청크 단위로 전송
            # ------------------------------
            elif experiment == "streaming":
                await websocket.send_text("streaming 파이프라인은 실험 2에서 구현 예정")

            # ------------------------------
            # 실험 3: 적응형 스트리밍 파이프라인 (미구현)
            # 네트워크 상태에 따라 청크 크기와 버퍼 전략을 동적으로 조절
            # ------------------------------
            elif experiment == "adaptive":
                await websocket.send_text("adaptive 파이프라인은 실험 3에서 구현 예정")

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

        # 이번 연결에서 누적된 서버 지표를 파일로 내보냄
        # 실험 2·3에서는 여기서 client_metrics도 함께 전달하도록 수정 예정
        result_logger.export(experiment=experiment)