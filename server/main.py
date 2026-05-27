# 서버 진입점

import logging
import os

import uvicorn
from dotenv import load_dotenv

load_dotenv()


# ========================================
# 로깅 설정
# ========================================

# 루트 로거 설정 시 모든 하위 모듈이 해당 설정 상속
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)


# ========================================
# 서버 실행
# ========================================

if __name__ == "__main__":
    # .env에 HOST / PORT가 없으면 기본값으로 폴백
    host: str = os.environ.get("HOST", "localhost")
    port: int = int(os.environ.get("PORT", "8765"))

    logger.info("서버 시작 — ws://%s:%d/ws", host, port)

    try:
        uvicorn.run(
            "server.connection_manager:app",
            host=host,
            port=port,
            log_level="info",
            # ws_ping_interval=None: 오디오 청크 전송 중 keepalive ping/pong이
            # 동시에 소켓에 쓰이면 websockets legacy drain waiter 충돌(AssertionError)이 발생함
            ws_ping_interval=None,
        )
    except (KeyboardInterrupt, SystemExit):
        logger.info("서버 종료")