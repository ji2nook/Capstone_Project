import asyncio
import logging

import pyaudio
from dotenv import load_dotenv

load_dotenv()

# 터미널에 로그 출력
logging.basicConfig(level=logging.DEBUG)

# server 디렉토리를 경로에 추가
import sys
sys.path.append(".")

from server.api_manager import generate_full_response, synthesize_full_audio

async def main():
    print("===== LLM 테스트 =====")
    text = await generate_full_response("너는 누구야? 간단한 자기소개 부탁해.")
    print(f"응답: {text}\n")
    
    print("===== TTS 테스트 =====")
    audio = await synthesize_full_audio(text)
    print(f"오디오 크기: {len(audio)} bytes\n")
    
    # 오디오 파일 저장
    with open("test_output.pcm", "wb") as f:
        f.write(audio)
    print("test_output.pcm으로 오디오 파일이 생성됨\n")

    # 오디오 재생
    print("===== 오디오 재생 =====")
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=24000,
        output=True
    )
    stream.write(audio)
    stream.stop_stream()
    stream.close()
    p.terminate()
    print("재생 완료")
    
asyncio.run(main())