import asyncio
import logging

import pyaudio
from dotenv import load_dotenv

load_dotenv()

# 터미널에 로그 출력
logging.basicConfig(level=logging.DEBUG)

import sys
sys.path.append(".")

from server.api_manager import (
    generate_full_response,
    synthesize_full_audio,
    generate_streaming_response,
    synthesize_streaming_audio,
)

TEST_PROMPT = "너는 누구야? 간단한 자기소개 부탁해."


async def main():
    # ===== 실험 1: LLM 전체 응답 =====
    print("===== 실험 1: LLM 전체 응답 =====")
    text = await generate_full_response(TEST_PROMPT)
    print(f"응답: {text}\n")

    # ===== 실험 1: TTS 전체 오디오 =====
    print("===== 실험 1: TTS 전체 오디오 =====")
    audio = await synthesize_full_audio(text)
    print(f"오디오 크기: {len(audio)} bytes\n")

    with open("test_output.pcm", "wb") as f:
        f.write(audio)
    print("test_output.pcm으로 오디오 파일이 생성됨\n")

    print("===== 실험 1: 오디오 재생 =====")
    pa = pyaudio.PyAudio()
    pa_stream = pa.open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)
    pa_stream.write(audio)
    pa_stream.stop_stream()
    pa_stream.close()
    pa.terminate()
    print("재생 완료\n")

    # ===== 실험 2: LLM 스트리밍 응답 =====
    print("===== 실험 2: LLM 스트리밍 응답 =====")
    tokens = []
    async for token in generate_streaming_response(TEST_PROMPT):
        print(token, end="", flush=True)
        tokens.append(token)
    streaming_text = "".join(tokens)
    print(f"\n\n총 토큰 수: {len(tokens)}\n")

    # ===== 실험 2: TTS 스트리밍 오디오 =====
    print("===== 실험 2: TTS 스트리밍 오디오 =====")
    audio_chunks = []
    async for chunk in synthesize_streaming_audio(streaming_text):
        audio_chunks.append(chunk)
        print(f"청크 {len(audio_chunks)}: {len(chunk)} bytes")
    total_bytes = sum(len(c) for c in audio_chunks)
    print(f"총 {total_bytes} bytes, {len(audio_chunks)}개 청크\n")

    print("===== 실험 2: 스트리밍 오디오 재생 =====")
    pa = pyaudio.PyAudio()
    pa_stream = pa.open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)
    for chunk in audio_chunks:
        pa_stream.write(chunk)
    pa_stream.stop_stream()
    pa_stream.close()
    pa.terminate()
    print("재생 완료")


asyncio.run(main())
