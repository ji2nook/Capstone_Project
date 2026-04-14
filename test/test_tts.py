import os

from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

import pyaudio

load_dotenv()

client = ElevenLabs(
    api_key=os.environ["TTS_API_KEY"]
)

audio = client.text_to_speech.convert(
    text="TTS 테스트입니다. 음성을 확인하세요.",
    voice_id="JBFqnCBsd6RMkjVDRZzb",
    model_id="eleven_flash_v2_5",
    output_format="pcm_24000"
)

audio_bytes = b"".join(audio)

p = pyaudio.PyAudio()
stream = p.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=24000,
    output=True
)

stream.write(audio_bytes)

stream.stop_stream()
stream.close()
p.terminate()