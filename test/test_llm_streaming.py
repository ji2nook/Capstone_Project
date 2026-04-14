import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["LLM_API_KEY"]
)

stream = client.chat.completions.create(
    model="openai/gpt-4o-mini-2024-07-18",
    messages=[
        {"role": "user", "content": "너는 누구야? 간단한 자기소개 부탁해."}
    ],
    stream=True,
    extra_body={
        "reasoning": {
            "exclude": True  # reasoning 토큰 제외
        }
    }
)

for chunk in stream:
    delta = chunk.choices[0].delta.content
    
    if delta is not None:
        print(delta, end="", flush=True)

print()