import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["LLM_API_KEY"]
)

response = client.chat.completions.create(
    model="openai/gpt-4o-mini-2024-07-18",
    messages=[
        {"role": "user", "content": "너는 누구야? 간단한 자기소개 부탁해."}
    ]
)

print(response.choices[0].message.content)