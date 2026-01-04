from openai import OpenAI
from config.settings import OPENAI_API_KEY, LLM_MODEL

def call_llm(prompt: str) -> str:
    client = OpenAI(api_key=OPENAI_API_KEY)

    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are a precise assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content
