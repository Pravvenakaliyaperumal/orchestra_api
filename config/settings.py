import os
from dotenv import load_dotenv

load_dotenv()

def get_env(name: str, default: str | None = None) -> str:
    val = os.getenv(name, default)
    if val is None or val == "":
        raise ValueError(f"Missing required env var: {name}")
    return val

def get_int_env(name: str, default: int) -> int:
    v = os.getenv(name)
    return int(v) if v else default

OPENAI_API_KEY = get_env("OPENAI_API_KEY", "")
LLM_MODEL = get_env("LLM_MODEL", "gpt-4o-mini")
TOP_K = get_int_env("TOP_K", 3)
