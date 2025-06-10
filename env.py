from os import getenv
from random import choice
LLM_API_KEY = getenv("LLM_API_KEY")
LLM_API_KEYS = getenv("LLM_API_KEYS").split()
GOOGLE_API_KEY = getenv("GOOGLE_API_KEY")
CX = getenv("CX")
IMSEARCH_API_KEY = getenv("IMSEARCH_API_KEY")
HOST_URL = getenv("HOST_URL")

# print(LLM_API_KEYS)

def get_llm_api_key():
    return choice(LLM_API_KEYS)
