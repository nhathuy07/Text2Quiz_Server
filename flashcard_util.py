import lang
# import requests
import json
import httpx
#import aiofiles
from imgsearch import search_img
from re import match, sub, findall, fullmatch
from re import split as re_split
from env import HOST_URL
from PIL import Image
from env import get_llm_api_key

def remove_numbering(text):
  """Removes list-like numberings (e.g., 1., 2. etc.) from a string.

  Args:
      text: The string to process.

  Returns:
      The string with numberings removed.
  """
  pattern = r"\d+\.\s?"  # Matches one or more digits followed by a dot and optional space
  return sub(pattern, "", text)

async def tldr(content, l=lang.VI_VN):
    async with httpx.AsyncClient() as client:
        _r = await client.post(
            url="https://api.awanllm.com/v1/chat/completions",
            headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {get_llm_api_key()}'},
            data=json.dumps({
                "model": "Meta-Llama-3-8B-Instruct",
                "messages": [
                    {"role": "user", "content": f"summarise in {l}: {content}"}
                ],
                "presence_penalty":0.3,
                "temperature":0.55
            }),
            timeout=None,
        )
    _summary = _r.json()
    # print(_summary)
    return _summary['choices'][0]['message']['content'].split('\n',1)[-1].strip()


async def fetch_img_for_words(words: list[str], __url_prefix=None):
    print("fetching images...")
    _img_link = [search_img(r) for r in words]
    return [(word,img) for (word, img) in zip(words, _img_link)]

async def get_definitions_from_words(words: list[str], summary: str = "", lang: str = lang.VI_VN):
    print("running inferrence")
    resp_form = r"{word}: {definition}"
    async with httpx.AsyncClient() as client:
        _r = await client.post(
            url="https://api.awanllm.com/v1/chat/completions",
            headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {get_llm_api_key()}'},
            data=json.dumps({
                "model": "Meta-Llama-3-8B-Instruct",
                "messages": [
                    {"role": "user", "content": f"{summary}. List concise definitions for the following words: {'; '.join(words)}.\nReturn a numbered list in the following format: {resp_form}\nDO NOT include the keywords inside their respective definition. Use {lang}."}
                ],
                "presence_penalty":0.3,
                "temperature":0.55
            }),
            timeout=None
        )

    # print(_r.json()['choices'][0]['message']['content'].split('\n'))

    out = _r.json()['choices'][0]['message']['content']
    rets = [x.replace('*', '').strip() for x in re_split(r"\r?\n|- |: |\> |\> ", out) if len(x.strip())]
    print(rets)
    rets_filtered = []
    require_numbering = True
    for e in rets:

        if bool(fullmatch(r"^\d+\.?\)? ?.+", e)) == require_numbering:
            if require_numbering:
                rets_filtered.append(sub(r"^\d+[\.?\)? ?]", "", e).strip())
            else:
                rets_filtered.append(e)
            require_numbering = not require_numbering
    
    return [(rets_filtered[i], rets_filtered[i+1]) for i in range(0, len(rets_filtered), 2)]

def get_imgs_from_words(words: list[str]):
    pass

def classify_words(words: list[str], deep: bool = False):
    pass
