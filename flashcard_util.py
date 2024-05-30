import lang
from asyncreq_util import async_post
import requests
import json

from imgsearch import search_img
from re import match, sub

from env import LLM_API_KEY

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
    _r = requests.post(
        url="https://api.awanllm.com/v1/chat/completions",
        headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {LLM_API_KEY}'},
        data=json.dumps({
            "model": "Meta-Llama-3-8B-Instruct",
            "messages": [
                {"role": "user", "content": f"tl;dr in {l}: {content}"}
            ],
            "presence_penalty":0.3,
            "temperature":0.55
        })
    )
    _summary = _r.json()
    # print(_summary)
    return _summary['choices'][0]['message']['content'].split('\n',1)[-1].strip()

async def fetch_img_for_words(words: list[str]):
    print("fetching images...")
    _img_link = [search_img(r) for r in words]
    return [(word,img) for (word, img) in zip(words, _img_link)]

async def get_definitions_from_words(words: list[str], summary: str = ""):
    print("running inferrence")
    _r = requests.post(
        url="https://api.awanllm.com/v1/chat/completions",
        headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {LLM_API_KEY}'},
        data=json.dumps({
            "model": "Meta-Llama-3-8B-Instruct",
            "messages": [
                {"role": "user", "content": f"{summary}. Based on this paragraph and your knowledge, give easy-to-understand definitions for the following words: {'; '.join(words)}"}
            ],
            "presence_penalty":0.3,
            "temperature":0.55
        })
    )

    # print(_r.json()['choices'][0]['message']['content'].split('\n'))
    print(_r.json()['choices'][0]['message']['content'].split('\n'))
    
    rets = []
    for _x in _r.json()['choices'][0]['message']['content'].split('\n'):
        try:
            k, v = _x.split(':')
            k = k.replace('*','').strip()
            k = remove_numbering(k)
            v = v.strip()
            if (v != ''):
                rets.append((k, v))
        except:
            continue
    return rets[:-1]

def get_imgs_from_words(words: list[str]):
    pass

def classify_words(words: list[str], deep: bool = False):
    pass