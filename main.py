from starlette.responses import PlainTextResponse, JSONResponse, FileResponse
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.exceptions import HTTPException

from gensim.models import KeyedVectors
"""Prompt templates for LLM"""
from env import LLM_API_KEY, get_llm_api_key
import prompt
from time import time

from re import split, match

from PIL import Image

import requests

import json

import pypandoc
import cv2
from io import BytesIO
import numpy as np
import os

import pytesseract
import lang

import httpx

from secrets import SystemRandom

from random import randint, sample

from enum import Enum
from re import sub, findall, escape

from functools import partial

import redis.asyncio as redis

import asyncio
import subprocess

pool = redis.ConnectionPool.from_url("redis://localhost")
r = redis.Redis.from_pool(pool)

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s %(levelname)s] %(message)s")

# Define a logger for your application (optional)
app_logger = logging.getLogger(__name__)

from flashcard_util import tldr, get_definitions_from_words, fetch_img_for_words

class QType(Enum):
    WH = 0
    STMT = 3
    FILL = 6

class APIGuardMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)

    async def dispatch(self, request, call_next):

        # Get current client url and client IP address
        client_url = request.url.path
        client_ip_addr = request.client.host
        
        # IP-based rate limitation
        async with r.pipeline(transaction=True) as pipeline:
            try:
                res = await pipeline.get(client_ip_addr).execute()
                res = int(res[-1])
            except:
                res = None
    
            if res == None:
                # lim = 25 if client_url.endswith('text2quiz-three.vercel.app') else 5
                # lim = 10 if client_url.endswith('localhost:8100') else 5

                app_logger.info(client_url)
                
                ok = await pipeline.set(client_ip_addr, 60).execute()
                await r.expire(client_ip_addr, 60)
            elif res > 0:
                ok = await pipeline.set(client_ip_addr, res-1).execute()
            else:
                raise HTTPException(status_code=429, detail="This IP address is rate-limited")
        
        # process the request and get the response    
        response = await call_next(request)
        return response


sys_random = SystemRandom()

# TODO: Change to environment variable in prod.

#pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Admin\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

async def __internal_tmp_w(id, content:any):
    try:
        async with r.pipeline(transaction=True) as pipeline:
            ok = await pipeline.set(id, json.dumps(content).encode("utf-8")).execute()
            await r.expire(id, 600)

        return ok
    except Exception as e:
        app_logger.info(e)

async def __internal_tmp_r(id):
    try:
        async with r.pipeline(transaction=True) as pipeline:
            res = await (pipeline.get(id).execute())
            if res[-1] == None:
                return [None, None, None]
            res = res[-1].decode("utf-8")
        return json.loads(res)
    except Exception as e:
        app_logger.info(e)
        return [None,None,None]

async def __internal_tmp_d(id):
    async with r.pipeline(transaction=True) as pipeline:
        res = await (pipeline.delete(id).execute())

async def __mltest(request):
    pass

async def __save_temp(request):
    file_id = sys_random.randbytes(20).hex()
    content = ""
    # async with request.form(max_fields=3) as form:
    form = await request.json()
    content = form['content']
    title = form['title']
    keywords = form['keywords']

    await __internal_tmp_w(file_id, [title, content, keywords])
    
    print(file_id)
    return PlainTextResponse(file_id, 200)
    
async def __get_temp(request, entry = 1):
    return JSONResponse(await __internal_tmp_r(request.path_params['id']))

async def __remove_temp(request):
    try:
        __internal_tmp_d(request.path_params['id'])
    except:
        return PlainTextResponse("", 500)
    
    return PlainTextResponse("", 200)

async def __convert_text(input, type_out="plain", type_in="html"):
    if (not input):
        app_logger.info("__convert_text: nothing to convert!")
        return ""
    # Create a subprocess
    process = await asyncio.create_subprocess_exec(
        # command to execute
        'pandoc', '-f', type_in, '-t', type_out,
        stdout=asyncio.subprocess.PIPE,  # redirect stdout
        stderr=asyncio.subprocess.STDOUT,
        stdin=asyncio.subprocess.PIPE,# redirect stderr
    )
    stdout, _ = await process.communicate(input=input.encode())
    #print("CONVERTED: ",stdout.decode("utf-8"))
    return (stdout.decode("utf-8"))

async def __convert_file(fname_in, type_out="plain"):
    proc = await asyncio.create_subprocess_exec(
        'pandoc', '-i', fname_in, '-t', type_out,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        stdin=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    return stdout.decode("utf-8")

async def __ocr(im, file_id):
    # Perform image preprocessing
    processed_im = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2GRAY)

    cv2.imwrite(f"{file_id}.png", processed_im)

    out = pytesseract.image_to_string(f"{file_id}.png", lang="vie", config=r'--psm 4')
    os.remove(f"{file_id}.png")

    return out

def convert_links_to_text(text):
    txt = text

    # Anything that isn't a square closing bracket
    name_regex = "[^]]+"
    # http:// or https:// followed by anything but a closing paren
    url_regex = "http[s]?://[^)]+"
    
    markup_regex = '\[({0})]\(\s*({1})\s*\)'.format(name_regex, url_regex)
    
    for match in findall(markup_regex,txt):
    	link_str = f"[{match[0]}]({match[1]})"
    	txt = txt.replace(link_str, match[0])

    return txt

def remove_wikipedia_footnote_ptrs(text):
    txt = text
    wiki_footnote_regex = r'\\\[\d+\\]'
    txt = sub(wiki_footnote_regex, '', txt)
    return txt

async def __convert2md(inp):
    # Use gfm-raw_html to strip styling data from source file
    converted = await __convert_text(inp, "gfm-raw_html", "html")
    converted_without_link = convert_links_to_text(converted)
    converted_without_footnote_ptr = remove_wikipedia_footnote_ptrs(converted_without_link)
    print("[CONVERT]:", converted_without_footnote_ptr)
    return converted_without_footnote_ptr

async def __convert2plain(inp):
    return await __convert_text(inp, "plain", "html")

def convert2md(req):
    pass

async def __parse_paragraphs (content: str, batching: bool = False):
    _p = ""
    _rp = content

    _rp = await __convert2md(_rp)
    _rp = _rp.replace('\r','')
    # remove empty lines and headers
    _p = [_x.strip() for _x in _rp.split('\n\n') if len(_x)!=0 and _x.strip().count('#') != len(_x)]

    _p_json = []
    h_cnt =0
    header=""
    for _n in _p:


        __h_cnt =0
        prev_h = ""
        # parse header for each paragraphs
        try:
            for _c in _n:
                if _c == '#': __h_cnt+=1
                else: break

            if (__h_cnt >= 1 and len(_n) > __h_cnt): 
                header=_n
                h_cnt = __h_cnt
                # print(_n, len(_n))
            elif (len(_n.replace('#','').strip())): 
                # remove accidental /n's in converted HTML content

                if (batching and len(_p_json) >= 1):
                    if (header == _p_json[-1]['header']):
                        # print(header)
                        _p_json[-1]['content'] += '\n'
                        _p_json[-1]['content'] += _n.replace('\n', ' ')
                        _p_json[-1]['count']+=1
                        continue

                _p_json.append({'header': header, 'h_cnt': h_cnt, 'content': _n.replace('\n',' '), 'count': 1})

            

        except:
            continue

    return _p_json

async def __query_ml_predict(qtype: QType, content: str, header: str, token_limit: int, num_qs=5, l=lang.VI_VN):
    """Get prediction from a third-party Llama3-8B-Instruct deployment"""
    app_logger.info('[PROC] ML prediction started')
    stopwatch = time()

    match qtype:
        case QType.WH:
            
            # Make request to Awan LLM endpoint
            async with httpx.AsyncClient() as client:
                _r = await client.post(
                    url="https://api.awanllm.com/v1/chat/completions",
                    headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {get_llm_api_key()}'},
                    data=json.dumps({
                        "model": "Meta-Llama-3-8B-Instruct",
                        "messages": [
                            {"role": "user", "content": prompt.gen_prompt_wh(content=content, header=header, num_qs=num_qs, lang=l)}
                        ],
                        "max_tokens": 4096,
                        "presence_penalty":0.3,
                        "temperature":0.55
                    }),
                    timeout=None
                )
            
            print(time() - stopwatch)
            if _r.status_code != 200: 
                app_logger.info(_r.json())
                return {"content": "", "style": None, "success": False}

            try:
                return {"content": _r.json()['choices'][0]['message']['content'], "style": QType.WH, "success": True}
            except:
                return {"content": "", "style": None, "success": False}
        
        case QType.STMT:

            # Make request to Awan LLM endpoint
            async with httpx.AsyncClient() as client:
                _r = await client.post(
                    url="https://api.awanllm.com/v1/chat/completions",
                    headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {get_llm_api_key()}'},
                    data=json.dumps({
                        "model": "Meta-Llama-3-8B-Instruct",
                        "messages": [
                            {"role": "user", "content": prompt.gen_prompt_statements(content=content, header=header, num_qs=num_qs, lang=l)}
                        ],
                        "max_tokens": 4096,
                        
                    }),
                    timeout=None
                )
            if _r.status_code//100 != 2: 
                app_logger.info(_r.json())
                return {"content": "", "style": QType.STMT, "success": False}

            try:
                _r_content = _r.json()['choices'][0]['message']['content']
            except:
                return {"content": "", "style":None, "success":False}
            
            try:
                _r_content = _r.json()['choices'][0]['message']['content'].split('\n\n',1)[1]
            except:
                _r_content = _r.json()['choices'][0]['message']['content'].split('\n',1)[1]
                
            async with httpx.AsyncClient() as client:
                _w = await client.post(
                    url="https://api.awanllm.com/v1/chat/completions",
                    headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {get_llm_api_key()}'},
                    data=json.dumps({
                        "model": "Meta-Llama-3-8B-Instruct",
                        "messages": [
                            {"role": "user", "content": prompt.gen_prompt_statements_false(content=_r_content, lang=l)}
                        ],
                        "max_tokens": 4096,
                        
                    }),
                    timeout=None
            )

            try:
                _ch = _w.json()['choices'][0]['message']['content']
            except:
                return {"content": "", "style": None, "success":False}
            
            try:
                _w_content = _w.json()['choices'][0]['message']['content'].split('\n\n',1)[1]
                #print(time() - stopwatch)
                return {"content": f"{_r_content}\n{_w_content}", "style": QType.STMT, "success": True}
            except:
                _w_content = _w.json()['choices'][0]['message']['content'].split('\n',1)[1]
                #print(time() - stopwatch)
                return {"content": f"{_r_content}\n{_w_content}", "style": QType.STMT, "success": True}


async def parse_wh_question(raw_qa_list, pgph_i):
    __ANS_KEY_MAPPING = {'A': 1, 'B':2, 'C':3,'D':4}
    __parsed_outputs = []
    for x in raw_qa_list:
        try:
            segments = x
            raw_key = segments[5]
            raw_key = 'A' if 'A' in raw_key else 'B' if 'B' in raw_key else 'D' if 'D' in raw_key else 'C'

            # print(segments) 
            match randint(0, 3):
                case 0 | 1:
    
                    __parsed_outputs.append(
                        {
                            "pgph_i": pgph_i,
                            "prompt": segments[0],
                            "type": "MCQ",
                            "choices": segments[1:5],
                            "keys": [segments[__ANS_KEY_MAPPING[raw_key]],],
                        }
                    )
                
                case 2 | 3:
                    __parsed_outputs.append(
                        {
                            "pgph_i": pgph_i,
                            "prompt": segments[0],
                            "type": "OPEN",
                            # Cleaning up ML output
                            "keys": [segments[__ANS_KEY_MAPPING[raw_key]].split(' ',1)[1]],
                            "choices": [segments[__ANS_KEY_MAPPING[raw_key]]]
                        }
                    )

        except:
            print("invalid: ", x)
            continue

    return __parsed_outputs

async def parse_stmt_question(stmts: list[str], pgph_i, __lang:str):
    print("starting inference...")
    
    if (stmts[0].__contains__('True: ') or stmts[0].__contains__('False: ')):
        __correct_stmts = [r[5:].strip() for r in stmts if r.__contains__('True: ')]
        __false_stmts = [r[5:].strip() for r in stmts if r.__contains__('False: ')]
    else:
        __correct_stmts = stmts[:len(stmts)//2]
        __false_stmts = stmts[len(stmts)//2:]
    __parsed_outputs = []

            # while len(__correct_stmts) >= 2:


    for c in range(0, len(__correct_stmts), 2):
        
        match randint(0, 6):
            
            case 6:
                try:
                    __parsed_outputs.append(
                        {
                            "pgph_i": pgph_i,
                            "prompt": prompt.USER_PROMPTS['AMEND'] if __lang==lang.VI_VN else prompt.USER_PROMPTS_EN['AMEND'],
                            "type": "AMEND",
                            "keys": __correct_stmts[c],
                            "choices": [__false_stmts[c]]
                        }
                    )
                    __parsed_outputs.append(
                        {
                            "pgph_i": pgph_i,
                            "prompt": prompt.USER_PROMPTS['AMEND'] if __lang==lang.VI_VN else prompt.USER_PROMPTS_EN['AMEND'],
                            "type": "AMEND",
                            "keys": __correct_stmts[c+1],
                            "choices": [__false_stmts[c+1]]
                        }
                    )
                except:
                    continue

            case 2|4:
                __c = __correct_stmts[c:c+2]
                # print(min(2, len(__false_stmts)))
                try:
                    __parsed_outputs.append(
                        {
                            "pgph_i": pgph_i,
                            "prompt": prompt.USER_PROMPTS['MULT'] if __lang==lang.VI_VN else prompt.USER_PROMPTS_EN['MULT'],
                            "type": "MULT",
                            "keys": __c,
                            "choices": sample([*__c,  *sample( __false_stmts, min(2, len(__false_stmts)) )], min(2, len(__false_stmts)) + len(__c))
                        }
                    )
                except:
                    continue

            case 3|5:
                try:
                    __c = sample(__false_stmts, 2)
                    __parsed_outputs.append(
                        {
                            "pgph_i": pgph_i,
                            "prompt": prompt.USER_PROMPTS['MULT_INV'] if __lang==lang.VI_VN else prompt.USER_PROMPTS_EN['MULT_INV'],
                            "type": "MULT",
                            "keys": __c,
                            "choices": sample([*__c, __correct_stmts[0], __correct_stmts[1]], 2+len(__c))
                        }
                    )
                except:
                    continue

            case 0|1:
                for aa in range(2):
                    try:
                        _prompt = __correct_stmts[c+aa]
                    except:
                        continue
                    # print(_prompt)
                    # FIXME: To circumvent some quirky 3rd party lib bugs around chunking phrases with quote, strip them from the sentences for the time being.
                    _prompt = _prompt.replace("\"", "").replace("\'", "")
                    _content_w = []
                    if __lang == lang.VI_VN:
                        _, _content_w = prompt.parse_content_words([_prompt])
                    else:
                        _, _content_w = prompt.parse_content_words_nltk([_prompt])
                    # print(_proper_n)
                    
                    for i, ns in enumerate(_content_w, 1):
                        try:
                            initials = "...".join([w[0] for w in ns.split(" ") if w])
                        except:
                            initials = "..."
                        _prompt = _prompt.replace(ns, f"({initials}...)", 1)

                    __parsed_outputs.append(
                        {
                            "pgph_i": pgph_i,
                            "prompt": _prompt,
                            "type": "OPEN",
                            "keys": _content_w,
                            "choices": []
                        }
                    )


    return __parsed_outputs


async def generate_questions(request):

    # parse paragraphs from document file
    try:
        __cont = await __internal_tmp_r(request.path_params['id'])
        __ps = await __parse_paragraphs(__cont[1], batching=True)
    except Exception as e:
        print(str(e))
        return JSONResponse({"err": str(e)}, 500)

    # Map asyncronous ML prediction function over list of paragraphs
    ptasks = [] 
    __raw_outputs = []
    __parsed_outputs = []

    # print(__ps)
    
    for z, _p in enumerate(__ps):
        # __query_ml_predict is an awaitable
        ptasks.append(__query_ml_predict(qtype=(QType.STMT if z%2==1 else QType.WH), content=_p['content'], header=_p['header'], l=request.path_params.get('lang', lang.VI_VN), num_qs=request.path_params.get('num_qs', 5 * _p.get('count', 1)), token_limit = int(1024 * _p.get('count', 1))))

    # __raw_outputs = [await p for p in ptasks]
    __raw_outputs = await asyncio.gather(*ptasks)
    print(__raw_outputs)
    for pgph_i, o in enumerate(__raw_outputs):
        # print(o)
        # print(pgph_i)
    # TODO: Parse ML output to JSON
        if (not o['success']):
            continue
        if (o['style'] == QType.WH):
            
            raw_qa_list = []
            # raw_segmented: list[str] = list(filter(lambda x: (len(x)>0), o['content'].split("\n\n")))[1:]

            # for i in range(len(raw_segmented)):
            #     if (len(raw_segmented[i]) and raw_segmented[i].count('\n') < 5):
            #         raw_segmented[i] += f'\n{raw_segmented[i+1]}'
            #         raw_segmented[i+1] = ""

            # print(raw_segmented)
            seg_index = 0
            seg_index_map = ['Q', 'A', 'B', 'C', 'D', '']
            raw_segmented = []
            raw_segmented_list = []
            

            for seg in o['content'].split('\n'):
                
                if seg.strip().startswith(seg_index_map[seg_index]):
                    if seg_index == 5:
                        if not ('A' in seg or 'B' in seg or 'C' in seg or 'D' in seg):
                            continue
                    print(seg)
                    raw_segmented.append(seg)
                    seg_index+=1
                if seg_index == 6:
                    raw_segmented_list.append(raw_segmented.copy())
                    raw_segmented = []
                    seg_index = 0

            __parsed_outputs.extend(await parse_wh_question(raw_segmented_list, pgph_i))
            seg_index = 0

        elif (o['style'] == QType.STMT):
            print(o['content'])
            # remove_after_dash_and_parentheses
            stmts = [ sub(r" - .*| \(.*\)", "", x.split('. ',1)[1]) for x in o['content'].split('\n') if bool(match("^\d+\.", x))]
            # print(stmts)
            __parsed_outputs.extend(await parse_stmt_question(stmts, pgph_i, request.path_params.get('lang', lang.VI_VN)))
    
    # Return the question data
    if len(__parsed_outputs):
        return JSONResponse({"questions": __parsed_outputs, "paragraphs": __ps, "title": __cont[0]})
    else:
        raise HTTPException(500)

async def scan2OCR(request):
    
    content = b''
    ret = []

    async with request.form(max_files=10, max_fields=20) as form:
        for i in range(int(form['uploads'])):

            # Get random file ID
            file_id = sys_random.randbytes(12).hex()

            # Load image using PIL and convert to opencv grayscale format
            im = Image.open(BytesIO(await form[f'upload_{i}'].read()))
            
            # # Perform image preprocessing
            # processed_im = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2GRAY)

            # cv2.imwrite(f"{file_id}.png", processed_im)

            # out = pytesseract.image_to_string(f"{file_id}.png", lang="vie", config=r'--psm 4')
            # os.remove(f"{file_id}.png")
            _loop = asyncio.get_running_loop()
            _out = await _loop.run_in_executor(None, partial(__ocr, im, file_id))
            out = await _out
            # adapt the output text to the HTML-based rich text editor
            ret.append({"content": out.replace('\n','<br/>')})
    
    return JSONResponse(ret, 200)


async def convert2html(request):
    content = b''
    filename = ""
    output = ""
    files = []

    rets = []

    async with request.form(max_files=10, max_fields=20) as form:

        print(form['uploads'])
        for i in range(int(form['uploads'])):
            # Get random file ID
            filename = sys_random.randbytes(12).hex()
            ext = form[f'upload_{i}'].filename.split(".")[-1]

            content = await form[f'upload_{i}'].read()

            with open(f"{filename}.{ext}", 'wb') as o:
                o.write(content)
            files.append(f"{filename}.{ext}")
    
    for file in files:
        try:
            output = await __convert_file(file, "html")
            print(output)

        except Exception as e:
            app_logger.error(e)
            return JSONResponse({"detail": ""}, status_code=422)
    
        # Extract image sources from document
        imgs = []
        start = -1

        for i in range(len(output)):
            if output[i:i+4] == "<img":
                
                start = i

            if output[i:i+2] == "/>" and start != -1:

                img_tag = output[start:i+2]
                imgs.append(img_tag)

                start = -1

        for x in imgs:
            output = output.replace(x, " ")
        
        # Remove upload file
        os.remove(file)
    
        rets.append({"content": output, "resources": imgs})

    return JSONResponse(rets)

async def llm_generate_text(request):
    async with httpx.AsyncClient() as client:
        o = await request.json()
        _r = await client.post(
                    url="https://api.awanllm.com/v1/chat/completions",
                    headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {get_llm_api_key()}'},
                    data=json.dumps({
                        "model": "Meta-Llama-3-8B-Instruct",
                        "messages": [
                            {"role": "user", "content": f"Generate a study note about {o['prompt']}. Use {o['lang']}."}
                        ],
                        "max_tokens": 4096,
                        "presence_penalty":0.3,
                        "temperature":0.5
                    }),
                    timeout=None)

        try:  
            if _r.status_code != 200:
                raise HTTPException(status_code=429, detail=str(_r.json()))
                
            repl = _r.json()['choices'][0]['message']['content']
            # input, type_out="plain", type_in="html"
            repl = await __convert_text(repl, "html-raw_html", "markdown")
            return PlainTextResponse(repl)
            
        except Exception as e:
            raise HTTPException(status_code=422, detail=str(e))

async def get_flashcards(request):
    # [title, content, keywords]

    __file = await __internal_tmp_r(request.path_params['id'])
    __content = __file[1]
    
    __lang = request.path_params['lang']
    __keywords = [r.strip() for r in __file[2] if len(r) > 0]

    __tldr = await tldr(__content, __lang)
    print(__tldr)
    __definitions = await get_definitions_from_words(__keywords, __tldr, __lang)
    print(__definitions)

    return JSONResponse({"tldr": __tldr, "defs": __definitions, "imgs": await fetch_img_for_words(__keywords)})

"""
Similarity validation
"""
w2v_vi = KeyedVectors.load_word2vec_format('wiki.vi.model.bin', binary=True)
# w2v_en = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',binary=True)
vocab_vi = w2v_vi.key_to_index
# vocab_en = w2v_en.vocab
from underthesea import word_tokenize
from nltk.tokenize import word_tokenize as word_tokenize_en
from numpy import zeros,zeros_like
from scipy.spatial.distance import cosine
import warnings
async def validate_similarity(request):
    req = await request.json()
    sent1, sent2 = req['sentences']
    l = req['lang']

    if (l == lang.VI_VN):
        tokens1 = word_tokenize(sent1.lower())
        tokens2 = word_tokenize(sent2.lower())
    else:
        tokens1 = word_tokenize_en(sent1.lower())
        tokens2 = word_tokenize_en(sent2.lower())

    vect1 = zeros_like(w2v_vi.get_vector('an'))
    vect2 = zeros_like(w2v_vi.get_vector('an'))

    for t in tokens1:
        if t in vocab_vi:
            vect1 += w2v_vi.get_vector(t)
    
    
    for t in tokens2:
        if t in vocab_vi:
            vect2 += w2v_vi.get_vector(t)

    # Calculate similarity using cosine similarity: This metric measures the cosine of the angle between two embedding vectors. A higher cosine similarity indicates more similar sentences.
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        try:
            sim = 1 - cosine(vect1, vect2) >= 0.8
        except RuntimeWarning as e:
            return JSONResponse({"isSimilar": "False"})
            
        return JSONResponse({"isSimilar": str(sim)})

async def get_cached_img_from_disk(request):
    _fn = request.path_params['fn']
    # /images/img_-3711971785602203114.webp HTTP/1.1"
    if _fn.startswith('img_') and _fn.endswith('.webp'):
        return FileResponse(_fn)
    else:
        raise HTTPException(404)

async def generate_redemption(request):
    req = await request.json()
    paragraphs = req['pgphs']

    ret_questions = []

    ptasks = []
    for paragraph in paragraphs:
        ptasks.append(__query_ml_predict(QType.WH, paragraph["content"], paragraph["header"], 4096, num_qs=paragraph.get("count", 1)*5, l=req['lang']))

    raw_questions: list[str] = await asyncio.gather(*ptasks)
    for query in raw_questions:
        if not query['success']:
            continue

        q = query['content']

        raw_segments = [x.strip() for x in q.split('\n') if x.strip()]
        filtered = []
        seg_cnt = 0
        seg_map = ['Q', 'A', 'B', 'C', 'D', '']
        for r in raw_segments:
            if r.startswith(seg_map[seg_cnt]):
                if seg_cnt == 5:
                    if not ('A' in r or 'B' in r or 'C' in r or 'D' in r):
                        continue
                filtered.append(r)
                seg_cnt += 1
            if seg_cnt == 6:
                _r = filtered[5]
                ans_index = 1 if 'A' in _r else 2 if 'B' in _r else 4 if 'D' in _r else 3
                ans_key = filtered[ans_index][2:].strip()

                if randint(0, 1) == 1:
                    ret_questions.append({'prompt': filtered[0], 'keys': [ans_key,]})
                else:
                    prompt_format = "\n".join(filtered[0:5])
                    ret_questions.append({'prompt': prompt_format, 'keys': [seg_map[ans_index]]})

                filtered.clear()
                seg_cnt = 0

    return JSONResponse({'questions': ret_questions})



async def root(requests):
    return PlainTextResponse("Success")


# Application entry point
routes = ...

middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=['http://localhost:8100', 'https://text2quiz-three.vercel.app'],
        allow_methods =['*'],
    ),
    Middleware(APIGuardMiddleware),
]

app = Starlette(debug=True,routes=[
    Route('/getFlashcards/{id}/{lang}', get_flashcards, methods=['GET']),
    Route('/convert2html',convert2html, methods=['POST']),
    Route('/scan2ocr', scan2OCR, methods=['POST']),
    Route('/temp', __save_temp, methods=['POST']),
    Route('/temp/{id}', __get_temp, methods=['GET']),
    Route('/temp/{id}', __remove_temp, methods=['DELETE']),
    # Route('/generateQuiz/{id}', generate_questions, methods=['GET']),
    Route('/generateQuiz/{id}/{lang}', generate_questions, methods=['GET']),
    # /images/img_-3711971785602203114.webp HTTP/1.1"
    Route('/images/{fn}', get_cached_img_from_disk, methods=['GET']),
    Route('/convert2md', convert2md, methods=['POST']),
    Route('/mltest', __mltest, methods=['GET']),
    Route('/validateSimilarity', validate_similarity, methods=['POST']),
    Route('/llmGenerateText', llm_generate_text, methods=['POST']),
    Route('/generateRedemption', generate_redemption, methods=['POST']),
    Route('/', root, methods=['GET']),
],
middleware=middleware)

import os
print("running at: " + os.getcwd())
