from starlette.responses import PlainTextResponse, JSONResponse
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

"""Prompt templates for LLM"""
from env import LLM_API_KEY
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

from secrets import SystemRandom

from random import randint, sample

from enum import Enum
from re import sub


from flashcard_util import tldr, get_definitions_from_words, fetch_img_for_words

class QType(Enum):
    WH = 0
    STMT = 3
    FILL = 6

routes = ...

temp_files = {}

middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=['http://localhost:8100'],
        allow_methods =['*'],
    ),
]

sys_random = SystemRandom()

# TODO: Change to environment variable in prod.

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Admin\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"


async def __mltest(request):
    pass

async def __save_temp(request):
    file_id = sys_random.randbytes(16).hex()
    content = ""
    # async with request.form(max_fields=3) as form:
    form = await request.json()
    content = form['content']
    title = form['title']
    keywords = form['keywords']
    temp_files[file_id] = [title, content, keywords]
    print(file_id)
    return PlainTextResponse(file_id, 200)
    
async def __get_temp(request, entry = 1):
    return JSONResponse(temp_files.get(request.path_params['id'], [None, None, None]))

async def __remove_temp(request):
    try:
        del temp_files[request.path_params['id']]
    except:
        return PlainTextResponse("", 500)
    
    return PlainTextResponse("", 200)

def __convert2md(inp):
    # Use gfm-raw_html to strip styling data from source file
    return pypandoc.convert_text(inp, "gfm-raw_html", "html")

def __convert2plain(inp):
    return pypandoc.convert_text(inp, "plain", "html")

def convert2md(req):
    pass

async def __parse_paragraphs (content: str, batching: bool = False):
    _p = ""
    _rp = content

    _rp = __convert2md(_rp).replace('\r','')
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
            elif (len(_n.replace('#',''))): 
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

async def __query_ml_predict(qtype:QType, content: str, header: str, token_limit: int, num_qs=5, l=lang.VI_VN):
    """Get prediction from a third-party Llama3-8B-Instruct deployment"""
    stopwatch = time()

    match qtype:
        case QType.WH:
            
            # Make request to Awan LLM endpoint
            _r = requests.post(
                url="https://api.awanllm.com/v1/chat/completions",
                headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {LLM_API_KEY}'},
                data=json.dumps({
                    "model": "Meta-Llama-3-8B-Instruct",
                    "messages": [
                        {"role": "user", "content": prompt.gen_prompt_wh(content=content, header=header, num_qs=num_qs, lang=l)}
                    ],
                    "max_tokens": max(token_limit, 4096),
                    "presence_penalty":0.3,
                    "temperature":0.55
                })
            )

            print(time() - stopwatch)
            return {"content": _r.json()['choices'][0]['message']['content'], "style": QType.WH}
        
        case QType.STMT:

            # Make request to Awan LLM endpoint
            _r = requests.post(
                url="https://api.awanllm.com/v1/chat/completions",
                headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {LLM_API_KEY}'},
                data=json.dumps({
                    "model": "Meta-Llama-3-8B-Instruct",
                    "messages": [
                        {"role": "user", "content": prompt.gen_prompt_statements(content=content, header=header, num_qs=num_qs, lang=l)}
                    ],
                    "max_tokens": max(token_limit, 4096),
                    
                })
            )

            _r_content = _r.json()['choices'][0]['message']['content'].split('\n\n',1)[1]

            _w = requests.post(
                url="https://api.awanllm.com/v1/chat/completions",
                headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {LLM_API_KEY}'},
                data=json.dumps({
                    "model": "Meta-Llama-3-8B-Instruct",
                    "messages": [
                        {"role": "user", "content": prompt.gen_prompt_statements_false(content=_r_content, lang=l)}
                    ],
                    "max_tokens": max(token_limit, 4096),
                    
                })
            )

            _w_content = _w.json()['choices'][0]['message']['content'].split('\n\n',1)[1]
            print(time() - stopwatch)
            return {"content": f"{_r_content}\n{_w_content}", "style": QType.STMT}


async def parse_wh_question(raw_qa_list, pgph_i):
    __ANS_KEY_MAPPING = {'A': 1, 'B':2, 'C':3,'D':4}
    __parsed_outputs = []
    for x in raw_qa_list:
        try:
            segments = [r for r in x.split('\n') if r.__len__()]
            raw_key = segments[5].strip()
            raw_key = 'A' if 'A' in raw_key else 'B' if 'B' in raw_key else 'D' if 'D' in raw_key else 'C'
        except:
            print("invalid: ", x)
            continue
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
        __ps = await __parse_paragraphs(temp_files[request.path_params['id']][1],batching=True)
    except:
        return JSONResponse({}, 500)

    # Map asyncronous ML prediction function over list of paragraphs
    ptasks = [] 
    __raw_outputs = []
    __parsed_outputs = []

    # print(__ps)
    
    for z, _p in enumerate(__ps):
        ptasks.append(__query_ml_predict(qtype=(QType.STMT if z%2==1 else QType.WH), content=_p['content'], header=_p['header'], l=request.path_params.get('lang', lang.VI_VN), num_qs=request.path_params.get('num_qs', 5 * _p.get('count', 1)), token_limit = int(1024 * _p.get('count', 1))))

    __raw_outputs = [await p for p in ptasks]
    
    for pgph_i, o in enumerate(__raw_outputs):
        # print(o)
        print(pgph_i)
    # TODO: Parse ML output to JSON
        if (o['style'] == QType.WH):
            
            raw_qa_list = []
            raw_segmented: list[str] = list(filter(lambda x: (len(x)>0), o['content'].split("\n\n")))[1:]
            print(raw_segmented)
            for i in range(len(raw_segmented)):
                if (len(raw_segmented[i]) and raw_segmented[i].count('\n') < 5):
                    raw_segmented[i] += f'\n{raw_segmented[i+1]}'
                    raw_segmented[i+1] = ""

            print(raw_segmented)
            
            __parsed_outputs.extend(await parse_wh_question(raw_segmented, pgph_i))

        elif (o['style'] == QType.STMT):
            print(o['content'])
            # remove_after_dash_and_parentheses
            stmts = [ sub(r" - .*| \(.*\)", "", x.split('. ',1)[1]) for x in o['content'].split('\n') if bool(match("^\d+\.", x))]
            # print(stmts)
            __parsed_outputs.extend(await parse_stmt_question(stmts, pgph_i, request.path_params.get('lang', lang.VI_VN)))
    
    # Return the question data

    return JSONResponse({"questions": __parsed_outputs, "paragraphs": __ps})

async def scan2OCR(request):
    
    content = b''
    ret = []

    async with request.form(max_files=10, max_fields=20) as form:
        for i in range(int(form['uploads'])):

            # Get random file ID
            file_id = sys_random.randbytes(12).hex()

            # Load image using PIL and convert to opencv grayscale format
            im = Image.open(BytesIO(await form[f'upload_{i}'].read()))
            
            # Perform image preprocessing
            processed_im = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2GRAY)

            cv2.imwrite(f"{file_id}.png", processed_im)

            out = pytesseract.image_to_string(f"{file_id}.png", lang="vie")
            os.remove(f"{file_id}.png")
            
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
            output = pypandoc.convert_file(file, "html")

        except: 
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


async def get_flashcards(request):
    # [title, content, keywords]
    __content = temp_files[request.path_params['id']][1]
    __lang = request.path_params['lang']
    __keywords = [r.strip() for r in temp_files[request.path_params['id']][2] if len(r) > 0]

    __tldr = await tldr(__content, __lang)
    print(__tldr)
    __definitions = await get_definitions_from_words(__keywords, __tldr)
    print(__definitions)

    return JSONResponse({"tldr": __tldr, "defs": __definitions, "imgs": await fetch_img_for_words(__keywords)})


app = Starlette(debug=True,routes=[
    Route('/getFlashcards/{id}/{lang}', get_flashcards, methods=['GET']),
    Route('/convert2html',convert2html, methods=['POST']),
    Route('/scan2ocr', scan2OCR, methods=['POST']),
    Route('/temp', __save_temp, methods=['POST']),
    Route('/temp/{id}', __get_temp, methods=['GET']),
    Route('/temp/{id}', __remove_temp, methods=['DELETE']),
    # Route('/generateQuiz/{id}', generate_questions, methods=['GET']),
    Route('/generateQuiz/{id}/{lang}', generate_questions, methods=['GET']),
    Route('/convert2md', convert2md, methods=['POST']),
    Route('/mltest', __mltest, methods=['GET'])

],
middleware=middleware)