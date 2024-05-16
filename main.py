from starlette.responses import PlainTextResponse, JSONResponse
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.middleware import Middleware
# from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.middleware.cors import CORSMiddleware
import prompt

import transformers
import 

from PIL import Image

import pypandoc
import tempfile
import cv2
from io import BytesIO
import numpy as np
import os

import tempfile
import pytesseract

from secrets import SystemRandom

routes = ...

# Ensure that all requests include an 'example.com' or
# '*.example.com' host header, and strictly enforce https-only access.

temp_files = {}

middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=['http://localhost:8100'],
        allow_methods =['*'],
    ),
]

temp_file = tempfile.NamedTemporaryFile(delete=False)  # Create a temporary file with a name (optional for some libraries)
temp_file_path = temp_file.name  # Get the path to the temporary file

print(temp_file_path)

sys_random = SystemRandom()

# TODO: Change to environment variable in prod.

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Admin\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# TODO: Move token to docker secrets in prod.
# mlClient = InferenceClient("meta-llama/Meta-Llama-3-8B-Instruct", token="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")

async def __mltest(request):
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": }
    )

async def __save_temp(request):
    file_id = sys_random.randbytes(16).hex()
    content = ""
    # async with request.form(max_fields=3) as form:
    form = await request.json()
    content = form['content']
    title = form['title']
    temp_files[file_id] = [title, content]
    
    return PlainTextResponse(file_id, 200)
    
async def __get_temp(request):
    return PlainTextResponse(temp_files.get(request.path_params['id'][1], ""))

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

async def __parse_paragraphs (content: str):
    _p = ""
    _rp = content

    _rp = __convert2md(_rp).replace('\r','')
    # remove empty lines and headers
    _p = [_x.strip() for _x in _rp.split('\n\n') if len(_x)!=0 and _x.strip().count('#') != len(_x)]
    print(_p)
    _p_json = []
    h_cnt =0
    header=""
    for _n in _p:


        __h_cnt =0
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
                _p_json.append({'header': header, 'h_cnt': h_cnt, 'content': _n.replace('\n',' ')})

        except:
            continue

    return _p_json

async def parseParagraph(request):
    w ="""
        <h1><b style="background-color: rgb(255, 255, 255);">Chariot overview</b></h1><div><p style="margin: 0.5em 0px 0px; padding-bottom: 0.5em; animation-delay: -0.01ms !important; animation-duration: 0.01ms !important; animation-iteration-count: 1 !important; scroll-behavior: auto !important; transition-duration: 0ms !important;"><font color="#202122" style=""><span style="font-size: 14px;"><span style="background-color: rgb(255, 255, 255);">A chariot is a type of </span><span style="background-color: rgb(255, 199, 1);">cart </span><span style="background-color: rgb(255, 255, 255);">driven by a </span><span style="background-color: rgb(255, 199, 1);">charioteer,</span></span><span style="font-size: 14px; background-color: rgb(255, 255, 255);"> usually using </span><span style="font-size: 14px; background-color: rgb(255, 199, 1);">horses</span><span style="font-size: 14px;"><span style="background-color: rgb(255, 255, 255);">[note 1] to provide </span><span style="background-color: rgb(255, 199, 1);">rapid </span></span><span style="font-size: 14px;"><span style="background-color: rgb(255, 199, 1);">motive power</span><span style="background-color: rgb(255, 255, 255);">. The oldest known chariots have been found in burials of the</span><span style="background-color: rgb(255, 199, 1);"> Sintashta culture</span><span style="background-color: rgb(255, 255, 255);"> in modern-day </span><span style="background-color: rgb(255, 199, 1);">Chelyabinsk</span><span style="background-color: rgb(255, 255, 255);"> Oblast, Russia, dated to c. 1950–1880 BCE[1][2] and are depicted on cylinder seals from Central Anatolia in Kültepe dated to c. 1900 BCE.[2] The critical invention that allowed the construction of light, horse-drawn chariots was the spoked wheel.</span></span></font></p><h1 style="margin: 0.5em 0px 0px; padding-bottom: 0.5em; animation-delay: -0.01ms !important; animation-duration: 0.01ms !important; animation-iteration-count: 1 !important; scroll-behavior: auto !important; transition-duration: 0ms !important;"><font color="#202122"><span style="font-size: 14px; background-color: rgb(255, 255, 255);"><b>Advantages of a chariot</b></span></font></h1><p style="margin: 0.5em 0px 0px; padding-bottom: 0.5em; animation-delay: -0.01ms !important; animation-duration: 0.01ms !important; animation-iteration-count: 1 !important; scroll-behavior: auto !important; transition-duration: 0ms !important;"><font color="#202122" style=""><span style="font-size: 14px; background-color: rgb(255, 255, 255);">The chariot was a fast, light, open,</span><span style="font-size: 14px; background-color: rgb(255, 199, 1);"> two-wheeled</span><span style="font-size: 14px; background-color: rgb(255, 255, 255);"> conveyance drawn by two or more equids (usually horses) that were hitched side by side, and was little more than a floor with a waist-high guard at the front and sides. It was initially used for ancient warfare during the Bronze and Iron Ages, but after its military capabilities had been superseded by light and </span><span style="font-size: 14px; background-color: rgb(255, 199, 1);">heavy cavalries</span><span style="font-size: 14px; background-color: rgb(255, 255, 255);">, chariots continued to be used for travel and transport, in processions, for games, and in races.</span></font></p></div>
    """
    # r = await __parse_paragraphs(temp_files[request.path_params['id']][1])
    r = await __parse_paragraphs(w)
    return JSONResponse(r)

async def generate_questions(request):
    # TODO: Uncomment in prod.
    # try:
    #     __ps = __parse_paragraphs(temp_files[request.path_params['id']])
    # except:
    #     return JSONResponse({}, 500)

    # TODO: Remove in prod
    _p_json = [
        
    ]
    __resp = []
    num_qs = 5
    for _p in _p_json[:1]:

        __resp.append(mlClient.text_generation(prompt.gen_prompt_vi(num_qs, _p['header'], _p['content']), max_new_tokens=2048))

    
    return JSONResponse(__resp)
    

async def generate_flashcards(request):
    pass

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

app = Starlette(debug=True,routes=[
    Route('/convert2html',convert2html, methods=['POST']),
    Route('/scan2ocr', scan2OCR, methods=['POST']),
    Route('/temp', __save_temp, methods=['POST']),
    Route('/temp/{id}', __get_temp, methods=['GET']),
    Route('/temp/{id}', __remove_temp, methods=['DELETE']),
    # Route('/generateQuiz/{id}', generate_questions, methods=['GET']),
    Route('/generateQuiz', generate_questions, methods=['GET']),
    Route('/convert2md', convert2md, methods=['POST']),
    Route('/parseParagraph/{id}', parseParagraph, methods=['GET']),
    Route('/mltest', __mltest, methods=['GET'])

],
middleware=middleware)