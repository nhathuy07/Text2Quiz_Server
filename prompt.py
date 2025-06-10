import lang
import pprint
from underthesea import  ner, pos_tag, sent_tokenize
from pprint import pprint

from nltk import pos_tag
from nltk import download as nltk_dl

from nltk.tokenize import word_tokenize

nltk_dl('punkt')
nltk_dl('averaged_perceptron_tagger')

USER_PROMPTS = {
    "MULT": "Hãy chọn những ý kiến đúng",
    "MULT_INV": "Hãy chọn những ý kiến KHÔNG đúng",
    "AMEND": "Hãy tìm và sửa lỗi sai trong đoạn sau"
}

USER_PROMPTS_EN = {
    "MULT": "Choose all correct statements",
    "MULT_INV": "Choose all wrong statements",
    "AMEND": "Rectify this statement:"
}

CONTENT_WORD_FILTER = {
    'V': 1, 'N': 1, 'Np': 1, 'Vp': 1, 'M': 1, 'A': 1
}

STOPWORDS = {}
STOPWORDS_EN = {}

with open("stopwords_en.txt",encoding="utf-8") as st:
    STOPWORDS_EN = {k.strip():1 for k in st.readlines()}

with open("stopwords.txt", encoding="utf-8") as st:
    STOPWORDS = {k.strip():1 for k in st.readlines()}

def parse_content_words_nltk(sentences: list[str], *_, **__):
    # text = "The outer giants are mostly gas and ice, while the inner ones are rocky. Jupiter is the biggest and Saturn has famous rings. Uranus and Neptune are icy and blue."
    tokens = [[x, y] for x, y in pos_tag(word_tokenize(" ".join(sentences)))]

    pos_tags = []
    wbuf = ""
    tbuf = ""

    for i in range(len(tokens)):
        # Classify tags to group of similar tags
        if (tokens[i][1].__contains__('NN')): tokens[i][1] = 'NN'
        elif (tokens[i][1].__contains__('VB')): tokens[i][1] = 'VB'
        elif (tokens[i][1].__contains__('JJ')): tokens[i][1] = 'JJ'
        elif (tokens[i][1].__contains__('CD')): tokens[i][1] = 'CD'
        else: tokens[i][1] = 'O'

        # group tags
        if (tbuf == tokens[i][1]):
            wbuf += f' {tokens[i][0]}'
        else:
            if wbuf:
                pos_tags.append(wbuf)
            tbuf = ""
            wbuf = ""
            if (tokens[i][1] != 'O'):
                tbuf = tokens[i][1]
                wbuf = tokens[i][0]
            
    return ([], pos_tags)

def parse_content_words(sentences: list[str], proper_n=True, content_w=True):

    proper_nouns = []
    content_words = []
    __ner_res = [ [[*phr] for phr in ner(sent) if phr[1] != 'CH'] for sent in sentences]

    _cur_phr = []
    _cur_type = ""

    if (not proper_n):
        return proper_nouns, content_words

    for i, ner_re in enumerate(__ner_res):
        for j, phr in enumerate(ner_re):
            if phr[3][0] == 'B' or phr[3][0] == 'I':

                if _cur_type == phr[3][-3:] or _cur_type == "":
                    _cur_phr.append(phr[0])
                    __ner_res[i][j][0] = f"{__ner_res[i][j-1][0]} {__ner_res[i][j][0]}"
                    __ner_res[i][j-1][0] = ""
                    _cur_type = phr[3][-3:]
                else:
                    if _cur_type != "":

                        _cur_type = ""

            else:
                _cur_type = ""


    for ner_re in __ner_res:
        for phr in ner_re:
            if ((phr[3] == 'O' and CONTENT_WORD_FILTER.get(phr[1], False) and not STOPWORDS.get(phr[0].lower(), False))
                or ((phr[3][0] == 'I' or phr[3][0] == 'B') and phr[0] != "")
                ):
                content_words.append(phr[0])

    # pprint([content_word for content_word in content_words if content_words[0] != ""])

    return (proper_nouns, content_words)

# parse_content_words([' Chiến xa có lẽ bắt nguồn ở Lưỡng Hà.', ' Sự mô tả sớm nhất về những cỗ xe trong bối cảnh chiến tranh là ở trên "Cờ hiệu của Ur".', ' Chúng được gọi một cách đúng đắn hơn là xe bò hay xe ngựa.', ' Bánh xe nan hoa không xuất hiện ở Lưỡng Hà cho đến những năm 2000 TCN.', ' Người Sumer cũng có một loại chiến xa 2 bánh nhẹ hơn.', ' Chiến xa có lẽ bắt nguồn ở Trung Á.', ' Sự mô tả sớm nhất về những cỗ xe trong bối cảnh chiến tranh là ở trên "Cờ hiệu của Babylon".', ' Chúng được gọi một cách đúng đắn hơn là xe ngựa hay xe trâu.', ' Bánh xe nan hoa xuất hiện ở Lưỡng Hà từ năm 3000 TCN.', ' Chiến xa được sử dụng cho chiến tranh thời hiện đại.'])

def gen_prompt_wh(num_qs, header, content, lang = lang.VI_VN):

    form = "Q:{question}\nA:{option a}\nB:{option b}\nC:{option c}\nD:{option d}\n{correct_option (A|B|C|D)}"

    # prompt = f"Given the following paragraph in {lang}:\n\n{header}\n{content}.\nGenerate {num_qs} multiple-choice, medium difficulty questions in {lang}. Do NOT include `All of the above`, `Neither of the above` and their equivalents as possible choices. Format your response as:\n{form}\nBe concise, DO NOT give further explanations.\n"
    
    prompt = f"""Given the following paragraph:\n{content}.\nGenerate {num_qs} multiple-choice, medium difficulty questions in {lang}. Do NOT include `All of the above`, `Neither of the above` and their equivalents as possible choices. Format your response in {lang} as:\n{form}\nDO NOT give further explanations.\n"""
    return prompt

def gen_prompt_statements(num_qs, header, content, lang = lang.VI_VN):
    prompt = f"{num_qs} pairs of true statement in {lang} based on this passage, no further explanations needed.:\n\n{header}\n{content}\n\n "
    return prompt

def gen_prompt_statements_false(content, lang = lang.VI_VN):
    prompt = f"Alter the following statements in {lang} so that they are false. Ensure changes are hard to notice. Avoid replacing words with their opposites. {content}\nWrite the statements in {lang}. No further explanations needed."
    return prompt