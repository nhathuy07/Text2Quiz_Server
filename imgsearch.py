import requests
from random import choice
from env import GOOGLE_API_KEY, CX, IMSEARCH_API_KEY

def search_img(term, orientation="landscape"):
    _t = f"{orientation} image of \"{term}\""
    _ret = requests.get(
        url=f"https://www.googleapis.com/customsearch/v1?key={IMSEARCH_API_KEY}&cx={CX}&q={_t}&searchType=image&fields=items(link)"
    )
    try:
        return choice(_ret.json()['items'][:10])['link']
    except:
        print(_ret.json())
        # return placeholder image URL if fail to find image
        return "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Placeholder_view_vector.svg/310px-Placeholder_view_vector.svg.png"
