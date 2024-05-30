import requests
from random import choice

GOOGLE_API_KEY = "AIzaSyDCeLIVjhVtoPFqrUym2OkfT2_7hRz1ORY"
CX = "54c182394edb24bc3"
def search_img(term):
    _ret = requests.get(
        url=f"https://www.googleapis.com/customsearch/v1?key={GOOGLE_API_KEY}&cx={CX}&q={term}&searchType=image&fields=items(link)"
    )
    try:
        return choice(_ret.json()['items'][:10])['link']
    except:
        print(_ret.json())
        return ""