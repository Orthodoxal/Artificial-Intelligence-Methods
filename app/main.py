from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.routers import parse
import math
import pandas as pd
from bitarray import bitarray

app = FastAPI()

templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(parse.router)


class BloomFilter(object):
    def __init__(self, size, number_expected_elements=100000):
        self.size = size
        self.number_expected_elements = number_expected_elements

        self.bloom_filter = bitarray(self.size)
        self.bloom_filter.setall(0)

        self.number_hash_functions = round((self.size / self.number_expected_elements) * math.log(2))

    def _hash_djb2(self, s):
        hash = 5381
        for x in s:
            hash = ((hash << 5) + hash) + ord(x)
        return hash % self.size

    def _hash(self, item, K):
        return self._hash_djb2(str(K) + item)

    def add_to_filter(self, item):
        for i in range(self.number_hash_functions):
            self.bloom_filter[self._hash(item, i)] = 1

    def check_is_not_in_filter(self, item):
        for i in range(self.number_hash_functions):
            if self.bloom_filter[self._hash(item, i)] == 0:
                return True
        return False


"""@app.get("/main", response_class=HTMLResponse)
def main(request: Request):
    return templates.TemplateResponse("main.html", {"request": request})"""


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/")
def result(request: Request, search_word_key: str = Form(...)):
    try:
        bloom_filter = BloomFilter(200, 20)
        key_word_array = ["торгующих", "продуктовый магазин", "США",
                          "гипермаркет", "ассортименте", "продуктовые магазины",
                          "товаров", "напитков", "продуктов", "магазин", "набор", "супермаркет",
                          "золото"]

        for i in range(len(key_word_array)):
            bloom_filter.add_to_filter(key_word_array[i])

        if not bloom_filter.check_is_not_in_filter(search_word_key):
            return templates.TemplateResponse('found.html', context={'request': request, 'search_word_key': search_word_key})
        return templates.TemplateResponse('error.html', context={'request': request})
    except Exception:
        return Exception


@app.post("/found")
def result(request: Request, search_word_key: str = Form(...)):
    try:
        dataframe = pd.read_csv("csv/df.csv", sep=",", encoding="windows-1251")
        filenames = []
        for i in range(dataframe.shape[0]):
            if search_word_key.lower() in dataframe.iloc[i]['description'].lower():
                filenames.append((dataframe.iloc[i]['file'], dataframe.iloc[i]['link']))

        if len(filenames) == 1 and filenames[0][0] == "9.csv":
            return templates.TemplateResponse('main.html', context={'request': request, 'file_path': filenames[0][0]})
        elif len(filenames) >= 1:
            names = []
            for index in range(len(filenames)):
                names.append(("Результат: " + filenames[index][0].replace('.csv', ''), filenames[index][1]))

            return templates.TemplateResponse('any.html', context={'request': request, 'filenames': names})
    except Exception:
        return Exception
