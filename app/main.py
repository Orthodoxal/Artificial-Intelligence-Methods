from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.routers import parse
import math
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
        bloom_filter = BloomFilter(200, 100)
        key_word_array = ["покупатель", "компании-супермаркета", "магазинов",
                          "номер магазина", "площадь магазина", "посетивших магазины",
                          "продажи", "долларах", "произведенные магазинами", "покупателей"]
        description = """В наборе данных вы получите данные о различных магазинах компании-супермаркета в соответствии 
        с их идентификаторами магазинов, которые для простоты были преобразованы в положительные целые числа.
        ID - Идентификационный номер магазина
        Store_Area - физическая площадь магазина в ядрах
        Items_Available - Количество различных предметов, доступных в соответствующем магазине.
        DailyCustomerCount - количество покупателей, посетивших магазины в среднем за месяц.
        Store_Sales - Продажи в (долларах США), произведенные магазинами."""

        for i in range(len(key_word_array)):
            bloom_filter.add_to_filter(key_word_array[i])

        if not bloom_filter.check_is_not_in_filter(search_word_key.lower()):
            if search_word_key.lower() in description.lower():
                return templates.TemplateResponse('main.html', context={'request': request})
            else:
                return templates.TemplateResponse('error.html', context={'request': request})
        else:
            return templates.TemplateResponse('error.html', context={'request': request})
    except Exception:
        return Exception
