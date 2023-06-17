import argparse
from bs4 import BeautifulSoup

import requests
import json

from constant import ROOT_URL
from create_embedding import create_embedding_from_pages


def extract_text_from(html):
    soup = BeautifulSoup(html, features="html.parser")
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train parameters")
    parser.add_argument("chunksize", type=int, default=1500, help="chunk size")
    args = parser.parse_args()

    r = requests.get(ROOT_URL)
    articles = json.loads(r.text)["articles"]
    pages = []
    for article in articles:
        url = article["url"]
        pages.append({"text": extract_text_from(article["body"]), "source": url})

    create_embedding_from_pages(pages, args.chunksize)
