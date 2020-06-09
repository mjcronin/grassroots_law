import requests
import ngram as ng
import pandas as pd
import sys
import pkg_resources
from goose3 import Goose
import string

ngrams = {}
def load_dictionaries():
    data = None
    with open("us-locations.csv", "r") as adjfile:
        data = pd.read_csv(adjfile)


    dict_fp = lambda fn : "./dictionaries/{}".format(fn)
    for colname in data.columns:
        fp = dict_fp("{}.txt".format(colname))
        terms = pd.read_csv(fp)
        terms.columns = ['index', colname]
        ngram = ng.NGram(list(terms[colname]))
        ngrams[colname] = ngram

def predict_location(text):
    strip_punctuation = lambda s : s.translate(str.maketrans('', '', string.punctuation))
    search_thresh = .5
    words = [w.lower() for w in text.split(' ')]
    for k, ngram in ngrams.items():
        print("Processing article for {}".format(k))
        for word in words:
            results = ngram.search(word, threshold=search_thresh) 
            if results:
                print(results) 


def main(url):
    load_dictionaries()

    url = "https://www.wired.com/story/black-investors-vc-funding-make-hire-send-wire/"
    page_content = requests.get(url).content
    extractor = Goose()
    article = extractor.extract(raw_html=page_content)
    locations = predict_location("This article is about an important event in portland, OR. Specifically in Benton county nearby the city of Salem, OR.")# article.cleaned_text)

if __name__ == "__main__":
    url =  sys.argv[0]
    main(url)
