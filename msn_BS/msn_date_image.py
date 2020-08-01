# # Naver Dictionary Crawling

import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from multiprocessing import Pool
import argparse
import json
import re
import sys

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('data', type=str, help='no data')
args = parser.parse_args()


def crawling_news(urls):
    for url in urls:
        req = requests.get(url)
        html = req.text
        soup = BeautifulSoup(html, 'html.parser')

        article_ = dict()
        # vert
        if 'refurl' not in url:
            url = url.split("/")[4:] 
        else:
            url = url.split("/")[5:]
        article_['verts'] = url[:2][0]
        # subvert
        article_['subverts'] = url[:2][1]
        # nid
        nid = url[-1]
        article_['nids'] = nid.split('?')[0].split('-')[-1]
        # title
        title = soup.find('header').text.strip()
        article_['titles'] = title
        # date
        date = soup.find('time').get('datetime')
        article_['dates'] = date
        # content
        content = soup.find('h2').text.strip()
        article_['contents'] = content
        # image
        image = soup.find('img').get('src')
        article_['images'] = image

    return article_

if __name__ == '__main__':
    newspaper = pd.read_csv('/Users/baeyuna/Documents/SNU_Dlab/Data/MINDlarge_{}/news.tsv'.format(args.data), delimiter='\t', header=None)
    urls = newspaper.iloc[:,5]

    with Pool(5) as p:
        articles_ = p.map(crawling_news, urls)

    with open('msn_{}.json'.format(args.data), 'w') as f:
        json.dump(articles_, f)
