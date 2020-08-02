
import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from multiprocessing import Pool, Manager
# import argparse
from time import sleep
import json
import re
import sys
import time 

#list를 공유 하기 위해
manager = Manager()
article_ = manager.dict()
# article_ = dict()
url_except =[]

def crawling_news(url):
    global article_
    # print('url', urls, '에 대한 작업 PID', os.getpid())

    try:
        req = requests.get(url)
        sleep(0.5)
        #정상적인 request 확인
        if req.ok:
            html = req.text
            soup = BeautifulSoup(html,'html.parser')

            # title
            if soup.find('h1').text.strip() is not None:
                title = soup.find('h1').text.strip()
                article_['title'] = title
            elif soup.find('header').text.strip() is not None:
                title = soup.find('header').text.strip()
                article_['title'] = title
            else:
                article_['title'] = None

            # date
            try:
                date = soup.find('time').get('datetime')
                article_['date'] = date
            except:
                article_['date'] = url

            # content
            if soup.find_all('p') is not None:
                content = []
                for el in soup.find_all('p'):
                    content.append(el.get_text()) 
                article_['content'] = ' '.join(content)
            elif soup.find('h2').text.strip() is not None:
                content = soup.find('h2').text.strip()
                article_['content'] = content
            else:
                article_['content'] = url

            # image
            try:
                image = soup.find('img').get('src')
                article_['image'] = image
            except:
                article_['image'] = 'no image'

            # nid
            nid = url.split('/')[-1].split('.')[0]
            article_['nid'] = nid
    except:
        sleep(1) 
        url_except.append(url)


    return article_

if __name__ == '__main__':
    start_time = int(time.time())
    print('start crawling')
    print(start_time)
    newspaper = pd.read_csv('/Users/baeyuna/Documents/SNU_Dlab/Data/MINDlarge_train/news.tsv', delimiter='\t', header=None)
    urls = newspaper.iloc[:,5].tolist()

    pool = Pool(processes=4) #4개의 프로세스 동시에 작동
    pool.map(crawling_news, urls) 
    # print(article_)

    with open('msn_train_pool.json', 'w') as f:
        json.dump(article_, f) 

    print('-------Seconds: %s--------' % (time.time() - start_time))