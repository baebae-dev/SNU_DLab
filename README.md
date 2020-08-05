# SNU_DLab

* 연구 활동 기록지 

<br>

### Task 1. MIND News Recommendation

| Task |  부연설명 | 파일  |
|------|---------|------------------|
| Data Crawler | 크롤러 생성 | [msn_crawler](SNU_Dlab/msn) |
| Bert Embedding | 기사 내용 임베딩 | [Bert_Embedding](SNU_Dlab/Bert_Embedding) |


```
* How To Use *

1. msn_large(SNU_Dlab/msn_large)
   : for large data set train & test
   <run code>
   scrapy crawl msn_large -o 저장할 파일명 -t 저장포멧


2. msn(SNU_Dlab/msn)
   : for small data set train & test
   <run code>
   scrapy crawl msn -o 저장할 파일명 -t 저장포멧

3. all using crawler(SNU_Dlab/Crawler)
   : for large. small data train & test
    estract features ['vert', 'subvert', 'nid', 'content', 'title', 'date;, 'image url']
    <run code>
    scrapy crawl msn -o 저장할 파일명 -t 저장포멧

4. BeautifulSoup crawling
   4-1. msn_BS 
         |_msn_train.py 
            <run code>
            python3 msn_train.py
   4-2.  msn_BS 
         |_msn_dev.py 
            <run code>
            python3 msn_dev.py 
```


```
* 해당 프로젝트를 하며 배운점 😊

- 해당 깃 코드 및 환경 설정에 문제가 있을경우 issue를 달고 해결하는게 좋다는 것을 깨달았다.
- os.environ([사용자 지정 PATH]) 지정? 
   : export 사용자 지정 PATH=해당 데이터 
- Scrapy 사용법 (crawler 레포 참조)
- zsh: no matches found: 에러 발생시 
   : unsetopt nomatch 코드 실행 

```

<br>
