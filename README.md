# SNU_DLab

* 연구 활동 기록지 

<br>

### Task 1. MIND News Recommendation

| Task |  부연설명 | 파일  |
|------|---------|------------------|
| Data Crawler | 크롤러 생성 | [msn_crawler](SNU_Dlab/msn) |
| NAML | recsys modeling | [NAML](SNU_Dlab/NAML) |
| NRMS | recsys modeling | [NRMS](SNU_Dlab/MIND2020) |


```
* How To Use *

BeautifulSoup crawling
📦msn_BS 
   |_msn_train.py 
      <run code>
      python3 msn_train.py
      python3 msn_dev.py 
   
   예외처리된 데이터 별도로 data.ipynb 실행 후 추가 데이터 수집 완료

최종 데이터와 합친 데이터 생성
📦msn_BS
   |_msn
      |_data.ipynb
 
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
