# SNU_DLab

* 연구 활동 기록지 

<br>

### Task 1. MIND News Recommendation

| Task |  부연설명 |  참조 사이트 및 자료 |
|------|---------|------------------|
| Data Crawler | 크롤러 생성 | [msn_crawler](SNU_Dlab/msn) |

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
