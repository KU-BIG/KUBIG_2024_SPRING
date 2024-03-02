# KUBIG CONTEST NLP TEAM 2 
## ITSUM: IT/Science Article Summarizaiton & Keyword-Extraction 

This repository is the implementation of Basic Study: NLP Team 2.
Our team have made a platform service which sends the summarization & keywords of IT/Science articles within 7 days through Steamlit, which is not shown in our presentation yet.

<img width="500" alt="image" src="https://github.com/KU-BIG/KUBIG_2024_SPRING/assets/90594374/e21cfa97-3a97-4abc-bac6-d83b1582cf0a">


---
### Using Model: 
- DBSCAN [1]
- KoBERT [2]
  <img width="500" alt="image" src="https://github.com/KU-BIG/KUBIG_2024_SPRING/assets/90594374/57cf5cf2-8840-48c5-898d-7395918c74b9">
- KoBART [3]
  <img width="500" alt="image" src="https://github.com/KU-BIG/KUBIG_2024_SPRING/assets/90594374/8c40e67d-8b68-4698-aa23-303b6bf85adb">

---
### Platform Flow Chart
![image](https://github.com/KU-BIG/KUBIG_2024_SPRING/assets/90594374/18781183-1e90-4940-8a01-1b4ea876f68f)


---
### Evaluation

- Rouge-1 Score(KoBART):

| Fine-Tuned KoBART | Rouge-1 | Rouge-2 | Rouge-L |
|-------------------|:-------:|:-------:|---------|
| Recall            |  0.6078 |  0.537  |  0.6078 |
| Precision         |    1    |  0.9667 |    1    |
| F1-Score          |  0.7561 |  0.6904 |  0.7561 |

- Examples:
```
input: (지디넷코리아=신영빈 기자) 국내 서빙로봇 선두기업 브이디컴퍼니가 외식업 프랜차이즈 기업 도야에프앤비의 가맹점 디지털전환(DX)에 나선다.브이디컴퍼니는 지난 20일 브이디컴퍼니 본사에서 도야에프앤비와 가맹점 매장 자동화 공동 추진을 위한 업무협약(MOU)을 체결했다고 21일 밝혔다.도야에프앤비는 이번 협약으로 전국 가맹점 340곳과 오픈 예정 매장을 대상으로 서빙로봇 '브이디로봇'과 테이블오더 '브이디메뉴'를 도입한다. 매장 자동화 솔루션을 공급해 직원들의 업무 강도는 낮추고 대면 서비스의 질을 개선해 매장 운영 시스템 효율을 극대화한다는 계획이다.함판식 브이디컴퍼니 대표는 "고물가에 최저임금 인상, 인력난까지 겹치며 매장 자동화 솔루션이 국내 외식업계에서 없어서는 안 될 솔루션으로 자리잡아 가고 있다"며 "많은 외식업장에서 매장 운영 효율화 및 비용 절감을 이룰 수 있기를 기대한다"고 말했다. 신영빈 기자(burger@zdnet.co.kr)

output: (지디넷코리아=신영빈 기자)국내 서빙로봇 선두기업 브이디컴퍼니가 외식업 프랜차이즈 기업 도야에프앤비의 가맹점 디지털전환(DX)에 나선다. 매장 자동화 솔루션을 공급해 직원들의 업무 강도는 낮추고 대면 서비스의 질을 개선해 매장 운영 시스템 효율을 극대화한다는 계획이다. 
```

---
### Role:
- 17기_서지민: KoBERT Modeling & Presentation
- 16기_임정준: KoBART Modeling & Crawling & Git
- 19기_최주희: Keyword Extraction & Streamlit

---
### Citation
[1] Ester, Martin; Kriegel, Hans-Peter; Sander, Jörg; Xu, Xiaowei (1996). Simoudis, Evangelos; Han, Jiawei; Fayyad, Usama M. (eds.). A density-based algorithm for discovering clusters in large spatial databases with noise (PDF). Proceedings of the Second International Conference on Knowledge Discovery and Data Mining (KDD-96). AAAI Press. pp. 226–231. CiteSeerX 10.1.1.121.9220. ISBN 1-57735-004-9.

[2] SKTBrain, https://github.com/SKTBrain/KoBERT

[3] SKT-AI, https://github.com/SKT-AI/KoBART
