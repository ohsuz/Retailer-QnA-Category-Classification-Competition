![image](https://user-images.githubusercontent.com/59340911/132998697-0338971b-11f0-4857-ad97-ed64bced3736.png)

---

## 소상공인 QnA 카테고리 분류 대회
> 소비자 문의사항 대화 내용의 카테고리 분류 모델 개발   
> 자연어 영역 | 개방형 문제 | Accuracy
- 문제정의
   - 디지털/가전 제품 관련 소비자 문의 질문-답변으로 이루어진 대화를 118가지 카테고리로 분류하는 과제
- 추진배경
   - 서비스 업계에서 고객의 문의사항은 체계적으로 수집, 분석, 활용 가능한 AI 모델의 필요성이 요구되어짐
   - 기업의 고객만족도와 충성도 향상을 위해 고객 문의사항을 카테고리화를 하여 체계적으로 분석하고자 함
- 활용 가능 사례
   - 콜센터 상담원 기록 기반 VOC 분석 모델
   - 고객 후기 내역 기발 감정분류 모델
---

## Code Description

### 선행되어야 하는 것
- wandb 코드를 따로 주석처리하지 않아 wandb 로그인이 필요합니다.
- wandb 로그인 후, train.py 파일 146번째 line에서 entity명 본인 계정명으로 수정

### 실행 순서
1. `bash train.sh`
2. `bash predict.sh`
3. `SoftVoting.ipynb` 노트북 전체 실행
4. `HardVoting.ipynb` 노트북 전체 실행 → 최종 result.csv 완성

---

## 🌳 Tree
```
aiconnect
|-- HardVoting.ipynb
|-- SoftVoting.ipynb
|-- config
|   |-- predict_config_bert.yml
|   |-- predict_config_bert_all.yml
|   |-- predict_config_electra.yml
|   |-- predict_config_electra_all.yml
|   |-- predict_config_funnel.yml
|   |-- train_config_bert.yml
|   |-- train_config_bert_all.yml
|   |-- train_config_electra.yml
|   |-- train_config_electra_all.yml
|   `-- train_config_funnel.yml
|-- data
|   |-- test
|   |   |-- test.csv
|   |   |-- test_kykim_bert_X.pt
|   |   |-- test_kykim_electra_X.pt
|   |   `-- test_kykim_funnel_X.pt
|   |-- train
|   |   |-- train.csv
|   |   |-- train_all.csv
|   |   |-- train_kykim_bert_X.pt
|   |   |-- train_kykim_bert_Y.pt
|   |   |-- train_kykim_bert_all_X.pt
|   |   |-- train_kykim_bert_all_Y.pt
|   |   |-- train_kykim_electra_X.pt
|   |   |-- train_kykim_electra_Y.pt
|   |   |-- train_kykim_electra_all_X.pt
|   |   |-- train_kykim_electra_all_Y.pt
|   |   |-- train_kykim_funnel_X.pt
|   |   `-- train_kykim_funnel_Y.pt
|   `-- val
|       |-- val.csv
|       |-- val_kykim_bert_X.pt
|       |-- val_kykim_bert_Y.pt
|       |-- val_kykim_electra_X.pt
|       |-- val_kykim_electra_Y.pt
|       |-- val_kykim_funnel_X.pt
|       `-- val_kykim_funnel_Y.pt
|-- logits
|   |-- bert.npy
|   |-- bert_all.npy
|   |-- electra.npy
|   |-- electra_all.npy
|   `-- funnel.npy
|-- models
|   |-- bert.pt
|   |-- bert_all.pt
|   |-- electra.pt
|   |-- electra_all.pt
|   |-- funnel.pt
|-- modules
|   |-- dataset.py
|   |-- earlystoppers.py
|   |-- metrics.py
|   |-- model.py
|   |-- recorders.py
|   |-- schedulers.py
|   |-- trainer.py
|   `-- utils.py
|-- predict.py
|-- predict.sh
|-- records
|-- submissions
|   |-- bert.csv
|   |-- bert_all.csv
|   |-- electra.csv
|   |-- electra_all.csv
|   |-- funnel.csv
|   |-- soft.csv
|   `-- result.csv
|-- train.py
|-- train.sh
`-- wandb
```
