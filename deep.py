# 필요한 라이브러리 설치 (코드를 실행하기 전에 다음 명령어를 실행해야 합니다.)
# pip install transformers torch pandas

import pandas as pd
from transformers import pipeline

# --- 1. 외부 데이터 불러오기 (예시: CSV 파일) ---
print(\"--- 1. 외부 텍스트 데이터 불러오기 및 준비 ---\")

# 더미 데이터프레임 생성 (실제 데이터를 대체)
# 사회정책 관련 댓글이나 의견이라고 가정합니다.
data = {
    'text': [
        \"이 새로운 교통 정책은 도시의 교통 체증을 확실히 줄일 수 있을 것입니다. 매우 긍정적입니다.\",
        \"정부의 최신 주택 정책은 청년층에게 너무 큰 부담을 주는 것 같습니다. 재고가 필요합니다.\",
        \"환경 규제 강화는 장기적으로 필요하지만, 단기적인 산업계 타격도 고려해야 합니다.\",
        \"복지 예산 증가는 환영하지만, 재원 마련 방안에 대한 구체적인 계획이 부족합니다.\",
        \"교육 개혁안은 미래 세대에게 희망을 줄 것입니다. 지지합니다.\",
        \"의료 시스템 개선은 시급하지만, 현재 제안된 방식은 비효율적입니다.\",
        \"지방자치 강화는 지역 균형 발전에 기여할 것입니다. 긍정적인 변화입니다.\"
    ]
}
df = pd.DataFrame(data)

print(\"불러온 텍스트 데이터:\")
print(df)
print(\"-\" * 50)

# --- 2. 트랜스포머 기반 모델 로드 (감성 분석 파이프라인) ---
print(\"--- 2. 트랜스포머 기반 감성 분석 모델 로드 ---\")
classifier = pipeline(\"sentiment-analysis\", model=\"cardiffnlp/twitter-roberta-base-sentiment-latest\")
print(\"모델 로드 완료: 'cardiffnlp/twitter-roberta-base-sentiment-latest'\")
print(\"-\" * 50)

# --- 3. 데이터 분석 (감성 예측 수행) ---
print(\"--- 3. 텍스트 데이터 감성 분석 결과 ---\")
df['sentiment'] = None
df['score'] = None

for i, row in df.iterrows():
    text = row['text']
    result = classifier(text)[0]
    df.at[i, 'sentiment'] = result['label']
    df.at[i, 'score'] = result['score']

print(df)
print(\"-\" * 50)

# --- 4. 분석 결과 활용 (예시) ---
print(\"--- 4. 분석 결과 요약 ---\")
sentiment_counts = df['sentiment'].value_counts()
print(\"\\n감성별 댓글 수:\")
print(sentiment_counts)

most_positive_comment = df.loc[df['sentiment'] == 'LABEL_2', 'text'].head(1).iloc[0] if 'LABEL_2' in sentiment_counts else \"없음\"
most_negative_comment = df.loc[df['sentiment'] == 'LABEL_0', 'text'].head(1).iloc[0] if 'LABEL_0' in sentiment_counts else \"없음\"

print(f\"\\n가장 긍정적인 댓글 (예시): '{most_positive_comment}'\")
print(f\"가장 부정적인 댓글 (예시): '{most_negative_comment}'\")

# --- 추가 설명: 모델 미세 조정 (Fine-tuning) ---
print(\"\\n--- 모델 미세 조정 (Fine-tuning)에 대한 추가 설명 ---\")
print(\"만약 특정 도메인(예: 한국 사회과학 논문, 특정 정책 보고서)에 특화된 분석을 원한다면,\")
print(\"사전 학습된 LLM을 해당 도메인의 레이블링된 데이터로 '미세 조정(Fine-tuning)'해야 합니다.\")
print(\"이는 모델이 특정 용어, 문맥, 감성을 더 정확하게 이해하고 분류하도록 훈련시키는 과정입니다.\")
print(\"Hugging Face `Trainer` API를 사용하면 미세 조정을 비교적 쉽게 수행할 수 있습니다.\")
print(\"예시 코드는 복잡하여 여기에 직접 포함하기 어렵지만, 주요 단계는 다음과 같습니다:\")
print(\"1. 특정 도메인의 레이블링된 데이터셋 준비 (텍스트와 해당 레이블).\")
print(\"2. 데이터셋을 토큰화하고 모델 입력 형식에 맞게 변환.\")
print(\"3. 사전 학습된 모델과 옵티마이저, 스케줄러 설정.\")
print(\"4. `Trainer` 객체를 사용하여 모델 훈련 (fine-tuning).\")
print(\"5. 훈련된 모델을 저장하고 새로운 데이터에 적용.\")
