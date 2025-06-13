# 필요한 라이브러리 설치 (코드를 실행하기 전에 다음 명령어를 실행해야 합니다.)
# pip install transformers torch pandas streamlit

import pandas as pd
from transformers import pipeline
import streamlit as st

# --- 1. 외부 데이터 불러오기 (예시: CSV 파일) ---
st.header("텍스트 데이터 감성 분석")

# 더미 데이터프레임 생성 (실제 데이터를 대체)
# 사회정책 관련 댓글이나 의견이라고 가정합니다.
data = {
    'text': [
        "이 새로운 교통 정책은 도시의 교통 체증을 확실히 줄일 수 있을 것입니다. 매우 긍정적입니다.",
        "정부의 최신 주택 정책은 청년층에게 너무 큰 부담을 주는 것 같습니다. 재고가 필요합니다.",
        "환경 규제 강화는 장기적으로 필요하지만, 단기적인 산업계 타격도 고려해야 합니다.",
        "복지 예산 증가는 환영하지만, 재원 마련 방안에 대한 구체적인 계획이 부족합니다.",
        "교육 개혁안은 미래 세대에게 희망을 줄 것입니다. 지지합니다.",
        "의료 시스템 개선은 시급하지만, 현재 제안된 방식은 비효율적입니다.",
        "지방자치 강화는 지역 균형 발전에 기여할 것입니다. 긍정적인 변화입니다."
    ]
}
df = pd.DataFrame(data)

st.subheader("1. 원본 데이터")
st.dataframe(df)


# --- 2. 트랜스포머 기반 모델 로드 (감성 분석 파이프라인) ---
# st.cache_resource를 사용하여 모델을 캐시에 저장하고 재실행 시 다시 로드하지 않도록 합니다.
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

classifier = load_model()


# --- 3. 데이터 분석 (감성 예측 수행) ---
if st.button("감성 분석 실행하기"):
    with st.spinner('감성 분석을 수행 중입니다...'):
        # 감성 분석 결과를 저장할 새로운 열 추가
        df['sentiment'] = None
        df['score'] = None

        for i, row in df.iterrows():
            text = row['text']
            # 모델이 한국어를 직접 지원하지 않으므로 결과가 부정확할 수 있습니다.
            # 여기서는 예시로 영문 모델을 사용합니다.
            result = classifier(text)[0]
            df.at[i, 'sentiment'] = result['label']
            df.at[i, 'score'] = result['score']

        st.subheader("3. 감성 분석 결과")
        st.dataframe(df)

        # --- 4. 분석 결과 활용 (예시) ---
        st.subheader("4. 분석 결과 요약")
        
        # 모델 레이블 설명 추가
        st.info("""
        **모델 레이블 의미**
        - **LABEL_2**: 긍정 (Positive)
        - **LABEL_1**: 중립 (Neutral)
        - **LABEL_0**: 부정 (Negative)
        """)
        
        sentiment_counts = df['sentiment'].value_counts()
        st.bar_chart(sentiment_counts)

        most_positive_comment = df.loc[df['sentiment'] == 'LABEL_2', 'text'].head(1).iloc[0] if 'LABEL_2' in sentiment_counts.index else "없음"
        most_negative_comment = df.loc[df['sentiment'] == 'LABEL_0', 'text'].head(1).iloc[0] if 'LABEL_0' in sentiment_counts.index else "없음"

        st.markdown(f"**가장 긍정적인 댓글 (예시):** '{most_positive_comment}'")
        st.markdown(f"**가장 부정적인 댓글 (예시):** '{most_negative_comment}'")
