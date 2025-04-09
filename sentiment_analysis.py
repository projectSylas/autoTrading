import logging
from newsapi import NewsApiClient
from transformers import pipeline
import pandas as pd
from datetime import datetime, timedelta

# config 모듈 로드 (NewsAPI 키)
try:
    import config
except ImportError:
    logging.error("config.py 파일을 찾을 수 없습니다.")
    config = None

# --- 전역 변수 (모델 로딩 등) ---
# 감성 분석 파이프라인 (FinBERT 모델 사용 예시)
# 모델 로딩은 초기화 시 한 번만 수행하는 것이 효율적
sentiment_pipeline = None
FINBERT_MODEL_NAME = "ProsusAI/finbert" # FinBERT 모델 경로

def initialize_sentiment_model():
    """감성 분석 모델을 초기화합니다."""
    global sentiment_pipeline
    if sentiment_pipeline is None:
        try:
            logging.info(f"감성 분석 모델 로딩 시도: {FINBERT_MODEL_NAME}")
            # device=-1 은 CPU 사용을 의미 (GPU 사용 시 0 이상)
            sentiment_pipeline = pipeline("sentiment-analysis", model=FINBERT_MODEL_NAME, device=-1)
            logging.info("✅ 감성 분석 모델 로딩 완료.")
        except Exception as e:
            logging.error(f"❌ 감성 분석 모델 로딩 실패: {e}. 'transformers' 및 'torch'/'tensorflow' 설치 확인 필요.")
            sentiment_pipeline = None # 실패 시 None 유지

# --- 뉴스 데이터 수집 --- 
def fetch_recent_news(keyword: str, days_ago: int = 1, language: str = 'en', page_size: int = 20) -> list[dict]:
    """NewsAPI를 사용하여 특정 키워드에 대한 최근 뉴스를 가져옵니다.

    Args:
        keyword (str): 검색할 키워드 (예: "Bitcoin", "Federal Reserve").
        days_ago (int): 며칠 전 뉴스까지 가져올지 (기본값: 1).
        language (str): 뉴스 언어 (기본값: 'en').
        page_size (int): 가져올 뉴스 기사 최대 개수 (기본값: 20, 최대 100).

    Returns:
        list[dict]: 뉴스 기사 리스트 (각 dict는 title, description, content 등 포함).
                   오류 발생 시 빈 리스트 반환.
    """
    if not config or not config.NEWS_API_KEY:
        logging.warning("NewsAPI 키가 설정되지 않아 뉴스를 가져올 수 없습니다.")
        return []

    try:
        newsapi = NewsApiClient(api_key=config.NEWS_API_KEY)
        from_date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')

        logging.info(f"'{keyword}' 관련 뉴스 검색 (from: {from_date}, lang: {language}, size: {page_size})...")
        # everything 엔드포인트 사용 (특정 키워드 검색)
        all_articles = newsapi.get_everything(
            q=keyword,
            from_param=from_date,
            language=language,
            sort_by='relevancy', # 관련성 높은 순
            page_size=min(page_size, 100) # API 최대 100개 제한
        )

        if all_articles['status'] == 'ok':
            articles = all_articles['articles']
            logging.info(f"뉴스 {len(articles)}건 수집 완료.")
            # 필요한 정보만 추출하여 반환 가능 (예: title, description)
            return articles
        else:
            logging.error(f"NewsAPI 오류: {all_articles.get('code')} - {all_articles.get('message')}")
            return []

    except Exception as e:
        logging.error(f"NewsAPI 호출 중 오류 발생: {e}", exc_info=True)
        return []

# --- 감성 분석 --- 
def analyze_sentiment(texts: list[str]) -> str:
    """주어진 텍스트 리스트의 전반적인 감성을 분석합니다. (FinBERT 사용 가정)

    Args:
        texts (list[str]): 분석할 텍스트(뉴스 제목, 내용 등) 리스트.

    Returns:
        str: 'positive', 'negative', 'neutral' 중 하나.
             모델 로딩 실패 또는 분석 중 오류 시 'neutral' 반환.
    """
    global sentiment_pipeline
    # 모델이 초기화되지 않았으면 초기화 시도
    if sentiment_pipeline is None:
        initialize_sentiment_model()
        # 그래도 초기화 실패 시
        if sentiment_pipeline is None:
            logging.warning("감성 분석 모델 사용 불가. 'neutral' 반환.")
            return "neutral"

    if not texts:
        return "neutral"

    try:
        logging.info(f"텍스트 {len(texts)}건 감성 분석 시작...")
        # 파이프라인을 사용하여 감성 분석 수행
        # truncation=True 옵션은 모델 최대 길이 초과 시 텍스트 자름
        results = sentiment_pipeline(texts, truncation=True)

        # 결과 집계 (예: positive, negative 개수 비교)
        sentiment_counts = pd.Series([r['label'] for r in results]).value_counts()
        positive_count = sentiment_counts.get('positive', 0)
        negative_count = sentiment_counts.get('negative', 0)
        neutral_count = sentiment_counts.get('neutral', 0)

        logging.info(f"감성 분석 결과: Positive={positive_count}, Negative={negative_count}, Neutral={neutral_count}")

        # 최종 감성 결정 로직 (개선 가능)
        if positive_count > negative_count and positive_count > neutral_count:
            return "positive"
        elif negative_count > positive_count and negative_count > neutral_count:
             # 요구사항: '공포' 키워드 -> 'negative'와 매핑
             logging.warning("** 부정적 뉴스 감성 감지! **")
             return "negative" # 'bearish' 대신 모델 결과 라벨 사용
        else:
            return "neutral"

    except Exception as e:
        logging.error(f"감성 분석 중 오류 발생: {e}", exc_info=True)
        return "neutral" # 오류 시 중립 반환

# --- 통합 함수 --- 
def get_market_sentiment(keyword: str, days_ago: int = 1, language: str = 'en') -> str:
    """특정 키워드 관련 뉴스를 가져와 전반적인 시장 감성을 판단합니다.

    Returns:
        str: 'positive', 'negative', 'neutral'
    """
    logging.info(f"--- '{keyword}' 시장 감성 분석 시작 --- ")
    articles = fetch_recent_news(keyword, days_ago, language)
    if not articles:
        logging.warning("분석할 뉴스 없음. 'neutral' 반환.")
        return "neutral"

    # 분석할 텍스트 선택 (예: 제목 + 설명)
    texts_to_analyze = []
    for article in articles:
        title = article.get('title', '') or ''
        description = article.get('description', '') or ''
        # 간단히 제목과 설명 합치기 (None 방지)
        text = f"{title}. {description}".strip()
        if text != ".":
            texts_to_analyze.append(text)

    if not texts_to_analyze:
        logging.warning("분석할 텍스트 없음. 'neutral' 반환.")
        return "neutral"

    overall_sentiment = analyze_sentiment(texts_to_analyze)
    logging.info(f"--- '{keyword}' 시장 감성 분석 완료: {overall_sentiment} ---")
    return overall_sentiment

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # NewsAPI 키 설정 확인
    if not config or not config.NEWS_API_KEY or config.NEWS_API_KEY == "YOUR_NEWS_API_KEY":
        logging.warning("테스트를 위해 .env 파일에 유효한 NEWS_API_KEY를 설정해주세요.")
    else:
        # 1. 뉴스 수집 테스트
        logging.info("\n--- 뉴스 수집 테스트 (Bitcoin) ---")
        bitcoin_news = fetch_recent_news("Bitcoin", days_ago=2, page_size=5)
        if bitcoin_news:
            for i, news in enumerate(bitcoin_news):
                print(f"  {i+1}. {news['title']}")
        else:
            print("  뉴스 수집 실패 또는 결과 없음.")

        # 2. 감성 분석 테스트 (모델 초기화 필요)
        logging.info("\n--- 감성 분석 테스트 (모델 로딩 시도) ---")
        # 모델 로딩 시 시간이 걸릴 수 있음
        test_texts = [
            "Stock market surges to new highs on positive economic data.",
            "Tech stocks plummet amid fears of rising interest rates.",
            "Company reports record profits, exceeding expectations.",
            "Regulatory concerns cast shadow over crypto market."
        ]
        sentiment_result = analyze_sentiment(test_texts)
        print(f"  테스트 텍스트 감성 분석 결과: {sentiment_result}")

        # 3. 통합 함수 테스트
        logging.info("\n--- 시장 감성 통합 테스트 (Bitcoin) ---")
        btc_sentiment = get_market_sentiment("Bitcoin", days_ago=1)
        print(f"  Bitcoin 최근 뉴스 기반 시장 감성: {btc_sentiment}")

        logging.info("\n--- 시장 감성 통합 테스트 (Federal Reserve) ---")
        fed_sentiment = get_market_sentiment("Federal Reserve", days_ago=3)
        print(f"  Federal Reserve 최근 뉴스 기반 시장 감성: {fed_sentiment}") 