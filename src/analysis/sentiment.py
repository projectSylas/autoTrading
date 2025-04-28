import logging
import torch # For device detection
from newsapi import NewsApiClient
from transformers import pipeline, AutoTokenizer
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import numpy as np

# 설정 모듈 로드
from src.config.settings import settings
# DB 로깅 함수 임포트 시도
try:
    from src.utils.database import log_sentiment_to_db
except ImportError:
    log_sentiment_to_db = None
    logging.warning("Database logging function (log_sentiment_to_db) not found. DB logging disabled for sentiment.")

# --- 전역 변수 (모델 로딩 등) ---
sentiment_pipeline = None
sentiment_tokenizer = None
_model_loaded = False

def _get_device() -> str:
    """Determine the device to run the model on based on settings."""
    if settings.HF_DEVICE == 'auto':
        return "cuda" if torch.cuda.is_available() else "cpu"
    return settings.HF_DEVICE

def initialize_sentiment_model(force_reload: bool = False):
    """Hugging Face 감성 분석 모델과 토크나이저를 초기화합니다."""
    global sentiment_pipeline, sentiment_tokenizer, _model_loaded
    
    if not settings.ENABLE_HF_SENTIMENT:
        logging.info("Hugging Face sentiment analysis is disabled in settings.")
        sentiment_pipeline = None
        sentiment_tokenizer = None
        _model_loaded = False
        return

    if _model_loaded and not force_reload:
        return

    try:
        device_name = _get_device()
        device_id = 0 if device_name == 'cuda' else -1 # pipeline expects device index
        logging.info(f"Initializing Hugging Face sentiment model: {settings.HF_SENTIMENT_MODEL_NAME} on device: {device_name}")

        # Load tokenizer first to get max length if needed (though pipeline handles truncation)
        sentiment_tokenizer = AutoTokenizer.from_pretrained(
            settings.HF_SENTIMENT_MODEL_NAME,
            cache_dir=settings.MODEL_WEIGHTS_DIR # Use configured cache/save directory
        )
        
        sentiment_pipeline = pipeline(
            "sentiment-analysis", 
            model=settings.HF_SENTIMENT_MODEL_NAME, 
            tokenizer=sentiment_tokenizer, # Reuse loaded tokenizer
            device=device_id, 
            cache_dir=settings.MODEL_WEIGHTS_DIR
        )
        _model_loaded = True
        logging.info("✅ Hugging Face sentiment model and tokenizer loaded successfully.")

    except Exception as e:
        logging.error(f"❌ Failed to load Hugging Face sentiment model: {e}. Check model name, paths, and dependencies ('transformers', 'torch').")
        sentiment_pipeline = None
        sentiment_tokenizer = None
        _model_loaded = False
        # Optionally send Slack alert here about model loading failure

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
    if not settings or not settings.NEWS_API_KEY:
        logging.warning("NewsAPI 키가 settings에 설정되지 않아 뉴스를 가져올 수 없습니다.")
        return []

    try:
        newsapi = NewsApiClient(api_key=settings.NEWS_API_KEY)
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

# --- 감성 분석 (수정됨) --- 
def analyze_sentiment_hf(texts: List[str]) -> List[Dict[str, float | str]]:
    """주어진 텍스트 리스트의 감성을 개별적으로 분석하고 점수를 정규화합니다.

    Args:
        texts (list[str]): 분석할 텍스트 리스트.

    Returns:
        list[dict]: 각 텍스트에 대한 분석 결과 리스트.
                    각 dict는 {'label': str, 'score': float} 형태.
                    - label: 'positive', 'neutral', 'negative'
                    - score: 0.0 (가장 부정적) ~ 1.0 (가장 긍정적) 사이의 값.
                    모델 로딩 실패 또는 분석 오류 시 빈 리스트 또는 기본값 리스트 반환.
    """
    global sentiment_pipeline, sentiment_tokenizer
    
    if not _model_loaded:
        initialize_sentiment_model() # Try to initialize if not loaded
        if not _model_loaded: # Still not loaded
             logging.warning("Sentiment model not available. Returning neutral scores.")
             return [{"label": "neutral", "score": 0.5} for _ in texts]

    if not texts:
        return []

    results_list = []
    try:
        logging.info(f"Analyzing sentiment for {len(texts)} texts using {settings.HF_SENTIMENT_MODEL_NAME}...")
        # Run inference
        predictions = sentiment_pipeline(
            texts, 
            truncation=True, 
            max_length=min(settings.HF_MAX_SEQ_LENGTH, sentiment_tokenizer.model_max_length)
        )

        # Process predictions
        for pred in predictions:
            raw_label = pred['label'].lower()
            raw_score = pred['score']
            final_label = "neutral" # Default to neutral
            normalized_score = 0.5 # Default score for neutral

            if raw_label == 'positive':
                # Map positive score (0~1) to normalized score (0.5 ~ 1.0)
                normalized_score = 0.5 + (raw_score / 2.0)
                if normalized_score >= settings.SENTIMENT_NEUTRAL_THRESHOLD_HIGH:
                    final_label = "positive"
            elif raw_label == 'negative':
                # Map negative score (0~1) to normalized score (0.0 ~ 0.5)
                normalized_score = 0.5 - (raw_score / 2.0)
                if normalized_score <= settings.SENTIMENT_NEUTRAL_THRESHOLD_LOW:
                    final_label = "negative"
            # Handle cases where the model might output 'neutral' directly if supported
            elif raw_label == 'neutral':
                 normalized_score = 0.5 # Keep score as 0.5 for neutral
                 # Label is already neutral

            results_list.append({"label": final_label, "score": round(normalized_score, 4)})
            
        logging.info(f"Sentiment analysis completed for {len(texts)} texts.")

    except Exception as e:
        logging.error(f"Error during sentiment analysis: {e}", exc_info=True)
        # Return neutral defaults on error for all texts
        results_list = [{"label": "neutral", "score": 0.5} for _ in texts]

    return results_list

# --- 통합 함수 (수정됨) --- 
def get_market_sentiment(keyword: str, days_ago: int = 1, language: str = 'en') -> Tuple[str, float, pd.DataFrame]:
    """특정 키워드 관련 뉴스를 가져와 전반적인 시장 감성 라벨, 평균 점수, 기사 DataFrame을 반환합니다.
       DB 로깅은 log_sentiment_to_db 함수가 있을 경우 시도합니다.

    Returns:
        Tuple[str, float, pd.DataFrame]: 
            (전체 감성 라벨, 평균 정규화 점수, 원본 뉴스 기사 DataFrame)
            오류 또는 분석 불가 시 ('neutral', 0.5, 빈 DataFrame) 반환.
    """
    global log_sentiment_to_db
    logging.info(f"--- Analyzing market sentiment for '{keyword}' --- ")
    
    # Check if HF sentiment is enabled, if not, return neutral immediately
    if not settings.ENABLE_HF_SENTIMENT:
         logging.info("HF Sentiment analysis disabled. Returning neutral.")
         # Log disabled status to DB if needed
         if log_sentiment_to_db:
              try: 
                  log_sentiment_to_db(keyword=keyword, sentiment='disabled', score=0.5, article_count=0, model_name='N/A')
              except Exception as db_err:
                  logging.error(f"DB logging failed for disabled sentiment: {db_err}")
         return "neutral", 0.5, pd.DataFrame()

    # Fetch news
    articles = fetch_recent_news(keyword, days_ago, language)
    articles_df = pd.DataFrame(articles)

    if not articles:
        logging.warning("No news found for analysis. Returning neutral.")
        if log_sentiment_to_db:
            try:
                 log_sentiment_to_db(keyword=keyword, sentiment='no_data', score=0.5, article_count=0, model_name=settings.HF_SENTIMENT_MODEL_NAME)
            except Exception as db_err:
                 logging.error(f"DB logging failed for no news data: {db_err}")
        return "neutral", 0.5, pd.DataFrame()

    # Prepare texts for analysis
    texts_to_analyze = []
    for article in articles:
        title = article.get('title', '') or ''
        description = article.get('description', '') or ''
        # Use content if available and description is short?
        # content = article.get('content', '') or '' 
        text = f"{title}. {description}".strip()
        # Skip empty or placeholder texts
        if text and text != ".":
            texts_to_analyze.append(text)

    if not texts_to_analyze:
        logging.warning("No valid text content found in fetched news. Returning neutral.")
        # Log appropriately if needed
        return "neutral", 0.5, articles_df

    # Analyze sentiment for all texts
    sentiment_results = analyze_sentiment_hf(texts_to_analyze)

    if not sentiment_results:
        logging.warning("Sentiment analysis returned no results. Returning neutral.")
        # Log appropriately if needed
        return "neutral", 0.5, articles_df

    # Calculate overall sentiment and average score
    scores = [res['score'] for res in sentiment_results]
    average_score = round(np.mean(scores), 4) if scores else 0.5
    
    # Determine overall label based on average score and thresholds
    overall_label = "neutral"
    if average_score >= settings.SENTIMENT_NEUTRAL_THRESHOLD_HIGH:
        overall_label = "positive"
    elif average_score <= settings.SENTIMENT_NEUTRAL_THRESHOLD_LOW:
        overall_label = "negative"
        # Log warning for strong negative sentiment
        logging.warning(f"** Strong negative sentiment detected for '{keyword}' (Avg Score: {average_score:.2f}) **")

    # Log results to DB if function is available
    if log_sentiment_to_db:
        try:
            log_sentiment_to_db(
                keyword=keyword,
                sentiment=overall_label,
                score=average_score,
                article_count=len(articles),
                model_name=settings.HF_SENTIMENT_MODEL_NAME
            )
            logging.info(f"[DB] Sentiment log saved for '{keyword}'.")
        except Exception as db_err:
            logging.error(f"Sentiment DB logging failed for '{keyword}': {db_err}")

    logging.info(f"--- Market sentiment analysis complete for '{keyword}': Label={overall_label}, Avg Score={average_score:.2f} --- ")
    return overall_label, average_score, articles_df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Ensure model initialization is attempted if enabled
    if settings.ENABLE_HF_SENTIMENT:
         initialize_sentiment_model()

    # NewsAPI Key Check (using settings)
    if not settings.NEWS_API_KEY:
        logging.warning("Please set a valid NEWS_API_KEY in the .env file for testing.")
    else:
        # 1. News Fetch Test
        logging.info("\n--- News Fetch Test (Bitcoin) ---")
        bitcoin_news = fetch_recent_news("Bitcoin", days_ago=2, page_size=5)
        if bitcoin_news:
            for i, news in enumerate(bitcoin_news):
                 print(f"  {i+1}. {news['title']}")
        else:
             print("  News fetch failed or no results.")

        # 2. Sentiment Analysis Test (using new function)
        logging.info("\n--- Sentiment Analysis Test ---")
        if settings.ENABLE_HF_SENTIMENT and _model_loaded:
            test_texts = [
                "Stock market surges to new highs on positive economic data.",
                "Tech stocks plummet amid fears of rising interest rates.",
                "Company reports record profits, exceeding expectations.",
                "Regulatory concerns cast shadow over crypto market.",
                "The weather today is quite pleasant."
            ]
            analysis_results = analyze_sentiment_hf(test_texts)
            print("  Individual Sentiment Analysis Results:")
            for text, result in zip(test_texts, analysis_results):
                print(f"    Text: {text[:50]}... -> Label: {result['label']}, Score: {result['score']:.3f}")
        elif not settings.ENABLE_HF_SENTIMENT:
             print("  Sentiment analysis disabled in settings.")
        else:
             print("  Sentiment model failed to load, analysis skipped.")

        # 3. Integrated Market Sentiment Test
        logging.info("\n--- Integrated Market Sentiment Test (Bitcoin) ---")
        btc_label, btc_score, _ = get_market_sentiment("Bitcoin", days_ago=1)
        print(f"  Bitcoin Market Sentiment: Label={btc_label}, Avg Score={btc_score:.3f}")

        logging.info("\n--- Integrated Market Sentiment Test (Federal Reserve) ---")
        fed_label, fed_score, _ = get_market_sentiment("Federal Reserve", days_ago=3)
        print(f"  Federal Reserve Market Sentiment: Label={fed_label}, Avg Score={fed_score:.3f}") 