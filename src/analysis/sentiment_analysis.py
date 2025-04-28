# src/analysis/sentiment_analysis.py

import os
import logging
import requests
from typing import List, Dict, Optional

# --- Project Structure Adjustment ---
# Ensure the project root is in the Python path for imports like settings
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ANALYSIS_DIR = SRC_DIR
PROJECT_ROOT = os.path.dirname(os.path.dirname(ANALYSIS_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# --- Import Configuration ---
try:
    from src.config.settings import settings
    NEWS_API_KEY = getattr(settings, 'NEWS_API_KEY', None)
except ImportError:
    print("[SentimentAnalysis ERROR] Failed to import settings. NEWS_API_KEY might not be available.")
    settings = None
    NEWS_API_KEY = None
except AttributeError:
     print("[SentimentAnalysis WARNING] NEWS_API_KEY not found in settings.")
     NEWS_API_KEY = None

# --- Logging Setup ---
# Using a specific logger for this module can be helpful
logger = logging.getLogger(__name__)
# Configure basic logging if no handlers are found (e.g., if run standalone)
if not logger.handlers:
     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def fetch_recent_news(keyword: str,
                      language: str = 'en',
                      page_size: int = 20,
                      api_key: Optional[str] = NEWS_API_KEY) -> List[Dict]:
    """
    Fetches recent news articles related to a specific keyword using the NewsAPI.

    Args:
        keyword (str): The keyword to search for (e.g., 'Bitcoin', 'stock market').
        language (str): The language of the news articles (e.g., 'en', 'ko'). Defaults to 'en'.
        page_size (int): The number of articles to fetch (max 100 for developer plan). Defaults to 20.
        api_key (Optional[str]): The NewsAPI key. Defaults to the value from settings.

    Returns:
        List[Dict]: A list of news articles (dictionaries), or an empty list if an error occurs or no key is found.
                    Each dictionary typically contains keys like 'title', 'description', 'content', 'url', 'publishedAt'.
    """
    if not api_key:
        logger.error("NewsAPI key is not configured. Cannot fetch news.")
        return []

    base_url = "https://newsapi.org/v2/everything"
    params = {
        'q': keyword,
        'language': language,
        'pageSize': min(page_size, 100), # Ensure page_size respects API limits
        'sortBy': 'publishedAt', # Get the most recent news first
        'apiKey': api_key
    }

    try:
        response = requests.get(base_url, params=params, timeout=10) # Added timeout
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()

        if data.get('status') == 'ok':
            articles = data.get('articles', [])
            logger.info(f"Successfully fetched {len(articles)} news articles for keyword '{keyword}'.")
            # Select relevant fields to return
            return [{'title': a.get('title'),
                     'description': a.get('description'),
                     'content': a.get('content'), # Note: Content might be truncated
                     'url': a.get('url'),
                     'publishedAt': a.get('publishedAt')}
                    for a in articles]
        else:
            error_message = data.get('message', 'Unknown error from NewsAPI')
            logger.error(f"NewsAPI returned an error: {data.get('code')} - {error_message}")
            return []

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching news from NewsAPI: {e}", exc_info=True)
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred during news fetching: {e}", exc_info=True)
        return []


def analyze_sentiment(texts: List[str]) -> str:
    """
    Analyzes the overall sentiment of a list of text snippets (e.g., news titles, descriptions).

    Args:
        texts (List[str]): A list of text strings to analyze.

    Returns:
        str: The overall sentiment summary ("bullish", "bearish", "neutral").
             Defaults to "neutral" if analysis is inconclusive or texts are empty.
    """
    if not texts:
        return "neutral"

    # --- Simple Keyword-Based Sentiment Analysis (Placeholder) ---
    # Define keywords associated with bullish and bearish sentiment
    # This is a very basic approach and should be replaced with a more robust model (e.g., FinBERT)
    bullish_keywords = ['surge', 'rally', 'optimism', 'gain', 'positive', 'upbeat', 'record', 'high', 'profit', 'boom', 'strong', 'rise', 'growth']
    bearish_keywords = ['plunge', 'fear', 'recession', 'loss', 'negative', 'downbeat', 'slump', 'low', 'crisis', 'weak', 'fall', 'decline', 'risk', 'warning', 'concern', 'drop']

    bullish_score = 0
    bearish_score = 0

    for text in texts:
        if not text: continue # Skip empty strings
        text_lower = text.lower()
        for keyword in bullish_keywords:
            if keyword in text_lower:
                bullish_score += 1
        for keyword in bearish_keywords:
            if keyword in text_lower:
                bearish_score += 1

    logger.debug(f"Sentiment analysis scores: Bullish={bullish_score}, Bearish={bearish_score} from {len(texts)} texts.")

    # Determine overall sentiment based on scores
    if bullish_score > bearish_score * 1.5: # Require a stronger bullish signal
        return "bullish"
    elif bearish_score > bullish_score * 1.2: # Bearish signal threshold
        return "bearish"
    else:
        return "neutral"

    # --- TODO: Replace with FinBERT or other sentiment analysis model ---
    # Example structure (requires transformers library and a pre-trained model):
    # from transformers import pipeline
    # sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    # results = sentiment_pipeline(texts)
    # # Process results to get an overall score...


def get_market_sentiment(keyword: str, use_content: bool = False) -> str:
    """
    Fetches recent news for a keyword and analyzes its sentiment to determine
    the overall market sentiment regarding that keyword.

    Args:
        keyword (str): The keyword to analyze market sentiment for.
        use_content (bool): Whether to include the (potentially truncated) article content
                            in the sentiment analysis, in addition to title and description.
                            Defaults to False as content might be noisy or incomplete.

    Returns:
        str: The determined market sentiment ("bullish", "bearish", "neutral").
    """
    logger.info(f"Getting market sentiment for keyword: '{keyword}'")
    articles = fetch_recent_news(keyword=keyword)

    if not articles:
        logger.warning("No news articles fetched, returning neutral sentiment.")
        return "neutral"

    # Extract relevant text for analysis
    texts_to_analyze = []
    for article in articles:
        if article.get('title'):
            texts_to_analyze.append(article['title'])
        if article.get('description'):
            texts_to_analyze.append(article['description'])
        if use_content and article.get('content'):
             # Be cautious with content as it might be truncated or irrelevant
             texts_to_analyze.append(article['content'])

    if not texts_to_analyze:
         logger.warning("No text could be extracted from fetched articles, returning neutral sentiment.")
         return "neutral"

    overall_sentiment = analyze_sentiment(texts_to_analyze)
    logger.info(f"Determined market sentiment for '{keyword}': {overall_sentiment}")
    return overall_sentiment


# --- Example Usage ---
if __name__ == "__main__":
    # Ensure logger is configured for direct execution
    if not logging.getLogger(__name__).handlers:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if not NEWS_API_KEY:
         logger.error("Cannot run example: NEWS_API_KEY is not set in environment variables or settings.")
    else:
        # Example: Get sentiment for Bitcoin
        btc_sentiment = get_market_sentiment("Bitcoin")
        print(f"\nExample: Market sentiment for Bitcoin: {btc_sentiment}")

        # Example: Get sentiment for the general stock market
        market_sentiment = get_market_sentiment("stock market")
        print(f"Example: Market sentiment for Stock Market: {market_sentiment}")

        # Example: Fetch news directly
        # news = fetch_recent_news("Federal Reserve", language='en', page_size=5)
        # print("\nExample: Fetched news for Federal Reserve:")
        # for i, article in enumerate(news):
        #     print(f"  {i+1}. {article.get('title')} ({article.get('publishedAt')})") 