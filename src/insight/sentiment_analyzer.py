import logging
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
# Import settings
try:
    from src.config import settings
except ImportError:
    # Fallback if settings cannot be imported (e.g., during initial setup)
    settings = type('obj', (object,), {
        'NEWS_API_KEY': os.getenv("NEWS_API_KEY"), 
        'SENTIMENT_CACHE_DIR': 'data/sentiment_cache/'
    })()
    logging.getLogger(__name__).warning("Could not import settings module. Using basic os.getenv fallbacks.")

# Setup logging according to project rules (assuming a central config exists)
# For now, use basic logging
logger = logging.getLogger(__name__)
# Configure logging only if it hasn't been configured by a central setup
if not logger.hasHandlers():
     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Constants ---
DEFAULT_FINBERT_MODEL = "ProsusAI/finbert" # Example model
# Use cache dir from settings
SENTIMENT_CACHE_DIR = settings.SENTIMENT_CACHE_DIR 

# --- Sentiment Analysis Function ---
def analyze_text_sentiment(text: str,
                           model_name: str = DEFAULT_FINBERT_MODEL,
                           device: int = -1 # -1 for CPU, 0 for CUDA device 0 etc.
                           ) -> dict[str, float]:
    """
    Analyzes the sentiment of a given text using a pre-trained financial model.

    Args:
        text (str): The input text to analyze.
        model_name (str): The name of the Hugging Face model to use (e.g., 'ProsusAI/finbert').
        device (int): Device for transformers pipeline (-1 CPU, >=0 GPU).

    Returns:
        dict[str, float]: A dictionary containing label ('positive', 'negative', 'neutral')
                          and its corresponding score. Returns empty dict on error.

    Raises:
        ImportError: If transformers library is not installed.
        Exception: For model loading or prediction errors.
    """
    if not text:
        logger.warning("Input text is empty, cannot analyze sentiment.")
        return {}

    try:
        # Load model and tokenizer using pipeline for simplicity
        # Ensure model exists or handle potential errors during loading
        sentiment_pipeline = pipeline(
            "text-classification",
            model=model_name,
            tokenizer=model_name,
            device=device # Use specified device
        )
        # logger.info(f"Sentiment analysis model loaded: {model_name}") # Reduced verbosity

        # Perform prediction
        results = sentiment_pipeline(text)

        # Assuming the pipeline returns a list with one dictionary like [{'label': 'positive', 'score': 0.9...}]
        if results and isinstance(results, list) and isinstance(results[0], dict):
            sentiment_result = results[0]
            # logger.info(f"Sentiment analysis result: Label={sentiment_result.get('label')}, Score={sentiment_result.get('score'):.4f}") # Reduced verbosity
            return {'label': sentiment_result.get('label'), 'score': sentiment_result.get('score')}
        else:
            logger.error(f"Unexpected output format from sentiment pipeline: {results}")
            return {}

    except ImportError:
        logger.error("Transformers library not found. Please install it: pip install transformers")
        raise
    except Exception as e:
        logger.error(f"Error during sentiment analysis using model {model_name}: {e}", exc_info=True)
        # Depending on desired behavior, could return default score or re-raise
        return {} # Return empty dict on error

# --- Placeholder: News Fetching Function ---
def fetch_news_for_ticker(ticker: str, date: str, api_key: str | None = None) -> list[str]:
    """
    Fetches news article headlines or snippets for a given ticker and date.
    Uses NewsAPI as an example source.

    Args:
        ticker (str): The stock ticker symbol.
        date (str): The target date (YYYY-MM-DD).
        api_key (str | None): API key for NewsAPI. If None, uses dummy data.

    Returns:
        list[str]: A list of news text snippets (headlines/titles).
    """
    if not api_key:
        logger.warning(f"NewsAPI key not provided for {ticker} on {date}. Returning dummy data.")
        # Return dummy data if no API key
        return [
            f"Placeholder news about {ticker} on {date}: Company announces positive earnings.",
            f"Another news snippet for {ticker}: Sector outlook remains uncertain."
        ]

    # logger.info(f"Fetching news for {ticker} on {date} using NewsAPI...") # Reduced verbosity
    try:
        from newsapi import NewsApiClient
        newsapi = NewsApiClient(api_key=api_key)
        # Search for news around the target date, limit results
        # Note: NewsAPI free tier has limitations (e.g., date range, sources)
        all_articles = newsapi.get_everything(q=ticker,
                                                # Use a range around the date for better results
                                                from_param=date, 
                                                to=date,
                                                language='en',
                                                sort_by='relevancy', # Or 'publishedAt'
                                                page_size=10) # Limit number of articles
        
        articles_found = all_articles.get('articles', [])
        headlines = [article['title'] for article in articles_found if article and article.get('title')]
        
        if not headlines:
            logger.warning(f"No articles found for {ticker} on {date} via NewsAPI.")
            return []
            
        logger.info(f"Fetched {len(headlines)} headlines for {ticker} on {date} from NewsAPI.")
        # logger.debug(f"Headlines: {headlines}") # Uncomment for debugging
        return headlines
        
    except ImportError:
        logger.error("newsapi-python library not found. Please install it: pip install newsapi-python")
        return [] # Return empty list if library is missing
    except Exception as e:
        # Catch potential API errors (rate limits, invalid key, etc.)
        logger.error(f"Error fetching news from NewsAPI for {ticker} on {date}: {e}", exc_info=True)
        return [] # Return empty list on error

# --- Sentiment Caching Functions ---
def load_cached_sentiment(ticker: str, date_str: str) -> float | None:
     """Loads sentiment score from a simple cache file if it exists."""
     # SENTIMENT_CACHE_DIR is now defined globally using settings
     cache_file = os.path.join(SENTIMENT_CACHE_DIR, f"{ticker.replace('/', '_')}_{date_str}.txt")
     if os.path.exists(cache_file):
         try:
             with open(cache_file, 'r') as f:
                 score = float(f.read().strip())
                 logger.debug(f"Loaded cached sentiment for {ticker} on {date_str}: {score:.4f}")
                 return score
         except Exception as e:
             logger.warning(f"Error reading cache file {cache_file}: {e}. Will refetch.")
     return None

def save_sentiment_to_cache(ticker: str, date_str: str, score: float):
     """Saves a sentiment score to a simple cache file."""
     try:
         os.makedirs(SENTIMENT_CACHE_DIR, exist_ok=True) # Ensure directory exists
         cache_file = os.path.join(SENTIMENT_CACHE_DIR, f"{ticker.replace('/', '_')}_{date_str}.txt")
         with open(cache_file, 'w') as f:
             f.write(f"{score:.4f}")
         logger.debug(f"Saved sentiment score to cache: {cache_file}")
     except Exception as e:
         logger.error(f"Error saving sentiment score to cache {cache_file}: {e}")

# --- Ticker/Date based Sentiment Score Function (Updated to use Settings) ---
def get_sentiment_score(ticker: str, date: str, news_api_key: str | None = None) -> float | None:
    """
    Fetches news, analyzes sentiment, and returns an aggregated score.
    Uses settings.NEWS_API_KEY if news_api_key is not provided.
    Uses caching.
    """
    logger.info(f"Getting sentiment score for {ticker} on {date}...")
    
    # Use provided key or fall back to settings
    api_key_to_use = news_api_key or settings.NEWS_API_KEY
    if not api_key_to_use:
         logger.warning("NEWS_API_KEY not provided and not found in settings/env. News fetching might fail or use dummy data.")
         # fetch_news_for_ticker will handle dummy data return

    # --- Check Cache First ---
    cached_score = load_cached_sentiment(ticker, date)
    if cached_score is not None:
         logger.info(f"Using cached sentiment score for {ticker} on {date}: {cached_score:.4f}")
         return cached_score

    # --- Fetch News if not cached ---
    news_texts = fetch_news_for_ticker(ticker, date, api_key=api_key_to_use)
    
    if not news_texts:
        logger.warning(f"No news found or fetched for {ticker} on {date}. Cannot calculate score.")
        # Optionally cache the fact that no news was found (e.g., save None or a special value)
        return None

    # --- Analyze Sentiment ---
    total_score = 0.0
    valid_analyses = 0
    sentiment_mapping = {'positive': 1.0, 'neutral': 0.0, 'negative': -1.0}

    logger.info(f"Analyzing sentiment for {len(news_texts)} fetched headlines...")
    for i, text in enumerate(news_texts):
        result = analyze_text_sentiment(text)
        if result and 'label' in result and 'score' in result:
             label_score = sentiment_mapping.get(result['label'].lower(), 0.0)
             probability = result['score']
             total_score += label_score * probability
             valid_analyses += 1
        else:
             logger.warning(f" -> Failed to analyze sentiment for text snippet: {text[:100]}...")

    if valid_analyses == 0:
        logger.error(f"Sentiment analysis failed for all fetched news snippets for {ticker} on {date}.")
        # Cache failure or return None
        # save_sentiment_to_cache(ticker, date, None) # Or a special value like -999
        return None

    aggregated_score = total_score / valid_analyses
    logger.info(f"Aggregated sentiment score for {ticker} on {date} (based on {valid_analyses} articles): {aggregated_score:.4f}")

    # --- Save to Cache ---
    save_sentiment_to_cache(ticker, date, aggregated_score)

    return aggregated_score

# Example usage (if run directly)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO) 
    print(f"Sentiment Cache Dir: {settings.SENTIMENT_CACHE_DIR}")
    print(f"News API Key Loaded: {'Yes' if settings.NEWS_API_KEY else 'No'}")
    
    print("\nAttempting to fetch sentiment score for TSLA (will use cache if exists, otherwise NewsAPI if key is set)...")
    ticker_score = get_sentiment_score("TSLA", "2024-04-09")
    print(f"Sentiment score for TSLA on 2024-04-09: {ticker_score}")
    
    print("\nAttempting to fetch sentiment score for NVDA...")
    ticker_score_nvda = get_sentiment_score("NVDA", "2024-04-08")
    print(f"Sentiment score for NVDA on 2024-04-08: {ticker_score_nvda}") 