import pandas as pd
import logging
import os
# Import FreqAI specific utilities if needed
# from freqtrade.strategy import IStrategy
# Import sentiment analysis function
try:
    from src.insight.sentiment_analyzer import get_sentiment_score
    SENTIMENT_ANALYZER_AVAILABLE = True
except ImportError:
    get_sentiment_score = None # Define as None if import fails
    SENTIMENT_ANALYZER_AVAILABLE = False
    logging.getLogger(__name__).warning("Sentiment analyzer module not found. Sentiment features will be disabled or use dummy data.")

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Constants ---
SENTIMENT_CACHE_DIR = "data/sentiment_cache/"

# --- Feature Generation Functions ---

def add_technical_indicators(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Adds standard technical indicators to the dataframe.
    (Example implementation - use libraries like TA-Lib or pandas-ta)
    """
    df = dataframe.copy()
    try:
        # Use pandas-ta if available (example)
        import pandas_ta as ta
        df.ta.sma(length=5, append=True, col_names=('sma5'))
        df.ta.sma(length=20, append=True, col_names=('sma20'))
        df.ta.rsi(length=14, append=True, col_names=('rsi'))
        # Add more indicators as needed using pandas_ta
        # df.ta.macd(append=True)
        # df.ta.bbands(append=True)
        logger.info("Added technical indicators using pandas-ta (SMA, RSI). Ensure library is installed.")
    except ImportError:
        logger.warning("pandas-ta library not found. Using basic rolling calculations for SMA/RSI.")
        # Fallback to basic rolling calculations
        df['sma5'] = df['close'].rolling(window=5).mean()
        df['sma20'] = df['close'].rolling(window=20).mean()
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
    except Exception as e:
        logger.error(f"Error calculating technical indicators: {e}", exc_info=True)
        # Return dataframe without indicators or partial indicators

    return df

def load_cached_sentiment(ticker: str, date_str: str) -> float | None:
     """Loads sentiment score from a simple cache file if it exists."""
     os.makedirs(SENTIMENT_CACHE_DIR, exist_ok=True)
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
         os.makedirs(SENTIMENT_CACHE_DIR, exist_ok=True)
         cache_file = os.path.join(SENTIMENT_CACHE_DIR, f"{ticker.replace('/', '_')}_{date_str}.txt")
         with open(cache_file, 'w') as f:
             f.write(f"{score:.4f}") # Save with precision
         logger.debug(f"Saved sentiment score to cache: {cache_file}")
     except Exception as e:
         logger.error(f"Error saving sentiment score to cache {cache_file}: {e}")

def add_sentiment_feature(dataframe: pd.DataFrame,
                           ticker: str, # Ticker needed to fetch relevant sentiment
                           # Removed sentiment_data_source, will call get_sentiment_score directly
                           # Requires NEWS_API_KEY to be configured (e.g., via os.environ or settings)
                           ) -> pd.DataFrame:
    """
    Adds a sentiment score feature by calling get_sentiment_score.
    Includes simple file-based caching for dates to avoid redundant API calls.

    Args:
        dataframe (pd.DataFrame): The OHLCV dataframe with datetime index.
        ticker (str): The crypto ticker symbol (e.g., "BTC/USDT").

    Returns:
        pd.DataFrame: Dataframe with 'sentiment_score' column added.
    """
    df = dataframe.copy()
    if not SENTIMENT_ANALYZER_AVAILABLE or get_sentiment_score is None:
        logger.warning("Sentiment analyzer not available. Adding dummy sentiment_score = 0.")
        df['sentiment_score'] = 0.0
        return df

    logger.info(f"Adding sentiment score feature for {ticker} by calling get_sentiment_score...")
    
    # Retrieve NEWS_API_KEY (example: from environment variables)
    # Make sure this is set in your environment where Freqtrade runs
    news_api_key = os.getenv("NEWS_API_KEY")
    if not news_api_key:
         logger.warning("NEWS_API_KEY environment variable not set. Sentiment analysis might fail or use limited sources.")
         # Depending on get_sentiment_score implementation, it might still work without key for some sources

    sentiment_scores = []
    # Iterate through dates present in the dataframe index
    unique_dates = dataframe.index.normalize().unique() # Get unique dates
    date_score_map = {}

    logger.info(f"Fetching/loading sentiment scores for {len(unique_dates)} unique dates...")
    for date in unique_dates:
        date_str = date.strftime('%Y-%m-%d')
        # 1. Try loading from cache
        cached_score = load_cached_sentiment(ticker, date_str)
        if cached_score is not None:
            score = cached_score
        else:
            # 2. If not cached, fetch and cache it
            try:
                # Call the imported function
                # Note: get_sentiment_score needs proper implementation for news fetching
                score = get_sentiment_score(ticker=ticker, date=date_str, news_api_key=news_api_key)
                # Use 0.0 as a neutral default if fetching fails or returns None
                score = score if score is not None else 0.0
                save_sentiment_to_cache(ticker, date_str, score) # Cache the result
            except Exception as e:
                logger.error(f"Error calling get_sentiment_score for {ticker} on {date_str}: {e}")
                score = 0.0 # Default to neutral on error
        date_score_map[date] = score

    # Map the scores back to the original dataframe index
    df['sentiment_score'] = df.index.normalize().map(date_score_map)

    # Ensure the column is filled (forward fill for missing intraday, then backfill)
    df['sentiment_score'] = df['sentiment_score'].ffill().bfill().fillna(0.0)
    logger.info(f"Sentiment score feature added. Example score head:\n{df['sentiment_score'].head().to_string()}")

    return df

# --- Main Feature Generation Function (called by FreqAI) ---
def generate_custom_features(dataframe: pd.DataFrame, metadata: dict | None = None) -> pd.DataFrame:
    """
    Generates all custom features for the FreqAI strategy.
    This function structure might be used within a Freqtrade strategy file.

    Args:
        dataframe (pd.DataFrame): OHLCV data from Freqtrade.
        metadata (dict | None): Optional metadata dict passed by Freqtrade
                                (might contain pair info, timeframe etc.).

    Returns:
        pd.DataFrame: Dataframe with all generated features.
    """
    ticker = metadata.get('pair', 'UNKNOWN_PAIR') if metadata else 'UNKNOWN_PAIR'
    logger.info(f"Generating custom features for pair: {ticker}")

    # 1. Add standard technical indicators
    df_with_ta = add_technical_indicators(dataframe)

    # 2. Add sentiment score feature by calling the sentiment analyzer
    df_with_sentiment = add_sentiment_feature(df_with_ta, ticker=ticker)

    logger.info(f"Custom feature generation complete for {ticker}. Final columns: {df_with_sentiment.columns.tolist()}")
    # logger.debug(f"Feature DataFrame head:\n{df_with_sentiment.head().to_string()}")
    return df_with_sentiment

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger.info("Running FreqAI Custom Feature Generator with Sentiment Integration...")
    # Create a dummy DataFrame for testing
    dummy_data = {
        'date': pd.to_datetime(['2024-01-01 10:00', '2024-01-01 11:00', '2024-01-02 12:00',
                           '2024-01-02 13:00', '2024-01-03 14:00']), # Multiple dates
        'open': [100, 101, 102, 103, 104],
        'high': [102, 103, 103, 105, 105],
        'low': [99, 100, 101, 102, 103],
        'close': [101, 102, 103, 104, 104],
        'volume': [1000, 1200, 1100, 1500, 1300]
    }
    dummy_df = pd.DataFrame(dummy_data).set_index('date')

    # Example usage:
    # Set a dummy NEWS_API_KEY for the example if needed by the placeholder fetcher
    # os.environ["NEWS_API_KEY"] = "YOUR_KEY_OR_DUMMY"
    features_df = generate_custom_features(dummy_df, metadata={'pair': 'BTC/USDT'})
    print("\n--- Dummy DataFrame with Features (including Sentiment) ---")
    print(features_df) 