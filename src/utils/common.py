import pandas as pd
import numpy as np
import logging
import yfinance as yf
import datetime
import os
import requests # For VIX
from requests.exceptions import RequestException # For VIX

# --- Setup logger for this module ---
# Assumes setup_logging() from logging_config has been called by the entry point
logger = logging.getLogger(__name__)
# --- End Setup logger ---

# --- Data Fetching Utilities ---

def get_historical_data(ticker: str, interval: str = '1h', period: str = '1y') -> pd.DataFrame:
    """Fetches historical data for a given ticker using yfinance.

    Args:
        ticker (str): The stock ticker symbol.
        interval (str): Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo').
        period (str): Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max').

    Returns:
        pd.DataFrame: DataFrame containing historical data, or an empty DataFrame on error.
    """
    logger.info(f"Fetching {interval} data for {ticker} over period {period}...")
    try:
        stock = yf.Ticker(ticker)
        hist_data = stock.history(period=period, interval=interval)
        if hist_data.empty:
            logger.warning(f"No {interval} data found for {ticker} for period {period}.")
            return pd.DataFrame()
        # Convert timezone-aware index to timezone-naive UTC (or keep timezone)
        # hist_data.index = hist_data.index.tz_convert(None) # Option 1: Convert to naive UTC
        # hist_data.index = hist_data.index.tz_localize(None) # Option 2: Make timezone-naive if localized
        # Option 3: Keep timezone-aware index (often preferred)

        # Data Cleaning (Example)
        hist_data.dropna(inplace=True) # Drop rows with any NaNs
        # Ensure standard column names (lowercase)
        hist_data.columns = hist_data.columns.str.lower()
        logger.info(f"Successfully fetched {len(hist_data)} rows of {interval} data for {ticker}.")
        return hist_data
    except Exception as e:
        logger.error(f"Error fetching historical data for {ticker}: {e}", exc_info=True)
        return pd.DataFrame() # Return empty DataFrame on failure

def get_current_vix() -> float:
    """Fetches the current VIX value from CBOE (approximate).

    Note: This scrapes a specific element from the CBOE website. It might break
          if the website structure changes.
          Consider more robust data sources for production.

    Returns:
        float: The current VIX value, or np.nan on failure.
    """
    # CBOE website endpoint might change - use a known source
    # Using Yahoo Finance for ^VIX as a more stable alternative
    logger.info("Fetching current VIX (^VIX) value from Yahoo Finance...")
    try:
        vix_ticker = yf.Ticker("^VIX")
        vix_data = vix_ticker.history(period="5d", interval="1d") # Get recent daily data
        if not vix_data.empty:
            last_vix = vix_data['Close'].iloc[-1]
            logger.info(f"Fetched last close VIX value: {last_vix:.2f}")
            return float(last_vix)
        else:
            logger.warning("Could not fetch recent VIX data from Yahoo Finance.")
            return np.nan

    except Exception as e:
        logger.error(f"Error fetching VIX data from Yahoo Finance: {e}", exc_info=True)
        return np.nan

# --- Logging Utilities ---

def save_log_to_csv(log_data: list, filename: str = 'trade_log.csv', log_dir: str = 'logs') -> None:
    """Saves a list of log entries (dictionaries) to a CSV file.

    Args:
        log_data (list): A list where each element is a dictionary representing a log entry.
                         Each dictionary should have the same keys.
        filename (str): The name of the CSV file.
        log_dir (str): The directory to save the log file in.
    """
    if not log_data:
        logger.debug("No log data provided to save.")
        return

    log_path = os.path.join(log_dir, filename)
    os.makedirs(log_dir, exist_ok=True) # Ensure the directory exists

    try:
        # Convert list of dicts to DataFrame
        log_df = pd.DataFrame(log_data)

        # Check if file exists to append or write header
        file_exists = os.path.isfile(log_path)

        # Append data to CSV, write header only if file doesn't exist
        log_df.to_csv(log_path, mode='a', header=not file_exists, index=False)
        logger.info(f"Successfully saved/appended {len(log_data)} log entries to {log_path}")

    except Exception as e:
        logger.error(f"Error saving log data to {log_path}: {e}", exc_info=True)

# Removed technical analysis functions - they are consolidated in src/utils/strategy_utils.py

# Example Usage (optional, can be removed or kept for testing)
if __name__ == '__main__':
    # Setup basic logging for testing this module directly
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger.info("Testing common utility functions...")

    # Test Data Fetching
    ticker_symbol = "AAPL"
    hist_data = get_historical_data(ticker_symbol, interval='1d', period='3mo')
    if not hist_data.empty:
        logger.info(f"Fetched {ticker_symbol} data head:\n{hist_data.head()}")
    else:
        logger.error(f"Failed to fetch data for {ticker_symbol}.")

    # Test VIX Fetching
    current_vix = get_current_vix()
    if not np.isnan(current_vix):
        logger.info(f"Current VIX (last close): {current_vix:.2f}")
    else:
        logger.warning("Failed to fetch VIX.")

    # Test Logging to CSV
    sample_logs = [
        {'timestamp': datetime.datetime.now(), 'action': 'BUY', 'ticker': 'MSFT', 'price': 300.50, 'quantity': 10, 'reason': 'SMA Crossover'},
        {'timestamp': datetime.datetime.now() - datetime.timedelta(hours=1), 'action': 'SELL', 'ticker': 'GOOG', 'price': 2800.00, 'quantity': 5, 'reason': 'RSI Overbought'}
    ]
    save_log_to_csv(sample_logs, filename="test_common_log.csv")
    save_log_to_csv([{'timestamp': datetime.datetime.now(), 'action': 'HOLD', 'ticker': 'AMZN', 'price': 3400.00, 'quantity': 0, 'reason': 'Neutral Signal'}], filename="test_common_log.csv")

    logger.info("Common utility function tests completed.") 