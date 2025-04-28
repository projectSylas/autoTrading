import pandas as pd
import yfinance as yf
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEFAULT_TICKER_UNIVERSE = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA', 'NVDA', 'META', 'JPM', 'V', 'JNJ'] # Example universe
DEFAULT_AVG_VOLUME_PERIOD = 20 # Days for moving average calculation
DEFAULT_VOLUME_SPIKE_RATIO = 2.0 # Ratio for detecting volume spike (e.g., current > 2x average)
DEFAULT_MIN_VOLUME_THRESHOLD = 1000000 # Minimum absolute volume threshold

def fetch_data_for_universe(tickers: List[str], period: str = "1mo", interval: str = "1d") -> Dict[str, pd.DataFrame]:
    """Fetches historical data for a list of tickers."""
    data_dict = {}
    logging.info(f"Fetching {period} data ({interval} interval) for {len(tickers)} tickers...")
    try:
        # Use yf.download for batch fetching
        data = yf.download(tickers, period=period, interval=interval, progress=False)
        if data.empty:
            logging.warning("yf.download returned empty data.")
            return {}
        
        # If only one ticker, the columns are flat. If multiple, they are multi-index.
        if len(tickers) == 1:
            data_dict[tickers[0]] = data
        else:
            # Group by ticker for multi-index dataframe
            for ticker in tickers:
                # Select columns for the specific ticker
                ticker_data = data.iloc[:, data.columns.get_level_values(1)==ticker]
                # Rename columns to remove the top level (e.g., ('Open', 'AAPL') -> 'Open')
                ticker_data.columns = ticker_data.columns.droplevel(1)
                if not ticker_data.empty:
                    data_dict[ticker] = ticker_data
        logging.info(f"Successfully fetched data for {len(data_dict)} tickers.")
    except Exception as e:
        logging.error(f"Error fetching data for universe: {e}")
    return data_dict

def scan_for_volume_spikes(ticker_universe: List[str] = DEFAULT_TICKER_UNIVERSE,
                             avg_volume_period: int = DEFAULT_AVG_VOLUME_PERIOD,
                             volume_spike_ratio: float = DEFAULT_VOLUME_SPIKE_RATIO,
                             min_volume_threshold: int = DEFAULT_MIN_VOLUME_THRESHOLD) -> List[str]:
    """Scans a list of tickers for significant volume increases."""
    
    potential_tickers = []
    
    # Fetch recent daily data (enough to calculate moving average)
    # Fetch slightly more than needed period for MA calculation
    fetch_period = f"{avg_volume_period + 5}d" 
    ticker_data_dict = fetch_data_for_universe(ticker_universe, period=fetch_period, interval="1d")
    
    if not ticker_data_dict:
        logging.warning("No data fetched for scanning. Returning empty list.")
        return []
        
    logging.info(f"Scanning {len(ticker_data_dict)} tickers for volume spikes...")
    
    for ticker, df in ticker_data_dict.items():
        if df is None or df.empty or 'Volume' not in df.columns:
            logging.warning(f"Skipping {ticker}: Data is missing or incomplete.")
            continue
            
        try:
            # Ensure data is sorted by date
            df = df.sort_index()
            
            # Calculate average volume
            df['Avg_Volume'] = df['Volume'].rolling(window=avg_volume_period, min_periods=avg_volume_period).mean()
            
            # Get the latest data point
            latest_data = df.iloc[-1]
            
            # Check conditions
            current_volume = latest_data['Volume']
            average_volume = latest_data['Avg_Volume']
            
            # Check if average volume is NaN (happens if not enough data)
            if pd.isna(average_volume) or average_volume == 0:
                logging.debug(f"Skipping {ticker}: Not enough data for {avg_volume_period}-day avg volume or avg volume is zero.")
                continue

            # Condition 1: Volume Spike Ratio
            is_spike = current_volume > (average_volume * volume_spike_ratio)
            # Condition 2: Minimum Absolute Volume
            is_above_threshold = current_volume > min_volume_threshold
            
            if is_spike and is_above_threshold:
                logging.info(f"Volume spike detected for {ticker}: Current={current_volume:,.0f}, Avg={average_volume:,.0f} (Ratio > {volume_spike_ratio:.1f}, Threshold > {min_volume_threshold:,})" )
                potential_tickers.append(ticker)
            else:
                 logging.debug(f"No significant volume spike for {ticker}: Current={current_volume:,.0f}, Avg={average_volume:,.0f}")
                 
        except Exception as e:
            logging.error(f"Error processing ticker {ticker}: {e}")
            
    logging.info(f"Scan complete. Found {len(potential_tickers)} potential tickers: {potential_tickers}")
    return potential_tickers

# Example usage:
if __name__ == '__main__':
    logging.info("Running ticker scanner example...")
    selected_tickers = scan_for_volume_spikes()
    print(f"Tickers meeting volume criteria: {selected_tickers}") 