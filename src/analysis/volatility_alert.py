# src/analysis/volatility_alert.py

import os
import sys
import pandas as pd
import logging
from typing import Optional

# --- Project Structure Adjustment ---
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ANALYSIS_DIR = SRC_DIR
PROJECT_ROOT = os.path.dirname(os.path.dirname(ANALYSIS_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# --- Necessary Imports with Error Handling ---
try:
    from prophet import Prophet
    prophet_installed = True
except ImportError:
    print("[VolatilityAlert ERROR] Prophet library is not installed. Please run 'pip install prophet'.")
    prophet_installed = False
    # Define a dummy Prophet class if import fails to avoid NameErrors later
    class Prophet:
        def __init__(self, *args, **kwargs): pass
        def fit(self, df): return self
        def make_future_dataframe(self, periods, freq): return pd.DataFrame({'ds': pd.to_datetime(['2023-01-01'])})
        def predict(self, future): return pd.DataFrame({'ds': pd.to_datetime(['2023-01-01']), 'yhat': [0], 'yhat_lower': [0], 'yhat_upper': [0]})

try:
    # Use get_historical_data from common utils for consistency
    from src.utils.common import get_historical_data
    common_utils_available = True
except ImportError as e:
    print(f"[VolatilityAlert ERROR] Failed to import get_historical_data from src.utils.common: {e}")
    common_utils_available = False
    # Define a dummy function if import fails
    def get_historical_data(*args, **kwargs):
        print("[VolatilityAlert ERROR] get_historical_data is not available.")
        return pd.DataFrame()

try:
    # Use the cleaned-up notifier function
    from src.utils.notifier import send_slack_notification
    notifier_available = True
except ImportError as e:
    print(f"[VolatilityAlert WARNING] Failed to import send_slack_notification from src.utils.notifier: {e}. Anomaly notifications will not be sent.")
    notifier_available = False
    # Dummy function signature matches the used one
    def send_slack_notification(message, channel=None, blocks=None, level='info'):
        print(f"[VolatilityAlert INFO] Slack Notification (Not Sent - Level: {level}): {message}")

# --- Logging Setup ---
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def format_data_for_prophet(df: pd.DataFrame) -> pd.DataFrame:
    """
    Formats a DataFrame with historical price data for Prophet.
    Requires a DatetimeIndex and a 'close' column.

    Args:
        df (pd.DataFrame): DataFrame with DatetimeIndex and 'close' column.

    Returns:
        pd.DataFrame: DataFrame with 'ds' (datetime) and 'y' (close price) columns.
                      Returns an empty DataFrame if input is invalid.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.error("Input DataFrame for Prophet must have a DatetimeIndex.")
        return pd.DataFrame()
    if 'close' not in df.columns:
        logger.error("Input DataFrame for Prophet must have a 'close' column.")
        return pd.DataFrame()

    # Reset index to get 'ds', rename 'close' to 'y'
    df_prophet = df.reset_index()
    df_prophet = df_prophet.rename(columns={df.index.name if df.index.name else 'index': 'ds', 'close': 'y'})

    # Keep only 'ds' and 'y'
    df_prophet = df_prophet[['ds', 'y']]
    logger.debug(f"Formatted DataFrame for Prophet with {len(df_prophet)} rows.")
    return df_prophet

def forecast_price(df_prophet: pd.DataFrame, periods: int = 24, freq: str = 'H') -> Optional[pd.DataFrame]:
    """
    Uses Prophet to forecast future prices based on historical data.

    Args:
        df_prophet (pd.DataFrame): DataFrame formatted for Prophet (columns 'ds', 'y').
        periods (int): Number of future periods to forecast.
        freq (str): Frequency of the periods (e.g., 'H' for hourly, 'D' for daily).

    Returns:
        Optional[pd.DataFrame]: DataFrame containing the forecast ('ds', 'yhat', 'yhat_lower', 'yhat_upper'),
                               or None if Prophet is not installed or an error occurs.
    """
    if not prophet_installed:
        logger.error("Prophet library is not available. Cannot generate forecast.")
        return None
    if df_prophet.empty or len(df_prophet) < 2:
         logger.warning("Not enough data points to generate a forecast.")
         return None

    try:
        # Initialize and fit the Prophet model
        # Consider adding hyperparameters like seasonality_mode, holidays etc. if needed
        model = Prophet(interval_width=0.95) # 95% confidence interval
        logger.info(f"Fitting Prophet model with {len(df_prophet)} data points...")
        model.fit(df_prophet)
        logger.info("Prophet model fitting complete.")

        # Create future dataframe
        future = model.make_future_dataframe(periods=periods, freq=freq)
        logger.info(f"Generating forecast for {periods} future periods with frequency '{freq}'.")

        # Make prediction
        forecast = model.predict(future)
        logger.info("Forecast generated successfully.")
        logger.debug(f"Forecast columns: {forecast.columns.tolist()}")
        logger.debug(f"Forecast tail:\n{forecast.tail()}")

        # Return relevant columns
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    except Exception as e:
        logger.error(f"Error during Prophet forecasting: {e}", exc_info=True)
        return None

def detect_anomaly(forecast_df: pd.DataFrame, actual_price: float) -> Optional[bool]:
    """
    Compares the latest actual price against the most recent forecast interval.

    Args:
        forecast_df (pd.DataFrame): DataFrame containing Prophet forecast results
                                    (must include 'ds', 'yhat_lower', 'yhat_upper').
                                    Assumes the last row corresponds to the most recent forecast point.
        actual_price (float): The latest actual price to compare against the forecast.

    Returns:
        Optional[bool]: True if the actual price is outside the forecast interval (anomaly),
                       False if it's within the interval.
                       None if forecast data is insufficient or invalid.
    """
    if forecast_df is None or forecast_df.empty:
        logger.warning("Cannot detect anomaly: Forecast data is missing or empty.")
        return None
    if len(forecast_df) < 1:
         logger.warning("Cannot detect anomaly: Forecast data has no rows.")
         return None

    # Get the most recent forecast interval
    latest_forecast = forecast_df.iloc[-1]
    try:
        yhat_lower = latest_forecast['yhat_lower']
        yhat_upper = latest_forecast['yhat_upper']
        forecast_time = latest_forecast['ds']

        logger.debug(f"Checking anomaly: Actual Price={actual_price:.2f} vs Forecast Interval=[{yhat_lower:.2f}, {yhat_upper:.2f}] for time {forecast_time}")

        # Check if actual price is outside the confidence interval
        is_anomaly = not (yhat_lower <= actual_price <= yhat_upper)

        if is_anomaly:
            logger.warning(f"Anomaly Detected! Actual Price {actual_price:.2f} is outside the forecast interval [{yhat_lower:.2f}, {yhat_upper:.2f}] for {forecast_time}.")
        else:
             logger.info(f"No anomaly detected. Actual Price {actual_price:.2f} is within the forecast interval.")

        return is_anomaly

    except KeyError as e:
        logger.error(f"Cannot detect anomaly: Forecast DataFrame is missing required columns ({e}). Columns found: {forecast_df.columns.tolist()}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during anomaly detection: {e}", exc_info=True)
        return None

def check_price_anomaly_and_notify(symbol: str,
                                   data_period: str = '30d', # How much historical data to use for fitting
                                   interval: str = '1h',
                                   forecast_periods: int = 1, # How many periods ahead to check anomaly (usually 1)
                                   threshold_pct: Optional[float] = None # 추가: main.py에서 전달받은 임계값
                                   ) -> bool: # 변경: 이상 감지 여부 반환
    """
    Fetches data, forecasts price, detects anomalies, and sends Slack notification if needed.

    Args:
        symbol (str): The trading symbol (e.g., 'BTC-USD').
        data_period (str): The amount of historical data to fetch for Prophet fitting (e.g., '30d', '90d').
        interval (str): The interval of the data (e.g., '1h', '15m'). Determines forecast frequency.
        forecast_periods (int): Number of future periods to forecast (typically 1 for immediate anomaly check).
        threshold_pct (Optional[float]): Anomaly detection threshold (percentage difference). If None, uses Prophet's interval.

    Returns:
        bool: True if an anomaly was detected, False otherwise.
    """
    if not prophet_installed:
        logger.error(f"Prophet not installed, skipping anomaly check for {symbol}.")
        return False # 이상 없음으로 간주
    if not common_utils_available:
        logger.error(f"Common utils (get_historical_data) not available, skipping anomaly check for {symbol}.")
        return False

    logger.info(f"Starting anomaly check for symbol: {symbol} (Interval: {interval}, History: {data_period})" )

    # 1. Fetch Historical Data
    hist_df = get_historical_data(symbol, period=data_period, interval=interval)

    if hist_df is None or hist_df.empty:
        logger.error(f"Failed to fetch historical data for {symbol}. Cannot perform anomaly check.")
        return False

    # Ensure data has 'close' column and DatetimeIndex
    if 'close' not in hist_df.columns or not isinstance(hist_df.index, pd.DatetimeIndex):
         logger.error(f"Historical data for {symbol} is not in the expected format (missing 'close' or DatetimeIndex).")
         return False

    # Check if enough data points exist
    if len(hist_df) < 2: # Need at least 2 for Prophet fitting
         logger.warning(f"Not enough historical data ({len(hist_df)} points) for {symbol} to run anomaly check.")
         return False

    latest_actual_price = hist_df['close'].iloc[-1]
    latest_timestamp = hist_df.index[-1]
    logger.info(f"Latest actual price for {symbol} at {latest_timestamp}: {latest_actual_price:.4f}")

    # 2. Format Data for Prophet
    df_prophet = format_data_for_prophet(hist_df)
    if df_prophet.empty:
         logger.error(f"Failed to format data for Prophet for {symbol}. Aborting anomaly check.")
         return False

    # 3. Generate Forecast
    freq_map = {'15m': '15T', '30m': '30T', '1h': 'H', '4h': '4H', '1d': 'D'}
    prophet_freq = freq_map.get(interval)
    if not prophet_freq:
        logger.error(f"Unsupported interval '{interval}' for Prophet frequency. Cannot forecast.")
        return False

    forecast_df = forecast_price(df_prophet, periods=forecast_periods, freq=prophet_freq)
    if forecast_df is None or forecast_df.empty:
        logger.error(f"Failed to generate forecast for {symbol}. Aborting anomaly check.")
        return False

    # Find the forecast row closest to the latest actual timestamp
    # forecast_df['ds'] is the predicted timestamp
    # latest_timestamp is the actual data timestamp
    # We need the forecast *for* the time period ending at latest_timestamp
    target_forecast_row = forecast_df[forecast_df['ds'] == latest_timestamp]

    if target_forecast_row.empty:
        # If exact match not found (might happen due to frequency alignment issues),
        # use the last row of the forecast dataframe, which represents the prediction
        # for the period immediately following the last known data point.
        logger.warning(f"Could not find exact forecast timestamp match for {latest_timestamp}. Using latest forecast point.")
        target_forecast_row = forecast_df.iloc[[-1]] # Keep as DataFrame
        if target_forecast_row.empty:
             logger.error("Forecast dataframe is unexpectedly empty after generation.")
             return False

    # 4. Detect Anomaly
    is_anomaly = False
    try:
        predicted_time = target_forecast_row['ds'].iloc[0]
        yhat = target_forecast_row['yhat'].iloc[0]
        yhat_lower = target_forecast_row['yhat_lower'].iloc[0]
        yhat_upper = target_forecast_row['yhat_upper'].iloc[0]

        if threshold_pct is not None:
            # Use custom percentage threshold
            lower_bound = yhat * (1 - threshold_pct / 100.0)
            upper_bound = yhat * (1 + threshold_pct / 100.0)
            is_anomaly = not (lower_bound <= latest_actual_price <= upper_bound)
            detection_method = f"Custom Threshold ({threshold_pct:.2f}%) [{lower_bound:.4f}, {upper_bound:.4f}]"
        else:
            # Use Prophet's confidence interval
            is_anomaly = not (yhat_lower <= latest_actual_price <= yhat_upper)
            detection_method = f"Prophet Interval [{yhat_lower:.4f}, {yhat_upper:.4f}]"

        log_msg = f"Anomaly Check for {symbol} at {latest_timestamp}: Actual={latest_actual_price:.4f}, Predicted={yhat:.4f}, Method={detection_method}"
        if is_anomaly:
            logger.warning(f"Anomaly Detected! {log_msg}")
            if notifier_available:
                alert_message = f"🚨 Price Anomaly Alert ({symbol} @ {interval})\n- Time: {latest_timestamp}\n- Actual Price: {latest_actual_price:.4f}\n- Predicted Price: {yhat:.4f}\n- Detection: {detection_method}"
                send_slack_notification(alert_message, level='warning') # Use the imported function
        else:
             logger.info(f"No anomaly detected. {log_msg}")

    except (KeyError, IndexError) as e:
        error_msg = f"Error accessing forecast data columns/rows: {e}. "
        error_msg += f"Forecast columns: {forecast_df.columns.tolist()}. "
        error_msg += f"Target row info: Index={target_forecast_row.index.tolist()}, Columns={target_forecast_row.columns.tolist()}"
        logger.error(error_msg)
        is_anomaly = False # Treat error as no anomaly for safety
    except Exception as e:
        logger.error(f"An unexpected error occurred during anomaly detection: {e}", exc_info=True)
        is_anomaly = False

    # DB 로깅은 여기서 호출하거나 main.py의 잡에서 반환값을 받아 처리
    # log_anomaly_to_db(...) # Placeholder

    return is_anomaly

# --- Example Usage ---
if __name__ == "__main__":
    if not logging.getLogger(__name__).handlers:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if not prophet_installed:
         logger.warning("Prophet is not installed. Cannot run example.")
    else:
        # Example: Check BTC/USD hourly data for anomalies
        check_price_anomaly_and_notify(symbol='BTC-USD', interval='1h', data_period='30d')

        # Example: Check ETH/USD daily data
        # check_price_anomaly_and_notify(symbol='ETH-USD', interval='1d', data_period='180d') 