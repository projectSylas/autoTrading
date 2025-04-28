# src/utils/strategy_utils.py

import pandas as pd
import numpy as np
import logging
import ta # Import for ta library functions used from common.py
from typing import Optional, List, Tuple, Dict
import pandas_ta as pta # Changed import to pta for consistency if used later
import numba
import vectorbt as vbt # Ensure vectorbt is imported if not already
from scipy.signal import find_peaks # For divergence detection from common.py

# --- Setup logger for this module ---
# Assumes setup_logging() from logging_config has been called by the entry point
logger = logging.getLogger(__name__)
# --- End Setup logger ---

# Setup basic logging if needed when run standalone or for debugging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_sma(df: pd.DataFrame, window: int, feature_col: str = 'Close') -> pd.Series:
    """
    Calculates the Simple Moving Average (SMA) for a given feature column.
    Returns a Series with the SMA values.
    """
    if df is None or df.empty or feature_col not in df.columns or window <= 0:
        logger.warning(f"calculate_sma: Invalid input. df empty: {df is None or df.empty}, col: {feature_col}, win: {window}")
        return pd.Series(dtype=float)
    if len(df) < window:
        logger.debug(f"calculate_sma: Data length ({len(df)}) < window ({window}).")
        # Return Series of NaNs with the same index
        return pd.Series(np.nan, index=df.index)

    sma_col_name = f'SMA_{window}'
    try:
        sma_series = df[feature_col].rolling(window=window, min_periods=window).mean() # Use min_periods=window for standard SMA
        logger.debug(f"Calculated {sma_col_name} using window {window} on column '{feature_col}'.")
        return sma_series
    except Exception as e:
        logger.error(f"calculate_sma: Error calculating SMA for window {window}: {e}", exc_info=True)
        return pd.Series(dtype=float) # Return empty Series on error

def calculate_rsi(df: pd.DataFrame, window: int = 14, feature_col: str = 'Close') -> pd.Series:
    """
    Calculates the Relative Strength Index (RSI) using Wilder's EMA method.
    Returns a Series with the RSI values.
    """
    if df is None or df.empty or feature_col not in df.columns or window <= 0:
        logger.warning(f"calculate_rsi: Invalid input. df empty: {df is None or df.empty}, col: {feature_col}, win: {window}")
        return pd.Series(dtype=float)
    if len(df) < window + 1:
        logger.debug(f"calculate_rsi: Data length ({len(df)}) insufficient for window ({window}).")
        return pd.Series(np.nan, index=df.index)

    try:
        delta = df[feature_col].diff(1).dropna()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        # Use Exponential Moving Average (EMA) for Wilder's RSI
        avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
        avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()

        # Handle potential division by zero if avg_loss is 0
        rs = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss) # Treat division by zero as infinite RS
        rsi = 100.0 - (100.0 / (1.0 + rs))
        rsi[rs == np.inf] = 100.0 # If RS is infinite, RSI is 100

        logger.debug(f"Calculated RSI using window {window} on column '{feature_col}'.")
        return pd.Series(rsi, index=avg_gain.index) # Ensure index alignment

    except Exception as e:
        logger.error(f"calculate_rsi: Error calculating RSI for window {window}: {e}", exc_info=True)
        return pd.Series(dtype=float)

def calculate_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std_dev: float = 2.0, feature_col: str = 'Close') -> pd.DataFrame:
    """
    Calculates Bollinger Bands (Middle, Upper, Lower).
    Returns a DataFrame with BB_Middle, BB_Upper, BB_Lower columns.
    """
    if df is None or df.empty or feature_col not in df.columns or window <= 0 or num_std_dev <= 0:
        logger.warning(f"calculate_bollinger_bands: Invalid input. df empty: {df is None or df.empty}, col: {feature_col}, win: {window}, std: {num_std_dev}")
        return pd.DataFrame(index=df.index if df is not None else None, columns=['BB_Middle', 'BB_Upper', 'BB_Lower'])

    middle_band_col = 'BB_Middle'
    upper_band_col = 'BB_Upper'
    lower_band_col = 'BB_Lower'

    bb_df = pd.DataFrame(index=df.index)

    try:
        bb_df[middle_band_col] = df[feature_col].rolling(window=window, min_periods=window).mean()
        rolling_std = df[feature_col].rolling(window=window, min_periods=window).std()
        bb_df[upper_band_col] = bb_df[middle_band_col] + (rolling_std * num_std_dev)
        bb_df[lower_band_col] = bb_df[middle_band_col] - (rolling_std * num_std_dev)
        logger.debug(f"Calculated Bollinger Bands (W:{window}, SD:{num_std_dev}) on '{feature_col}'.")
        return bb_df

    except Exception as e:
        logger.error(f"calculate_bollinger_bands: Error calculating Bollinger Bands: {e}", exc_info=True)
        return pd.DataFrame(index=df.index, columns=['BB_Middle', 'BB_Upper', 'BB_Lower']) # Return empty BB df on error

# --- Signal Detection Functions ---

# Merged Divergence Detection (Using common.py version as a base, improved)
def detect_rsi_divergence(prices: pd.Series, rsi: pd.Series, window: int = 14, peak_distance: int = 5, peak_prominence_ratio: float = 0.03) -> str:
    """Detects RSI divergence using scipy.signal.find_peaks.

    Args:
        prices (pd.Series): Close price series.
        rsi (pd.Series): RSI series.
        window (int): Lookback window for finding peaks/troughs.
        peak_distance (int): Minimum distance between peaks/troughs.
        peak_prominence_ratio (float): Minimum prominence ratio (relative to price range) for peaks/troughs.

    Returns:
        str: 'bullish', 'bearish', or 'none'.
    """
    if prices.empty or rsi.empty or len(prices) < window or len(rsi) < window:
        logger.debug("detect_rsi_divergence: Insufficient data.")
        return 'none'

    # Ensure alignment
    prices, rsi = prices.align(rsi, join='inner')
    if prices.empty:
         logger.debug("detect_rsi_divergence: Data alignment resulted in empty series.")
         return 'none'

    prices_window = prices.tail(window)
    rsi_window = rsi.tail(window)

    if prices_window.isnull().all() or rsi_window.isnull().all():
        logger.debug("detect_rsi_divergence: Window contains only NaNs.")
        return 'none'

    price_range = prices_window.max() - prices_window.min()
    if price_range == 0: # Avoid division by zero if price is flat
        price_range = 1.0

    peak_prominence = price_range * peak_prominence_ratio
    rsi_prominence = 1 # Use a fixed prominence for RSI or make it relative too?

    try:
        # Find peaks (highs)
        price_peaks_idx, _ = find_peaks(prices_window, prominence=peak_prominence, distance=peak_distance)
        rsi_peaks_idx, _ = find_peaks(rsi_window, prominence=rsi_prominence, distance=peak_distance)

        # Find troughs (lows)
        price_troughs_idx, _ = find_peaks(-prices_window, prominence=peak_prominence, distance=peak_distance)
        rsi_troughs_idx, _ = find_peaks(-rsi_window, prominence=rsi_prominence, distance=peak_distance)

        # Convert relative window indices to original series indices for value lookup
        price_peak_indices = prices_window.index[price_peaks_idx]
        rsi_peak_indices = rsi_window.index[rsi_peaks_idx]
        price_trough_indices = prices_window.index[price_troughs_idx]
        rsi_trough_indices = rsi_window.index[rsi_troughs_idx]

        # Check Bearish Divergence (Price Higher High, RSI Lower High)
        if len(price_peak_indices) >= 2 and len(rsi_peak_indices) >= 2:
            # Get the last two peaks
            last_price_peak_time = price_peak_indices[-1]
            prev_price_peak_time = price_peak_indices[-2]

            # Find corresponding RSI peaks (this matching can be complex, simple approach: use nearest)
            # For simplicity, find RSI peak closest in time to price peak
            def find_nearest(array, value):
                array = np.asarray(array)
                idx = (np.abs(array - value)).argmin()
                return array[idx]

            # Find RSI peaks closest to the price peaks in time (or use the price peak time directly)
            last_rsi_peak_val = rsi.get(last_price_peak_time, np.nan) # Use get for safety
            prev_rsi_peak_val = rsi.get(prev_price_peak_time, np.nan)

            if not pd.isna(last_rsi_peak_val) and not pd.isna(prev_rsi_peak_val):
                 if prices[last_price_peak_time] > prices[prev_price_peak_time] and \
                    last_rsi_peak_val < prev_rsi_peak_val:
                    # Check if the last peak is recent enough (e.g., within last 3 bars of window)
                    if (prices_window.index[-1] - last_price_peak_time) <= pd.Timedelta(days=3 * (prices_window.index[1]-prices_window.index[0]).days) : # Adjust based on timeframe
                         logger.info(f"🐻 Bearish RSI Divergence detected: Price HH ({prev_price_peak_time.date()}->{last_price_peak_time.date()}), RSI LH")
                         return 'bearish'

        # Check Bullish Divergence (Price Lower Low, RSI Higher Low)
        if len(price_trough_indices) >= 2 and len(rsi_trough_indices) >= 2:
            last_price_trough_time = price_trough_indices[-1]
            prev_price_trough_time = price_trough_indices[-2]

            last_rsi_trough_val = rsi.get(last_price_trough_time, np.nan)
            prev_rsi_trough_val = rsi.get(prev_price_trough_time, np.nan)

            if not pd.isna(last_rsi_trough_val) and not pd.isna(prev_rsi_trough_val):
                if prices[last_price_trough_time] < prices[prev_price_trough_time] and \
                   last_rsi_trough_val > prev_rsi_trough_val:
                   if (prices_window.index[-1] - last_price_trough_time) <= pd.Timedelta(days=3 * (prices_window.index[1]-prices_window.index[0]).days):
                        logger.info(f"🐂 Bullish RSI Divergence detected: Price LL ({prev_price_trough_time.date()}->{last_price_trough_time.date()}), RSI HL")
                        return 'bullish'

    except Exception as e:
        logger.error(f"detect_rsi_divergence: Error during detection: {e}", exc_info=True)
        return 'none'

    return 'none'


# Merged Volume Spike Detection (Using common.py version for now)
def detect_volume_spike(df: pd.DataFrame, window: int = 20, factor: float = 2.0, volume_col: str = 'Volume') -> bool:
    """Detects if the latest volume is significantly higher than the recent average volume.

    Args:
        df (pd.DataFrame): DataFrame containing 'Volume' column.
        window (int): Rolling window for average volume calculation.
        factor (float): Multiplier for average volume to define a spike.
        volume_col (str): Name of the volume column.

    Returns:
        bool: True if a volume spike is detected, False otherwise.
    """
    if df is None or df.empty or volume_col not in df.columns or len(df) < window + 1:
        # logger.debug(f"detect_volume_spike: Insufficient data or missing '{volume_col}' column.")
        return False # Not necessarily a warning, might just not have enough data yet

    try:
        avg_volume = df[volume_col].rolling(window=window, closed='left').mean().iloc[-1] # Use closed='left' to exclude current bar
        latest_volume = df[volume_col].iloc[-1]

        if pd.isna(avg_volume):
             # logger.debug("detect_volume_spike: Average volume is NaN.")
             return False

        is_spike = latest_volume > (avg_volume * factor)
        # if is_spike:
        #     logger.debug(f"Volume Spike Detected: Latest={latest_volume:.0f}, Avg={avg_volume:.0f}, Factor={factor}")
        return is_spike

    except Exception as e:
        logger.error(f"detect_volume_spike: Error calculating volume spike: {e}", exc_info=True)
        return False


# Keeping Numba optimized trendline detection from strategy_utils.py
@numba.njit('i1[:](f8[:], f8[:], f8[:], i8, i8, f8)')
def find_breakout_pullback_nb(high: np.ndarray,
                              low: np.ndarray,
                              close: np.ndarray,
                              lookback_breakout: int,
                              lookback_pullback: int,
                              pullback_threshold: float) -> np.ndarray:
    """
    Numba-optimized function to find trendline breakout and pullback signals.

    Args:
        high (np.ndarray): High prices.
        low (np.ndarray): Low prices.
        close (np.ndarray): Close prices.
        lookback_breakout (int): Window to find highs/lows for trendline.
        lookback_pullback (int): Window to check for pullback after breakout.
        pullback_threshold (float): Relative threshold for pullback validation.

    Returns:
        np.ndarray: Signal array (0: None, 1: Breakout Up, 2: Pullback Support,
                      -1: Breakout Down, -2: Pullback Resistance).
    """
    n = len(close)
    signals = np.zeros(n, dtype=np.int8)

    if n < lookback_breakout + lookback_pullback:
        return signals # Not enough data

    for i in range(lookback_breakout + lookback_pullback -1, n):
        # --- Upper Trendline (Resistance) ---
        highs_window = high[i - lookback_breakout + 1 : i + 1]
        idx_window = np.arange(len(highs_window))

        # Find indices of the two highest points in the window
        if len(highs_window) >= 2:
             # Simple approach: use argpartition, but might not find peaks well.
             # A better approach would involve peak finding like in divergence.
             # For simplicity here, assume we can connect first and last points or highest two.
             # Let's try connecting highest two distinct points
             sorted_indices = np.argsort(highs_window)[::-1]
             idx_h1 = sorted_indices[0]
             idx_h2 = -1
             for k in range(1, len(sorted_indices)):
                 if highs_window[sorted_indices[k]] < highs_window[idx_h1]: # Ensure distinct high value
                     idx_h2 = sorted_indices[k]
                     break

             if idx_h2 != -1:
                # Ensure chronological order for slope calculation
                p1_idx, p1_val = (idx_h2, highs_window[idx_h2]) if idx_h2 < idx_h1 else (idx_h1, highs_window[idx_h1])
                p2_idx, p2_val = (idx_h1, highs_window[idx_h1]) if idx_h2 < idx_h1 else (idx_h2, highs_window[idx_h2])

                if p2_idx > p1_idx: # Avoid division by zero
                    slope_upper = (p2_val - p1_val) / (p2_idx - p1_idx)
                    intercept_upper = p1_val - slope_upper * p1_idx

                    # Trendline value at current point (relative index within window)
                    current_relative_idx = len(highs_window) - 1
                    trendline_val_upper = slope_upper * current_relative_idx + intercept_upper

                    # Check for Breakout Up
                    if close[i] > trendline_val_upper and close[i-1] <= (slope_upper * (current_relative_idx - 1) + intercept_upper):
                        signals[i] = 1 # Breakout Up

                        # Check for Pullback to Support after Breakout Up
                        if i >= lookback_breakout + lookback_pullback:
                            broke_out = False
                            pullback_valid = False
                            for j in range(1, lookback_pullback + 1):
                                prev_idx_rel = current_relative_idx - j
                                prev_trend_val = slope_upper * prev_idx_rel + intercept_upper
                                # Check if price was above trendline previously
                                if close[i-j] > prev_trend_val:
                                     broke_out = True
                                # Check if price came back near or below the trendline recently
                                if broke_out and low[i] <= trendline_val_upper * (1 + pullback_threshold) and low[i] >= trendline_val_upper * (1-pullback_threshold):
                                     pullback_valid = True
                                     break # Found valid pullback
                            if pullback_valid:
                                signals[i] = 2 # Pullback Support

        # --- Lower Trendline (Support) ---
        lows_window = low[i - lookback_breakout + 1 : i + 1]
        # Find indices of the two lowest points
        if len(lows_window) >= 2:
            sorted_indices_low = np.argsort(lows_window)
            idx_l1 = sorted_indices_low[0]
            idx_l2 = -1
            for k in range(1, len(sorted_indices_low)):
                if lows_window[sorted_indices_low[k]] > lows_window[idx_l1]:
                    idx_l2 = sorted_indices_low[k]
                    break

            if idx_l2 != -1:
                p1_idx_low, p1_val_low = (idx_l2, lows_window[idx_l2]) if idx_l2 < idx_l1 else (idx_l1, lows_window[idx_l1])
                p2_idx_low, p2_val_low = (idx_l1, lows_window[idx_l1]) if idx_l2 < idx_l1 else (idx_l2, lows_window[idx_l2])

                if p2_idx_low > p1_idx_low:
                    slope_lower = (p2_val_low - p1_val_low) / (p2_idx_low - p1_idx_low)
                    intercept_lower = p1_val_low - slope_lower * p1_idx_low
                    trendline_val_lower = slope_lower * current_relative_idx + intercept_lower

                    # Check for Breakout Down
                    if close[i] < trendline_val_lower and close[i-1] >= (slope_lower * (current_relative_idx - 1) + intercept_lower):
                        signals[i] = -1 # Breakout Down

                        # Check for Pullback to Resistance after Breakout Down
                        if i >= lookback_breakout + lookback_pullback:
                            broke_down = False
                            pullback_valid_res = False
                            for j in range(1, lookback_pullback + 1):
                                prev_idx_rel = current_relative_idx - j
                                prev_trend_val_low = slope_lower * prev_idx_rel + intercept_lower
                                if close[i-j] < prev_trend_val_low:
                                     broke_down = True
                                if broke_down and high[i] >= trendline_val_lower * (1 - pullback_threshold) and high[i] <= trendline_val_lower * (1 + pullback_threshold):
                                     pullback_valid_res = True
                                     break
                            if pullback_valid_res:
                                 signals[i] = -2 # Pullback Resistance

    return signals

def detect_trendline_breakout_pullback(df: pd.DataFrame,
                                       price_col: str = 'Close',
                                       high_col: str = 'High',
                                       low_col: str = 'Low',
                                       lookback_breakout: int = 20,
                                       lookback_pullback: int = 5,
                                       pullback_threshold: float = 0.01) -> pd.Series:
    """
    Wrapper for the Numba-optimized trendline breakout/pullback detection.

    Args:
        df (pd.DataFrame): DataFrame with High, Low, Close columns.
        price_col, high_col, low_col (str): Column names.
        lookback_breakout, lookback_pullback (int): Lookback periods.
        pullback_threshold (float): Threshold for pullback validation.

    Returns:
        pd.Series: Series containing signals (0, 1, 2, -1, -2) with the same index as df.
    """
    if df is None or df.empty or not all(c in df.columns for c in [high_col, low_col, price_col]):
        logger.warning("detect_trendline_breakout_pullback: Invalid input.")
        return pd.Series(dtype=np.int8)

    high_np = df[high_col].to_numpy(dtype=np.float64)
    low_np = df[low_col].to_numpy(dtype=np.float64)
    close_np = df[price_col].to_numpy(dtype=np.float64)

    # Handle potential NaNs before passing to Numba
    if np.isnan(high_np).any() or np.isnan(low_np).any() or np.isnan(close_np).any():
         logger.warning("detect_trendline_breakout_pullback: NaN values found in input data. Attempting to fill.")
         # Simple forward fill, consider better NaN handling if necessary
         df_filled = df[[high_col, low_col, price_col]].ffill().bfill()
         high_np = df_filled[high_col].to_numpy(dtype=np.float64)
         low_np = df_filled[low_col].to_numpy(dtype=np.float64)
         close_np = df_filled[price_col].to_numpy(dtype=np.float64)
         if np.isnan(high_np).any() or np.isnan(low_np).any() or np.isnan(close_np).any():
             logger.error("detect_trendline_breakout_pullback: Failed to fill NaNs. Cannot proceed.")
             return pd.Series(dtype=np.int8)


    signals_np = find_breakout_pullback_nb(
        high_np, low_np, close_np,
        lookback_breakout, lookback_pullback, pullback_threshold
    )
    return pd.Series(signals_np, index=df.index)


# POC Calculation (Keeping strategy_utils version) - Note: Multiple definitions found, consolidating
# Removed duplicate/older POC function definitions. Keeping one primary version.
def calculate_poc(df: pd.DataFrame, lookback: int = 50, price_col: str = 'Close', volume_col: str = 'Volume', num_bins: int = 20) -> Optional[float]:
    """
    Calculates the Point of Control (POC) over a lookback period.
    POC is the price level with the highest traded volume.

    Args:
        df (pd.DataFrame): DataFrame with price and volume data.
        lookback (int): Lookback period for calculation.
        price_col (str): Price column (e.g., 'Close', 'High', 'Low', or typical price like HLC/3).
        volume_col (str): Volume column.
        num_bins (int): Number of price bins for volume profile calculation.

    Returns:
        Optional[float]: The Point of Control price level, or None if calculation fails.
    """
    if df is None or df.empty or not all(c in df.columns for c in [price_col, volume_col]):
        logger.warning(f"calculate_poc: Invalid input or missing columns '{price_col}', '{volume_col}'.")
        return None
    if len(df) < lookback:
        logger.debug(f"calculate_poc: Insufficient data ({len(df)}) for lookback ({lookback}).")
        return None

    try:
        # Select the lookback period
        df_lookback = df.iloc[-lookback:]

        # Determine price range and create bins
        price_min = df_lookback[price_col].min()
        price_max = df_lookback[price_col].max()
        if pd.isna(price_min) or pd.isna(price_max) or price_max == price_min:
            logger.debug("calculate_poc: Cannot determine price range or price is flat.")
            return None

        # Create price bins - linspace includes endpoints
        bins = np.linspace(price_min, price_max, num_bins + 1)

        # Assign each price to a bin - use bin centers later
        df_lookback['price_bin'] = pd.cut(df_lookback[price_col], bins=bins, labels=False, include_lowest=True)

        # Calculate volume per bin
        volume_profile = df_lookback.groupby('price_bin')[volume_col].sum()

        if volume_profile.empty:
            logger.debug("calculate_poc: Volume profile is empty after grouping.")
            return None

        # Find the bin index with the maximum volume
        poc_bin_index = volume_profile.idxmax()

        # Calculate the center price of the POC bin
        # Bin edges are in `bins`. Bin `i` corresponds to edges `bins[i]` and `bins[i+1]`
        poc_price = (bins[int(poc_bin_index)] + bins[int(poc_bin_index) + 1]) / 2.0

        logger.debug(f"Calculated POC over lookback {lookback}: {poc_price:.4f}")
        return poc_price

    except Exception as e:
        logger.error(f"calculate_poc: Error calculating POC: {e}", exc_info=True)
        return None

# --- Other Numba Optimized Functions (Keep from strategy_utils) ---

@numba.njit('boolean(f8[:], f8[:], i8)')
def check_last_crossing_nb(series1: np.ndarray, series2: np.ndarray, lookback: int) -> bool:
    """
    Numba-optimized check if series1 crossed above series2 within the recent lookback period.
    Assumes series are aligned and non-NaN within the lookback.
    """
    n = len(series1)
    if n < lookback + 1: return False # Need at least lookback+1 points to check crossing

    crossed_within_lookback = False
    for i in range(n - lookback, n):
         # Check if crossed between i-1 and i
         if series1[i-1] <= series2[i-1] and series1[i] > series2[i]:
             crossed_within_lookback = True
             break
    return crossed_within_lookback

def check_sma_crossover(df: pd.DataFrame, short_window: int, long_window: int, price_col: str = 'Close', lookback: int = 3) -> str:
    """
    Checks for Golden Cross (short SMA crosses above long SMA) or Death Cross
    (short SMA crosses below long SMA) within the recent `lookback` period.

    Args:
        df (pd.DataFrame): DataFrame containing the price data.
        short_window (int): Window for the short-term SMA.
        long_window (int): Window for the long-term SMA.
        price_col (str): Column to calculate SMAs on.
        lookback (int): How many recent periods to check for a crossing event.

    Returns:
        str: 'golden', 'death', or 'none'.
    """
    if df is None or df.empty or price_col not in df.columns:
        logger.warning("check_sma_crossover: Invalid input.")
        return 'none'
    if len(df) < long_window + lookback: # Need enough data for long SMA and lookback
        logger.debug("check_sma_crossover: Insufficient data.")
        return 'none'

    try:
        sma_short = calculate_sma(df, window=short_window, feature_col=price_col)
        sma_long = calculate_sma(df, window=long_window, feature_col=price_col)

        if sma_short.empty or sma_long.empty:
             logger.debug("check_sma_crossover: SMA calculation failed.")
             return 'none'

        # Align series and check for NaNs in the lookback period
        sma_short, sma_long = sma_short.align(sma_long, join='inner')
        recent_short = sma_short.iloc[-(lookback+1):]
        recent_long = sma_long.iloc[-(lookback+1):]

        if recent_short.isnull().any() or recent_long.isnull().any():
             logger.debug("check_sma_crossover: NaNs found in recent SMA data.")
             return 'none'

        # Check for Golden Cross
        crossed_above = False
        for i in range(1, len(recent_short)):
            if recent_short.iloc[i-1] <= recent_long.iloc[i-1] and recent_short.iloc[i] > recent_long.iloc[i]:
                crossed_above = True
                break
        if crossed_above:
            logger.info(f"Golden Cross detected within last {lookback} periods.")
            return 'golden'

        # Check for Death Cross
        crossed_below = False
        for i in range(1, len(recent_short)):
             if recent_short.iloc[i-1] >= recent_long.iloc[i-1] and recent_short.iloc[i] < recent_long.iloc[i]:
                 crossed_below = True
                 break
        if crossed_below:
            logger.info(f"Death Cross detected within last {lookback} periods.")
            return 'death'

        return 'none'

    except Exception as e:
        logger.error(f"check_sma_crossover: Error checking crossover: {e}", exc_info=True)
        return 'none'


@numba.njit('i1[:](f8[:], f8[:], f8[:], f8[:], i8, i8, f8, i8, f8)', cache=True)
def find_volume_breakout_pullback_nb(high: np.ndarray,
                                     low: np.ndarray,
                                     close: np.ndarray,
                                     volume: np.ndarray,
                                     lookback_breakout: int, # For trendline
                                     lookback_pullback: int, # For pullback check
                                     pullback_threshold: float, # For pullback validation
                                     volume_lookback: int, # For volume average
                                     volume_multiplier: float # For volume spike detection
                                     ) -> np.ndarray:
    """
    Numba-optimized function combining trendline breakout/pullback with volume spike confirmation.

    Args:
        high, low, close, volume (np.ndarray): Price and volume data.
        lookback_breakout (int): Window for trendline highs/lows.
        lookback_pullback (int): Window to check pullback.
        pullback_threshold (float): Relative threshold for pullback.
        volume_lookback (int): Window for average volume calculation.
        volume_multiplier (float): Factor to detect volume spike.

    Returns:
        np.ndarray: Signal array similar to find_breakout_pullback_nb, but only
                      returns signals if confirmed by a volume spike.
                      (0: None, 1: Vol Breakout Up, 2: Vol Pullback Support,
                       -1: Vol Breakout Down, -2: Vol Pullback Resistance).
    """
    n = len(close)
    signals = np.zeros(n, dtype=np.int8)
    min_len = max(lookback_breakout, volume_lookback) + lookback_pullback

    if n < min_len:
        return signals # Not enough data

    for i in range(min_len -1, n):
        # --- Volume Spike Check ---
        if i < volume_lookback: continue # Need enough history for volume average
        volume_window = volume[i - volume_lookback : i] # Exclude current bar volume
        avg_volume = np.mean(volume_window)
        current_volume = volume[i]
        is_volume_spike = current_volume > (avg_volume * volume_multiplier) if avg_volume > 0 else False

        if not is_volume_spike:
            continue # Skip if no volume spike

        # --- Trendline Check (copied logic from find_breakout_pullback_nb) ---
        current_relative_idx = lookback_breakout - 1 # Index within the breakout window

        # Upper Trendline
        highs_window = high[i - lookback_breakout + 1 : i + 1]
        if len(highs_window) >= 2:
            # (Simplified peak finding - assumes last two points for trend)
            # Replace with robust peak finding if needed
             p1_idx, p1_val = 0, highs_window[0]
             p2_idx, p2_val = len(highs_window)-2, highs_window[-2] # Example: connect first and second to last high
             if p2_idx > p1_idx:
                 slope_upper = (p2_val - p1_val) / (p2_idx - p1_idx) if (p2_idx - p1_idx) != 0 else 0
                 intercept_upper = p1_val - slope_upper * p1_idx
                 trendline_val_upper = slope_upper * current_relative_idx + intercept_upper

                 # Check for Volume Confirmed Breakout Up
                 if close[i] > trendline_val_upper and close[i-1] <= (slope_upper * (current_relative_idx - 1) + intercept_upper):
                     signals[i] = 1 # Vol Breakout Up

                     # Check for Volume Confirmed Pullback to Support
                     if i >= min_len:
                         broke_out = False
                         pullback_valid = False
                         for j in range(1, lookback_pullback + 1):
                              prev_idx_rel = current_relative_idx - j
                              prev_trend_val = slope_upper * prev_idx_rel + intercept_upper
                              if close[i-j] > prev_trend_val: broke_out = True
                              if broke_out and low[i] <= trendline_val_upper * (1 + pullback_threshold) and low[i] >= trendline_val_upper * (1-pullback_threshold):
                                  pullback_valid = True
                                  break
                         if pullback_valid: signals[i] = 2 # Vol Pullback Support

        # Lower Trendline
        lows_window = low[i - lookback_breakout + 1 : i + 1]
        if len(lows_window) >= 2:
            # (Simplified trough finding)
            p1_idx_low, p1_val_low = 0, lows_window[0]
            p2_idx_low, p2_val_low = len(lows_window)-2, lows_window[-2]
            if p2_idx_low > p1_idx_low:
                slope_lower = (p2_val_low - p1_val_low) / (p2_idx_low - p1_idx_low) if (p2_idx_low - p1_idx_low) != 0 else 0
                intercept_lower = p1_val_low - slope_lower * p1_idx_low
                trendline_val_lower = slope_lower * current_relative_idx + intercept_lower

                # Check for Volume Confirmed Breakout Down
                if close[i] < trendline_val_lower and close[i-1] >= (slope_lower * (current_relative_idx - 1) + intercept_lower):
                    signals[i] = -1 # Vol Breakout Down

                    # Check for Volume Confirmed Pullback to Resistance
                    if i >= min_len:
                        broke_down = False
                        pullback_valid_res = False
                        for j in range(1, lookback_pullback + 1):
                             prev_idx_rel = current_relative_idx - j
                             prev_trend_val_low = slope_lower * prev_idx_rel + intercept_lower
                             if close[i-j] < prev_trend_val_low: broke_down = True
                             if broke_down and high[i] >= trendline_val_lower * (1 - pullback_threshold) and high[i] <= trendline_val_lower * (1 + pullback_threshold):
                                  pullback_valid_res = True
                                  break
                        if pullback_valid_res: signals[i] = -2 # Vol Pullback Resistance

    return signals 