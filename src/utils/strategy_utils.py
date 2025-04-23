# src/utils/strategy_utils.py

import pandas as pd
import numpy as np
import logging
from typing import Optional
import pandas_ta as ta # pandas-ta import 추가
import numba
import vectorbt as vbt # Ensure vectorbt is imported if not already

# --- Setup logger for this module --- 
# Assumes setup_logging() from logging_config has been called by the entry point
logger = logging.getLogger(__name__)
# --- End Setup logger --- 

# Setup basic logging if needed when run standalone or for debugging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_sma(df: pd.DataFrame, window: int, feature_col: str = 'Close') -> pd.DataFrame:
    """
    Calculates the Simple Moving Average (SMA) for a given feature column.

    Args:
        df (pd.DataFrame): DataFrame containing the price data with a DatetimeIndex.
                           Must contain the `feature_col`.
        window (int): The rolling window size for the SMA calculation.
        feature_col (str): The column name to calculate the SMA on (default: 'Close').

    Returns:
        pd.DataFrame: The original DataFrame with an additional column named f'SMA_{window}'.
                      Returns the original DataFrame if calculation fails or prerequisites not met.
    """
    if df is None or df.empty:
        # Use module logger
        logger.warning("calculate_sma: Input DataFrame is empty.")
        return df
    if feature_col not in df.columns:
        # Use module logger
        logger.warning(f"calculate_sma: Feature column '{feature_col}' not found in DataFrame.")
        return df
    if window <= 0:
        # Use module logger
        logger.warning("calculate_sma: Window size must be positive.")
        return df
    if len(df) < window:
        # Use module logger
        logger.warning(f"calculate_sma: Data length ({len(df)}) is shorter than window size ({window}). SMA will contain NaNs.")
        # Continue calculation, but result will be NaN until enough data

    sma_col_name = f'SMA_{window}'
    try:
        # Calculate SMA using pandas rolling mean
        df[sma_col_name] = df[feature_col].rolling(window=window, min_periods=1).mean() # Use min_periods=1 to get SMA even for initial points
        # Use module logger
        logger.debug(f"Calculated {sma_col_name} using window {window} on column '{feature_col}'.")
    except Exception as e:
        # Use module logger
        logger.error(f"calculate_sma: Error calculating SMA for window {window}: {e}", exc_info=True)
        # Return original df without the problematic column if error occurs
        if sma_col_name in df.columns:
             del df[sma_col_name]

    return df

def calculate_rsi(df: pd.DataFrame, window: int = 14, feature_col: str = 'Close') -> pd.DataFrame:
    """
    Calculates the Relative Strength Index (RSI).

    Args:
        df (pd.DataFrame): DataFrame containing price data. Must include `feature_col`.
        window (int): The period for RSI calculation (default: 14).
        feature_col (str): The column to calculate RSI on (default: 'Close').

    Returns:
        pd.DataFrame: The original DataFrame with an additional 'RSI' column.
    """
    if df is None or df.empty:
        # Use module logger
        logger.warning("calculate_rsi: Input DataFrame is empty.")
        return df
    if feature_col not in df.columns:
        # Use module logger
        logger.warning(f"calculate_rsi: Feature column '{feature_col}' not found.")
        return df
    if window <= 0:
        # Use module logger
        logger.warning("calculate_rsi: Window size must be positive.")
        return df
    if len(df) < window + 1: # Need at least window+1 periods for first delta
        # Use module logger
        logger.warning(f"calculate_rsi: Data length ({len(df)}) is insufficient for window size ({window}). RSI will be NaN.")
        df['RSI'] = np.nan # Add NaN column and return
        return df

    rsi_col_name = 'RSI'
    try:
        delta = df[feature_col].diff(1)
        delta = delta.dropna() # Remove first NaN

        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Use Exponential Moving Average (EMA) for Wilder's RSI
        avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
        avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        df[rsi_col_name] = rsi
        # Use module logger
        logger.debug(f"Calculated {rsi_col_name} using window {window} on column '{feature_col}'.")

    except Exception as e:
        # Use module logger
        logger.error(f"calculate_rsi: Error calculating RSI for window {window}: {e}", exc_info=True)
        if rsi_col_name in df.columns:
             del df[rsi_col_name]

    return df

# --- Placeholder for other utility functions ---

def calculate_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std_dev: int = 2, feature_col: str = 'Close') -> pd.DataFrame:
    """
    Calculates Bollinger Bands (Middle, Upper, Lower).

    Args:
        df (pd.DataFrame): DataFrame with price data.
        window (int): Rolling window for SMA and standard deviation.
        num_std_dev (int): Number of standard deviations for upper/lower bands.
        feature_col (str): Column to use for calculation.

    Returns:
        pd.DataFrame: Original DataFrame with 'BB_Middle', 'BB_Upper', 'BB_Lower' columns.
    """
    if df is None or df.empty or feature_col not in df.columns:
         # Use module logger
         logger.warning("calculate_bollinger_bands: Invalid input.")
         return df
    if window <= 0 or num_std_dev <= 0:
        # Use module logger
        logger.warning("calculate_bollinger_bands: Window and num_std_dev must be positive.")
        return df

    middle_band_col = 'BB_Middle'
    upper_band_col = 'BB_Upper'
    lower_band_col = 'BB_Lower'

    try:
         # Calculate Middle Band (SMA)
         df[middle_band_col] = df[feature_col].rolling(window=window, min_periods=1).mean()

         # Calculate Standard Deviation
         rolling_std = df[feature_col].rolling(window=window, min_periods=1).std()

         # Calculate Upper and Lower Bands
         df[upper_band_col] = df[middle_band_col] + (rolling_std * num_std_dev)
         df[lower_band_col] = df[middle_band_col] - (rolling_std * num_std_dev)

         # Use module logger
         logger.debug(f"Calculated Bollinger Bands (W:{window}, SD:{num_std_dev}) on '{feature_col}'.")

    except Exception as e:
        # Use module logger
        logger.error(f"calculate_bollinger_bands: Error calculating Bollinger Bands: {e}", exc_info=True)
        # Clean up columns if error occurred
        for col in [middle_band_col, upper_band_col, lower_band_col]:
             if col in df.columns: del df[col]

    return df

def detect_rsi_divergence(df: pd.DataFrame, price_col: str = 'Close', rsi_col: str = 'RSI', lookback: int = 14, threshold: float = 0.01) -> str:
    """
    Detects simple RSI divergence by comparing the last two major lows/highs
    within the lookback period.

    Args:
        df (pd.DataFrame): DataFrame containing price and RSI data. Index must be sorted.
        price_col (str): Column name for price data (e.g., 'Close', 'Low', 'High').
        rsi_col (str): Column name for RSI data.
        lookback (int): Period to look back for finding the last two lows/highs.
        threshold (float): Minimum relative difference threshold to consider highs/lows significantly different.

    Returns:
        str: 'bullish', 'bearish', or 'none' indicating the type of divergence found.
    """
    if df is None or df.empty or rsi_col not in df.columns or price_col not in df.columns:
        # Use module logger
        logger.warning("detect_rsi_divergence: Invalid input DataFrame or missing columns.")
        return 'none'
    if len(df) < lookback * 2: # Need enough data for comparison
        # Use module logger
        logger.warning(f"detect_rsi_divergence: Insufficient data ({len(df)}) for lookback ({lookback}).")
        return 'none'

    # Use the last `lookback` periods for finding peaks/troughs
    recent_df = df.iloc[-lookback:]

    try:
        # --- Find last two significant lows for Bullish Divergence ---
        # Using rolling min might be too sensitive. Let's try finding actual low points.
        # Simple approach: find the index of the min price and RSI in the period
        price_low_idx = recent_df[price_col].idxmin()
        rsi_low_idx = recent_df[rsi_col].idxmin()
        price_low_val = recent_df.loc[price_low_idx, price_col]
        rsi_low_val = recent_df.loc[rsi_low_idx, rsi_col]

        # Find previous low before the most recent one in the full df (excluding the last identified low point)
        prev_df_lows = df.iloc[-lookback*2:-lookback] # Look in the period before the recent one
        if not prev_df_lows.empty:
            prev_price_low_idx = prev_df_lows[price_col].idxmin()
            prev_rsi_low_idx = prev_df_lows[rsi_col].idxmin() # Using same index for simplicity, refinement possible
            prev_price_low_val = prev_df_lows.loc[prev_price_low_idx, price_col]
            prev_rsi_low_val = prev_df_lows.loc[prev_rsi_low_idx, rsi_col] # Use RSI at the same time as prev price low

            # Check for Bullish Divergence (Price Lower Low, RSI Higher Low)
            # Ensure lows are significantly different using threshold
            price_diff_ratio = abs(price_low_val - prev_price_low_val) / max(abs(prev_price_low_val), 1e-9) # Avoid zero division
            rsi_diff_ratio = abs(rsi_low_val - prev_rsi_low_val) / max(abs(prev_rsi_low_val), 1e-9)

            if price_low_val < prev_price_low_val and rsi_low_val > prev_rsi_low_val and price_diff_ratio > threshold and rsi_diff_ratio > threshold:
                 # Use module logger
                 logger.info(f"Bullish RSI Divergence Detected: Price ({prev_price_low_val:.2f} -> {price_low_val:.2f}), RSI ({prev_rsi_low_val:.2f} -> {rsi_low_val:.2f})")
                 return 'bullish'

        # --- Find last two significant highs for Bearish Divergence ---
        price_high_idx = recent_df[price_col].idxmax()
        rsi_high_idx = recent_df[rsi_col].idxmax()
        price_high_val = recent_df.loc[price_high_idx, price_col]
        rsi_high_val = recent_df.loc[rsi_high_idx, rsi_col]

        # Find previous high before the most recent one
        prev_df_highs = df.iloc[-lookback*2:-lookback]
        if not prev_df_highs.empty:
            prev_price_high_idx = prev_df_highs[price_col].idxmax()
            prev_rsi_high_idx = prev_df_highs[rsi_col].idxmax() # Use RSI at the same time as prev price high
            prev_price_high_val = prev_df_highs.loc[prev_price_high_idx, price_col]
            prev_rsi_high_val = prev_df_highs.loc[prev_rsi_high_idx, rsi_col]

            # Check for Bearish Divergence (Price Higher High, RSI Lower High)
            price_diff_ratio = abs(price_high_val - prev_price_high_val) / max(abs(prev_price_high_val), 1e-9)
            rsi_diff_ratio = abs(rsi_high_val - prev_rsi_high_val) / max(abs(prev_rsi_high_val), 1e-9)

            if price_high_val > prev_price_high_val and rsi_high_val < prev_rsi_high_val and price_diff_ratio > threshold and rsi_diff_ratio > threshold:
                # Use module logger
                logger.info(f"Bearish RSI Divergence Detected: Price ({prev_price_high_val:.2f} -> {price_high_val:.2f}), RSI ({prev_rsi_high_val:.2f} -> {rsi_high_val:.2f})")
                return 'bearish'

    except Exception as e:
        # Use module logger
        logger.error(f"detect_rsi_divergence: Error detecting divergence: {e}", exc_info=True)
        return 'none' # Return 'none' on error

    return 'none' # Default if no divergence found

def detect_trendline_breakout_pullback(df: pd.DataFrame,
                                       price_col: str = 'Close',
                                       high_col: str = 'High',
                                       low_col: str = 'Low',
                                       lookback_breakout: int = 20,
                                       lookback_pullback: int = 5,
                                       pullback_threshold: float = 0.01) -> str:
    """
    Detects a breakout of a recent high/low followed by a pullback towards the breakout level.
    Improved logic: Identifies the breakout bar first, then checks for pullback after.

    Args:
        df (pd.DataFrame): DataFrame with price data (Close, High, Low). Index must be sorted.
        price_col (str): Column for current price checks ('Close').
        high_col (str): Column for high prices.
        low_col (str): Column for low prices.
        lookback_breakout (int): Period before the potential breakout to find the high/low level.
        lookback_pullback (int): Period after the potential breakout to look for pullback.
                                  Also defines how recently the breakout must have occurred.
        pullback_threshold (float): Relative threshold defining how close the pullback must be to the breakout level.

    Returns:
        str: 'bullish_pullback', 'bearish_breakdown', or 'none'.
    """
    if df is None or df.empty or not all(c in df.columns for c in [price_col, high_col, low_col]):
        # Use module logger
        logger.warning("detect_trendline_breakout_pullback: Invalid input DataFrame or missing columns.")
        return 'none'
    # Need enough data for breakout period + pullback lookback period + current bar
    required_len = lookback_breakout + lookback_pullback + 1
    if len(df) < required_len:
        # Use module logger
        logger.debug(f"detect_trendline_breakout_pullback: Insufficient data ({len(df)}) for lookbacks ({required_len}).")
        return 'none'

    try:
        current_index = len(df) - 1 # 현재 봉의 인덱스 (0부터 시작)
        current_close = df[price_col].iloc[current_index]

        # Iterate backwards from `lookback_pullback` bars ago up to yesterday to find a potential breakout bar
        # Example: lookback_pullback=5 -> Check bars at index -5, -4, -3, -2, -1
        for i in range(lookback_pullback, 0, -1):
            potential_breakout_index = current_index - i
            if potential_breakout_index < lookback_breakout: # Need enough preceding data to define breakout level
                continue

            potential_breakout_high = df[high_col].iloc[potential_breakout_index]
            potential_breakout_low = df[low_col].iloc[potential_breakout_index]

            # --- Define Breakout Level --- 
            # Look back `lookback_breakout` bars *before* the potential breakout bar
            breakout_level_period_start = potential_breakout_index - lookback_breakout
            breakout_level_period_end = potential_breakout_index # exclusive
            breakout_level_df = df.iloc[breakout_level_period_start:breakout_level_period_end]

            if breakout_level_df.empty: continue # Should not happen with length check, but be safe

            highest_high_before_breakout = breakout_level_df[high_col].max()
            lowest_low_before_breakout = breakout_level_df[low_col].min()

            # --- Check for Bullish Breakout Event --- 
            # Did the high of the potential breakout bar exceed the previous highest high?
            is_bullish_breakout_bar = potential_breakout_high > highest_high_before_breakout

            if is_bullish_breakout_bar:
                # Breakout identified at `potential_breakout_index`
                # Now, check for pullback in the period *after* the breakout bar, up to the current bar
                pullback_check_start = potential_breakout_index + 1
                pullback_check_end = current_index + 1 # inclusive
                pullback_period_df = df.iloc[pullback_check_start:pullback_check_end]

                if not pullback_period_df.empty:
                    lowest_low_after_breakout = pullback_period_df[low_col].min()

                    # Check if the pullback low touched near the breakout level
                    # and if the current close is above the pullback low (bounce)
                    pullback_valid = lowest_low_after_breakout <= highest_high_before_breakout * (1 + pullback_threshold)
                    bounce_valid = current_close > lowest_low_after_breakout

                    if pullback_valid and bounce_valid:
                         # Use module logger
                         logger.info(f"Bullish Pullback Detected: Breakout Bar {potential_breakout_index}, Broke Level {highest_high_before_breakout:.2f}, Pulled back to {lowest_low_after_breakout:.2f}, Current {current_close:.2f}")
                         return 'bullish_pullback'
                # If a bullish breakout was found, no need to check further back for other breakouts
                break # Exit the loop once the most recent valid breakout is checked

            # --- Check for Bearish Breakdown Event --- 
            # Did the low of the potential breakout bar go below the previous lowest low?
            is_bearish_breakdown_bar = potential_breakout_low < lowest_low_before_breakout

            if is_bearish_breakdown_bar:
                # Breakdown identified at `potential_breakout_index`
                # Check for pullback (rally attempt) after the breakdown bar
                pullback_check_start = potential_breakout_index + 1
                pullback_check_end = current_index + 1
                pullback_period_df = df.iloc[pullback_check_start:pullback_check_end]

                if not pullback_period_df.empty:
                    highest_high_after_breakdown = pullback_period_df[high_col].max()

                    # Check if the rally attempt reached near the breakdown level
                    # and if the current close is below the rally high (failed rally)
                    pullback_valid = highest_high_after_breakdown >= lowest_low_before_breakout * (1 - pullback_threshold)
                    rejection_valid = current_close < highest_high_after_breakdown

                    if pullback_valid and rejection_valid:
                        # Use module logger
                        logger.info(f"Bearish Breakdown Detected: Breakdown Bar {potential_breakout_index}, Broke Level {lowest_low_before_breakout:.2f}, Pulled back to {highest_high_after_breakdown:.2f}, Current {current_close:.2f}")
                        return 'bearish_breakdown'
                # If a bearish breakdown was found, no need to check further back
                break

    except Exception as e:
        # Use module logger
        logger.error(f"detect_trendline_breakout_pullback: Error detecting: {e}", exc_info=True)
        return 'none'

    return 'none' # No valid breakout/pullback pattern found within the lookback period

def calculate_poc(df: pd.DataFrame, lookback: int = 50, price_col: str = 'Close', volume_col: str = 'Volume') -> Optional[float]:
    """
    Calculates the Point of Control (POC) approximated by Volume Weighted Average Price (VWAP)
    over the specified lookback period.

    Args:
        df (pd.DataFrame): DataFrame containing price and volume data.
        lookback (int): The number of periods to look back for calculation.
        price_col (str): The column name for price (default: 'Close').
        volume_col (str): The column name for volume (default: 'Volume').

    Returns:
        Optional[float]: The calculated POC (VWAP) value for the last point in the lookback period,
                         or None if calculation is not possible (e.g., insufficient data, zero volume).
    """
    if df is None or df.empty:
        logger.warning("calculate_poc: Input DataFrame is empty.")
        return None
    if not all(col in df.columns for col in [price_col, volume_col]):
        logger.warning(f"calculate_poc: Required columns ('{price_col}', '{volume_col}') not found.")
        return None
    if len(df) < lookback:
        # logger.debug(f"calculate_poc: Insufficient data ({len(df)}) for lookback ({lookback}).") # Debug level
        return None # Not enough data to calculate over the full lookback

    # Get the slice for the lookback period (last 'lookback' rows)
    df_slice = df.iloc[-lookback:]

    # --- Check for NaN values in the slice --- 
    if df_slice[[price_col, volume_col]].isnull().values.any():
        logger.warning(f"calculate_poc: NaN values found in price or volume within the lookback period. Unable to calculate POC accurately.")
        # Decide how to handle NaNs: return None, fillna(0), etc.
        # For now, return None if any NaN exists in the required columns for the slice
        return None

    try:
        total_volume = df_slice[volume_col].sum()

        # --- Check for zero total volume before division --- 
        if total_volume == 0:
            logger.warning(f"calculate_poc: Total volume within the lookback period ({lookback}) is zero. Cannot calculate POC.")
            return None

        # Calculate VWAP as POC proxy
        vwap = (df_slice[price_col] * df_slice[volume_col]).sum() / total_volume

        # Check if the result is NaN (could happen due to other numeric issues)
        if pd.isna(vwap):
             logger.warning(f"calculate_poc: Calculated VWAP (POC proxy) is NaN despite non-zero volume. Check input data.")
             return None

        # logger.debug(f"Calculated POC (VWAP proxy): {vwap:.4f} over last {lookback} periods.") # Debug level
        return vwap

    except Exception as e:
        logger.error(f"calculate_poc: Error calculating POC (VWAP proxy): {e}", exc_info=True)
        return None

# Add other functions like calculate_vpvr etc. here later 

# Numba-optimized version of the core logic
# Add explicit type signature
@numba.njit('i1[:](f8[:], f8[:], f8[:], i8, i8, f8)')
def find_breakout_pullback_nb(high: np.ndarray,
                              low: np.ndarray,
                              close: np.ndarray,
                              lookback_breakout: int,
                              lookback_pullback: int,
                              pullback_threshold: float) -> np.ndarray:
    """
    Numba-optimized function to detect breakout/pullbacks.
    Returns an array where:
    1 = bullish pullback
    -1 = bearish breakdown pullback
    0 = none
    """
    n = len(close)
    signals = np.zeros(n, dtype=np.int8) # 0: none, 1: bullish, -1: bearish

    # Need enough data for breakout period + pullback lookback period + current bar
    required_len = lookback_breakout + lookback_pullback + 1

    # Start from the first index where enough data is available
    for current_index in range(required_len - 1, n):
        # Iterate backwards from `lookback_pullback` bars ago up to yesterday to find a potential breakout bar
        found_signal_in_inner = False # Added flag
        for i in range(lookback_pullback, 0, -1):
            if found_signal_in_inner: break # Exit if signal found in inner loop

            potential_breakout_index = current_index - i
            # Ensure we don't index before the start of the array
            if potential_breakout_index < 0:
                 continue
            # Check if we have enough history for the breakout level calculation itself
            if potential_breakout_index < lookback_breakout:
                continue

            potential_breakout_high = high[potential_breakout_index]
            potential_breakout_low = low[potential_breakout_index]

            # --- Define Breakout Level ---
            breakout_level_period_start = potential_breakout_index - lookback_breakout
            breakout_level_period_end = potential_breakout_index
            # Numba works with numpy slices
            breakout_level_highs = high[breakout_level_period_start:breakout_level_period_end]
            breakout_level_lows = low[breakout_level_period_start:breakout_level_period_end]

            if breakout_level_highs.size == 0 or breakout_level_lows.size == 0: continue # Safety check

            # Use np.nanmax/np.nanmin if NaNs are possible, otherwise np.max/np.min
            highest_high_before_breakout = np.max(breakout_level_highs)
            lowest_low_before_breakout = np.min(breakout_level_lows)

            # --- Check for Bullish Breakout Event ---
            is_bullish_breakout_bar = potential_breakout_high > highest_high_before_breakout
            if is_bullish_breakout_bar:
                # Check for pullback after the breakout bar
                pullback_check_start = potential_breakout_index + 1
                pullback_check_end = current_index + 1 # Numba slicing is exclusive at the end
                pullback_period_lows = low[pullback_check_start:pullback_check_end]

                if pullback_period_lows.size > 0:
                    lowest_low_after_breakout = np.min(pullback_period_lows)
                    current_close_val = close[current_index]

                    # Check conditions
                    pullback_valid = lowest_low_after_breakout <= highest_high_before_breakout * (1 + pullback_threshold)
                    bounce_valid = current_close_val > lowest_low_after_breakout

                    if pullback_valid and bounce_valid:
                        signals[current_index] = 1 # Bullish pullback
                        found_signal_in_inner = True # Set flag
                        break # Exit inner loop, found most recent pattern for this current_index
                # Even if pullback check fails, break because we found the most recent breakout bar
                break # Exit inner loop, checked the most recent breakout

            # --- Check for Bearish Breakdown Event ---
            is_bearish_breakdown_bar = potential_breakout_low < lowest_low_before_breakout
            if is_bearish_breakdown_bar:
                # Check for pullback (rally attempt) after the breakdown bar
                pullback_check_start = potential_breakout_index + 1
                pullback_check_end = current_index + 1
                pullback_period_highs = high[pullback_check_start:pullback_check_end]

                if pullback_period_highs.size > 0:
                    highest_high_after_breakdown = np.max(pullback_period_highs)
                    current_close_val = close[current_index]

                    # Check conditions
                    pullback_valid = highest_high_after_breakdown >= lowest_low_before_breakout * (1 - pullback_threshold)
                    rejection_valid = current_close_val < highest_high_after_breakdown

                    if pullback_valid and rejection_valid:
                        signals[current_index] = -1 # Bearish breakdown pullback
                        found_signal_in_inner = True # Set flag
                        break # Exit inner loop
                # Even if pullback check fails, break because we found the most recent breakdown bar
                break # Exit inner loop

        # If the inner loop finished without break, no signal was found for current_index

    return signals

# Create a vectorbt Indicator using the Numba function
# BreakoutPullback = vbt.IndicatorFactory(
#     class_name="BreakoutPullback",
#     short_name="bopb",
#     input_names=["high", "low", "close"], # Input price series
#     param_names=["lookback_breakout", "lookback_pullback", "pullback_threshold"], # Parameters for the function
#     output_names=["signal"]
# ).from_apply_func(
#     find_breakout_pullback_nb, # The Numba function
#     # Default parameter values (can be overridden)
#     lookback_breakout=20,
#     lookback_pullback=5,
#     pullback_threshold=0.01,
#     keep_pd=True # Output as pd.Series
# ) 

# --- NEW RSI Divergence Function (as requested) ---
def detect_rsi_divergence_v2(close: pd.Series, rsi: pd.Series, window: int = 14):
    '''
    Detects simple bullish RSI divergence over a rolling window.
    - MODIFIED: Price makes a lower low (< 1% below window min), RSI makes a higher low (> 1% above window min).

    Args:
        close (pd.Series): Series of closing prices.
        rsi (pd.Series): Series of RSI values.
        window (int): Lookback window for divergence detection.

    Returns:
        pd.Series: Boolean series indicating where bullish divergence is detected.
                   True means divergence detected at that point.
    '''
    if not isinstance(close, pd.Series) or not isinstance(rsi, pd.Series):
        logger.error("detect_rsi_divergence_v2: Inputs must be pandas Series.")
        return pd.Series(False, index=close.index if isinstance(close, pd.Series) else None)
    if not close.index.equals(rsi.index):
        logger.error("detect_rsi_divergence_v2: Price and RSI Series must have the same index.")
        # Attempt to align if possible, otherwise return False
        try:
            rsi = rsi.reindex(close.index)
        except Exception:
             return pd.Series(False, index=close.index)

    divergence = pd.Series(False, index=close.index)

    # Ensure input series are numpy arrays for Numba compatibility if we decide to use it
    close_np = close.to_numpy(dtype=np.float64, na_value=np.nan)
    rsi_np = rsi.to_numpy(dtype=np.float64, na_value=np.nan)

    # Basic loop implementation (can be optimized with Numba later if needed)
    for i in range(window, len(close_np)):
        # Define the lookback window slice (inclusive of current point i)
        window_slice_indices = range(i - window, i + 1)
        price_window = close_np[window_slice_indices]
        rsi_window = rsi_np[window_slice_indices]

        # Check for NaNs in the lookback window
        if np.isnan(price_window).any() or np.isnan(rsi_window).any():
            continue # Skip if NaN values exist in the window

        price_now = close_np[i]
        rsi_now = rsi_np[i]

        # Calculate window minimums (handling NaNs)
        min_price_in_window = np.nanmin(price_window)
        min_rsi_in_window = np.nanmin(rsi_window)

        # === MODIFIED Bullish Divergence Condition ===
        # Price is more than 1% below the window minimum price
        price_cond = price_now < (min_price_in_window * 0.99)
        # RSI is more than 1% above the window minimum RSI
        rsi_cond = rsi_now > (min_rsi_in_window * 1.01)

        if price_cond and rsi_cond:
            divergence.iloc[i] = True

    logger.info(f"Calculated RSI Bullish Divergence signals (window={window}, modified logic). Found {divergence.sum()} divergence points.")
    return divergence
# --- END NEW FUNCTION ---

# === Placeholder for Volume Profile / PoC Calculation ===
def calculate_poc(df: pd.DataFrame, lookback: int = 50, price_col: str = 'Close', volume_col: str = 'Volume') -> Optional[float]:
    """
    Calculates the Point of Control (PoC) using a simple volume-weighted approach
    over a given lookback period. This is a simplified approximation.

    Args:
        df (pd.DataFrame): DataFrame with price and volume data.
        lookback (int): Number of periods to look back.
        price_col (str): Column name for price (used for grouping).
        volume_col (str): Column name for volume.

    Returns:
        Optional[float]: The price level with the highest volume (PoC) in the lookback,
                         or None if calculation fails.
    """
    if df is None or df.empty or price_col not in df.columns or volume_col not in df.columns:
        logger.warning("calculate_poc: Invalid input DataFrame or missing columns.")
        return None
    if len(df) < lookback:
        logger.warning(f"calculate_poc: Insufficient data ({len(df)}) for lookback ({lookback}).")
        return None

    recent_df = df.iloc[-lookback:]

    try:
        # Group by price level (can be sensitive to exact price; rounding might be needed)
        # For simplicity, let's round the price to a reasonable number of decimals
        # Assuming crypto, maybe 2 decimal places? Adjust as needed.
        if recent_df[price_col].max() > 1000: # Heuristic for crypto
            price_rounded = recent_df[price_col].round(2)
        else: # Assume stocks/lower priced assets
            price_rounded = recent_df[price_col].round(4)

        volume_at_price = recent_df.groupby(price_rounded)[volume_col].sum()

        if volume_at_price.empty:
            logger.warning("calculate_poc: No volume data found after grouping by price.")
            return None

        poc_price = volume_at_price.idxmax()
        logger.debug(f"Calculated PoC over last {lookback} periods: {poc_price}")
        return float(poc_price)

    except Exception as e:
        logger.error(f"calculate_poc: Error calculating PoC: {e}", exc_info=True)
        return None

# =========================================================
# === SECTION: Functions Potentially Useful for Backtrader ===
# These might need adjustments to work directly within backtrader's structure
# =========================================================

@numba.njit('boolean(f8[:], f8[:], i8)')
def check_last_crossing_nb(series1: np.ndarray, series2: np.ndarray, lookback: int) -> bool:
    """Numba optimized check for recent crossover/crossunder."""
    n = len(series1)
    if n < lookback + 1:
        return False # Not enough data
    
    current_diff = series1[n-1] - series2[n-1]
    
    for i in range(n - 2, n - lookback - 1, -1):
        prev_diff = series1[i] - series2[i]
        # Check if the sign of the difference changed (indicates a cross)
        if np.sign(current_diff) != np.sign(prev_diff) and prev_diff != 0:
            return True # A cross happened within the lookback period
    return False

def check_sma_crossover(df: pd.DataFrame, short_window: int, long_window: int, lookback: int = 3) -> str:
    """Checks for recent SMA crossovers (Golden/Death Cross)."""
    sma_short_col = f'SMA_{short_window}'
    sma_long_col = f'SMA_{long_window}'

    if sma_short_col not in df.columns or sma_long_col not in df.columns:
        logger.warning("check_sma_crossover: Required SMA columns not found.")
        return 'none'
    if len(df) < max(short_window, long_window) + lookback:
         logger.warning("check_sma_crossover: Insufficient data for lookback check.")
         return 'none'

    sma_short = df[sma_short_col].values
    sma_long = df[sma_long_col].values
    
    # Check for Golden Cross (Short SMA crosses above Long SMA)
    if sma_short[-1] > sma_long[-1]: # Currently above
        # Use Numba function to check if a crossunder happened recently
        crossed_recently = check_last_crossing_nb(sma_short, sma_long, lookback)
        if crossed_recently and sma_short[-lookback-1] < sma_long[-lookback-1]: # Check direction before cross
             logger.info(f"Golden Cross detected within last {lookback} periods.")
             return 'golden'
            
    # Check for Death Cross (Short SMA crosses below Long SMA)
    elif sma_short[-1] < sma_long[-1]: # Currently below
         crossed_recently = check_last_crossing_nb(sma_short, sma_long, lookback)
         if crossed_recently and sma_short[-lookback-1] > sma_long[-lookback-1]: # Check direction before cross
             logger.info(f"Death Cross detected within last {lookback} periods.")
             return 'death'

    return 'none'

# Placeholder for BreakoutPullback adapted for backtrader (might need class structure)
# This needs access to `self.data` within a backtrader Strategy
# def detect_breakout_pullback_backtrader(data, lookback_breakout, lookback_pullback, threshold):
#     high = data.high.get(size=lookback_breakout + lookback_pullback + 1)
#     low = data.low.get(size=lookback_breakout + lookback_pullback + 1)
#     close = data.close.get(size=lookback_breakout + lookback_pullback + 1)
#     # ... rest of the logic using array indexing ...
#     # This would likely be part of the next() method in a Strategy
#     pass

# === NEW Numba function including Volume Surge ===
@numba.njit('i1[:](f8[:], f8[:], f8[:], f8[:], i8, i8, f8, i8, f8)', cache=True)
def find_volume_breakout_pullback_nb(high: np.ndarray,
                                     low: np.ndarray,
                                     close: np.ndarray,
                                     volume: np.ndarray,
                                     lookback_breakout: int,
                                     lookback_pullback: int,
                                     pullback_threshold: float,
                                     volume_lookback: int,
                                     volume_multiplier: float) -> np.ndarray:
    """
    Numba-optimized function to detect breakouts with volume surge, followed by a pullback.
    Returns an array where:
    1 = bullish breakout with volume surge + pullback
    0 = none
    (Currently only implements bullish scenario)
    """
    n = len(close)
    signals = np.zeros(n, dtype=np.int8)
    required_len = max(lookback_breakout, volume_lookback) + lookback_pullback + 1

    for current_index in range(required_len - 1, n):
        found_signal_in_inner = False
        for i in range(lookback_pullback, 0, -1):
            if found_signal_in_inner: break

            potential_breakout_index = current_index - i
            if potential_breakout_index < 0: continue
            if potential_breakout_index < lookback_breakout or potential_breakout_index < volume_lookback:
                continue

            potential_breakout_high = high[potential_breakout_index]
            # potential_breakout_low = low[potential_breakout_index] # Not used for bullish only
            breakout_volume = volume[potential_breakout_index]

            # --- Define Breakout Level --- 
            breakout_level_period_start = potential_breakout_index - lookback_breakout
            breakout_level_period_end = potential_breakout_index
            breakout_level_highs = high[breakout_level_period_start:breakout_level_period_end]
            if breakout_level_highs.size == 0: continue
            highest_high_before_breakout = np.max(breakout_level_highs)

            # --- Define Average Volume --- 
            volume_period_start = potential_breakout_index - volume_lookback
            volume_period_end = potential_breakout_index
            recent_volumes = volume[volume_period_start:volume_period_end]
            if recent_volumes.size == 0: continue
            avg_volume = np.mean(recent_volumes)
            # Avoid division by zero or issues with very low volume
            if avg_volume <= 1e-9: avg_volume = 1e-9

            # --- Check for Bullish Breakout with Volume Surge --- 
            is_bullish_breakout_bar = potential_breakout_high > highest_high_before_breakout
            volume_surge = breakout_volume > (avg_volume * volume_multiplier)

            if is_bullish_breakout_bar and volume_surge:
                # Breakout+Volume Surge identified at `potential_breakout_index`
                # Now, check for pullback
                pullback_check_start = potential_breakout_index + 1
                pullback_check_end = current_index + 1
                pullback_period_lows = low[pullback_check_start:pullback_check_end]

                if pullback_period_lows.size > 0:
                    lowest_low_after_breakout = np.min(pullback_period_lows)
                    current_close_val = close[current_index]

                    # Pullback must touch near the breakout level
                    pullback_valid = lowest_low_after_breakout <= highest_high_before_breakout * (1 + pullback_threshold)
                    # Current price must be above the lowest point of the pullback (bounce)
                    bounce_valid = current_close_val > lowest_low_after_breakout

                    if pullback_valid and bounce_valid:
                        signals[current_index] = 1 # Bullish signal
                        found_signal_in_inner = True
                        break # Exit inner loop, found signal
                # Break inner loop even if pullback failed, as we processed the most recent potential breakout
                break

            # --- Bearish Check (Skipped for this strategy) ---
            # is_bearish_breakdown_bar = potential_breakout_low < lowest_low_before_breakout
            # ... (volume surge check for bearish)
            # ... (pullback check for bearish)

    return signals
# --- END NEW Numba function ---

# --- NEW RSI Divergence Function (as requested) ---
def detect_rsi_divergence_v2(close: pd.Series, rsi: pd.Series, window: int = 14):
    '''
    Detects simple bullish RSI divergence over a rolling window.
    - MODIFIED: Price makes a lower low (< 1% below window min), RSI makes a higher low (> 1% above window min).

    Args:
        close (pd.Series): Series of closing prices.
        rsi (pd.Series): Series of RSI values.
        window (int): Lookback window for divergence detection.

    Returns:
        pd.Series: Boolean series indicating where bullish divergence is detected.
                   True means divergence detected at that point.
    '''
    if not isinstance(close, pd.Series) or not isinstance(rsi, pd.Series):
        logger.error("detect_rsi_divergence_v2: Inputs must be pandas Series.")
        return pd.Series(False, index=close.index if isinstance(close, pd.Series) else None)
    if not close.index.equals(rsi.index):
        logger.error("detect_rsi_divergence_v2: Price and RSI Series must have the same index.")
        # Attempt to align if possible, otherwise return False
        try:
            rsi = rsi.reindex(close.index)
        except Exception:
             return pd.Series(False, index=close.index)

    divergence = pd.Series(False, index=close.index)

    # Ensure input series are numpy arrays for Numba compatibility if we decide to use it
    close_np = close.to_numpy(dtype=np.float64, na_value=np.nan)
    rsi_np = rsi.to_numpy(dtype=np.float64, na_value=np.nan)

    # Basic loop implementation (can be optimized with Numba later if needed)
    for i in range(window, len(close_np)):
        # Define the lookback window slice (inclusive of current point i)
        window_slice_indices = range(i - window, i + 1)
        price_window = close_np[window_slice_indices]
        rsi_window = rsi_np[window_slice_indices]

        # Check for NaNs in the lookback window
        if np.isnan(price_window).any() or np.isnan(rsi_window).any():
            continue # Skip if NaN values exist in the window

        price_now = close_np[i]
        rsi_now = rsi_np[i]

        # Calculate window minimums (handling NaNs)
        min_price_in_window = np.nanmin(price_window)
        min_rsi_in_window = np.nanmin(rsi_window)

        # === MODIFIED Bullish Divergence Condition ===
        # Price is more than 1% below the window minimum price
        price_cond = price_now < (min_price_in_window * 0.99)
        # RSI is more than 1% above the window minimum RSI
        rsi_cond = rsi_now > (min_rsi_in_window * 1.01)

        if price_cond and rsi_cond:
            divergence.iloc[i] = True

    logger.info(f"Calculated RSI Bullish Divergence signals (window={window}, modified logic). Found {divergence.sum()} divergence points.")
    return divergence
# --- END NEW FUNCTION ---

# === Placeholder for Volume Profile / PoC Calculation ===
def calculate_poc(df: pd.DataFrame, lookback: int = 50, price_col: str = 'Close', volume_col: str = 'Volume') -> Optional[float]:
    """
    Calculates the Point of Control (PoC) using a simple volume-weighted approach
    over a given lookback period. This is a simplified approximation.

    Args:
        df (pd.DataFrame): DataFrame with price and volume data.
        lookback (int): Number of periods to look back.
        price_col (str): Column name for price (used for grouping).
        volume_col (str): Column name for volume.

    Returns:
        Optional[float]: The price level with the highest volume (PoC) in the lookback,
                         or None if calculation fails.
    """
    if df is None or df.empty or price_col not in df.columns or volume_col not in df.columns:
        logger.warning("calculate_poc: Invalid input DataFrame or missing columns.")
        return None
    if len(df) < lookback:
        logger.warning(f"calculate_poc: Insufficient data ({len(df)}) for lookback ({lookback}).")
        return None

    recent_df = df.iloc[-lookback:]

    try:
        # Group by price level (can be sensitive to exact price; rounding might be needed)
        # For simplicity, let's round the price to a reasonable number of decimals
        # Assuming crypto, maybe 2 decimal places? Adjust as needed.
        if recent_df[price_col].max() > 1000: # Heuristic for crypto
            price_rounded = recent_df[price_col].round(2)
        else: # Assume stocks/lower priced assets
            price_rounded = recent_df[price_col].round(4)

        volume_at_price = recent_df.groupby(price_rounded)[volume_col].sum()

        if volume_at_price.empty:
            logger.warning("calculate_poc: No volume data found after grouping by price.")
            return None

        poc_price = volume_at_price.idxmax()
        logger.debug(f"Calculated PoC over last {lookback} periods: {poc_price}")
        return float(poc_price)

    except Exception as e:
        logger.error(f"calculate_poc: Error calculating PoC: {e}", exc_info=True)
        return None

# =========================================================
# === SECTION: Functions Potentially Useful for Backtrader ===
# These might need adjustments to work directly within backtrader's structure
# =========================================================

@numba.njit('boolean(f8[:], f8[:], i8)')
def check_last_crossing_nb(series1: np.ndarray, series2: np.ndarray, lookback: int) -> bool:
    """Numba optimized check for recent crossover/crossunder."""
    n = len(series1)
    if n < lookback + 1:
        return False # Not enough data
    
    current_diff = series1[n-1] - series2[n-1]
    
    for i in range(n - 2, n - lookback - 1, -1):
        prev_diff = series1[i] - series2[i]
        # Check if the sign of the difference changed (indicates a cross)
        if np.sign(current_diff) != np.sign(prev_diff) and prev_diff != 0:
            return True # A cross happened within the lookback period
    return False

def check_sma_crossover(df: pd.DataFrame, short_window: int, long_window: int, lookback: int = 3) -> str:
    """Checks for recent SMA crossovers (Golden/Death Cross)."""
    sma_short_col = f'SMA_{short_window}'
    sma_long_col = f'SMA_{long_window}'

    if sma_short_col not in df.columns or sma_long_col not in df.columns:
        logger.warning("check_sma_crossover: Required SMA columns not found.")
        return 'none'
    if len(df) < max(short_window, long_window) + lookback:
         logger.warning("check_sma_crossover: Insufficient data for lookback check.")
         return 'none'

    sma_short = df[sma_short_col].values
    sma_long = df[sma_long_col].values
    
    # Check for Golden Cross (Short SMA crosses above Long SMA)
    if sma_short[-1] > sma_long[-1]: # Currently above
        # Use Numba function to check if a crossunder happened recently
        crossed_recently = check_last_crossing_nb(sma_short, sma_long, lookback)
        if crossed_recently and sma_short[-lookback-1] < sma_long[-lookback-1]: # Check direction before cross
             logger.info(f"Golden Cross detected within last {lookback} periods.")
             return 'golden'
            
    # Check for Death Cross (Short SMA crosses below Long SMA)
    elif sma_short[-1] < sma_long[-1]: # Currently below
         crossed_recently = check_last_crossing_nb(sma_short, sma_long, lookback)
         if crossed_recently and sma_short[-lookback-1] > sma_long[-lookback-1]: # Check direction before cross
             logger.info(f"Death Cross detected within last {lookback} periods.")
             return 'death'

    return 'none'

# Placeholder for BreakoutPullback adapted for backtrader (might need class structure)
# This needs access to `self.data` within a backtrader Strategy
# def detect_breakout_pullback_backtrader(data, lookback_breakout, lookback_pullback, threshold):
#     high = data.high.get(size=lookback_breakout + lookback_pullback + 1)
#     low = data.low.get(size=lookback_breakout + lookback_pullback + 1)
#     close = data.close.get(size=lookback_breakout + lookback_pullback + 1)
#     # ... rest of the logic using array indexing ...
#     # This would likely be part of the next() method in a Strategy
#     pass 