# src/utils/strategy_utils.py

import pandas as pd
import numpy as np
import logging
from typing import Optional
import pandas_ta as ta # pandas-ta import 추가

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
        logging.warning("calculate_sma: Input DataFrame is empty.")
        return df
    if feature_col not in df.columns:
        logging.warning(f"calculate_sma: Feature column '{feature_col}' not found in DataFrame.")
        return df
    if window <= 0:
        logging.warning("calculate_sma: Window size must be positive.")
        return df
    if len(df) < window:
        logging.warning(f"calculate_sma: Data length ({len(df)}) is shorter than window size ({window}). SMA will contain NaNs.")
        # Continue calculation, but result will be NaN until enough data

    sma_col_name = f'SMA_{window}'
    try:
        # Calculate SMA using pandas rolling mean
        df[sma_col_name] = df[feature_col].rolling(window=window, min_periods=1).mean() # Use min_periods=1 to get SMA even for initial points
        logging.debug(f"Calculated {sma_col_name} using window {window} on column '{feature_col}'.")
    except Exception as e:
        logging.error(f"calculate_sma: Error calculating SMA for window {window}: {e}", exc_info=True)
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
        logging.warning("calculate_rsi: Input DataFrame is empty.")
        return df
    if feature_col not in df.columns:
        logging.warning(f"calculate_rsi: Feature column '{feature_col}' not found.")
        return df
    if window <= 0:
        logging.warning("calculate_rsi: Window size must be positive.")
        return df
    if len(df) < window + 1: # Need at least window+1 periods for first delta
        logging.warning(f"calculate_rsi: Data length ({len(df)}) is insufficient for window size ({window}). RSI will be NaN.")
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
        logging.debug(f"Calculated {rsi_col_name} using window {window} on column '{feature_col}'.")

    except Exception as e:
        logging.error(f"calculate_rsi: Error calculating RSI for window {window}: {e}", exc_info=True)
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
         logging.warning("calculate_bollinger_bands: Invalid input.")
         return df
    if window <= 0 or num_std_dev <= 0:
        logging.warning("calculate_bollinger_bands: Window and num_std_dev must be positive.")
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

         logging.debug(f"Calculated Bollinger Bands (W:{window}, SD:{num_std_dev}) on '{feature_col}'.")

    except Exception as e:
        logging.error(f"calculate_bollinger_bands: Error calculating Bollinger Bands: {e}", exc_info=True)
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
        logging.warning("detect_rsi_divergence: Invalid input DataFrame or missing columns.")
        return 'none'
    if len(df) < lookback * 2: # Need enough data for comparison
        logging.warning(f"detect_rsi_divergence: Insufficient data ({len(df)}) for lookback ({lookback}).")
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
                 logging.info(f"Bullish RSI Divergence Detected: Price ({prev_price_low_val:.2f} -> {price_low_val:.2f}), RSI ({prev_rsi_low_val:.2f} -> {rsi_low_val:.2f})")
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
                logging.info(f"Bearish RSI Divergence Detected: Price ({prev_price_high_val:.2f} -> {price_high_val:.2f}), RSI ({prev_rsi_high_val:.2f} -> {rsi_high_val:.2f})")
                return 'bearish'

    except Exception as e:
        logging.error(f"detect_rsi_divergence: Error detecting divergence: {e}", exc_info=True)
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
        logging.warning("detect_trendline_breakout_pullback: Invalid input DataFrame or missing columns.")
        return 'none'
    # Need enough data for breakout period + pullback lookback period + current bar
    required_len = lookback_breakout + lookback_pullback + 1
    if len(df) < required_len:
        logging.debug(f"detect_trendline_breakout_pullback: Insufficient data ({len(df)}) for lookbacks ({required_len}).")
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
                         logging.info(f"Bullish Pullback Detected: Breakout Bar {potential_breakout_index}, Broke Level {highest_high_before_breakout:.2f}, Pulled back to {lowest_low_after_breakout:.2f}, Current {current_close:.2f}")
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
                        logging.info(f"Bearish Breakdown Detected: Breakdown Bar {potential_breakout_index}, Broke Level {lowest_low_before_breakout:.2f}, Pulled back to {highest_high_after_breakdown:.2f}, Current {current_close:.2f}")
                        return 'bearish_breakdown'
                # If a bearish breakdown was found, no need to check further back
                break

    except Exception as e:
        logging.error(f"detect_trendline_breakout_pullback: Error detecting: {e}", exc_info=True)
        return 'none'

    return 'none' # No valid breakout/pullback pattern found within the lookback period

def calculate_poc(df: pd.DataFrame, lookback: int = 50, price_col: str = 'Close', volume_col: str = 'Volume') -> Optional[float]:
    """
    Calculates the Point of Control (POC) using pandas_ta library's VWAP calculation
    as a proxy (most traded price level within the period).
    Note: pandas_ta doesn't have a direct VPVR function, VWAP on recent data can serve
          as an approximation of the highest volume price area if interpreted carefully.
          For true VPVR, a custom histogram calculation or another library might be needed.
          This function provides a basic POC approximation based on recent VWAP behavior.

    Args:
        df (pd.DataFrame): DataFrame containing price and volume data.
                           Needs 'High', 'Low', 'Close', 'Volume' columns typically.
        lookback (int): Period over which to calculate the VWAP/POC proxy.
        price_col (str): Price column (usually 'Close') for reference.
        volume_col (str): Volume column.

    Returns:
        Optional[float]: The approximated Point of Control (highest volume price level),
                         or None if calculation fails.
    """
    if df is None or df.empty or len(df) < lookback:
        logging.warning("calculate_poc: Insufficient data for POC calculation.")
        return None
    if not all(col in df.columns for col in ['High', 'Low', price_col, volume_col]):
         logging.warning(f"calculate_poc: Missing required columns (High, Low, {price_col}, {volume_col}).")
         return None

    poc_value = None
    try:
        # Use pandas_ta vwap as a POC proxy (most volume-weighted price) over the lookback period
        # Ensure column names match what pandas_ta expects (lowercase) or pass them
        df_copy = df.copy()
        df_copy.columns = [col.lower() for col in df_copy.columns] # Convert columns to lowercase for ta

        # Calculate VWAP for the lookback period
        vwap_series = ta.vwap(high=df_copy['high'], low=df_copy['low'], close=df_copy['close'], volume=df_copy['volume'], length=lookback)

        if vwap_series is not None and not vwap_series.empty:
            # Simple POC proxy: Use the last calculated VWAP value as the area of high recent activity
            poc_value = vwap_series.iloc[-1]
            if pd.isna(poc_value):
                 logging.warning(f"calculate_poc: Calculated VWAP (POC proxy) is NaN for the last period.")
                 poc_value = None # Ensure None if NaN
            else:
                 logging.debug(f"Calculated POC (VWAP proxy) over last {lookback} periods: {poc_value:.4f}")
        else:
             logging.warning(f"calculate_poc: pandas_ta.vwap returned None or empty series.")

    except Exception as e:
        logging.error(f"calculate_poc: Error calculating POC (VWAP proxy): {e}", exc_info=True)
        poc_value = None

    return poc_value

# Add other functions like calculate_vpvr etc. here later 