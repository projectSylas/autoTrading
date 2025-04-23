# src/backtesting/backtest_runner_vbt.py

import vectorbt as vbt
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import logging
# import numba # Removed numba import
import itertools # Import itertools for parameter combinations
import time # To measure optimization time
import seaborn as sns # For heatmap
import matplotlib.pyplot as plt # For heatmap

# Import utility functions - Ensure these work with pandas Series/DataFrames directly
# from src.utils import strategy_utils # Keep original import if other utils are used
# from src.utils.strategy_utils import BreakoutPullback # Import the new indicator
from src.config.settings import settings # For potential parameters
from src.utils.logging_config import setup_logging, APP_LOGGER_NAME
# === UPDATED: Import the new volume-based function and remove old divergence import ===
# from src.utils.strategy_utils import detect_rsi_divergence_v2 # Removed
# from src.utils.strategy_utils import find_breakout_pullback_nb # Removed original Numba func import
# from src.utils.strategy_utils import find_volume_breakout_pullback_nb # Removed volume Numba func import
# ================================================
# === ADDED: Import AI Signal Generator ===
from src.signals.ai_signal_generator import generate_ai_entry_signals, generate_ai_exit_signals
# =======================================

# --- Setup Logging ---
setup_logging()
logger = logging.getLogger(APP_LOGGER_NAME + ".vbt_runner")
# --- End Setup Logging ---

# --- Numba function DEFINED HERE (Re-enabled JIT) ---
# @numba.njit('i1[:](f8[:], f8[:], f8[:], i8, i8, f8)', cache=True) # DEBUG: Temporarily disable JIT
# REMOVED: Numba function definition for find_breakout_pullback_nb
# ... (entire function removed) ...

# === UPDATED Function Signature: Final Correction ===
def run_vectorbt_backtest(
    # --- Required Context & Strategy Parameters (from MCP) ---
    symbol: str,
    start_date: str,
    end_date: str,
    # --- Removed parameters no longer used by AI signal --- 
    # pullback_threshold: float,
    # breakout_lookback: int, 
    # pullback_lookback: int,
    # volume_lookback: int | None = None, 
    # volume_multiplier: float | None = None,
    # --- Existing or potentially relevant parameters --- 
    rsi_period: int,
    sma_short_period: int,
    sma_long_period: int,
    adx_threshold: float | None = None,
    # --- AI Model Parameters --- 
    ai_model_path: str = "models/buy_signal_gru.pth",
    ai_threshold: float = 0.8,
    ai_window_size: int = 48,
    ai_device: str = "cpu",
    # --- Execution Control & Defaults ---
    interval: str = '1h',
    initial_cash: float = 10000.0,
    commission: float = 0.001,
    use_trend_filter: bool = False, # Potentially keep for filtering AI signals
    use_atr_exit: bool = False,
    atr_window: int = 14,
    atr_sl_multiplier: float = 1.5,
    atr_tp_multiplier: float = 2.5,
    sl_stop: float = settings.CHALLENGE_SL_RATIO, # Default fixed, used if use_atr_exit=False
    tp_stop: float = settings.CHALLENGE_TP_RATIO, # Default fixed, used if use_atr_exit=False
    plot: bool = True,
    return_stats_only: bool = False, 
    **kwargs # Catch potentially unused params from MCP
):
    """
    Runs a vectorbt backtest for the challenge strategy using AI-generated signals.
    Accepts strategy parameters explicitly, suitable for use with MCP runner.
    Includes extensive debug logging.
    """
    run_signature = f"{symbol}_{start_date}_{end_date}_{interval}"
    logger.info(f"[{run_signature}] Starting vectorbt backtest with AI Signals...")

    # --- Log unused parameters caught by kwargs ---
    if kwargs:
        # Filter out standard vbt params that might be passed
        ignored_params = {k: v for k, v in kwargs.items() if k not in ['pullback_threshold', 'breakout_lookback', 'pullback_lookback', 'volume_lookback', 'volume_multiplier']}
        if ignored_params:
             logger.warning(f"[{run_signature}] Ignoring unused parameters passed via kwargs: {ignored_params}")

    # --- 1. Fetch and Validate Data ---
    try:
        logger.info(f"[{run_signature}] Fetching data using yfinance...")
        price_data = yf.download(symbol, start=start_date, end=end_date, interval=interval)
        if price_data.empty:
            error_msg = f"[{run_signature}] No data fetched from yfinance."
            logger.error(error_msg)
            return {'error': error_msg} # Return error dict

        # --- Flatten MultiIndex Columns if they exist ---
        if isinstance(price_data.columns, pd.MultiIndex):
            price_data.columns = price_data.columns.droplevel(1)
            logger.info(f"[{run_signature}] Flattened MultiIndex columns.")

        # --- Ensure lowercase columns and check required columns ---
        price_data.columns = price_data.columns.str.lower()
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in price_data.columns]
        if missing_cols:
            error_msg = f"[{run_signature}] Missing required columns: {missing_cols}. Cannot proceed."
            logger.error(error_msg)
            return {'error': error_msg} # Return error dict
        logger.info(f"[{run_signature}] Data fetched successfully. Shape: {price_data.shape}. Columns: {price_data.columns.tolist()}")

        # --- Log Head/Tail/Info ---
        logger.debug(f"[{run_signature}] Data head:\n{price_data.head().to_string()}")
        logger.debug(f"[{run_signature}] Data tail:\n{price_data.tail().to_string()}")
        # logger.debug(f"[{run_signature}] Data info:\n{price_data.info()}") # Can be verbose

        # --- NaN Check and Handling ---
        nan_counts = price_data[required_cols].isnull().sum()
        if nan_counts.sum() > 0:
            logger.warning(f"[{run_signature}] NaN values found in OHLCV data:\n{nan_counts[nan_counts > 0].to_string()}. Attempting ffill/bfill.")
            price_data.ffill(inplace=True)
            price_data.bfill(inplace=True) # Fill remaining NaNs at the beginning
            nan_counts_after_fill = price_data[required_cols].isnull().sum()
            if nan_counts_after_fill.sum() > 0:
                error_msg = f"[{run_signature}] NaN values still present after fill:\n{nan_counts_after_fill[nan_counts_after_fill > 0].to_string()}. Cannot proceed."
                logger.error(error_msg)
                return {'error': error_msg} # Return error dict
            else:
                logger.info(f"[{run_signature}] NaN values successfully handled with ffill/bfill.")
        else:
            logger.info(f"[{run_signature}] No NaN values found in OHLCV data.")

        # --- Price Data Validation (Close <= 0) ---
        invalid_close_count = (price_data['close'] <= 0).sum()
        if invalid_close_count > 0:
            error_msg = f"[{run_signature}] Invalid price data detected ({invalid_close_count} rows with Close <= 0). Aborting backtest."
            logger.error(error_msg)
            # logger.debug(f"[{run_signature}] Rows with invalid close price:\n{price_data[price_data['close'] <= 0].head().to_string()}")
            return {'error': error_msg} # Return error dict
        logger.info(f"[{run_signature}] Price data validation (Close > 0) passed.")

    except Exception as e:
        error_msg = f"[{run_signature}] Error fetching or validating data: {e}."
        logger.error(error_msg, exc_info=True)
        return {'error': error_msg} # Return error dict

    # --- 2. Calculate Indicators (Keep relevant ones) ---
    logger.info(f"[{run_signature}] Calculating indicators (SMA, RSI, ADX if needed)...")
    try:
        # --- SMA Calculation ---
        logger.debug(f"[{run_signature}] Calculating SMA (Long: {sma_long_period}, Short: {sma_short_period})...")
        sma_long = vbt.MA.run(price_data['close'], sma_long_period, short_name='sma_long').ma
        sma_short = vbt.MA.run(price_data['close'], sma_short_period, short_name='sma_short').ma
        logger.debug(f"[{run_signature}] SMA Long - Head:\n{sma_long.head().to_string()}")
        logger.debug(f"[{run_signature}] SMA Short - Head:\n{sma_short.head().to_string()}")
        if sma_long.isnull().all() or sma_short.isnull().all():
            logger.warning(f"[{run_signature}] SMA calculation resulted in all NaNs (Long NaNs: {sma_long.isnull().sum()}, Short NaNs: {sma_short.isnull().sum()}). Check periods and data length.")
        else:
            logger.debug(f"[{run_signature}] SMA NaNs - Long: {sma_long.isnull().sum()}, Short: {sma_short.isnull().sum()}")

        # --- RSI Calculation ---
        logger.debug(f"[{run_signature}] Calculating RSI (Period: {rsi_period})...")
        rsi = vbt.RSI.run(price_data['close'], rsi_period, short_name='rsi').rsi
        logger.debug(f"[{run_signature}] RSI - Head:\n{rsi.head().to_string()}")
        if rsi.isnull().all():
            logger.warning(f"[{run_signature}] RSI calculation resulted in all NaNs (NaNs: {rsi.isnull().sum()}). Check period and data length.")
        else:
             logger.debug(f"[{run_signature}] RSI NaNs: {rsi.isnull().sum()}")

        # --- ADX Calculation (Conditional) ---
        adx = None
        adx_value = None # Store the final ADX series
        if adx_threshold is not None:
            try:
                logger.debug(f"[{run_signature}] Calculating ADX (Window: 14)...")
                adx = vbt.indicators.ADX.run(price_data['high'], price_data['low'], price_data['close'], window=14)
                if adx is None or adx.dxi is None:
                    logger.warning(f"[{run_signature}] ADX calculation failed or returned unexpected result. Skipping ADX filter.")
                    adx = None # Ensure adx is None if calculation failed
                else:
                    adx_value = adx.dxi # Store the dxi series
                    logger.debug(f"[{run_signature}] ADX calculated. DX NaN count: {adx_value.isnull().sum()}. Head:\n{adx_value.head().to_string()}")
                    # Handle potential NaNs in ADX.DXI
                    if adx_value.isnull().all():
                        logger.warning(f"[{run_signature}] ADX.DXI calculation resulted in all NaNs. Skipping ADX filter.")
                        adx = None
                        adx_value = None
                    elif adx_value.isnull().any():
                        logger.warning(f"[{run_signature}] ADX.DXI contains {adx_value.isnull().sum()} NaNs. Forward/Backward filling...")
                        adx_value.ffill(inplace=True)
                        adx_value.bfill(inplace=True)
                        if adx_value.isnull().any():
                            logger.error(f"[{run_signature}] NaNs still present in ADX.DXI after ffill/bfill. Skipping ADX filter.")
                            adx = None
                            adx_value = None
                        else:
                            logger.info(f"[{run_signature}] ADX.DXI NaNs handled.")
            except AttributeError:
                 logger.error(f"[{run_signature}] Error calculating ADX: `vbt.indicators.ADX.run` failed. Is ADX available? Skipping ADX filter.")
                 adx = None
                 adx_value = None
            except Exception as e:
                logger.error(f"[{run_signature}] Error calculating ADX: {e}. Skipping ADX filter.", exc_info=True)
                adx = None
                adx_value = None

        # --- Generate AI Entry Signals ---
        logger.info(f"[{run_signature}] Generating AI Entry Signals (model={ai_model_path}, threshold={ai_threshold}, window={ai_window_size})...")
        # Rename columns temporarily if needed by the signal generator
        # The generator expects 'Open', 'High', 'Low', 'Close', 'Volume'
        temp_price_data = price_data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
        ai_entry_signals = generate_ai_entry_signals(
            price_df=temp_price_data,
            model_path=ai_model_path,
            threshold=ai_threshold,
            window_size=ai_window_size,
            device=ai_device
        )
        logger.info(f"[{run_signature}] AI Entry Signal generation completed.")
        # Log the distribution of generated signals
        logger.info(f"[{run_signature}] AI Entry Signal Distribution:\n{ai_entry_signals.value_counts().to_string()}")
        
        # --- Generate AI Exit Signals (Placeholder) ---
        # Currently returns all zeros, can be integrated later
        ai_exit_signals = generate_ai_exit_signals(price_df=temp_price_data, device=ai_device)

        # --- ATR Calculation (for exits) ---
        atr = None
        if use_atr_exit:
            logger.debug(f"[{run_signature}] Calculating ATR (Window: {atr_window})...")
            atr = vbt.ATR.run(price_data['high'], price_data['low'], price_data['close'], atr_window).atr
            if atr.isnull().all():
                logger.warning(f"[{run_signature}] ATR calculation resulted in all NaNs (NaNs: {atr.isnull().sum()}). Disabling ATR exit.")
                use_atr_exit = False
                atr = None
            elif atr.isnull().any():
                logger.warning(f"[{run_signature}] ATR contains {atr.isnull().sum()} NaNs. Forward/Backward filling...")
                atr.ffill(inplace=True)
                atr.bfill(inplace=True)
                if atr.isnull().any():
                    logger.error(f"[{run_signature}] NaNs still present in ATR after ffill/bfill. Disabling ATR exit.")
                    use_atr_exit = False
                    atr = None
                else:
                    logger.info(f"[{run_signature}] ATR NaNs handled.")
                    logger.debug(f"[{run_signature}] ATR - Head:\n{atr.head().to_string()}")
            else:
                 logger.debug(f"[{run_signature}] ATR NaNs: {atr.isnull().sum()}")
                 logger.debug(f"[{run_signature}] ATR - Head:\n{atr.head().to_string()}")


    except FileNotFoundError as e:
        error_msg = f"[{run_signature}] AI Model file not found: {e}. Ensure the model exists at the specified path."
        logger.error(error_msg, exc_info=True)
        return {'error': error_msg}
    except Exception as e:
        error_msg = f"[{run_signature}] Error calculating indicators or generating AI signals: {e}."
        logger.error(error_msg, exc_info=True)
        return {'error': error_msg}

    # --- 3. Define Entry and Exit Signals (Combining AI and optional filters) ---
    logger.info(f"[{run_signature}] Defining final entry and exit signals...")

    # --- Base Entry Signal (from AI) ---
    entries = ai_entry_signals == 1
    logger.debug(f"[{run_signature}] Initial AI Entries: {entries.sum()}")

    # --- Apply Optional Trend Filter (SMA Cross) ---
    if use_trend_filter:
        trend_signal = sma_short > sma_long
        entries = entries & trend_signal
        logger.info(f"[{run_signature}] Applied Trend Filter (SMA Short > Long). Entries after filter: {entries.sum()}")
        logger.debug(f"[{run_signature}] Trend Signal (SMA Short > Long): {trend_signal.sum()} periods")

    # --- Apply Optional ADX Filter ---
    if adx_threshold is not None and adx_value is not None:
        adx_filter_signal = adx_value > adx_threshold
        entries = entries & adx_filter_signal
        logger.info(f"[{run_signature}] Applied ADX Filter (ADX > {adx_threshold}). Entries after filter: {entries.sum()}")
        logger.debug(f"[{run_signature}] ADX Filter Signal (> {adx_threshold}): {adx_filter_signal.sum()} periods")

    # --- Define Exits ---
    # Basic Exit: Opposite signal (AI exit - currently placeholder, or rule-based like SMA cross under)
    # simple_exits = sma_short < sma_long # Example rule-based exit
    simple_exits = ai_exit_signals == 1 # Use AI exit signal (currently all False)
    logger.debug(f"[{run_signature}] Initial Exits (AI based or simple rule): {simple_exits.sum()}")

    # --- SL/TP Calculation (Use ATR or fixed stops) ---
    sl = np.nan
    tp = np.nan
    sl_points = np.nan
    tp_points = np.nan

    if use_atr_exit and atr is not None:
        logger.info(f"[{run_signature}] Calculating ATR-based SL/TP (SL Mult: {atr_sl_multiplier}, TP Mult: {atr_tp_multiplier})...")
        # SL/TP points based on entry price and ATR at entry
        sl_points = atr * atr_sl_multiplier
        tp_points = atr * atr_tp_multiplier

        # Check for NaN/inf values in calculated SL/TP points
        sl_nan_count = sl_points.isnull().sum()
        tp_nan_count = tp_points.isnull().sum()
        sl_inf_count = np.isinf(sl_points).sum()
        tp_inf_count = np.isinf(tp_points).sum()

        if sl_nan_count > 0 or tp_nan_count > 0 or sl_inf_count > 0 or tp_inf_count > 0:
            logger.warning(f"[{run_signature}] NaN/inf values detected in calculated ATR SL/TP points.")
            logger.warning(f"[{run_signature}] SL NaNs: {sl_nan_count}, TP NaNs: {tp_nan_count}")
            logger.warning(f"[{run_signature}] SL Infs: {sl_inf_count}, TP Infs: {tp_inf_count}")
            # Attempt to fill - might not be the best strategy, depends on cause
            sl_points.ffill(inplace=True)
            sl_points.bfill(inplace=True)
            tp_points.ffill(inplace=True)
            tp_points.bfill(inplace=True)
            # Replace any remaining inf/-inf with NaN, then fill again
            sl_points.replace([np.inf, -np.inf], np.nan, inplace=True)
            tp_points.replace([np.inf, -np.inf], np.nan, inplace=True)
            sl_points.ffill(inplace=True)
            sl_points.bfill(inplace=True)
            if sl_points.isnull().any() or tp_points.isnull().any():
                 logger.error(f"[{run_signature}] Failed to handle all NaN/inf in ATR SL/TP points. Falling back to fixed SL/TP.")
                 use_atr_exit = False # Fallback
                 sl = sl_stop
                 tp = tp_stop
            else:
                 logger.info(f"[{run_signature}] NaN/inf values in ATR SL/TP points handled.")
        else:
            logger.info(f"[{run_signature}] ATR SL/TP points calculated successfully.")
        
        # Vectorbt expects SL/TP as percentages or fixed points relative to entry
        # Here we pass the dynamically calculated points per bar
        sl = sl_points
        tp = tp_points

    else:
        logger.info(f"[{run_signature}] Using fixed SL/TP ({sl_stop*100:.2f}% / {tp_stop*100:.2f}%)...")
        sl = sl_stop # Fixed percentage stop loss
        tp = tp_stop # Fixed percentage take profit

    # --- Final Signal Checks ---
    logger.info(f"[{run_signature}] Final Entry Signals Count: {entries.sum()}")
    logger.info(f"[{run_signature}] Final Exit Signals (Simple Rule/AI) Count: {simple_exits.sum()}")
    if entries.sum() == 0:
        logger.warning(f"[{run_signature}] No entry signals generated after applying all filters. Backtest will likely show no trades.")

    # --- 4. Run Backtest ---
    logger.info(f"[{run_signature}] Running vectorbt Portfolio simulation...")
    try:
        # Use from_signals for flexibility
        portfolio = vbt.Portfolio.from_signals(
            close=price_data['close'],
            entries=entries,
            exits=simple_exits, # Use simple exits or placeholder AI exits
            sl_stop=sl,         # Can be Series (ATR) or float (fixed)
            tp_stop=tp,         # Can be Series (ATR) or float (fixed)
            init_cash=initial_cash,
            fees=commission,
            freq=interval       # Pass interval for correct time-based calculations
        )
        logger.info(f"[{run_signature}] Portfolio simulation completed.")

        # --- 5. Get Stats --- 
        # Removed freq=interval based on previous findings
        logger.info(f"[{run_signature}] Calculating portfolio stats...")
        stats = portfolio.stats()
        # Convert stats Series to dictionary for JSON serialization
        stats_dict = stats.to_dict()

        # Add context info to stats
        stats_dict['symbol'] = symbol
        stats_dict['start_date'] = start_date
        stats_dict['end_date'] = end_date
        stats_dict['interval'] = interval
        stats_dict['strategy_params'] = { # Log relevant parameters used
            'rsi_period': rsi_period,
            'sma_short_period': sma_short_period,
            'sma_long_period': sma_long_period,
            'adx_threshold': adx_threshold,
            'ai_model_path': ai_model_path,
            'ai_threshold': ai_threshold,
            'ai_window_size': ai_window_size,
            'use_trend_filter': use_trend_filter,
            'use_atr_exit': use_atr_exit,
            'atr_window': atr_window if use_atr_exit else None,
            'atr_sl_multiplier': atr_sl_multiplier if use_atr_exit else None,
            'atr_tp_multiplier': atr_tp_multiplier if use_atr_exit else None,
            'sl_stop': sl_stop if not use_atr_exit else None,
            'tp_stop': tp_stop if not use_atr_exit else None,
        }
        logger.info(f"[{run_signature}] Portfolio stats calculated successfully.")
        # logger.debug(f"[{run_signature}] Full Stats:\n{stats}") # Can be very verbose
        logger.info(f"[{run_signature}] Key Stats: Return={stats.get('Total Return [%]', 'N/A'):.2f}%, Win Rate={stats.get('Win Rate [%]', 'N/A'):.2f}%, Sharpe={stats.get('Sharpe Ratio', 'N/A'):.2f}")

        # --- 6. Plotting (Optional) ---
        if plot:
            try:
                logger.info(f"[{run_signature}] Generating plot...")
                # Use portfolio.plot() - might require interactive backend or saving to file
                # fig = portfolio.plot()
                # fig.show() # Requires interactive backend
                # OR save to file:
                # plot_filename = f"results/{symbol}_{start_date}_{end_date}_plot.png"
                # portfolio.plot().write_image(plot_filename)
                # logger.info(f"[{run_signature}] Plot saved to {plot_filename}")
                pass # Placeholder for plotting logic if needed
            except Exception as e:
                logger.warning(f"[{run_signature}] Error during plotting: {e}")

        # --- 7. Return Results ---
        if return_stats_only:
            logger.info(f"[{run_signature}] Returning stats dictionary.")
            return stats_dict
        else:
            logger.info(f"[{run_signature}] Returning portfolio object and stats dictionary.")
            return portfolio, stats_dict

    except vbt.base.errors.DataMismatchError as e:
        error_msg = f"[{run_signature}] Data mismatch error during Portfolio simulation: {e}. Check alignment of signals and price data."
        logger.error(error_msg, exc_info=True)
        # Log shapes and indices for debugging
        logger.error(f"Price Close shape: {price_data['close'].shape}, index head: {price_data['close'].index[:5]}, index tail: {price_data['close'].index[-5:]}")
        logger.error(f"Entries shape: {entries.shape}, index head: {entries.index[:5]}, index tail: {entries.index[-5:]}")
        logger.error(f"Exits shape: {simple_exits.shape}, index head: {simple_exits.index[:5]}, index tail: {simple_exits.index[-5:]}")
        if isinstance(sl, pd.Series):
            logger.error(f"SL shape: {sl.shape}, index head: {sl.index[:5]}, index tail: {sl.index[-5:]}")
        if isinstance(tp, pd.Series):
            logger.error(f"TP shape: {tp.shape}, index head: {tp.index[:5]}, index tail: {tp.index[-5:]}")
        return {'error': error_msg}

    except Exception as e:
        error_msg = f"[{run_signature}] An unexpected error occurred during backtest execution: {e}."
        logger.error(error_msg, exc_info=True)
        return {'error': error_msg} # Return error dict

# --- Helper Function for Stats Extraction (Optional) ---
def extract_stats(portfolio, symbol, period_label, params):
    """Extracts key performance metrics from a vectorbt portfolio object."""
    if portfolio is None:
        return {
            'Symbol': symbol,
            'Period': period_label,
            'Pullback Threshold': params['pb'],
            'SL Stop': params['sl'],
            'TP Stop': params['tp'],
            'Total Return [%]': 0.0,
            'Max Drawdown [%]': None,
            'Total Trades': 0,
            'Win Rate [%]': None,
            'Sharpe Ratio': None,
            'Error': 'Portfolio object is None'
        }
    try:
        stats = portfolio.stats()
        # Handle cases where metrics might be NaN or Inf (e.g., no trades)
        total_return = stats.get('Total Return [%]', 0.0)
        max_drawdown = stats.get('Max Drawdown [%]')
        total_trades = stats.get('Total Trades', 0)
        win_rate = stats.get('Win Rate [%]')
        sharpe_ratio = stats.get('Sharpe Ratio')

        return {
            'Symbol': symbol,
            'Period': period_label,
            'Pullback Threshold': params['pb'],
            'SL Stop': params['sl'],
            'TP Stop': params['tp'],
            'Total Return [%]': total_return if pd.notna(total_return) else 0.0,
            'Max Drawdown [%]': max_drawdown if pd.notna(max_drawdown) else None,
            'Total Trades': total_trades,
            'Win Rate [%]': win_rate if pd.notna(win_rate) else None,
            'Sharpe Ratio': sharpe_ratio if pd.notna(sharpe_ratio) and np.isfinite(sharpe_ratio) else None
        }
    except Exception as e:
        logger.error(f"Error extracting stats for {symbol} ({period_label}): {e}")
        return {
            'Symbol': symbol,
            'Period': period_label,
            'Pullback Threshold': params['pb'],
            'SL Stop': params['sl'],
            'TP Stop': params['tp'],
            'Error': str(e)
        }


# --- NEW: Function for ATR Parameter Optimization ---
def optimize_atr_params(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str,
    initial_cash: float,
    commission: float,
    atr_windows: list[int],
    atr_sl_multipliers: list[float],
    atr_tp_multipliers: list[float],
    # Pass other fixed strategy parameters needed by run_vectorbt_backtest
    **kwargs
):
    """Runs backtests for combinations of ATR parameters and finds the best based on Sharpe Ratio."""
    logger.info(f"Starting ATR parameter optimization for {symbol} ({start_date} to {end_date})...")
    optimization_start_time = time.time()

    param_combinations = list(itertools.product(atr_windows, atr_sl_multipliers, atr_tp_multipliers))
    logger.info(f"Total parameter combinations to test: {len(param_combinations)}")

    results = []

    for i, (window, sl_mult, tp_mult) in enumerate(param_combinations):
        logger.info(f"Running test {i+1}/{len(param_combinations)}: window={window}, sl_mult={sl_mult}, tp_mult={tp_mult}")
        try:
            # Call the updated function which now has enhanced logging
            stats = run_vectorbt_backtest(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                initial_cash=initial_cash,
                commission=commission,
                # --- Key settings for this optimization ---
                use_trend_filter=False, # Keep trend filter off
                use_atr_exit=True,      # Enable ATR exits
                atr_window=window,
                atr_sl_multiplier=sl_mult,
                atr_tp_multiplier=tp_mult,
                # --- Performance settings ---
                plot=False,             # Disable plotting for speed
                return_stats_only=True, # Return only stats dict
                # --- Pass other fixed params ---
                **kwargs
            )

            if stats is not None:
                # Add the tested parameters to the results dictionary
                stats['atr_window'] = window
                stats['atr_sl_multiplier'] = sl_mult
                stats['atr_tp_multiplier'] = tp_mult
                # Ensure required stats keys exist, provide defaults if missing
                required_keys = ['Total Return [%]', 'Max Drawdown [%]', 'Total Trades', 'Win Rate [%]', 'Sharpe Ratio']
                for key in required_keys:
                     if key not in stats: stats[key] = None # Or np.nan
                # Handle potential inf Sharpe Ratios (e.g., no trades or zero std dev)
                if stats.get('Sharpe Ratio') is not None and not np.isfinite(stats['Sharpe Ratio']):
                    stats['Sharpe Ratio'] = np.nan # Treat inf/-inf as NaN for sorting
                results.append(stats)
            else:
                logger.warning(f"Backtest failed for combination: window={window}, sl_mult={sl_mult}, tp_mult={tp_mult}. Skipping.")

        except Exception as e:
            logger.error(f"Error during optimization run for window={window}, sl={sl_mult}, tp={tp_mult}: {e}", exc_info=True)

    optimization_end_time = time.time()
    logger.info(f"ATR parameter optimization finished in {optimization_end_time - optimization_start_time:.2f} seconds.")

    if not results:
        logger.error("No results collected from ATR optimization run.")
        return None

    # --- Process and Display Results ---
    results_df = pd.DataFrame(results)

    # Define columns to display, including parameters
    cols_to_display = [
        'atr_window', 'atr_sl_multiplier', 'atr_tp_multiplier',
        'Total Return [%]', 'Max Drawdown [%]', 'Total Trades',
        'Win Rate [%]', 'Sharpe Ratio'
    ]
    # Ensure columns exist before selecting/sorting
    results_df = results_df.reindex(columns=cols_to_display)

    # Sort by Sharpe Ratio (descending, NaNs last)
    results_df = results_df.sort_values(by='Sharpe Ratio', ascending=False, na_position='last')

    print("\n--- ATR Parameter Optimization Results (Top 5 by Sharpe Ratio) --- \n")
    # Format numbers for display
    pd.options.display.float_format = '{:,.4f}'.format
    print(results_df.head(5).to_string(index=False))
    pd.reset_option('display.float_format') # Reset formatting

    # Return the full results DataFrame
    return results_df


# --- Main Execution Block ---
# (Keep __main__ block as is or modify for specific testing)
if __name__ == "__main__":
    # Example: Run a single test with the enhanced logging
    logger.info("=== Starting Manual Test Run with Enhanced Logging ===")
    test_params = {
        'symbol': 'BTC-USD',
        'start_date': '2023-07-01', # Use a period suspected to fail
        'end_date': '2023-12-31',
        'interval': '1h',
        'rsi_period': 14,
        'sma_short_period': 7,
        'sma_long_period': 50,
        'adx_threshold': 0.5,
        'ai_model_path': "models/buy_signal_gru.pth",
        'ai_threshold': 0.8,
        'ai_window_size': 48,
        'ai_device': "cpu",
        'initial_cash': 10000.0,
        'commission': 0.001,
        'use_trend_filter': False,
        'use_atr_exit': True,
        'atr_window': 10,
        'atr_sl_multiplier': 2.0,
        'atr_tp_multiplier': 2.5,
        'plot': False, # Disable plot for manual run
        'return_stats_only': True
    }
    stats_result = run_vectorbt_backtest(**test_params)

    if stats_result:
        print("\n--- Manual Test Run Successful ---")
        print(pd.Series(stats_result))
    else:
        print("\n--- Manual Test Run Failed (Check Logs) ---")

    logger.info("=== Main Execution Block Finished ===")