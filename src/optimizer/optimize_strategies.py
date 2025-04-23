"""
Strategy Optimizer using vectorbt backtester.

Iterates through parameter combinations, runs backtests, saves results to CSV,
and prints top performers based on Sharpe Ratio and Win Rate.
"""

import itertools
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import os
import logging
import numpy as np
import time

# --- Project Specific Imports ---
try:
    from src.backtesting.backtest_runner_vbt import run_vectorbt_backtest
    from src.config.settings import settings
    from src.utils.logging_config import setup_logging, APP_LOGGER_NAME
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Ensure script is run from project root (e.g., 'poetry run python src/optimizer/optimize_strategies.py')")
    import sys
    sys.exit(1)

# === 로깅 설정 강제 초기화 ===
# 스크립트 시작 시 기존 핸들러를 모두 제거하고 새로 설정
logger = setup_logging(force_reset=True)
# logger = logging.getLogger(APP_LOGGER_NAME + ".optimizer") # setup_logging이 로거를 반환하므로 이 줄은 필요 없어짐
# ============================

# --- Configuration ---
OUTPUT_DIR = "results" # Output directory
OUTPUT_FILENAME = "experiment_summary.csv" # Summary CSV file
# Individual results per run can be saved if needed, but focusing on summary for now
# OUTPUT_DIR_INDIVIDUAL = "mcp/results" # Example if saving individual JSONs

# --- Experiment Context --- 
SYMBOL = "BTC-USD"
TIMEFRAME = "1h"
START_DATE = "2023-10-01"
END_DATE = "2023-12-30"
INITIAL_CASH = 10000.0
COMMISSION = 0.001

# --- Fixed Parameters for this Optimization Run --- 
FIXED_PARAMS = {
    'breakout_lookback': 20,
    'pullback_lookback': 5,
    'volume_lookback': 10,
    'volume_multiplier': 2.0,
    'atr_window': 10,
    # These might be needed by the function signature even if not actively used
    'sma_short_period': 7, 
    'sma_long_period': 50
}

# --- Parameters to Optimize (Grid) --- 
param_grid = {
    'pullback_threshold': [0.03, 0.05, 0.07],
    'rsi_period': [7, 10, 14],
    'atr_sl_multiplier': [1.8, 2.0, 2.2],
    'atr_tp_multiplier': [3.5, 4.0, 4.5]
}
# Total combinations = 3 * 3 * 3 * 3 = 81

# --- Metrics to Collect --- 
# Include varying params + desired performance metrics
METRICS_TO_COLLECT = list(param_grid.keys()) + [
    'Total Trades', 'Total Return [%]', 'Win Rate [%]', 'Sharpe Ratio', 'Max Drawdown [%]'
]

def run_optimization():
    """Runs the parameter optimization process."""
    logger.info("=== Starting Strategy Optimization (breakout-volume-surge) ===")
    optimization_start_time = time.time()

    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(itertools.product(*param_values))
    total_combinations = len(param_combinations)
    logger.info(f"Total parameter combinations to test: {total_combinations}")

    results_list = []

    for i, params in enumerate(tqdm(param_combinations, desc="Running Backtests")):
        current_params = dict(zip(param_names, params))
        run_id = i + 1

        try:
            # --- Prepare arguments for run_vectorbt_backtest --- 
            # Start with fixed params, then add current combination
            backtest_args = {
                **FIXED_PARAMS,
                **current_params,
                # --- Add execution context and run controls ---
                'symbol': SYMBOL,
                'start_date': START_DATE,
                'end_date': END_DATE,
                'interval': TIMEFRAME,
                'initial_cash': INITIAL_CASH,
                'commission': COMMISSION,
                'use_trend_filter': False, # Explicitly off for this strategy focus
                'use_atr_exit': True, # Explicitly using ATR SL/TP
                'plot': False,
                'return_stats_only': True
            }
            # logger.debug(f"Run {run_id}: Args: {backtest_args}")

            # --- Run the backtest --- 
            stats = run_vectorbt_backtest(**backtest_args)

            if stats is None:
                logger.warning(f"Run {run_id} failed (returned None) for params: {current_params}")
                failure_entry = {**current_params, 'Error': 'Backtest Failed'}
                for metric in METRICS_TO_COLLECT: failure_entry.setdefault(metric, np.nan)
                results_list.append(failure_entry)
                continue

            # --- Extract results --- 
            result_entry = {**current_params} # Start with the parameters
            for metric in METRICS_TO_COLLECT:
                if metric not in current_params:
                    value = stats.get(metric)
                    if isinstance(value, (np.integer, np.int_)): value = int(value)
                    elif isinstance(value, (np.floating, np.float_)):
                        if pd.isna(value): value = np.nan
                        elif not np.isfinite(value): value = np.nan # Treat Inf as NaN for sorting
                        else: value = float(value)
                    elif pd.isna(value): value = np.nan
                    result_entry[metric] = value
            
            # Ensure Sharpe is NaN if non-finite
            if 'Sharpe Ratio' in result_entry and not np.isfinite(result_entry['Sharpe Ratio']):
                result_entry['Sharpe Ratio'] = np.nan
                
            results_list.append(result_entry)

        except Exception as e:
            logger.error(f"Run {run_id} failed with exception for params: {current_params}. Error: {e}", exc_info=False) # Hide traceback for cleaner logs during optim
            failure_entry = {**current_params, 'Error': str(e)}
            for metric in METRICS_TO_COLLECT: failure_entry.setdefault(metric, np.nan)
            results_list.append(failure_entry)

    optimization_end_time = time.time()
    logger.info(f"Optimization loop finished in {optimization_end_time - optimization_start_time:.2f} seconds.")

    if not results_list:
        logger.error("No results were collected during the optimization.")
        return

    # --- Process and Save Results --- 
    results_df = pd.DataFrame(results_list)
    for col in METRICS_TO_COLLECT:
        if col not in results_df.columns: results_df[col] = np.nan
    # Ensure correct column order
    results_df = results_df[METRICS_TO_COLLECT + [col for col in results_df.columns if col not in METRICS_TO_COLLECT]]

    output_filepath = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        results_df.to_csv(output_filepath, index=False, encoding='utf-8', float_format='%.4f')
        logger.info(f"Optimization results summary saved to: {output_filepath}")
    except Exception as e:
        logger.error(f"Error saving results to CSV: {e}")

    # --- Display Top Results --- 
    # Set display options
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    pd.set_option('display.float_format', '{:.4f}'.format)

    # Top 5 by Sharpe Ratio
    top_sharpe = results_df.sort_values(by='Sharpe Ratio', ascending=False, na_position='last').head(5)
    print("\n=== Top 5 Combinations by Sharpe Ratio ===")
    print(top_sharpe[METRICS_TO_COLLECT].to_string(index=False))

    # Top 3 by Win Rate
    top_winrate = results_df.sort_values(by='Win Rate [%]', ascending=False, na_position='last').head(3)
    print("\n=== Top 3 Combinations by Win Rate [%] ===")
    print(top_winrate[METRICS_TO_COLLECT].to_string(index=False))

    # Reset display options
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.float_format')

    logger.info("=== Strategy Optimization Finished ===")

if __name__ == "__main__":
    run_optimization() 