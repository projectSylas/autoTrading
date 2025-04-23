"""
Performs robustness testing for a trading strategy by running backtests
across different assets and time periods using MCP configuration files.

Iterates through predefined experiment combinations:
1. Generates 'mcp/experiment_context.json' for each combination.
2. Executes 'mcp/run_mcp_experiment.py' to run the backtest.
3. Collects results from the generated JSON output files.
4. Summarizes and prints the best performing combinations based on
   Sharpe Ratio and Win Rate.
"""

import json
import os
import subprocess
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm

# --- Project Specific Imports ---
try:
    from src.utils.logging_config import setup_logging, APP_LOGGER_NAME
    # Assuming settings might provide defaults if needed, though context file is primary
    from src.config.settings import settings 
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Ensure script is run from project root (e.g., 'poetry run python src/experiments/run_robustness_test.py')")
    import sys
    sys.exit(1)

# --- Setup Logging ---
setup_logging()
logger = logging.getLogger(APP_LOGGER_NAME + ".robustness_test")
# --- End Setup Logging ---

# --- Configuration ---
CONTEXT_FILE_PATH = "mcp/experiment_context.json"
RESULTS_DIR = "mcp/results"
MCP_RUNNER_SCRIPT = "mcp/run_mcp_experiment.py"
STRATEGY_NAME = "breakout-volume-surge" # Strategy being tested
SUMMARY_FILENAME = "robustness_summary_breakout_volume_surge.csv" # 명확한 파일명

# --- Fixed Strategy Parameters for Robustness Test (Updated Optimal Params) ---
# These are the 'optimized' parameters we want to test across contexts
FIXED_STRATEGY_PARAMS = {
    # Optimized from previous runs
    'atr_sl_multiplier': 2.0,
    'atr_tp_multiplier': 4.5,
    # Other params from the best combo (though they seemed less sensitive)
    'pullback_threshold': 0.03,
    'rsi_period': 14,
    # Base fixed params
    'breakout_lookback': 20,
    'pullback_lookback': 5,
    'volume_lookback': 10,
    'volume_multiplier': 2.0,
    'atr_window': 10,
    'sma_short_period': 7, # Required by function signature
    'sma_long_period': 50, # Required by function signature
    'use_trend_filter': False, # Explicitly off for this strategy focus
    'use_atr_exit': True,
    'initial_cash': 10000.0,
    'commission': 0.001,
}

# --- Experiment Combinations for Robustness Testing (User Provided) ---
EXPERIMENT_COMBINATIONS = [
    # 암호화폐 (고변동성, 24h)
    {"symbol": "BTC-USD", "start_date": "2023-10-01", "end_date": "2023-12-30", "interval": "1h"},
    {"symbol": "ETH-USD", "start_date": "2023-10-01", "end_date": "2023-12-30", "interval": "1h"},
    {"symbol": "SOL-USD", "start_date": "2023-10-01", "end_date": "2023-12-30", "interval": "1h"},

    # 전통 금융 (낮은 변동성, 일봉 기준)
    {"symbol": "SPY", "start_date": "2023-10-01", "end_date": "2023-12-30", "interval": "1d"},
    {"symbol": "QQQ", "start_date": "2023-10-01", "end_date": "2023-12-30", "interval": "1d"},

    # 미국 개별주식 (트렌드 추적 가능한 고변동 개별주)
    {"symbol": "TSLA", "start_date": "2023-10-01", "end_date": "2023-12-30", "interval": "1d"},
    {"symbol": "NVDA", "start_date": "2023-10-01", "end_date": "2023-12-30", "interval": "1d"},

    # 알트코인 (불안정, 저유동성 테스트용)
    {"symbol": "DOGE-USD", "start_date": "2023-10-01", "end_date": "2023-12-30", "interval": "1h"},
]

# --- Metrics to Collect from Result JSONs ---
METRICS_TO_EXTRACT = ['symbol', 'start_date', 'end_date', 'Total Trades', 'Total Return [%]', 'Win Rate [%]', 'Sharpe Ratio', 'Max Drawdown [%]']

def generate_mcp_context(combination):
    """Generates the mcp/experiment_context.json file for a given combination."""
    context = {
        "strategy_name": STRATEGY_NAME,
        **combination, # symbol, start_date, end_date, interval
        **FIXED_STRATEGY_PARAMS # All the fixed strategy parameters
    }
    try:
        os.makedirs(os.path.dirname(CONTEXT_FILE_PATH), exist_ok=True)
        with open(CONTEXT_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(context, f, indent=4, ensure_ascii=False)
        logger.debug(f"Generated context file: {CONTEXT_FILE_PATH} for {combination['symbol']} {combination['start_date']}")
        return True
    except Exception as e:
        logger.error(f"Error generating context file {CONTEXT_FILE_PATH}: {e}")
        return False

def run_mcp_script():
    """Runs the mcp/run_mcp_experiment.py script using subprocess."""
    try:
        # Use poetry run to ensure correct environment and dependencies
        # Using | cat is generally not needed with subprocess if capturing output, 
        # but kept here if the script itself has interactive elements we want to bypass.
        # Better to run without pipe if possible. Let's try without it first.
        # command = ["poetry", "run", "python", MCP_RUNNER_SCRIPT, "|", "cat"] 
        command = ["poetry", "run", "python", MCP_RUNNER_SCRIPT]
        logger.info(f"Executing command: {' '.join(command)}")
        
        # Run the command, capture output and errors
        result = subprocess.run(command, capture_output=True, text=True, check=False, encoding='utf-8')

        if result.returncode != 0:
            logger.error(f"MCP Runner script failed with exit code {result.returncode}.")
            logger.error(f"Stderr:\n{result.stderr}")
            logger.error(f"Stdout:\n{result.stdout}")
            return False
        else:
            logger.info(f"MCP Runner script executed successfully.")
            logger.debug(f"Stdout:\n{result.stdout}") # Log stdout on success for debugging if needed
            return True

    except FileNotFoundError:
         logger.error(f"Error: 'poetry' command not found. Is poetry installed and in PATH?")
         return False
    except Exception as e:
        logger.error(f"An unexpected error occurred running the MCP script: {e}", exc_info=True)
        return False


def collect_result(combination):
    """Collects the result from the generated JSON file."""
    symbol = combination['symbol']
    start_date = combination['start_date']
    end_date = combination['end_date']
    result_filename = f"{symbol}_{start_date}_{end_date}_result.json"
    result_filepath = os.path.join(RESULTS_DIR, result_filename)

    try:
        with open(result_filepath, 'r', encoding='utf-8') as f:
            result_data = json.load(f)
        
        # Add context info to the results
        extracted_data = {
            'symbol': symbol,
            'start_date': start_date,
            'end_date': end_date,
        }
        for metric in METRICS_TO_EXTRACT:
             if metric not in ['symbol', 'start_date', 'end_date']:
                 # Handle potential missing keys or None values gracefully
                 value = result_data.get(metric)
                 # Convert 'inf', '-inf', None back to np.nan for calculations if needed
                 if value == 'inf': value = np.inf
                 elif value == '-inf': value = -np.inf
                 elif value is None: value = np.nan
                 
                 # Attempt conversion to float, fallback to NaN
                 try:
                     extracted_data[metric] = float(value) if value is not None else np.nan
                 except (ValueError, TypeError):
                      extracted_data[metric] = np.nan # If conversion fails

        # Ensure Sharpe Ratio is NaN if infinite
        if 'Sharpe Ratio' in extracted_data and not np.isfinite(extracted_data['Sharpe Ratio']):
            extracted_data['Sharpe Ratio'] = np.nan
            
        return extracted_data

    except FileNotFoundError:
        logger.warning(f"Result file not found: {result_filepath}")
        return None
    except json.JSONDecodeError:
        logger.warning(f"Error decoding JSON from result file: {result_filepath}")
        return None
    except Exception as e:
        logger.error(f"Error collecting result from {result_filepath}: {e}")
        return None

def run_robustness_test():
    """Main function to run the robustness tests."""
    logger.info("=== Starting Robustness Testing ===")
    logger.info(f"Strategy: {STRATEGY_NAME}")
    logger.info(f"Number of combinations to test: {len(EXPERIMENT_COMBINATIONS)}")

    all_results = []

    for combination in tqdm(EXPERIMENT_COMBINATIONS, desc="Running Robustness Tests"):
        logger.info(f"--- Running Test for: {combination['symbol']} ({combination['start_date']} to {combination['end_date']}) ---")
        
        # 1. Generate MCP context file
        if not generate_mcp_context(combination):
            logger.warning(f"Skipping combination due to context generation failure: {combination}")
            continue

        # 2. Run the MCP experiment script
        if not run_mcp_script():
            logger.warning(f"Skipping combination due to MCP script execution failure: {combination}")
            continue

        # 3. Collect results
        result = collect_result(combination)
        if result:
            all_results.append(result)
        else:
             logger.warning(f"Could not collect results for combination: {combination}")

    logger.info("--- Robustness Testing Loop Completed ---")

    if not all_results:
        logger.error("No results were collected during the robustness testing.")
        return

    # --- Process and Summarize Results ---
    results_df = pd.DataFrame(all_results)
    
    # Ensure numeric types where expected
    for col in ['Total Trades', 'Total Return [%]', 'Win Rate [%]', 'Sharpe Ratio', 'Max Drawdown [%]']:
         if col in results_df.columns:
              results_df[col] = pd.to_numeric(results_df[col], errors='coerce')

    # --- ADDED: Save summary results to CSV ---
    summary_filepath = os.path.join(RESULTS_DIR, SUMMARY_FILENAME)
    try:
        # Ensure the results directory exists (though it should by now)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        results_df.to_csv(summary_filepath, index=False, encoding='utf-8', float_format='%.4f')
        logger.info(f"Robustness test summary saved to: {summary_filepath}")
    except Exception as e:
        logger.error(f"Error saving robustness summary CSV to {summary_filepath}: {e}")
    # --- END ADDED ---

    # --- Display Top Results ---
    logger.info("=== Robustness Test Summary ===")
    
    # Set display options for potentially wider tables
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    pd.set_option('display.float_format', '{:.4f}'.format)

    print("\n--- All Test Results ---")
    print(results_df[METRICS_TO_EXTRACT].to_string(index=False))

    # Top by Sharpe Ratio (handle NaNs)
    top_sharpe = results_df.sort_values(by='Sharpe Ratio', ascending=False, na_position='last').head(3)
    print("\n--- Top 3 by Sharpe Ratio --- ")
    if not top_sharpe.empty:
        print(top_sharpe[METRICS_TO_EXTRACT].to_string(index=False))
    else:
        print("No valid Sharpe Ratio results found.")

    # Top by Win Rate (handle NaNs)
    top_winrate = results_df.sort_values(by='Win Rate [%]', ascending=False, na_position='last').head(3)
    print("\n--- Top 3 by Win Rate [%] --- ")
    if not top_winrate.empty:
        print(top_winrate[METRICS_TO_EXTRACT].to_string(index=False))
    else:
        print("No valid Win Rate results found.")

    # Reset display options
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.float_format')

    logger.info("=== Robustness Testing Finished ===")


if __name__ == "__main__":
    run_robustness_test() 