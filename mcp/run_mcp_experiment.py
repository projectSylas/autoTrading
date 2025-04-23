"""
Runs a single backtest experiment based on the configuration
specified in mcp/experiment_context.json.

Reads the context, executes the vectorbt backtest, and saves the results
to a JSON file in the mcp/results directory.
"""

import json
import os
import sys
import pandas as pd
import numpy as np
import logging

# --- Project Specific Imports ---
try:
    # Assuming run_vectorbt_backtest is correctly implemented and accessible
    from src.backtesting.backtest_runner_vbt import run_vectorbt_backtest
    from src.config.settings import settings # Import settings for defaults if needed
    from src.utils.logging_config import setup_logging, APP_LOGGER_NAME
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Please ensure the script is run from the project root directory (e.g., using 'poetry run python mcp/run_mcp_experiment.py') or PYTHONPATH is set correctly.")
    sys.exit(1)

# --- Setup Logging ---
setup_logging()
logger = logging.getLogger(APP_LOGGER_NAME + ".mcp_runner")
# --- End Setup Logging ---

# --- Constants ---
CONTEXT_FILE = "mcp/experiment_context.json"
RESULTS_DIR = "mcp/results"

def load_context(filepath):
    """Loads the experiment context from a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            context = json.load(f)
        logger.info(f"Successfully loaded experiment context from: {filepath}")
        return context
    except FileNotFoundError:
        logger.error(f"Context file not found: {filepath}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {filepath}: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred loading context: {e}")
        return None

def save_results(results, filepath):
    """Saves the backtest results dictionary to a JSON file."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Convert numpy types to standard Python types for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, (np.integer, np.int_)):
                serializable_results[key] = int(value)
            elif isinstance(value, (np.floating, np.float_)):
                # Handle NaN and Inf
                if pd.isna(value):
                    serializable_results[key] = None # Represent NaN as null
                elif not np.isfinite(value):
                     serializable_results[key] = str(value) # Represent Inf as string 'inf' or '-inf'
                else:
                     serializable_results[key] = float(value)
            elif isinstance(value, np.bool_):
                serializable_results[key] = bool(value)
            elif pd.isna(value):
                 serializable_results[key] = None # Handle pandas NA
            else:
                serializable_results[key] = value # Assume other types are serializable

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=4, ensure_ascii=False)
        logger.info(f"Successfully saved experiment results to: {filepath}")
    except TypeError as e:
        logger.error(f"Type error during JSON serialization: {e}. Results: {results}")
    except Exception as e:
        logger.error(f"An unexpected error occurred saving results to {filepath}: {e}")

def run_experiment():
    """Loads context, runs backtest, and saves results."""
    logger.info("=== Starting MCP Experiment Run ===")

    context = load_context(CONTEXT_FILE)
    if context is None:
        logger.error("Failed to load experiment context. Aborting.")
        return

    # --- Extract Key Information for Filename and Logging ---
    symbol = context.get('symbol', 'UNKNOWN')
    start_date = context.get('start_date', 'UNKNOWN')
    end_date = context.get('end_date', 'UNKNOWN')
    strategy_name = context.get('strategy_name', 'UNKNOWN_STRATEGY')
    logger.info(f"Running experiment for: {strategy_name} on {symbol} ({start_date} to {end_date})")

    # --- Prepare Arguments for Backtester ---
    # Use context directly as arguments, assuming keys match run_vectorbt_backtest signature
    # Add necessary defaults or fixed arguments if not in context
    backtest_args = context.copy() # Start with all context parameters
    backtest_args.pop('strategy_name', None) # Remove strategy_name as it's not a backtest arg
    backtest_args['plot'] = False # Ensure plotting is off for automated runs
    backtest_args['return_stats_only'] = True # We only need the stats dict

    # --- Run the Backtest --- 
    try:
        logger.info(f"Executing run_vectorbt_backtest with args: { {k:v for k,v in backtest_args.items() if k not in ['api_key', 'secret_key']} }") # Avoid logging keys
        stats = run_vectorbt_backtest(**backtest_args)

        if stats is None:
            logger.error("Backtest function returned None. No results to save.")
            return
        logger.info(f"Backtest completed successfully. Stats: {stats}")

    except Exception as e:
        logger.error(f"Exception occurred during backtest execution: {e}", exc_info=True)
        return # Stop processing if backtest fails

    # --- Save the Results --- 
    result_filename = f"{symbol}_{start_date}_{end_date}_result.json"
    result_filepath = os.path.join(RESULTS_DIR, result_filename)

    # --- ADDED: Check if backtest returned an error before saving --- 
    if isinstance(stats, dict) and 'error' in stats:
        logger.error(f"Backtest for {strategy_name} failed with error: {stats['error']}. Skipping result saving.")
    elif stats is None:
        # This case might still happen if an exception occurred before returning the error dict
        logger.error(f"Backtest for {strategy_name} returned None unexpectedly. Skipping result saving.")
    else:
        # Proceed with saving only if stats is a valid results dictionary
        save_results(stats, result_filepath)
    # --- END ADDED --- 

    logger.info("=== MCP Experiment Run Finished ===")

if __name__ == "__main__":
    run_experiment() 