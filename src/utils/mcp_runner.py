"""
MCP Runner Agent v2: Loads experiment context from a JSON file, validates,
dynamically maps parameters, selects strategy function, runs a backtest,
and saves the results or errors.
Designed for automated execution, potentially by an LLM.
"""
import json
import sys
import os
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import inspect # To inspect function signatures

# --- Project Specific Imports ---
try:
    # Import all potential backtesting functions
    from src.backtesting.backtest_runner_vbt import run_vectorbt_backtest
    # Add imports for other strategy functions if they exist
    # from src.strategies.volume_breakout import run_volume_breakout_backtest # Example

    from src.config.settings import settings # Import settings for potential defaults
    from src.utils.logging_config import setup_logging, APP_LOGGER_NAME
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Please ensure the script is run from the project root directory (e.g., using 'poetry run python src/utils/mcp_runner.py ...') or PYTHONPATH is set correctly.")
    sys.exit(1)

# --- Setup Logging ---
setup_logging()
logger = logging.getLogger(APP_LOGGER_NAME + ".mcp_runner_v2")
# --- End Setup Logging ---

# --- Configuration ---
OUTPUT_DIR = "mcp/results" # Output path relative to project root
DEFAULT_OUTPUT_FILENAME = "experiment_result.json"

# --- Strategy Function Mapping --- 
# Map strategy_core names from MCP to actual Python functions
STRATEGY_MAP = {
    "breakout-pullback": run_vectorbt_backtest, # Assumes this handles the base case
    "breakout-volume-surge": run_vectorbt_backtest, # Assumes this function handles volume params
    # Add other mappings here, e.g.:
    # "mean-reversion-rsi": run_mean_reversion_backtest 
}

def load_mcp(filepath: str) -> dict | None:
    """Loads and validates the MCP JSON file."""
    logger.info(f"Loading MCP context from: {filepath}")
    if not os.path.exists(filepath):
        logger.error(f"MCP file not found at: {filepath}")
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            mcp_data = json.load(f)
        logger.info("MCP context loaded successfully.")

        # --- Validation --- 
        required_keys = ['project', 'strategy_core', 'execution', 'sl_tp', 'strategy_params', 'metrics']
        missing_keys = [key for key in required_keys if key not in mcp_data]
        if missing_keys:
            logger.error(f"MCP file is missing required top-level keys: {missing_keys}")
            return None

        if mcp_data['strategy_core'] not in STRATEGY_MAP:
            logger.error(f"Unknown 'strategy_core' specified in MCP: {mcp_data['strategy_core']}. Available: {list(STRATEGY_MAP.keys())}")
            return None
            
        exec_params = mcp_data.get('execution', {})
        if not all(k in exec_params for k in ['symbol', 'timeframe', 'date_range']) or \
           not isinstance(exec_params['date_range'], list) or len(exec_params['date_range']) != 2:
            logger.error("MCP 'execution' section is invalid or missing required keys/structure.")
            return None

        sl_tp_params = mcp_data.get('sl_tp', {})
        if 'type' not in sl_tp_params:
            logger.error("MCP 'sl_tp' section is missing required key: type.")
            return None
        if sl_tp_params['type'] == "ATR-dynamic" and not all(k in sl_tp_params and sl_tp_params[k] is not None for k in ['atr_window', 'sl_multiplier', 'tp_multiplier']):
            logger.error("MCP 'sl_tp' is ATR-dynamic but missing required keys/values: atr_window, sl_multiplier, tp_multiplier.")
            return None

        if not isinstance(mcp_data.get('strategy_params'), dict):
             logger.error("MCP 'strategy_params' must be a dictionary.")
             return None

        logger.info("MCP validation passed.")
        return mcp_data

    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from MCP file {filepath}: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during MCP loading/validation: {e}", exc_info=True)
        return None

def prepare_backtest_args(mcp: dict, target_function) -> dict | None:
    """Prepares the arguments dictionary specifically for the target backtest function."""
    logger.info(f"Preparing arguments for target function: {target_function.__name__}")
    try:
        exec_params = mcp['execution']
        sl_tp_params = mcp['sl_tp']
        strategy_params = mcp['strategy_params']

        # Get expected arguments from the target function signature
        sig = inspect.signature(target_function)
        func_params = sig.parameters
        expected_arg_names = set(func_params.keys())

        # --- Start building args --- 
        args = {
            'symbol': exec_params['symbol'],
            'start_date': exec_params['date_range'][0],
            'end_date': exec_params['date_range'][1],
            'interval': exec_params['timeframe'],
            'initial_cash': mcp.get('initial_cash', settings.INITIAL_CASH if hasattr(settings, 'INITIAL_CASH') else 10000.0),
            'commission': mcp.get('commission', settings.COMMISSION if hasattr(settings, 'COMMISSION') else 0.001),
        }

        # --- Add SL/TP Parameters based on function signature --- 
        if 'use_atr_exit' in expected_arg_names:
            args['use_atr_exit'] = sl_tp_params['type'] == "ATR-dynamic"
            if args['use_atr_exit']:
                if 'atr_window' in expected_arg_names: args['atr_window'] = sl_tp_params['atr_window']
                if 'atr_sl_multiplier' in expected_arg_names: args['atr_sl_multiplier'] = sl_tp_params['sl_multiplier']
                if 'atr_tp_multiplier' in expected_arg_names: args['atr_tp_multiplier'] = sl_tp_params['tp_multiplier']
            else:
                 # Pass fixed SL/TP only if the function expects them
                if 'sl_stop' in expected_arg_names: args['sl_stop'] = sl_tp_params.get('sl_stop', settings.CHALLENGE_SL_RATIO)
                if 'tp_stop' in expected_arg_names: args['tp_stop'] = sl_tp_params.get('tp_stop', settings.CHALLENGE_TP_RATIO)
        else:
            logger.warning(f"Target function {target_function.__name__} does not accept 'use_atr_exit'. SL/TP logic might be handled internally or differently.")

        # --- Add Strategy Parameters --- 
        # Only pass strategy params that are actual arguments of the target function
        valid_strategy_params = {}
        unknown_strategy_params = {}
        for key, value in strategy_params.items():
            if key in expected_arg_names:
                valid_strategy_params[key] = value
            else:
                unknown_strategy_params[key] = value
        
        args.update(valid_strategy_params)
        if unknown_strategy_params:
            logger.warning(f"Ignoring unknown strategy parameters from MCP for function {target_function.__name__}: {unknown_strategy_params}")
        logger.debug(f"Added strategy params to args: {valid_strategy_params}")

        # --- Add Run Control Parameters (if function accepts them) ---
        if 'plot' in expected_arg_names: args['plot'] = False
        if 'return_stats_only' in expected_arg_names: args['return_stats_only'] = True
        if 'use_trend_filter' in expected_arg_names: args['use_trend_filter'] = mcp.get('use_trend_filter', False)

        logger.info("Argument preparation complete.")
        logger.debug(f"Final arguments for {target_function.__name__}: {args}")
        return args

    except KeyError as e:
        logger.error(f"Missing expected key during argument preparation: {e}. Check MCP structure.")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during argument preparation: {e}", exc_info=True)
        return None

def run_experiment_from_mcp(mcp: dict) -> dict | None:
    """Selects the strategy function and runs the backtest."""
    logger.info("Running experiment based on loaded MCP context...")

    strategy_name = mcp.get('strategy_core')
    target_function = STRATEGY_MAP.get(strategy_name)

    if target_function is None: # Should be caught by validation, but double-check
        logger.error(f"Could not find target function for strategy: {strategy_name}")
        return {"Error": f"Target function not found for strategy {strategy_name}"}

    backtest_args = prepare_backtest_args(mcp, target_function)
    if backtest_args is None:
        logger.error("Failed to prepare backtest arguments.")
        return {"Error": "Failed to prepare backtest arguments."}

    try:
        # --- Call the Correct Backtest Function --- 
        logger.info(f"Calling {target_function.__name__} for {backtest_args['symbol']}...")
        stats = target_function(**backtest_args)

        if stats is None:
            logger.error(f"{target_function.__name__} returned None. Backtest likely failed or produced no trades/results.")
            return {"Error": "Backtest failed or yielded no stats."}

        # --- Extract Required Metrics --- 
        metrics_to_extract = mcp.get('metrics', [])
        result_metrics = {}
        missing_metrics = []
        for metric in metrics_to_extract:
            if metric in stats:
                value = stats[metric]
                # Convert types for JSON
                if isinstance(value, (np.integer, np.int_)): value = int(value)
                elif isinstance(value, (np.floating, np.float_)):
                    if pd.isna(value): value = None
                    elif not np.isfinite(value): value = str(value)
                    else: value = float(value)
                elif pd.isna(value): value = None
                result_metrics[metric] = value
            else:
                logger.warning(f"Metric '{metric}' specified in MCP not found in results.")
                result_metrics[metric] = None
                missing_metrics.append(metric)

        if missing_metrics: logger.warning(f"Missing metrics: {missing_metrics}")

        logger.info(f"Experiment backtest completed. Extracted metrics: {result_metrics}")
        return result_metrics

    except TypeError as e:
        # This error now strongly suggests a real mismatch that wasn't caught by prepare_args
        # or an internal issue in the called function.
        logger.error(f"TypeError during {target_function.__name__} call: {e}. Check function signature vs prepared args.")
        logger.error(f"Arguments passed: {backtest_args}")
        return {"Error": f"TypeError: {e}"}
    except Exception as e:
        logger.error(f"An unexpected error occurred during experiment execution ({target_function.__name__}): {e}", exc_info=True)
        return {"Error": f"Unexpected error: {e}"}

def save_result(result: dict, context_path: str, output_dir: str = OUTPUT_DIR, output_filename: str = DEFAULT_OUTPUT_FILENAME):
    """Saves the experiment result to a JSON file."""
    if result is None:
        logger.error("Cannot save result because the result dictionary is None.")
        return False # Indicate failure

    output_filepath = os.path.join(output_dir, output_filename)
    logger.info(f"Saving experiment result to: {output_filepath}")

    try:
        os.makedirs(output_dir, exist_ok=True)
        output_data = {
            "context_file": os.path.abspath(context_path), # Store absolute path
            "executed_at": datetime.now().isoformat(),
            "result": result # Contains metrics or error message
        }
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        logger.info("Experiment result saved successfully.")
        return True # Indicate success

    except IOError as e:
        logger.error(f"Error writing result to file {output_filepath}: {e}")
        return False # Indicate failure
    except Exception as e:
        logger.error(f"An unexpected error occurred while saving result: {e}", exc_info=True)
        return False # Indicate failure


if __name__ == "__main__":
    logger.info("MCP Runner Agent v2 started.")

    if len(sys.argv) < 2:
        logger.error("Usage: poetry run python src/utils/mcp_runner.py <path_to_mcp_context.json>")
        print("Error: Please provide the path to the MCP context file.")
        sys.exit(1)

    context_path = sys.argv[1]

    mcp_context = load_mcp(context_path)
    if mcp_context is None:
        print(f"Error: Failed to load or validate MCP context from {context_path}")
        sys.exit(1)

    experiment_result = run_experiment_from_mcp(mcp_context)

    # --- Save Result --- 
    if save_result(experiment_result, context_path):
        if isinstance(experiment_result, dict) and "Error" in experiment_result:
             print(f"Experiment finished with an error. Error details saved to {os.path.join(OUTPUT_DIR, DEFAULT_OUTPUT_FILENAME)}")
             logger.error(f"Experiment execution failed: {experiment_result['Error']}")
             # sys.exit(1) # Optionally exit with error code
        else:
             print(f"Experiment finished successfully. Results saved to {os.path.join(OUTPUT_DIR, DEFAULT_OUTPUT_FILENAME)}")
             logger.info("MCP Runner Agent v2 finished successfully.")
    else:
        print("Error: Failed to save experiment results.")
        sys.exit(1) 