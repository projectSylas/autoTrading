import subprocess
import sys
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the list of tickers and their corresponding trained model steps
TICKER_STEPS = {
    'AAPL': 500000,
    'MSFT': 100000, # Added MSFT with its trained steps
    # Add other tickers and their steps here once trained
    # 'GOOG': 500000,
    # 'AMZN': 500000,
    # 'TSLA': 500000
}
# Define the path to the backtest script relative to the project root
BACKTEST_SCRIPT_PATH = "src/ai_models/backtest_custom_env.py"

def run_single_backtest(ticker, model_steps):
    """Runs the backtest script for a single ticker with specific model steps."""
    logging.info(f"--- Starting backtest for {ticker} using model with {model_steps} steps ---")

    # Construct expected model path
    expected_model_path = f"models/custom_env/ppo_{ticker.upper()}_custom_env_{model_steps}.zip"
    if not os.path.exists(expected_model_path):
        logging.warning(f"Model file for {ticker} with {model_steps} steps not found at {expected_model_path}. Skipping backtest.")
        return False # Indicate failure for this ticker

    command = [
        sys.executable, # Use the same python interpreter
        BACKTEST_SCRIPT_PATH,
        "--ticker",
        ticker,
        "--model_steps", # Pass the correct steps to the backtest script
        str(model_steps)
    ]
    try:
        # Run the script and capture output
        # Increased timeout in case backtesting takes longer
        result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8', timeout=600)
        logging.info(f"Backtest for {ticker} ({model_steps} steps) completed successfully.")

        # Print summary lines from the output
        for line in result.stdout.splitlines():
             if "Final Portfolio Value:" in line or "Total Return:" in line or "Number of Trades (Closed):" in line:
                 logging.info(f"[{ticker} Result ({model_steps} steps)] {line.strip()}")

    except FileNotFoundError:
         logging.error(f"Error: Backtest script not found at {BACKTEST_SCRIPT_PATH}")
         return False
    except subprocess.TimeoutExpired:
        logging.error(f"Timeout expired while running backtest for {ticker} ({model_steps} steps).")
        return False
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running backtest for {ticker} ({model_steps} steps). Return code: {e.returncode}")
        logging.error(f"Command: {' '.join(e.cmd)}")
        # Log stderr for detailed error information
        logging.error(f"Stderr:\n{e.stderr}")
        # Also log stdout if it contains useful info despite error
        if e.stdout:
             logging.error(f"Stdout:\n{e.stdout}")
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred while running backtest for {ticker} ({model_steps} steps): {e}")
        return False
    return True

if __name__ == "__main__":
    logging.info("Starting multi-ticker backtesting process...")
    successful_backtests = 0
    failed_backtests = 0
    tickers_to_process = list(TICKER_STEPS.keys())

    # Ensure the backtest script exists before starting the loop
    if not os.path.exists(BACKTEST_SCRIPT_PATH):
        logging.error(f"Fatal: Backtest script not found at {BACKTEST_SCRIPT_PATH}. Aborting.")
        sys.exit(1)

    for ticker in tickers_to_process:
        steps = TICKER_STEPS[ticker]
        if run_single_backtest(ticker, steps):
            successful_backtests += 1
        else:
            failed_backtests += 1
        logging.info(f"--- Finished processing for {ticker} ---")

    logging.info("=" * 50)
    logging.info(" Multi-Ticker Backtest Summary ")
    logging.info("=" * 50)
    logging.info(f"Total tickers configured: {len(tickers_to_process)}")
    logging.info(f"Successful backtests run: {successful_backtests}")
    logging.info(f"Skipped/Failed backtests: {failed_backtests}")
    logging.info("=" * 50) 