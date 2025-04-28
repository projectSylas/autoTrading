import logging
import argparse # Import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# Project specific imports
from src.utils.common import get_historical_data
from src.ai_models.rl_environment import TradingEnv # Import custom environment

# Change log level to DEBUG for detailed step information
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Parameters ---
# TICKER = "AAPL" # Removed hardcoded ticker
DEFAULT_TICKER = "AAPL" # Define a default ticker
BACKTEST_PERIOD = "1y" # Use last 1 year for backtesting
BACKTEST_INTERVAL = "1h" # Use 1-hour data
# MODEL_PATH = "models/custom_env/ppo_AAPL_custom_env_500000.zip" # Path will be constructed dynamically
INITIAL_BALANCE = 10000 # Same as training
RSI_PERIOD = 14 # Same as training
SMA_PERIOD = 7 # Same as training
TRANSACTION_COST = 0.001 # Same as training
# Add parameters for new indicators, matching train script defaults
ATR_WINDOW = 14
BB_WINDOW = 20
BB_DEV = 2

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Run backtesting for a trained PPO agent.")
parser.add_argument('--ticker', type=str, default=DEFAULT_TICKER,
                    help=f'Ticker symbol to backtest (default: {DEFAULT_TICKER})')
parser.add_argument('--model_steps', type=int, default=500000,
                    help='Number of training steps for the model file (default: 500000)')
args = parser.parse_args()

# --- Use parsed arguments ---
TICKER = args.ticker.upper() # Ensure ticker is uppercase
MODEL_STEPS = args.model_steps
MODEL_PATH = f"models/custom_env/ppo_{TICKER}_custom_env_{MODEL_STEPS}.zip" # Construct model path dynamically

def run_backtest(ticker: str, model_path: str):
    """Runs the backtesting process for the trained PPO agent."""
    logging.info("=" * 50)
    logging.info(f" Starting Custom TradingEnv Agent Backtesting for {ticker} ")
    logging.info("=" * 50)

    # --- 1. Load Data ---
    logging.info(f"Loading backtest data for {ticker}: Period={BACKTEST_PERIOD}, Interval={BACKTEST_INTERVAL}")
    df_backtest = get_historical_data(ticker, period=BACKTEST_PERIOD, interval=BACKTEST_INTERVAL)

    if df_backtest is None or df_backtest.empty:
        logging.error(f"Failed to load backtesting data for {ticker}.")
        return

    logging.info(f"Backtest data loaded successfully. Shape: {df_backtest.shape}")

    # --- 2. Create Environment ---
    logging.info("Creating TradingEnv for backtesting...")
    try:
        env = TradingEnv(
            df=df_backtest,
            initial_balance=INITIAL_BALANCE,
            transaction_cost=TRANSACTION_COST,
            rsi_period=RSI_PERIOD,
            sma_period=SMA_PERIOD,
            # Pass new indicator parameters
            atr_window=ATR_WINDOW,
            bb_window=BB_WINDOW,
            bb_dev=BB_DEV
        )
        logging.info("Backtest environment created successfully.")
    except ValueError as e:
        logging.error(f"Error creating backtest environment: {e}")
        return
    except Exception as e:
         logging.error(f"Unexpected error creating environment: {e}")
         return

    # --- 3. Load Model ---
    logging.info(f"Loading trained model from: {model_path}")
    try:
        model = PPO.load(model_path, env=env) # Provide env for observation/action space checks
        logging.info("Model loaded successfully.")
    except FileNotFoundError:
        logging.error(f"Model file not found at {model_path}")
        return
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return

    # --- 4. Run Backtesting Simulation ---
    logging.info("Starting backtesting simulation...")
    obs, info = env.reset()
    terminated = False
    truncated = False
    total_steps = 0
    portfolio_values = [INITIAL_BALANCE] # Track portfolio value over time
    trade_log = [] # Store trade details

    while not terminated and not truncated:
        action, _states = model.predict(obs, deterministic=True) # Use deterministic actions for evaluation
        obs, reward, terminated, truncated, info = env.step(action)
        total_steps += 1

        # --- Enhanced Logging ---
        current_price = info.get('price', 0)
        current_portfolio_value = env.balance
        held_value = 0
        if env.position_held == 1:
            # Use the price at which the step occurred (most recent price)
            # Ensure current_step reflects the step just taken before this check
            step_index_for_price = env.current_step - 1 if env.current_step > 0 else 0
            if step_index_for_price < len(env.df):
                 current_price_for_value = env.df.iloc[step_index_for_price]['Close']
                 held_value = current_price_for_value # Value of the single held unit
                 current_portfolio_value += held_value
            else:
                 # Handle edge case where index might be out of bounds (shouldn't happen ideally)
                 logging.warning(f"Step {total_steps}: Could not get current price for held value calculation.")

        portfolio_values.append(current_portfolio_value) # Track the more accurately calculated portfolio value
        logging.debug(f"Step {total_steps}: Action={action}, Reward={reward:.4f}, Balance={env.balance:.2f}, Held={env.position_held}, HeldValue={held_value:.2f}, PortValue={current_portfolio_value:.2f}, TotalPnL={env.total_pnl:.4f}")


        # Log trade info if available
        if 'trade' in info:
             if info['trade'] == 'buy':
                 trade_log.append(info)
                 logging.info(f"Step {info['step']} - TRADE EXECUTED: BUY at price {info.get('entry_price', 'N/A'):.2f}")
             elif info['trade'] == 'sell':
                 trade_log.append(info)
                 logging.info(f"Step {info['step']} - TRADE EXECUTED: SELL at price {info.get('exit_price', 'N/A'):.2f}, PnL: {info.get('pnl', 0):.4f}")
             elif 'invalid' in info['trade']:
                  logging.debug(f"Step {info['step']} - Invalid trade attempt: {info['trade']}")
             # No need to log simple 'hold' unless debugging verbosely
        # --- End Enhanced Logging ---

        # Simplified portfolio value tracking (using PnL, less precise for intra-step)
        # current_portfolio_value_pnl_based = INITIAL_BALANCE * (1 + env.total_pnl)
        # portfolio_values.append(current_portfolio_value_pnl_based)


        if terminated or truncated:
            logging.info(f"Backtesting loop finished after {total_steps} steps.")
            # Log final state
            logging.info(f"Final State: Balance={env.balance:.2f}, Position Held={env.position_held}, Total PnL={env.total_pnl:.4f}, Final Portfolio Value={current_portfolio_value:.2f}")
            break

    # --- 5. Analyze Results ---
    logging.info("Analyzing backtesting results...")

    # Calculate performance metrics
    final_portfolio_value = portfolio_values[-1]
    total_return_pct = ((final_portfolio_value - INITIAL_BALANCE) / INITIAL_BALANCE) * 100
    num_trades = len([t for t in trade_log if t.get('trade') == 'sell']) # Count closed trades (sell actions)
    # Use .get() for safety, though 'trade' should exist if appended

    # TODO: Add more metrics: Sharpe Ratio, Max Drawdown, Win Rate etc.

    logging.info("-" * 30)
    logging.info(f"Backtest Performance Summary ({ticker})") # Use ticker variable
    logging.info("-" * 30)
    logging.info(f"Initial Balance: ${INITIAL_BALANCE:,.2f}")
    logging.info(f"Final Portfolio Value: ${final_portfolio_value:,.2f}")
    logging.info(f"Total Return: {total_return_pct:.2f}%")
    logging.info(f"Total Steps: {total_steps}")
    logging.info(f"Number of Trades (Closed): {num_trades}")
    # logging.info(f"Win Rate: ...")
    # logging.info(f"Max Drawdown: ...")
    logging.info("-" * 30)


    # --- 6. Visualize Results ---
    logging.info("Generating visualization...")
    plt.figure(figsize=(14, 10))

    # Plot 1: Portfolio Value Over Time
    plt.subplot(2, 1, 1)
    plt.plot(portfolio_values)
    plt.title(f'{ticker} Backtest: Portfolio Value Over Time ({BACKTEST_PERIOD} - {BACKTEST_INTERVAL})') # Use ticker variable
    plt.ylabel('Portfolio Value ($)')
    plt.xlabel('Steps')
    plt.grid(True)

    # Plot 2: Price with Buy/Sell Signals
    plt.subplot(2, 1, 2)
    valid_df = env.df # Use the dataframe from the env after indicator calculation and NaN drop
    plt.plot(valid_df.index, valid_df['Close'], label='Close Price', alpha=0.7)

    buy_signals = [(t['step'], t['price']) for t in env.trade_history if t['type'] == 'buy']
    sell_signals = [(t['step'], t['price']) for t in env.trade_history if t['type'] == 'sell']

    if buy_signals:
        # Adjusted index access for potentially dropped initial rows
        buy_steps = [s[0] for s in buy_signals if s[0] < len(valid_df.index)]
        buy_indices = [valid_df.index[step] for step in buy_steps]
        buy_prices = [s[1] for s in buy_signals if s[0] < len(valid_df.index)]
        if buy_indices: # Ensure lists are not empty before scattering
            plt.scatter(buy_indices, buy_prices, label='Buy Signal', marker='^', color='green', s=100, alpha=1)

    if sell_signals:
        # Adjusted index access for potentially dropped initial rows
        sell_steps = [s[0] for s in sell_signals if s[0] < len(valid_df.index)]
        sell_indices = [valid_df.index[step] for step in sell_steps]
        sell_prices = [s[1] for s in sell_signals if s[0] < len(valid_df.index)]
        if sell_indices: # Ensure lists are not empty before scattering
            plt.scatter(sell_indices, sell_prices, label='Sell Signal', marker='v', color='red', s=100, alpha=1)

    plt.title(f'{ticker} Price with Trading Signals') # Use ticker variable
    plt.ylabel('Price ($)')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    save_path = f'backtest_results_{ticker}_{BACKTEST_PERIOD}_{BACKTEST_INTERVAL}.png' # Use ticker variable
    plt.savefig(save_path)
    logging.info(f"Visualization saved to {save_path}")
    # plt.show() # Uncomment to display the plot directly

if __name__ == "__main__":
    # Call run_backtest with parsed arguments
    run_backtest(ticker=TICKER, model_path=MODEL_PATH) 