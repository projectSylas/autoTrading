import logging
import pandas as pd
import numpy as np
import os
from stable_baselines3 import PPO, A2C, DDPG # Import algorithms used for loading
from typing import Optional

# Project specific imports
from src.config import settings
from src.utils.common import get_historical_data, calculate_rsi, calculate_sma # For state calculation
from src.utils.database import log_trade_to_db # For logging trades
from src.utils.notifier import send_slack_notification
# Import functions for actual order execution (e.g., from the original challenge_trading.py or a new exchange interaction module)
# from src.challenge.exchange_api import create_futures_order, get_current_positions, get_symbol_ticker, close_position # Placeholder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Mapping from string name to SB3 algorithm class (must match trainer)
SB3_ALGORITHMS = {
    "PPO": PPO,
    "A2C": A2C,
    "DDPG": DDPG,
}

class RLTradingStrategy:
    """Manages the execution of the RL trading strategy using a pre-trained model."""

    def __init__(self):
        self.model: Optional[BaseAlgorithm] = None
        self.scaler = None # Placeholder for potential future use (e.g., MinMaxScaler)
        self.current_position: int = 0
        self.entry_price: float = 0.0
        # --- Added: Load reference price for normalization --- 
        self.reference_price: float = settings.RL_NORMALIZATION_REFERENCE_PRICE
        if self.reference_price <= 0.0:
            logging.error(
                f"RL Normalization Reference Price (RL_NORMALIZATION_REFERENCE_PRICE in .env) "
                f"is not set or invalid ({self.reference_price}). RL strategy cannot function correctly. "
                f"Please train the model first and set this value in .env based on the training logs."
            )
            # Consider raising an error or preventing model loading if reference price is critical
            # self.model = None # Force model unload if ref price invalid?
        else:
            logging.info(f"RL Normalization Reference Price loaded: {self.reference_price:.8f}")
        # --- End Added ---
        self.load_model()
        # TODO: Initialize scaler (e.g., load from training or fit on recent data)
        # self.initialize_scaler()

    def load_model(self) -> None:
        """Loads the pre-trained RL model."""
        AlgorithmClass = SB3_ALGORITHMS.get(settings.RL_ALGORITHM)
        if AlgorithmClass is None:
            logging.error(f"Unsupported RL algorithm for loading: {settings.RL_ALGORITHM}")
            return

        if os.path.exists(settings.RL_MODEL_PATH):
            try:
                self.model = AlgorithmClass.load(settings.RL_MODEL_PATH)
                logging.info(f"RL model loaded successfully from {settings.RL_MODEL_PATH}")
            except Exception as e:
                logging.error(f"Failed to load RL model from {settings.RL_MODEL_PATH}: {e}")
        else:
            logging.error(f"RL model file not found at {settings.RL_MODEL_PATH}. Cannot run RL strategy.")

    # def initialize_scaler(self):
    #     """Initializes the scaler for state normalization."""
    #     # Option 1: Load scaler saved during training
    #     # Option 2: Fit scaler on recent historical data
    #     logging.info("Initializing state scaler...")
    #     # Placeholder: Needs actual implementation
    #     pass

    def _get_live_observation(self) -> Optional[np.ndarray]:
        """Gets the current market state and formats it using the fixed reference price."""
        # --- Added: Check for valid reference price --- 
        if self.reference_price <= 0.0:
            logging.error("Cannot get observation: RL Normalization Reference Price is invalid.")
            return None
        # --- End Added ---

        logging.debug("Getting live observation...")
        try:
            # 1. Fetch recent data
            df_live = get_historical_data(settings.RL_ENV_SYMBOL, period="1d", interval="1m") # Adjust as needed
            if df_live is None or df_live.empty or len(df_live) < max(14, 7): # Assuming RSI 14, SMA 7
                logging.warning("Could not fetch sufficient live data for observation.")
                return None

            # 2. Calculate indicators
            rsi = calculate_rsi(dataframe=df_live, window=14, column='Close')
            sma = calculate_sma(dataframe=df_live, window=7, column='Close')
            if rsi is None or sma is None or rsi.empty or sma.empty:
                logging.warning("Failed to calculate live indicators.")
                return None

            latest_rsi = rsi.iloc[-1]
            latest_sma = sma.iloc[-1]
            latest_price = df_live['Close'].iloc[-1]

            # 3. Calculate current PnL
            current_pnl = 0.0
            if self.current_position == 1 and self.entry_price > 0:
                current_pnl = (latest_price - self.entry_price) / self.entry_price

            # 4. Normalize features using the FIXED reference price from settings
            price_norm = latest_price / self.reference_price
            rsi_norm = latest_rsi # RSI is kept as is (0-100 range expected by env)
            sma_norm = latest_sma / self.reference_price
            pnl_norm = current_pnl * 10 # Simple scaling, matches env

            state = np.array([
                price_norm,
                rsi_norm,
                sma_norm,
                float(self.current_position),
                pnl_norm
            ], dtype=np.float32)

            if not np.all(np.isfinite(state)):
                logging.warning(f"Constructed state contains non-finite values: {state}. Replacing with zeros.")
                state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0) # Match env NaN handling

            logging.debug(f"Live observation created: PriceNorm={price_norm:.4f}, RSI={rsi_norm:.2f}, SmaNorm={sma_norm:.4f}, Pos={state[3]}, PnLNorm={pnl_norm:.4f}")
            return state

        except Exception as e:
            logging.error(f"Error getting live observation: {e}", exc_info=True)
            return None

    def _trigger_conditional_fallback(self, reason: str) -> None:
        # ... (implementation) ...
        pass

    def run_step(self) -> None:
        """Executes one step of the RL strategy: get state, predict action, execute action."""
        if not self.model:
            logging.error("RL Model not loaded. Cannot execute step.")
            return

        logging.info("--- RL Strategy Step --- ")
        # 1. Get Current State
        observation = self._get_live_observation()
        if observation is None:
            logging.warning("Failed to get valid observation. Skipping step.")
            return

        # 2. Predict Action
        # Use deterministic=True for exploitation during live trading
        action, _states = self.model.predict(observation, deterministic=True)
        action = int(action) # Convert numpy int to standard int

        action_map = {0: "Hold", 1: "Buy", 2: "Sell"}
        logging.info(f"RL Agent Action: {action_map[action]}")

        # 3. Execute Action
        # This requires integration with actual exchange API functions
        # Placeholder logic:
        current_price = 0 # Get actual current price from ticker
        try:
            # ticker = get_symbol_ticker(settings.RL_ENV_SYMBOL)
            # if ticker and 'price' in ticker:
            #     current_price = float(ticker['price'])
            # else:
            #      logging.error("Failed to get current price for execution.")
            #      return
            pass # Replace with actual price fetching
        except Exception as e:
             logging.error(f"Error fetching price for execution: {e}")
             return

        executed_trade = False
        trade_info = {
            'symbol': settings.RL_ENV_SYMBOL,
            'strategy': 'rl_challenge',
            'action_taken': action_map[action],
            'timestamp': datetime.now()
        }

        if action == 1: # Buy Signal
            if self.current_position == 0:
                logging.info(f"Executing BUY order for {settings.RL_ENV_SYMBOL} based on RL model.")
                # --- TODO: Implement Actual Buy Order --- #
                # quantity = calculate_rl_position_size(...) # Need position sizing logic
                # order_result = create_futures_order(settings.RL_ENV_SYMBOL, "BUY", quantity)
                order_result = {"status": "FILLED", "orderId": "rl_buy_test"} # Placeholder
                # --- End TODO --- #
                if order_result and order_result.get('status') in ['FILLED', 'PARTIALLY_FILLED']: # Check successful execution
                    self.current_position = 1
                    # Fetch execution price if possible, otherwise use current_price approx
                    self.entry_price = current_price # Update entry price
                    executed_trade = True
                    trade_info.update({'side': 'buy', 'status': 'open', 'entry_price': self.entry_price})
                    if send_slack_notification:
                         send_slack_notification(f"🤖 RL Buy Signal Executed", f"Symbol: {settings.RL_ENV_SYMBOL} at ~${self.entry_price:.2f}")
                else:
                     logging.error("RL Buy order execution failed or status unknown.")
                     trade_info.update({'side': 'buy', 'status': 'failed'})
            else:
                logging.info("RL Buy signal ignored (already holding position).")

        elif action == 2: # Sell Signal
            if self.current_position == 1:
                logging.info(f"Executing SELL order for {settings.RL_ENV_SYMBOL} based on RL model.")
                 # --- TODO: Implement Actual Sell Order --- #
                # current_pos_details = get_current_position_details(...) # Need details like quantity
                # quantity_to_close = current_pos_details['quantity']
                # order_result = close_position(settings.RL_ENV_SYMBOL, quantity_to_close, side_to_close='BUY') # Close long means buy back
                order_result = {"status": "FILLED", "orderId": "rl_sell_test"} # Placeholder
                # --- End TODO --- #
                if order_result and order_result.get('status') in ['FILLED']: # Check successful execution
                    exit_price = current_price # Use actual exit price if available
                    pnl = (exit_price - self.entry_price) / self.entry_price if self.entry_price > 0 else 0
                    self.current_position = 0
                    executed_trade = True
                    trade_info.update({'side': 'sell', 'status': 'closed', 'exit_price': exit_price, 'pnl_percent': pnl})
                    logging.info(f"RL Position Closed. PnL: {pnl:.4f}")
                    if send_slack_notification:
                         send_slack_notification(f"🤖 RL Sell Signal Executed", f"Symbol: {settings.RL_ENV_SYMBOL} at ~${exit_price:.2f}, PnL: {pnl:.2%}")
                    self.entry_price = 0.0
                else:
                     logging.error("RL Sell order execution failed or status unknown.")
                     trade_info.update({'side': 'sell', 'status': 'failed'})
            else:
                logging.info("RL Sell signal ignored (no position held).")
        else: # Hold Signal
            logging.info("RL Hold signal. No action taken.")

        # 4. Log trade if executed
        if executed_trade:
            try:
                # Use appropriate columns for log_trade_to_db
                log_trade_to_db(
                    strategy=trade_info['strategy'],
                    symbol=trade_info['symbol'],
                    side=trade_info['side'],
                    status=trade_info['status'],
                    entry_price=trade_info.get('entry_price'),
                    exit_price=trade_info.get('exit_price'),
                    pnl_percent=trade_info.get('pnl_percent'),
                    reason=f"RL Action: {action_map[action]}"
                    # Add quantity, order_id if available
                )
                logging.info("[DB] RL trade action logged.")
            except Exception as db_err:
                 logging.error(f"Failed to log RL trade to DB: {db_err}")

# --- Main Execution Logic (Example) --- #
def run_rl_trading_loop():
    """Runs the RL trading strategy in a loop."""
    strategy = RLTradingStrategy()
    if not strategy.model:
        logging.error("Exiting RL trading loop: Model not loaded.")
        return

    # Loop based on schedule or time interval
    # For demonstration, run a few steps
    logging.info("Starting RL Trading Loop (Demo: 5 steps)")
    for i in range(5):
        strategy.run_step()
        # Add sleep based on data frequency (e.g., sleep 60s for 1m data)
        import time
        time.sleep(10) # Simple 10 second delay for demo
    logging.info("Finished RL Trading Loop (Demo).")

if __name__ == "__main__":
    # Run the live trading loop (requires pre-trained model and exchange integration)
    run_rl_trading_loop() 