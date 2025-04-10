import logging
import pandas as pd
import numpy as np
import os
from datetime import datetime # Corrected import
from typing import Optional, Tuple, Dict, Any # Added for type hints
from stable_baselines3 import PPO, A2C, DDPG # Import algorithms used for loading
from stable_baselines3.common.base_class import BaseAlgorithm # For type hinting model

# Project specific imports
from src.config import settings
from src.utils.common import get_historical_data, calculate_rsi, calculate_sma # For state calculation
from src.utils.database import log_trade_to_db # For logging trades
from src.utils.notifier import send_slack_notification
# Placeholder for exchange interaction
# from src.challenge.exchange_api import ...

# --- Import Conditional Strategy Functions for Fallback ---
try:
    from src.challenge.strategy import (
        check_entry_conditions,
        calculate_position_size,
        create_futures_order,
        get_symbol_ticker, # Needed for price and position size calculation
        get_futures_account_balance # Needed for equity
    )
    conditional_strategy_available = True
    logging.info("Conditional strategy functions loaded successfully for fallback.")
except ImportError as e:
    logging.warning(f"Failed to import conditional strategy functions: {e}. Fallback to conditional strategy disabled.")
    check_entry_conditions = None
    calculate_position_size = None
    create_futures_order = None
    get_symbol_ticker = None
    get_futures_account_balance = None
    conditional_strategy_available = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Mapping from string name to SB3 algorithm class (must match trainer)
SB3_ALGORITHMS: Dict[str, type[BaseAlgorithm]] = {
    "PPO": PPO,
    "A2C": A2C,
    "DDPG": DDPG,
}

class RLTradingStrategy:
    # ... (existing __init__) ...
    def __init__(self):
        self.model: Optional[BaseAlgorithm] = None # Type hint added
        self.scaler = None # Placeholder for state scaler
        self.current_position: int = 0 # 0: None, 1: Long (Assuming only long for now)
        self.entry_price: float = 0.0
        self.load_model()
        # self.initialize_scaler()

    def load_model(self) -> None:
        """Loads the pre-trained RL model."""
        AlgorithmClass: Optional[type[BaseAlgorithm]] = SB3_ALGORITHMS.get(settings.RL_ALGORITHM)
        if AlgorithmClass is None:
            logging.error(f"Unsupported RL algorithm for loading: {settings.RL_ALGORITHM}")
            self.model = None # Ensure model is None
            return

        if os.path.exists(settings.RL_MODEL_PATH):
            try:
                self.model = AlgorithmClass.load(settings.RL_MODEL_PATH)
                logging.info(f"RL model loaded successfully from {settings.RL_MODEL_PATH}")
            except Exception as e:
                logging.error(f"Failed to load RL model from {settings.RL_MODEL_PATH}: {e}")
                self.model = None # Ensure model is None on load failure
        else:
            logging.error(f"RL model file not found at {settings.RL_MODEL_PATH}. Cannot run RL strategy.")
            self.model = None # Ensure model is None if file not found

    # ... (existing initialize_scaler placeholder) ...

    def _get_live_observation(self) -> Optional[np.ndarray]: # Return type hint added
        """Gets the current market state and formats it for the model.
        Requires matching normalization from training environment.
        """
        logging.debug("Getting live observation...")
        try:
            # Fetch data (adjust period/interval as needed)
            df_live = get_historical_data(settings.RL_ENV_SYMBOL, period="1d", interval="1m")
            if df_live is None or df_live.empty or len(df_live) < 15: # Need enough data for indicators (e.g., RSI 14)
                logging.warning("Could not fetch sufficient live data for observation.")
                return None

            # Calculate indicators
            rsi = calculate_rsi(df_live, window=14, column='Close')
            sma = calculate_sma(df_live, window=7, column='Close')
            if rsi is None or sma is None or rsi.empty or sma.empty:
                logging.warning("Failed to calculate live indicators.")
                return None

            latest_rsi = rsi.iloc[-1]
            latest_sma = sma.iloc[-1]
            latest_price = df_live['Close'].iloc[-1]

            # Calculate PnL
            current_pnl = 0.0
            if self.current_position == 1 and self.entry_price > 0:
                current_pnl = (latest_price - self.entry_price) / self.entry_price
            elif self.current_position == 1:
                logging.warning("Holding position but entry price is not valid.")

            # --- Crucial: Normalize features matching TradingEnv --- #
            # This needs to be implemented correctly, using the same scaler/method
            # Placeholder normalization (highly likely incorrect):
            price_norm = (latest_price - df_live['Close'].mean()) / df_live['Close'].std() if df_live['Close'].std() != 0 else 0
            rsi_norm = (latest_rsi - 50) / 50 # Simple scale to approx [-1, 1]
            sma_norm = (latest_sma - df_live['Close'].mean()) / df_live['Close'].std() if df_live['Close'].std() != 0 else 0
            pnl_norm = np.clip(current_pnl * 10, -1, 1) # Clip PnL, scale by 10
            # -------------------------------------------------------- #

            state = np.array([
                price_norm,
                rsi_norm,
                sma_norm,
                float(self.current_position),
                pnl_norm
            ], dtype=np.float32)

            if not np.all(np.isfinite(state)):
                logging.warning(f"Constructed state contains non-finite values: {state}")
                return None

            logging.debug(f"Live observation created: {state}")
            return state

        except Exception as e:
            logging.error(f"Error getting live observation: {e}", exc_info=True)
            return None

    def _trigger_conditional_fallback(self, reason: str) -> None:
        """Attempts to run the conditional strategy as a fallback."""
        logging.warning(f"Fallback Triggered: {reason}. Attempting conditional strategy check.")
        if not conditional_strategy_available:
            logging.error("Conditional strategy functions not available. Cannot execute fallback.")
            return

        try:
            # Check conditional entry conditions
            conditional_signal = check_entry_conditions(settings.CHALLENGE_SYMBOL)

            if conditional_signal and 'side' in conditional_signal:
                logging.info(f"Conditional Fallback Signal: {conditional_signal}")

                # Get necessary info for ordering
                balance_info = get_futures_account_balance()
                ticker_info = get_symbol_ticker(settings.CHALLENGE_SYMBOL)

                if not balance_info or not ticker_info or 'availableBalance' not in balance_info or 'price' not in ticker_info:
                     logging.error("Fallback Error: Failed to get balance or ticker info for conditional order.")
                     return

                total_equity = float(balance_info['availableBalance']) # Use available balance as approximation
                current_price = float(ticker_info['price'])

                # Calculate position size
                quantity = calculate_position_size(total_equity, current_price)
                if quantity <= 0:
                    logging.warning("Fallback: Calculated quantity is zero or less. Skipping conditional order.")
                    return

                # Execute conditional order
                logging.info(f"Fallback: Executing conditional {conditional_signal['side'].upper()} order, Qty: {quantity}")
                order_result = create_futures_order(
                    symbol=settings.CHALLENGE_SYMBOL,
                    side=conditional_signal['side'].upper(), # BUY or SELL
                    quantity=quantity
                )

                if order_result and order_result.get('orderId'):
                     logging.info(f"Fallback: Conditional order placed successfully. Order ID: {order_result.get('orderId')}")
                     # Log this trade via database
                     log_trade_to_db(
                         strategy='challenge_fallback', # Indicate it's a fallback trade
                         symbol=settings.CHALLENGE_SYMBOL,
                         side=conditional_signal['side'],
                         quantity=quantity,
                         entry_price=current_price, # Approximate entry price
                         status='open_fallback', # Custom status
                         reason=f"Fallback due to: {reason}. Cond: {conditional_signal.get('reason', 'N/A')}",
                         order_id=order_result.get('orderId')
                     )
                     if send_slack_notification:
                         send_slack_notification(
                            f"⚠️ Fallback Trade Executed ({conditional_signal['side']})",
                            f"Reason: {reason}\nSymbol: {settings.CHALLENGE_SYMBOL}, Qty: {quantity}\nCond. Reason: {conditional_signal.get('reason', 'N/A')}",
                            level="warning"
                        )
                else:
                    logging.error(f"Fallback: Conditional order placement failed. Result: {order_result}")
            else:
                logging.info("Fallback: No conditional entry signal found.")

        except Exception as e:
            logging.error(f"Error during conditional fallback execution: {e}", exc_info=True)
            if send_slack_notification:
                 send_slack_notification("🚨 Fallback Logic Error", f"Error executing fallback strategy: {e}", level="error")

    def run_step(self) -> None: # Return type hint added
        """Executes one step of the RL strategy, with fallback to conditional strategy."""
        if not self.model:
            self._trigger_conditional_fallback(reason="RL model not loaded")
            return # Stop RL execution for this step

        logging.info("--- RL Strategy Step --- ")
        observation = self._get_live_observation()
        if observation is None:
            self._trigger_conditional_fallback(reason="Failed to get valid RL observation")
            return

        action = -1 # Initialize action to an invalid value
        try:
            action_raw, _states = self.model.predict(observation, deterministic=True)
            action = int(action_raw)
        except Exception as e:
            logging.error(f"Error during RL model prediction: {e}", exc_info=True)
            self._trigger_conditional_fallback(reason=f"RL prediction error: {e}")
            return

        action_map = {0: "Hold", 1: "Buy", 2: "Sell"}
        action_str = action_map.get(action, f'Unknown({action})')
        logging.info(f"RL Agent Action: {action_str}")

        # --- Execute RL Action (if prediction was successful) ---
        if action in action_map: # Check if action is valid
            # Placeholder for price fetching and order execution
            current_price = 0
            try:
                ticker = get_symbol_ticker(settings.RL_ENV_SYMBOL)
                if ticker and 'price' in ticker:
                    current_price = float(ticker['price'])
                else:
                     logging.error("RL Action: Failed to get current price for execution.")
                     # Optionally trigger fallback here too if price fetch fails critically?
                     # self._trigger_conditional_fallback(reason="Failed to fetch price for RL action")
                     return # Skip RL action if price is unavailable
            except Exception as e:
                 logging.error(f"RL Action: Error fetching price: {e}")
                 return

            executed_trade = False
            trade_info: Dict[str, Any] = {
                'symbol': settings.RL_ENV_SYMBOL,
                'strategy': 'rl_challenge',
                'action_taken': action_str,
                'timestamp': datetime.now()
            }

            # RL Action Execution Logic (Simplified)
            if action == 1 and self.current_position == 0: # Buy
                logging.info(f"Executing RL BUY order for {settings.RL_ENV_SYMBOL} @ ~{current_price:.2f}")
                # order_result = create_futures_order(..., "BUY", ...) # TODO: Use RL-specific size?
                order_result = {"status": "FILLED", "orderId": f"rl_buy_{int(time.time())}"} # Placeholder
                if order_result and order_result.get('status') == 'FILLED':
                    self.current_position = 1
                    self.entry_price = current_price
                    executed_trade = True
                    trade_info.update({'side': 'buy', 'status': 'open', 'entry_price': self.entry_price, 'order_id': order_result.get('orderId')})
                    if send_slack_notification: send_slack_notification(f"🤖 RL Buy Executed", f"Symbol: {settings.RL_ENV_SYMBOL} at ${self.entry_price:.2f}")
                else:
                    logging.error("RL Buy order failed.")
                    trade_info.update({'side': 'buy', 'status': 'failed'})

            elif action == 2 and self.current_position == 1: # Sell
                logging.info(f"Executing RL SELL order for {settings.RL_ENV_SYMBOL} @ ~{current_price:.2f}")
                # order_result = close_position(...) # TODO
                order_result = {"status": "FILLED", "orderId": f"rl_sell_{int(time.time())}"} # Placeholder
                if order_result and order_result.get('status') == 'FILLED':
                    exit_price = current_price
                    pnl = (exit_price - self.entry_price) / self.entry_price if self.entry_price > 0 else 0
                    executed_trade = True
                    trade_info.update({'side': 'sell', 'status': 'closed', 'exit_price': exit_price, 'pnl_percent': pnl, 'order_id': order_result.get('orderId')})
                    self.current_position = 0
                    self.entry_price = 0.0
                    if send_slack_notification: send_slack_notification(f"🤖 RL Sell Executed", f"Symbol: {settings.RL_ENV_SYMBOL} PnL: {pnl:.2%}")
                else:
                    logging.error("RL Sell order failed.")
                    trade_info.update({'side': 'sell', 'status': 'failed'})
            else: # Hold or invalid action or already in position
                logging.info(f"RL Signal: {action_str}. No trade execution needed.")

            # Log RL trade action/status to DB
            if executed_trade or trade_info.get('status') == 'failed':
                try:
                    log_trade_to_db(
                        strategy=trade_info.pop('strategy'),
                        symbol=trade_info.pop('symbol'),
                        **{k: v for k, v in trade_info.items() if k not in ['action_taken', 'timestamp']} # Pass relevant fields only
                    )
                    logging.info("[DB] RL trade action/status logged.")
                except Exception as db_err:
                    logging.error(f"Failed to log RL trade action/status to DB: {db_err}")
        else:
             logging.warning(f"Invalid or unknown RL action received: {action}. Skipping execution.")
             # Optionally trigger fallback for unknown actions?
             # self._trigger_conditional_fallback(reason=f"Unknown RL action: {action}")

# --- Main Execution Logic (Example) --- #
def run_rl_trading_loop() -> None:
    """Runs the RL trading strategy in a loop (example)."""
    strategy = RLTradingStrategy()
    # No need to check strategy.model here, run_step handles it with fallback

    logging.info("Starting RL Trading Loop (Demo: 5 steps)")
    for i in range(5):
        try:
            strategy.run_step() # run_step now includes fallback logic
        except Exception as e:
            logging.critical(f"Unhandled exception in RL trading step {i+1}: {e}", exc_info=True)
            if send_slack_notification:
                send_slack_notification("🚨 CRITICAL RL Loop Error", f"Error in step {i+1}: {e}", level="error")
            break # Stop loop on critical error
        import time
        time.sleep(10) # Simple delay
    logging.info("Finished RL Trading Loop (Demo).") 