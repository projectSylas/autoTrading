import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging
import ta # Import ta library
from ta.volatility import AverageTrueRange, BollingerBands # Import ATR and BB
from sklearn.preprocessing import StandardScaler # Import StandardScaler
from typing import List, Tuple

# Project specific imports
from src.utils.common import get_historical_data, calculate_rsi, calculate_sma
# Add other necessary indicators if needed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TradingEnv(gym.Env):
    """A custom trading environment for Reinforcement Learning based on Gymnasium.

    State space: [price, rsi, sma, volume, macd, macd_signal, macd_hist, atr, bb_bbm, bb_bbh, bb_bbl, position_held, current_pnl, balance_norm]
    Action space: 0: Hold, 1: Buy, 2: Sell
    """
    metadata = {'render_modes': ['human', 'none'], 'render_fps': 4}

    def __init__(self, df: pd.DataFrame, initial_balance=10000, transaction_cost=0.001, rsi_period=14, sma_period=7,
                 macd_fast=12, macd_slow=26, macd_sign=9, atr_window=14, bb_window=20, bb_dev=2): # Add ATR and BB params
        super().__init__()

        # Ensure necessary columns are present
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_columns):
             missing = [col for col in required_columns if col not in df.columns]
             raise ValueError(f"DataFrame missing required columns: {missing}")

        # Initial length check before adding indicators - Include ATR and BB
        min_len_for_macd = macd_slow + macd_sign - 1 # Minimum length needed for MACD
        min_len_for_bb = bb_window # Minimum length needed for Bollinger Bands
        min_len_for_atr = atr_window # Minimum length needed for ATR
        min_len_required = max(rsi_period, sma_period, min_len_for_macd, min_len_for_bb, min_len_for_atr)
        if len(df) < min_len_required:
            raise ValueError(f"DataFrame is too short ({len(df)} rows) for indicator calculation (min required: {min_len_required}).")

        self.df = df.copy()
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.rsi_period = rsi_period
        self.sma_period = sma_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_sign = macd_sign
        self.atr_window = atr_window
        self.bb_window = bb_window
        self.bb_dev = bb_dev

        # Calculate indicators
        self._add_indicators()
        # Drop rows with NaN indicators (important after adding MACD)
        initial_rows = len(self.df)
        self.df.dropna(inplace=True)
        if self.df.empty:
            raise ValueError("DataFrame became empty after adding indicators and dropping NaNs.")
        dropped_rows = initial_rows - len(self.df)
        logging.info(f"Dropped {dropped_rows} rows with NaNs after indicator calculation.")

        # --- State Normalization --- 
        # Define columns to normalize (excluding position and scaled PnL)
        self.feature_columns = [
            'Close', 'rsi', 'sma', 'volume',
            'macd', 'macd_signal', 'macd_hist', # Existing features
            'atr', 'bb_bbm', 'bb_bbh', 'bb_bbl' # Added ATR and Bollinger Bands
            # Note: balance_norm is added dynamically, not normalized with scaler
        ]
        # Ensure lowercase 'close' if needed, yfinance usually provides 'Close'
        if 'close' in self.df.columns and 'Close' not in self.feature_columns:
            self.feature_columns[self.feature_columns.index('Close')] = 'close' # Adjust if necessary
            
        self.scaler = StandardScaler()
        # Fit scaler only on the features used in the observation, using the training data
        # Ensure all feature columns exist before fitting
        missing_feature_cols = [col for col in self.feature_columns if col not in self.df.columns]
        if missing_feature_cols:
             raise ValueError(f"Missing columns required for feature scaling: {missing_feature_cols}")
        self.scaler.fit(self.df[self.feature_columns])
        logging.info("StandardScaler fitted to training data features.")
        # --- End Normalization Setup ---

        self.current_step = 0
        self.max_steps = len(self.df) - 1 # Adjust for 0-based indexing

        # Define action space: 0: Hold, 1: Buy, 2: Sell
        self.action_space = spaces.Discrete(3)

        # Define observation space (normalized features + position + scaled PnL + balance_norm)
        # Normalized values generally center around 0 with std 1, but can vary.
        # Using -inf/inf bounds is safest initially.
        num_features = len(self.feature_columns)
        self.observation_space = spaces.Box(
            low=np.array([-np.inf] * num_features + [0, -np.inf, -np.inf]), # Normalized features + position + pnl_norm + balance_norm
            high=np.array([np.inf] * num_features + [1, np.inf, np.inf]),
            shape=(num_features + 3,), dtype=np.float32 # Updated shape for balance_norm
        )
        # Log final shape for confirmation
        logging.info(f"Observation space shape: {self.observation_space.shape}") 

        # Internal state
        self.balance = 0.0
        self.position_held = 0 # 0: No position, 1: Long position
        self.entry_price = 0.0
        self.total_pnl = 0.0
        self.total_reward = 0.0
        self.trade_history = [] # List to store trade details

        logging.info(f"TradingEnv initialized with {len(self.df)} steps.")

    def _add_indicators(self):
        """Calculate and add technical indicators to the DataFrame."""
        logging.info("Calculating indicators (RSI, SMA, MACD, Volume, ATR, Bollinger Bands)...")
        # RSI and SMA
        self.df['rsi'] = calculate_rsi(dataframe=self.df, window=self.rsi_period, column='Close')
        self.df['sma'] = calculate_sma(dataframe=self.df, window=self.sma_period, column='Close')
        # MACD using ta library
        macd = ta.trend.MACD(self.df['Close'], window_slow=self.macd_slow, window_fast=self.macd_fast, window_sign=self.macd_sign)
        self.df['macd'] = macd.macd()
        self.df['macd_signal'] = macd.macd_signal()
        self.df['macd_hist'] = macd.macd_diff()

        # ATR using ta library
        # Ensure High, Low, Close are present for ATR
        if not all(col in self.df.columns for col in ['High', 'Low', 'Close']):
             logging.warning("Missing High/Low/Close columns, cannot calculate ATR. Setting ATR to 0.")
             self.df['atr'] = 0.0
        else:
             self.df['atr'] = AverageTrueRange(high=self.df['High'], low=self.df['Low'], close=self.df['Close'], window=self.atr_window).average_true_range()

        # Bollinger Bands using ta library
        bbands = BollingerBands(close=self.df['Close'], window=self.bb_window, window_dev=self.bb_dev)
        self.df['bb_bbm'] = bbands.bollinger_mavg()   # Middle Band
        self.df['bb_bbh'] = bbands.bollinger_hband()   # Upper Band
        self.df['bb_bbl'] = bbands.bollinger_lband()   # Lower Band

        # Volume is assumed to be present from yfinance
        if 'Volume' not in self.df.columns:
             logging.warning("'Volume' column not found in DataFrame. State representation will be incomplete.")
             self.df['volume'] = 0 # Add dummy volume column if missing
        else:
            self.df['volume'] = self.df['Volume'] # Ensure lowercase column name if needed

    def _get_observation(self) -> np.ndarray:
        """Returns the current state observation, now normalized."""
        obs_data = self.df.iloc[self.current_step]
        current_price = obs_data['Close'] # Use the correct case as per dataframe
        current_pnl = 0.0
        if self.position_held == 1:
            # Avoid division by zero if entry_price is somehow 0
            if self.entry_price != 0:
                current_pnl = (current_price - self.entry_price) / self.entry_price
            else:
                 current_pnl = 0.0

        # Get feature values for normalization
        # Ensure all feature columns exist in obs_data
        missing_obs_cols = [col for col in self.feature_columns if col not in obs_data]
        if missing_obs_cols:
             logging.error(f"Missing feature columns in observation data at step {self.current_step}: {missing_obs_cols}. Using zeros.")
             # Create a zero array with the correct shape if data is missing
             feature_values = np.zeros((1, len(self.feature_columns)))
             # Try to fill existing ones if possible
             for i, col in enumerate(self.feature_columns):
                 if col in obs_data:
                     feature_values[0, i] = obs_data[col]
        else:
            feature_values = obs_data[self.feature_columns].values.reshape(1, -1) # Reshape for scaler

        # Normalize the features using the fitted scaler
        normalized_features = self.scaler.transform(feature_values).flatten()
        
        # PnL scaling (keep as is for now)
        pnl_norm = current_pnl * 10

        # Balance scaling
        # Normalize balance relative to the initial balance, scale by 10 like PnL
        # Ensure initial_balance is not zero to prevent division by zero
        if self.initial_balance > 0:
            balance_norm = (self.balance / self.initial_balance - 1.0) * 10
        else:
            balance_norm = 0.0 # Handle case where initial balance might be zero

        # Combine normalized features, position, scaled PnL, and scaled Balance
        state = np.concatenate((normalized_features, [float(self.position_held), pnl_norm, balance_norm]), dtype=np.float32)

        # Check if state is valid (finite)
        if not np.all(np.isfinite(state)):
             # Use feature_columns names for better debug message
             feature_dict = dict(zip(self.feature_columns, feature_values.flatten()))
             logging.warning(f"Invalid state encountered at step {self.current_step}. Raw features: {feature_dict}. Scaled: {normalized_features}. PnL: {current_pnl}. State: {state}. Replacing NaNs/Infs with 0.")
             state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
             # Ensure the shape is still correct after potential nan_to_num
             if state.shape[0] != self.observation_space.shape[0]:
                 logging.error(f"State shape mismatch after nan_to_num! Expected {self.observation_space.shape[0]}, got {state.shape[0]}. Filling with zeros.")
                 state = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)

        return state

    def _calculate_reward(self, current_price: float, next_price: float, action: int) -> float:
        """Calculates the reward for the taken action, incentivizing profitable sells."""
        reward = 0.0
        pnl_change = (next_price - current_price) / current_price

        if self.position_held == 1: # Currently holding a position
            # 1. Reward/Penalty for Holding (Adjusted Multiplier)
            # Smaller reward for just holding during price increase
            reward += pnl_change * 10 # Reduced multiplier (e.g., from 100 to 10)

            if action == 2: # Sell action taken
                # Calculate PnL based on entry price for this specific trade
                # Ensure entry_price is valid before division
                if self.entry_price > 0:
                    # Use the actual exit price considering transaction cost from step info if available
                    # Re-calculating here for clarity based on current price before action logic executes fully
                    exit_price_simulated = current_price * (1 - self.transaction_cost)
                    exit_pnl = (exit_price_simulated - self.entry_price) / self.entry_price
                else:
                    exit_pnl = 0 # Avoid division by zero if entry price is invalid
                    logging.warning(f"Step {self.current_step}: Sell action (2) chosen but entry_price is 0. Cannot calculate PnL reward.")

                # 2. Large Reward for Profitable Sell
                if exit_pnl > 0:
                    reward += exit_pnl * 150  # Significant reward for closing at profit (adjust multiplier as needed)
                    # Alternatively, add a fixed large bonus: reward += 5
                    logging.debug(f"Step {self.current_step}: Rewarding profitable sell. PnL: {exit_pnl:.4f}, Reward Add: {exit_pnl * 150:.4f}")
                # 3. Penalty for Selling at a Loss (Increased Multiplier)
                else: # exit_pnl <= 0
                    # Increased penalty for closing at loss
                    reward += exit_pnl * 30 # Penalty scales with the loss size (Increased from 5 to 30)
                    logging.debug(f"Step {self.current_step}: Penalizing sell at loss. PnL: {exit_pnl:.4f}, Reward Add: {exit_pnl * 30:.4f}")

        elif self.position_held == 0: # Not holding a position
            # Optional: Minor penalty for holding cash if price is rising?
            # if action == 0 and pnl_change > 0:
            #     reward -= pnl_change * 1 # Small penalty for missing out
            pass # Primarily focus on rewards/penalties related to active positions and trades

        # 4. Penalty for Transaction Costs REMOVED again to encourage trading
        # if action == 1 or action == 2:
        #      # Penalty should reflect the cost occurred in the step function
        #      # We can directly subtract a small constant or use self.transaction_cost
        #      # reward -= 0.01 # Example of very small penalty
        #      pass

        # Ensure reward is finite
        if not np.isfinite(reward):
            logging.warning(f"Step {self.current_step}: Non-finite reward calculated ({reward}). Resetting to 0.")
            reward = 0.0

        return reward

    def reset(self, seed=None, options=None):
        """Resets the environment to the initial state."""
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.position_held = 0
        self.entry_price = 0.0
        self.total_pnl = 0.0
        self.total_reward = 0.0
        self.trade_history = []

        observation = self._get_observation()
        info = {}
        logging.debug(f"Environment reset. Initial observation: {observation}")
        return observation, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Executes one time step within the environment."""
        current_price = self.df.iloc[self.current_step]['Close']
        terminated = False # Episode ends naturally (e.g., out of data)
        truncated = False # Episode ends unnaturally (e.g., time limit)
        info = {'step': self.current_step, 'price': current_price, 'action': action}

        # --- Action Logic --- #
        if action == 1: # Buy
            if self.position_held == 0:
                self.position_held = 1
                self.entry_price = current_price * (1 + self.transaction_cost) # Account for cost
                self.balance -= current_price # Simplified balance update
                info['trade'] = 'buy'
                info['entry_price'] = self.entry_price
                self.trade_history.append({'step': self.current_step, 'type': 'buy', 'price': self.entry_price})
                logging.debug(f"Step {self.current_step}: BUY at {current_price:.2f}")
            else:
                 info['trade'] = 'hold_invalid_buy'
                 logging.debug(f"Step {self.current_step}: Invalid BUY attempt (already holding).")

        elif action == 2: # Sell
            if self.position_held == 1:
                exit_price = current_price * (1 - self.transaction_cost) # Account for cost
                pnl = (exit_price - self.entry_price) / self.entry_price
                self.total_pnl += pnl
                self.balance += current_price # Simplified balance update
                self.position_held = 0
                info['trade'] = 'sell'
                info['exit_price'] = exit_price
                info['pnl'] = pnl
                self.trade_history.append({'step': self.current_step, 'type': 'sell', 'price': exit_price, 'pnl': pnl})
                logging.debug(f"Step {self.current_step}: SELL at {current_price:.2f}, PnL: {pnl:.4f}")
                self.entry_price = 0.0
            else:
                info['trade'] = 'hold_invalid_sell'
                logging.debug(f"Step {self.current_step}: Invalid SELL attempt (not holding).")

        else: # Hold (action == 0)
            info['trade'] = 'hold'
            logging.debug(f"Step {self.current_step}: HOLD at {current_price:.2f}")

        # --- Calculate Reward --- #
        next_price = self.df.iloc[min(self.current_step, self.max_steps)]['Close']
        reward = self._calculate_reward(current_price, next_price, action)
        self.total_reward += reward

        # --- Increment Step & Check Termination --- #
        self.current_step += 1
        terminated = self.current_step > self.max_steps

        # --- Get Next Observation --- #
        if not terminated:
            # Only get observation if the episode is not yet done
            observation = self._get_observation() # Uses the incremented current_step
        else:
            # If terminated, SB3 ignores the returned observation.
            # Return a dummy observation (e.g., zeros) matching the space.
            observation = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
            logging.info(f"Reached end of data (step {self.current_step}). Returning zero observation.")

        # --- Final PnL Calculation (only if terminated) --- #
        if terminated:
            logging.info("Processing final step PnL calculation.") # Add log
            # Finalize PnL if still holding position at the end
            if self.position_held == 1:
                 # Use the last valid price (at max_steps) for final PnL calc
                 try:
                     final_price = self.df.iloc[self.max_steps]['Close']
                     final_pnl = (final_price - self.entry_price) / self.entry_price
                     self.total_pnl += final_pnl
                     info['final_pnl_on_close'] = final_pnl
                     logging.info(f"Final PnL calculated: {final_pnl:.4f} using price {final_price:.2f}")
                 except IndexError:
                     logging.error(f"IndexError during final PnL calculation at index {self.max_steps}. DF length: {len(self.df)}")
                 except Exception as e:
                     logging.error(f"Error during final PnL calculation: {e}")

        info['total_pnl'] = self.total_pnl
        info['total_reward'] = self.total_reward

        # truncated is False as we handle termination by data end
        truncated = False

        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        """Renders the environment (e.g., plots, prints)."""
        if mode == 'human':
            print(f"Step: {self.current_step}, Price: {self.df.iloc[self.current_step-1]['Close']:.2f}, "
                  f"Position: {self.position_held}, Balance: {self.balance:.2f}, Total PnL: {self.total_pnl:.4f}, "
                  f"Total Reward: {self.total_reward:.4f}")
        elif mode == 'none':
            pass # No rendering

    def close(self):
        """Performs any necessary cleanup."""
        logging.info("Closing TradingEnv.")
        # Close plots, release resources etc.

# --- Example Usage --- #
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    # Load sample data
    try:
        # Use a relatively short period for quick testing
        btc_data = get_historical_data("BTC-USD", period="1mo", interval="1h")
    except Exception as e:
        logging.error(f"Failed to load data for environment test: {e}")
        btc_data = None

    if btc_data is not None and not btc_data.empty:
        env = TradingEnv(btc_data)
        # Check environment compatibility with stable-baselines3
        try:
            from stable_baselines3.common.env_checker import check_env
            check_env(env)
            logging.info("Environment check passed!")
        except ImportError:
            logging.warning("stable-baselines3 not found, skipping env check.")
        except Exception as check_err:
             logging.error(f"Environment check failed: {check_err}")

        # Simple random agent test
        logging.info("\nTesting environment with random actions...")
        obs, _ = env.reset()
        n_steps = 20
        total_reward = 0
        for step in range(n_steps):
            action = env.action_space.sample() # Random action
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            env.render(mode='human')
            if terminated or truncated:
                logging.info("Episode finished.")
                break
        logging.info(f"Random agent test finished after {step+1} steps. Total reward: {total_reward:.4f}")
        env.close()
    else:
         logging.error("Could not run environment test due to data loading failure.") 