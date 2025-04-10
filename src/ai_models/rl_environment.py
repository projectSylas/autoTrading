import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging

# Project specific imports
from src.utils.common import get_historical_data, calculate_rsi, calculate_sma
# Add other necessary indicators if needed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TradingEnv(gym.Env):
    """A custom trading environment for Reinforcement Learning based on Gymnasium.

    State space: [normalized price, normalized RSI, normalized SMA, position_held, current_pnl]
    Action space: 0: Hold, 1: Buy, 2: Sell
    """
    metadata = {'render_modes': ['human', 'none'], 'render_fps': 4}

    def __init__(self, df: pd.DataFrame, initial_balance=10000, transaction_cost=0.001, rsi_period=14, sma_period=7):
        super().__init__()

        if df is None or df.empty or len(df) < max(rsi_period, sma_period):
             raise ValueError("DataFrame is empty or too short for indicator calculation.")

        self.df = df.copy()
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.rsi_period = rsi_period
        self.sma_period = sma_period

        # Calculate indicators
        self._add_indicators()
        # Drop rows with NaN indicators
        self.df.dropna(inplace=True)
        if self.df.empty:
            raise ValueError("DataFrame became empty after adding indicators and dropping NaNs.")

        self.current_step = 0
        self.max_steps = len(self.df) - 1 # Adjust for 0-based indexing

        # Define action space: 0: Hold, 1: Buy, 2: Sell
        self.action_space = spaces.Discrete(3)

        # Define state space (observation space)
        # Example: [price_norm, rsi_norm, sma_norm, position_held (0 or 1), current_pnl_norm]
        # Adjust bounds as needed, especially for pnl
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, 0, 0, 0, -np.inf]), # Price, RSI, SMA can be normalized differently
            high=np.array([np.inf, 100, np.inf, 1, np.inf]), # Normalize pnl carefully
            shape=(5,), dtype=np.float32
        )

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
        logging.info("Calculating indicators (RSI, SMA)...")
        # Use functions from common utils or implement here
        self.df['rsi'] = calculate_rsi(dataframe=self.df, window=self.rsi_period, column='Close')
        self.df['sma'] = calculate_sma(dataframe=self.df, window=self.sma_period, column='Close')
        # Add more indicators if needed (e.g., MACD, Bollinger Bands)
        # self.df['macd'] = ...

    def _get_observation(self) -> np.ndarray:
        """Returns the current state observation."""
        obs_data = self.df.iloc[self.current_step]
        current_price = obs_data['Close']
        current_pnl = 0.0
        if self.position_held == 1:
            current_pnl = (current_price - self.entry_price) / self.entry_price

        # Normalize features (Example: simple scaling - consider MinMaxScaler)
        # Normalization should be done carefully, potentially based on training data stats
        price_norm = current_price / self.df['Close'].iloc[0] # Normalize relative to start price
        rsi_norm = obs_data['rsi']
        sma_norm = obs_data['sma'] / self.df['Close'].iloc[0]
        pnl_norm = current_pnl * 10 # Scale PnL for reward signal strength (adjust as needed)

        state = np.array([
            price_norm,
            rsi_norm,
            sma_norm,
            float(self.position_held),
            pnl_norm
        ], dtype=np.float32)

        # Check if state is valid (finite)
        if not np.all(np.isfinite(state)):
             logging.warning(f"Invalid state encountered at step {self.current_step}: {state}. Replacing NaNs/Infs with 0.")
             state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)

        return state

    def _calculate_reward(self, current_price: float, next_price: float, action: int) -> float:
        """Calculates the reward for the taken action."""
        reward = 0.0

        # Reward based on PnL if holding a position
        if self.position_held == 1:
            # Reward for holding during price increase
            pnl_change = (next_price - current_price) / current_price
            reward += pnl_change * 10 # Scale reward

            # Penalty for selling at a loss (relative to entry)
            if action == 2: # Sell
                exit_pnl = (current_price - self.entry_price) / self.entry_price
                if exit_pnl < 0:
                    reward += exit_pnl * 5 # Stronger penalty for closing at loss

        # Penalty for taking invalid actions (e.g., buying when already long, selling when short/none)
        if (action == 1 and self.position_held == 1) or \
           (action == 2 and self.position_held == 0):
            reward -= 0.1 # Small penalty for invalid action attempt

        # Penalty for transaction costs
        if action == 1 or action == 2:
            reward -= self.transaction_cost * 2 # Penalize entry/exit cost

        # Consider adding Sharpe ratio or other risk-adjusted metrics to reward

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

        # --- Move to next step --- #
        self.current_step += 1
        if self.current_step > self.max_steps:
            terminated = True
            logging.info("Reached end of data.")
            # Finalize PnL if still holding position at the end
            if self.position_held == 1:
                 final_price = self.df.iloc[self.max_steps]['Close']
                 final_pnl = (final_price - self.entry_price) / self.entry_price
                 self.total_pnl += final_pnl
                 info['final_pnl_on_close'] = final_pnl

        # --- Calculate Reward --- #
        next_price = self.df.iloc[min(self.current_step, self.max_steps)]['Close']
        reward = self._calculate_reward(current_price, next_price, action)
        self.total_reward += reward

        # --- Get Next Observation --- #
        if not terminated:
            observation = self._get_observation()
        else:
            # Return last valid observation if terminated
            observation = self._get_observation() # Or a zero observation?

        info['total_pnl'] = self.total_pnl
        info['total_reward'] = self.total_reward

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