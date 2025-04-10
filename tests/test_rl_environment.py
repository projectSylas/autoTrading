import pytest
import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import numpy.testing as npt

# Import the class to test
# Assuming rl_environment is structured correctly
try:
    from src.ai_models.rl_environment import TradingEnv
    # Import necessary functions if env uses them internally during init/reset
    # from src.utils.common import calculate_rsi, calculate_sma
except ImportError as e:
     pytest.skip(f"Skipping RL environment tests, failed to import: {e}", allow_module_level=True)

# --- Fixtures --- #
@pytest.fixture
def valid_df() -> pd.DataFrame:
    """Creates a valid DataFrame for initializing the environment."""
    dates = pd.date_range(start='2023-01-01', periods=50, freq='h') # Longer data for indicators
    data = {'Close': np.linspace(100, 150, 50) + np.random.normal(0, 2, 50)}
    df = pd.DataFrame(data, index=dates)
    # Add required columns if TradingEnv expects them (Open, High, Low, Volume)
    df['Open'] = df['Close'] - np.random.uniform(0, 1, 50)
    df['High'] = df['Close'] + np.random.uniform(0, 1, 50)
    df['Low'] = df['Close'] - np.random.uniform(1, 2, 50)
    df['Volume'] = np.random.randint(1000, 5000, 50)
    return df

@pytest.fixture
def short_df(valid_df: pd.DataFrame) -> pd.DataFrame:
    """Creates a DataFrame too short for default indicators."""
    # Default RSI=14, SMA=7. Need at least 14 non-NaN rows after indicator calculation.
    return valid_df.head(10)

@pytest.fixture
def env_params(valid_df: pd.DataFrame) -> dict:
    """Default parameters for TradingEnv initialization."""
    return {
        'df': valid_df,
        'initial_balance': 10000,
        'transaction_cost': 0.001,
        'rsi_period': 14,
        'sma_period': 7
    }

@pytest.fixture
def initialized_env(env_params) -> TradingEnv:
    """Provides an initialized TradingEnv instance."""
    env = TradingEnv(**env_params)
    env.reset() # Start at step 0
    return env

# --- Tests for __init__ --- #

def test_env_initialization(env_params):
    """Test successful environment initialization."""
    env = TradingEnv(**env_params)
    assert isinstance(env, gym.Env)
    # Check spaces
    assert isinstance(env.action_space, Discrete)
    assert env.action_space.n == 3
    assert isinstance(env.observation_space, Box)
    assert env.observation_space.shape == (5,) # Based on current state def
    assert env.observation_space.dtype == np.float32
    # Check if indicators were added and NaNs dropped
    assert 'rsi' in env.df.columns
    assert 'sma' in env.df.columns
    assert env.df.isnull().sum().sum() == 0 # No NaNs left
    assert env.max_steps == len(env.df) - 1

def test_env_init_short_dataframe(short_df):
    """Test initialization failure with short DataFrame."""
    params = {'df': short_df}
    # Depending on where the check happens (init vs _add_indicators), the error might differ
    with pytest.raises(ValueError, match="too short|empty after adding indicators"):
        TradingEnv(**params)

def test_env_init_empty_dataframe():
    """Test initialization failure with empty DataFrame."""
    params = {'df': pd.DataFrame()}
    with pytest.raises(ValueError, match="DataFrame is empty"):
        TradingEnv(**params)

# --- Tests for reset --- #

def test_env_reset(env_params):
    """Test the reset method."""
    env = TradingEnv(**env_params)
    # Take a few steps to change the state
    env.reset()
    env.step(1) # Buy
    env.step(0) # Hold

    # Now reset
    observation, info = env.reset()

    # Check observation shape and type
    assert isinstance(observation, np.ndarray)
    assert observation.shape == env.observation_space.shape
    assert observation.dtype == env.observation_space.dtype

    # Check if internal states are reset
    assert env.current_step == 0
    assert env.balance == env.initial_balance
    assert env.position_held == 0
    assert env.entry_price == 0.0
    assert env.total_pnl == 0.0
    assert env.total_reward == 0.0
    assert len(env.trade_history) == 0

    # Check if the returned observation matches the expected initial observation
    # This requires calculating the expected first observation manually or via _get_observation
    expected_initial_obs = env._get_observation() # Get obs at step 0 after reset
    np.testing.assert_allclose(observation, expected_initial_obs, rtol=1e-6)

    assert isinstance(info, dict)

# --- Tests for _get_observation (more specific tests needed) --- #
# TODO: Add tests specifically for _get_observation normalization logic

def test_get_observation_structure(initialized_env: TradingEnv):
    """Test the basic structure and finiteness of the observation."""
    obs = initialized_env._get_observation()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == initialized_env.observation_space.shape
    assert obs.dtype == initialized_env.observation_space.dtype
    assert np.all(np.isfinite(obs)) # Ensure no NaN/Inf

# --- Tests for step --- #

def test_step_action_buy(initialized_env: TradingEnv):
    """Test the state transition and info dict after a BUY action."""
    env = initialized_env
    initial_step = env.current_step
    initial_balance = env.balance
    current_price = env.df.iloc[initial_step]['Close']

    observation, reward, terminated, truncated, info = env.step(1) # Action 1: Buy

    assert env.current_step == initial_step + 1
    assert env.position_held == 1 # Position should be opened
    expected_entry_price = current_price * (1 + env.transaction_cost)
    assert np.isclose(env.entry_price, expected_entry_price)
    # Balance update depends on implementation (simple subtraction used here)
    assert np.isclose(env.balance, initial_balance - current_price)

    # Check info dict
    assert info['step'] == initial_step
    assert info['price'] == current_price
    assert info['action'] == 1
    assert info['trade'] == 'buy'
    assert np.isclose(info['entry_price'], expected_entry_price)

    # Check returned values
    assert isinstance(observation, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)

def test_step_action_sell(initialized_env: TradingEnv):
    """Test the state transition and info dict after a SELL action when holding."""
    env = initialized_env
    # First, buy to hold a position
    env.step(1)
    initial_step = env.current_step
    initial_balance = env.balance
    entry_price = env.entry_price
    current_price = env.df.iloc[initial_step]['Close']

    observation, reward, terminated, truncated, info = env.step(2) # Action 2: Sell

    assert env.current_step == initial_step + 1
    assert env.position_held == 0 # Position should be closed
    assert env.entry_price == 0.0 # Entry price reset
    # Balance update
    assert np.isclose(env.balance, initial_balance + current_price)

    # Check info dict
    expected_exit_price = current_price * (1 - env.transaction_cost)
    expected_pnl = (expected_exit_price - entry_price) / entry_price
    assert info['step'] == initial_step
    assert info['price'] == current_price
    assert info['action'] == 2
    assert info['trade'] == 'sell'
    assert np.isclose(info['exit_price'], expected_exit_price)
    assert np.isclose(info['pnl'], expected_pnl)

def test_step_action_hold(initialized_env: TradingEnv):
    """Test the state transition after a HOLD action."""
    env = initialized_env
    initial_step = env.current_step
    initial_balance = env.balance
    initial_position = env.position_held
    initial_entry_price = env.entry_price

    observation, reward, terminated, truncated, info = env.step(0) # Action 0: Hold

    assert env.current_step == initial_step + 1
    assert env.position_held == initial_position # Position unchanged
    assert env.entry_price == initial_entry_price # Entry price unchanged
    assert env.balance == initial_balance # Balance unchanged by hold
    assert info['trade'] == 'hold'

def test_step_invalid_actions(initialized_env: TradingEnv):
    """Test behavior for invalid actions (buy when holding, sell when not)."""
    env = initialized_env
    # Test invalid sell (not holding)
    obs1, reward1, _, _, info1 = env.step(2) # Sell when not holding
    assert env.position_held == 0
    assert info1['trade'] == 'hold_invalid_sell'
    assert reward1 < 0 # Should be penalized

    # Buy first
    env.step(1)
    initial_pos = env.position_held
    initial_entry = env.entry_price
    # Test invalid buy (already holding)
    obs2, reward2, _, _, info2 = env.step(1) # Buy again when holding
    assert env.position_held == initial_pos # Should remain holding
    assert env.entry_price == initial_entry # Entry price unchanged
    assert info2['trade'] == 'hold_invalid_buy'
    assert reward2 < 0 # Should be penalized

def test_step_termination(env_params):
    """Test if the environment terminates at the end of data."""
    # Use a shorter dataframe for faster testing
    short_df_term = env_params['df'].head(20) # e.g., 20 steps
    params = {**env_params, 'df': short_df_term}
    env = TradingEnv(**params)
    obs, info = env.reset()
    terminated = False
    steps_taken = 0
    while not terminated:
        obs, reward, terminated, truncated, info = env.step(0) # Just hold
        steps_taken += 1
        if steps_taken > env.max_steps + 5: # Safety break
            pytest.fail("Environment did not terminate as expected.")

    assert terminated is True
    assert env.current_step == env.max_steps + 1

# --- Tests for _calculate_reward --- #
# Reward calculation is complex, test key scenarios

def test_calculate_reward_hold_profit(initialized_env: TradingEnv):
    """Test reward when holding a position and price increases."""
    env = initialized_env
    env.step(1) # Buy at price p0
    p0 = env.entry_price / (1 + env.transaction_cost) # Approx entry price before cost
    p1 = p0 * 1.01 # Price increases
    reward = env._calculate_reward(current_price=p0, next_price=p1, action=0) # Hold action
    assert reward > 0 # Expect positive reward for price increase while holding

def test_calculate_reward_hold_loss(initialized_env: TradingEnv):
    """Test reward when holding a position and price decreases."""
    env = initialized_env
    env.step(1) # Buy at price p0
    p0 = env.entry_price / (1 + env.transaction_cost)
    p1 = p0 * 0.99 # Price decreases
    reward = env._calculate_reward(current_price=p0, next_price=p1, action=0) # Hold action
    assert reward < 0 # Expect negative reward

def test_calculate_reward_sell_profit(initialized_env: TradingEnv):
    """Test reward when selling a profitable position."""
    env = initialized_env
    env.step(1) # Buy at p0
    p0 = env.entry_price / (1 + env.transaction_cost)
    p1 = p0 * 1.02 # Sell price (higher than entry)
    env.position_held = 1 # Manually set position for reward calc logic
    env.entry_price = p0 * (1 + env.transaction_cost) # Ensure entry price is set
    reward = env._calculate_reward(current_price=p1, next_price=p1*1.01, action=2) # Sell action
    # Reward includes holding gain (p1-p0) - transaction cost penalty
    # Exact value depends on scaling factors in _calculate_reward
    assert reward > - (env.transaction_cost * 2) # Should be positive excluding transaction cost penalty

def test_calculate_reward_sell_loss(initialized_env: TradingEnv):
    """Test reward (penalty) when selling a losing position."""
    env = initialized_env
    env.step(1) # Buy at p0
    p0 = env.entry_price / (1 + env.transaction_cost)
    p1 = p0 * 0.98 # Sell price (lower than entry)
    env.position_held = 1
    env.entry_price = p0 * (1 + env.transaction_cost)
    reward = env._calculate_reward(current_price=p1, next_price=p1*0.99, action=2) # Sell action
    # Reward includes holding loss (p1-p0) + loss penalty + transaction cost penalty
    assert reward < 0 # Expect significant negative reward

def test_calculate_reward_invalid_action_penalty(initialized_env: TradingEnv):
    """Test penalty for invalid actions."""
    env = initialized_env
    # Invalid sell
    reward_invalid_sell = env._calculate_reward(current_price=100, next_price=101, action=2)
    assert reward_invalid_sell < 0
    # Buy first
    env.step(1)
    # Invalid buy
    reward_invalid_buy = env._calculate_reward(current_price=100, next_price=101, action=1)
    assert reward_invalid_buy < 0 