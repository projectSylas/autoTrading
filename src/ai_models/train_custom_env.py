import logging
import os
import pandas as pd
import sys

# --- Project Structure Adjustment ---
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(SRC_DIR)) # Assumes this file is in src/ai_models/
if PROJECT_ROOT not in sys.path:
     sys.path.append(PROJECT_ROOT)

# --- Import Custom Environment and Utils ---
try:
    from src.ai_models.rl_environment import TradingEnv
    from src.utils.common import get_historical_data # Assuming data loading utility exists
    # Import SB3 components
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
    SB3_AVAILABLE = True
except ImportError as e:
    SB3_AVAILABLE = False
    IMPORT_ERROR = e
    # Define dummy classes if SB3 is not available
    class TradingEnv:
        def __init__(self, *args, **kwargs): pass
    class PPO:
        def __init__(self, *args, **kwargs): pass
        def learn(self, *args, **kwargs): pass
        def save(self, *args, **kwargs): pass
    def make_vec_env(*args, **kwargs): return None
    class EvalCallback: pass
    class StopTrainingOnRewardThreshold: pass
    def get_historical_data(*args, **kwargs): return pd.DataFrame()

# --- Load Settings --- 
try:
    # Import the settings instance directly
    from src.config.settings import settings
    if settings is None:
        raise ImportError("Settings object is None after import.")
except ImportError:
    settings = None
    logging.getLogger(__name__).warning("Could not import settings instance. Using default paths and parameters.")

# --- Setup Logging --- 
logger = logging.getLogger(__name__)
# Ensure basicConfig format string is standard Python format string
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Training Configuration --- 
DEFAULT_TICKER = "AAPL" # Example ticker
DEFAULT_START_DATE = "2020-01-01"
DEFAULT_END_DATE = "2023-12-31"
DEFAULT_TIMESTEPS = 500000 # Further increased training duration for deeper learning
DEFAULT_RSI_PERIOD = 14
DEFAULT_SMA_PERIOD = 7
DEFAULT_MACD_FAST = 12 # Default MACD fast period
DEFAULT_MACD_SLOW = 26 # Default MACD slow period
DEFAULT_MACD_SIGN = 9  # Default MACD signal period
DEFAULT_ATR_WINDOW = 14 # Default ATR window
DEFAULT_BB_WINDOW = 20  # Default Bollinger Bands window
DEFAULT_BB_DEV = 2     # Default Bollinger Bands std deviation
DEFAULT_INITIAL_BALANCE = 10000
# Use the imported settings instance for paths
DEFAULT_MODEL_DIR = os.path.join(settings.MODEL_SAVE_DIR if settings else "models", "custom_env")
DEFAULT_LOG_DIR = os.path.join(settings.LOG_DIR if settings else "logs", "sb3_custom_env")
DEFAULT_LOG_PATH = "logs/custom_env/"
DEFAULT_MODEL_PATH = "models/custom_env/"
DEFAULT_LEARNING_RATE = 0.0003
DEFAULT_BATCH_SIZE = 64

# Ensure directories exist
os.makedirs(DEFAULT_MODEL_DIR, exist_ok=True)
os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)

def train_custom_environment_agent(
    ticker: str = DEFAULT_TICKER,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    rsi_period: int = DEFAULT_RSI_PERIOD,
    sma_period: int = DEFAULT_SMA_PERIOD,
    initial_balance: float = DEFAULT_INITIAL_BALANCE,
    total_timesteps: int = DEFAULT_TIMESTEPS,
    model_save_path: str | None = None,
    log_path: str = DEFAULT_LOG_DIR
) -> str | None:
    """
    Trains a Stable-Baselines3 agent using the custom TradingEnv.
    """
    if not SB3_AVAILABLE:
        logger.error(f"Stable-Baselines3 or TradingEnv not available due to import error: {IMPORT_ERROR}")
        return None
        
    logger.info(f"Starting training for {ticker} using custom TradingEnv ({start_date} to {end_date})")
    logger.info(f" > Indicators: RSI({rsi_period}), SMA({sma_period}), MACD({DEFAULT_MACD_FAST},{DEFAULT_MACD_SLOW},{DEFAULT_MACD_SIGN}), ATR({DEFAULT_ATR_WINDOW}), BB({DEFAULT_BB_WINDOW},{DEFAULT_BB_DEV})")
    logger.info(f" > State Space: [price, rsi, sma, volume, macd, macd_signal, macd_hist, atr, bb_bbm, bb_bbh, bb_bbl, position_held, pnl_norm]")
    logger.info(f" > Total Timesteps: {total_timesteps}")

    # --- 1. Load Data ---
    # Use standard interval string '1h' and appropriate period for get_historical_data
    # Yahoo finance limits 1h data to the last 730 days
    train_period = "729d" # Reduced period to comply with yfinance limit for 1h data
    train_interval = "1h" # Match backtesting interval
    logger.info(f"Step 1: Loading historical data for period: {train_period}, interval: {train_interval}...")
    df = get_historical_data(ticker, period=train_period, interval=train_interval)
    if df is None or df.empty:
        logger.error(f"Failed to load data for {ticker}.")
        return None
    logger.info(f"Data loaded successfully. Shape: {df.shape}")

    # --- 2. Create Environment ---
    logger.info("Step 2: Creating custom TradingEnv environment...")
    try:
        # Use make_vec_env for SB3 compatibility (can specify n_envs=1)
        env = make_vec_env(
            lambda: TradingEnv(
                df=df,
                initial_balance=initial_balance,
                rsi_period=rsi_period,
                sma_period=sma_period,
                macd_fast=DEFAULT_MACD_FAST,
                macd_slow=DEFAULT_MACD_SLOW,
                macd_sign=DEFAULT_MACD_SIGN,
                atr_window=DEFAULT_ATR_WINDOW,
                bb_window=DEFAULT_BB_WINDOW,
                bb_dev=DEFAULT_BB_DEV
                # Add other env params if needed (transaction_cost etc.)
            ),
            n_envs=1
        )
        logger.info("TradingEnv created successfully.")
    except ValueError as ve:
        logger.error(f"Error creating TradingEnv: {ve}. Check data length and indicator periods.")
        return None
    except Exception as e:
        logger.error(f"Unexpected error creating TradingEnv: {e}", exc_info=True)
        return None

    # --- 3. Setup Model (PPO example) ---
    logger.info("Step 3: Setting up PPO model...")
    # PPO hyperparameters can be tuned - Adjusting defaults to encourage exploration
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=log_path,
        learning_rate=DEFAULT_LEARNING_RATE,
        ent_coef=0.01 # Increased entropy coefficient for exploration
    )
    logger.info(f"PPO model initialized with custom hyperparameters: lr={DEFAULT_LEARNING_RATE}, ent_coef={0.01}")

    # --- 4. Training ---
    logger.info(f"Step 4: Starting training for {total_timesteps} timesteps...")
    # Optional callbacks for evaluation or early stopping
    # eval_callback = EvalCallback(env, best_model_save_path=DEFAULT_MODEL_DIR, log_path=log_path, eval_freq=5000)
    # stop_callback = StopTrainingOnRewardThreshold(reward_threshold=YOUR_TARGET_REWARD, verbose=1)
    try:
        model.learn(total_timesteps=total_timesteps, log_interval=10) #, callback=[eval_callback])
        logger.info("Training finished.")
    except Exception as e:
        logger.error(f"Error during model training: {e}", exc_info=True)
        return None

    # --- 5. Save Model ---
    if model_save_path is None:
        model_save_path = os.path.join(DEFAULT_MODEL_DIR, f"ppo_{ticker}_custom_env_{total_timesteps}.zip")
        
    logger.info(f"Step 5: Saving trained model to {model_save_path}")
    try:
        model.save(model_save_path)
        logger.info("Model saved successfully.")
        return model_save_path
    except Exception as e:
        logger.error(f"Error saving model: {e}", exc_info=True)
        return None

if __name__ == "__main__":
    logger.info("=======================================================")
    logger.info(" Starting Custom TradingEnv Agent Training Script  ")
    logger.info("=======================================================")
    
    # Example: Train PPO for Apple with default settings
    saved_path = train_custom_environment_agent(ticker="AAPL")
    
    if saved_path:
        # Use standard f-string formatting
        print(f"\nTraining complete. Model saved to: {saved_path}")
    else:
        print("\nTraining failed. Check logs for details.") 