from typing import Optional, List, Dict, Any # Added for type hints
import pandas as pd
import logging
import os
from stable_baselines3 import PPO, A2C, DDPG, SAC, TD3 # Example algorithms - Added SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm # For type hinting model
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv # For type hinting env
from stable_baselines3.common.callbacks import BaseCallback # For type hinting callbacks
import gymnasium as gym

# Project specific imports
from src.config import settings
from src.utils.common import get_historical_data
from src.ai_models.rl_environment import TradingEnv # Import custom environment

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Mapping from string name to SB3 algorithm class
SB3_ALGORITHMS: Dict[str, type[BaseAlgorithm]] = {
    "PPO": PPO,
    "A2C": A2C,
    "DDPG": DDPG,
    "SAC": SAC,
    "TD3": TD3,
    # Add other SB3 algorithms as needed
}

def train_rl_model(data_period: str = "6mo", data_interval: str = "1h", save_model: bool = True) -> None:
    """Trains the Reinforcement Learning trading agent.

    Args:
        data_period (str): Period string for fetching historical data (e.g., "6mo", "1y").
        data_interval (str): Interval string for fetching historical data (e.g., "1h", "1d").
        save_model (bool): Whether to save the trained model.
    """
    logging.info("===== 🤖 RL Agent Training Start =====")

    # 1. Load Data
    df_train: Optional[pd.DataFrame] = None
    try:
        logging.info(f"Loading training data for {settings.RL_ENV_SYMBOL} ({data_period} {data_interval})...")
        df_train = get_historical_data(settings.RL_ENV_SYMBOL, period=data_period, interval=data_interval)
        if df_train is None or df_train.empty:
            logging.error("Failed to load training data. Aborting training.")
            return
    except Exception as e:
        logging.error(f"Error loading training data: {e}")
        return

    # 2. Create and Vectorize Environment
    env: Optional[TradingEnv | VecEnv] = None # Can be single or vectorized
    try:
        # Single environment
        env = TradingEnv(df_train)
        # Optional: Vectorize
        # n_envs = 4
        # env = make_vec_env(lambda: TradingEnv(df_train), n_envs=n_envs)
        logging.info("Trading environment created.")
    except ValueError as ve:
         logging.error(f"Failed to create TradingEnv: {ve}. Check data length and indicators.")
         return
    except Exception as e:
        logging.error(f"Error creating trading environment: {e}")
        return

    # 3. Select Algorithm and Initialize Model
    AlgorithmClass: Optional[type[BaseAlgorithm]] = SB3_ALGORITHMS.get(settings.RL_ALGORITHM)
    if AlgorithmClass is None:
        logging.error(f"Unsupported RL algorithm: {settings.RL_ALGORITHM}. Supported: {list(SB3_ALGORITHMS.keys())}")
        return

    model_params: Dict[str, Any] = {
        "policy": settings.RL_POLICY,
        "env": env,
        "verbose": 1,
        "learning_rate": settings.RL_LEARNING_RATE,
        # batch_size is handled specifically for PPO below
        "tensorboard_log": "./rl_tensorboard_logs/",
    }

    if settings.RL_ALGORITHM == "PPO":
        model_params["n_steps"] = 2048 # Example value, might need tuning
        model_params["batch_size"] = settings.RL_BATCH_SIZE # PPO uses this for minibatch size
    elif settings.RL_ALGORITHM in ["SAC", "TD3", "DDPG"]:
        # These often use buffer_size and batch_size differently
        model_params["batch_size"] = settings.RL_BATCH_SIZE
        # model_params["buffer_size"] = 1_000_000 # Example buffer size
        # Add other specific params like tau, gradient_steps etc. if needed
    else: # A2C
         model_params["n_steps"] = 5 # A2C default
         # A2C doesn't use batch_size in the same way

    model: Optional[BaseAlgorithm] = None
    try:
        logging.info(f"Initializing {settings.RL_ALGORITHM} model with policy {settings.RL_POLICY}...")
        model = AlgorithmClass(**model_params)
    except Exception as e:
        logging.error(f"Error initializing RL model: {e}")
        return

    # 4. Define Callbacks (Optional)
    callback_list: Optional[List[BaseCallback]] = None # Initialize as None
    # Add specific callbacks here if needed (EvalCallback, etc.)

    # 5. Train the Model
    try:
        logging.info(f"Training model for {settings.RL_TRAIN_TIMESTEPS} timesteps...")
        model.learn(
            total_timesteps=settings.RL_TRAIN_TIMESTEPS,
            callback=callback_list,
            progress_bar=True
        )
        logging.info("Training finished.")

        # 6. Save the Trained Model
        if save_model:
            os.makedirs(os.path.dirname(settings.RL_MODEL_PATH), exist_ok=True)
            model.save(settings.RL_MODEL_PATH)
            logging.info(f"Trained RL model saved to {settings.RL_MODEL_PATH}")

    except Exception as e:
        logging.error(f"Error during RL model training: {e}", exc_info=True)
    finally:
        # Close the environment(s)
        if env:
            env.close()

    logging.info("===== 🤖 RL Agent Training End =====")

# --- Example Execution --- #
if __name__ == "__main__":
    train_rl_model(data_period="1y", data_interval="1h") 