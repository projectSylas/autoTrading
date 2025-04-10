import pandas as pd
import logging
import os
from stable_baselines3 import PPO, A2C, DDPG # Example algorithms
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import gymnasium as gym
from typing import Optional, List, Dict, Any

# Project specific imports
from src.config import settings
from src.utils.common import get_historical_data
from src.ai_models.rl_environment import TradingEnv # Import custom environment

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Mapping from string name to SB3 algorithm class
SB3_ALGORITHMS = {
    "PPO": PPO,
    "A2C": A2C,
    "DDPG": DDPG,
    # Add other SB3 algorithms as needed
}

def train_rl_model(data_period: str = "6mo", data_interval: str = "1h", save_model: bool = True) -> None:
    """Trains the Reinforcement Learning trading agent and logs the normalization reference price."""
    logging.info("===== 🤖 RL Agent Training Start =====")

    # 1. Load Data
    df_train: Optional[pd.DataFrame] = None
    try:
        logging.info(f"Loading training data for {settings.RL_ENV_SYMBOL} ({data_period} {data_interval})...")
        df_train = get_historical_data(settings.RL_ENV_SYMBOL, period=data_period, interval=data_interval)
        if df_train is None or df_train.empty:
            logging.error("Failed to load training data. Aborting training.")
            return
        if 'Close' in df_train.columns and not df_train.empty:
            reference_price = df_train['Close'].iloc[0]
            logging.info(f"========= IMPORTANT =========")
            logging.info(f"RL 정규화 기준 가격 (훈련 데이터 첫 종가): {reference_price:.8f}")
            logging.info(f"이 값을 .env 파일의 RL_NORMALIZATION_REFERENCE_PRICE 변수에 설정해야 합니다!")
            logging.info(f"예: RL_NORMALIZATION_REFERENCE_PRICE={reference_price:.8f}")
            logging.info(f"===========================")
        else:
             logging.warning("Could not determine the reference price for normalization from training data.")

    except Exception as e:
        logging.error(f"Error loading training data: {e}")
        return

    # 2. Create Environment
    env: Optional[TradingEnv | VecEnv] = None
    try:
        # Pass the dataframe directly
        env = TradingEnv(df_train)
        # Optional: Vectorize env = make_vec_env(lambda: TradingEnv(df_train), n_envs=n_envs)
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

    # Define model parameters based on settings
    model_params: Dict[str, Any] = {
        "policy": settings.RL_POLICY,
        "env": env,
        "verbose": 1, # 0: none, 1: training logs, 2: debug
        "learning_rate": settings.RL_LEARNING_RATE,
        "batch_size": settings.RL_BATCH_SIZE, # Often specific to algorithm (PPO uses n_steps)
        "tensorboard_log": "./rl_tensorboard_logs/",
        # Add other algorithm-specific hyperparameters from settings if needed
        # "gamma": 0.99,
        # "ent_coef": 0.0,
        # "vf_coef": 0.5,
        # "max_grad_norm": 0.5,
    }
    # Adjust PPO specific params if needed
    if settings.RL_ALGORITHM == "PPO":
        model_params["n_steps"] = 2048 # Default PPO steps, adjust batch_size relation
        model_params.pop("batch_size", None) # PPO uses n_steps * n_envs / batch_size for minibatch

    model: Optional[BaseAlgorithm] = None
    try:
        logging.info(f"Initializing {settings.RL_ALGORITHM} model with policy {settings.RL_POLICY}...")
        model = AlgorithmClass(**model_params)
    except Exception as e:
        logging.error(f"Error initializing RL model: {e}")
        # Ensure env is closed if model init fails after env creation
        if env: env.close()
        return

    # 4. Define Callbacks (Optional)
    # Example: Evaluate model periodically and save the best one
    # eval_env = TradingEnv(df_train) # Use a separate eval env if possible
    # eval_callback = EvalCallback(eval_env, best_model_save_path='./model_weights/best_rl_model',
    #                              log_path='./rl_eval_logs/', eval_freq=5000,
    #                              deterministic=True, render=False)
    # Example: Stop training if a reward threshold is reached
    # reward_threshold = 1000 # Example threshold
    # stop_callback = StopTrainingOnRewardThreshold(reward_threshold=reward_threshold, verbose=1)
    # callback_list = [eval_callback, stop_callback]
    callback_list: Optional[List[BaseCallback]] = None # No callbacks for simplicity now

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
        # if 'eval_env' in locals(): eval_env.close()

    logging.info("===== 🤖 RL Agent Training End =====")

# --- Example Execution --- #
if __name__ == "__main__":
    # Load data for a longer period for training
    train_rl_model(data_period="1y", data_interval="1h") 