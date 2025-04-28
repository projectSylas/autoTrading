import logging
import os
import sys
import optuna
import numpy as np
import pandas as pd
import stable_baselines3 as sb3
from stable_baselines3.common.env_util import make_vec_env
import argparse

# --- Project Structure Adjustment ---
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(SRC_DIR))
if PROJECT_ROOT not in sys.path:
     sys.path.append(PROJECT_ROOT)

# --- Import Custom Environment and Utils ---
try:
    from src.ai_models.rl_environment import TradingEnv
    from src.utils.common import get_historical_data
    SB3_AVAILABLE = True
except ImportError as e:
    SB3_AVAILABLE = False
    IMPORT_ERROR = e
    # Define dummy classes if SB3 is not available (Corrected)
    class TradingEnv:
        def __init__(self, *args, **kwargs):
            pass
    def get_historical_data(*args, **kwargs):
        return pd.DataFrame()

# --- Setup Logging ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Suppress Optuna INFO messages to avoid cluttering the logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- Default Parameters --- 
DEFAULT_TICKER = "AAPL" # Ticker to tune for
DEFAULT_TRIALS = 50      # Number of tuning trials
DEFAULT_TUNE_TIMESTEPS = 50000 # Shorter duration for tuning
DEFAULT_EVAL_TIMESTEPS = 10000 # Steps for evaluation within objective

# --- Argument Parser --- 
parser = argparse.ArgumentParser(description="Tune PPO hyperparameters for the TradingEnv.")
parser.add_argument("--ticker", type=str, default=DEFAULT_TICKER,
                    help=f"Ticker symbol to tune for (default: {DEFAULT_TICKER}).")
parser.add_argument("--n_trials", type=int, default=DEFAULT_TRIALS,
                    help=f"Number of Optuna trials (default: {DEFAULT_TRIALS}).")
parser.add_argument("--tune_timesteps", type=int, default=DEFAULT_TUNE_TIMESTEPS,
                    help=f"Training timesteps per trial (default: {DEFAULT_TUNE_TIMESTEPS}).")
args = parser.parse_args()

# --- Objective Function for Optuna --- 
def objective(trial: optuna.Trial) -> float:
    """Objective function for Optuna hyperparameter optimization."""
    if not SB3_AVAILABLE:
        logger.error(f"Stable-Baselines3 or TradingEnv not available: {IMPORT_ERROR}")
        # Return a very bad score to indicate failure
        return -np.inf
        
    logger.info(f"Starting Trial {trial.number}...")

    # --- 1. Suggest Hyperparameters ---
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_categorical("n_steps", [256, 512, 1024, 2048])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    ent_coef = trial.suggest_float("ent_coef", 0.0, 0.1)
    vf_coef = trial.suggest_float("vf_coef", 0.1, 1.0)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3])
    gamma = trial.suggest_categorical("gamma", [0.99, 0.995, 0.999, 0.9999])
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.9, 0.92, 0.95, 0.98, 1.0])
    # Define network architecture (example: 2 hidden layers)
    # net_arch_width = trial.suggest_categorical("net_arch_width", [64, 128, 256])
    # net_arch = [net_arch_width, net_arch_width]

    # Log suggested parameters for this trial
    logger.debug(f"Trial {trial.number} Parameters: lr={learning_rate:.6f}, n_steps={n_steps}, batch_size={batch_size}, ent_coef={ent_coef:.4f}, vf_coef={vf_coef:.4f}, clip={clip_range}, gamma={gamma}, gae={gae_lambda}")

    # --- 2. Load Data & Create Env ---
    try:
        # Load data (adjust period/interval as needed for tuning)
        df_tune = get_historical_data(args.ticker, period="729d", interval="1h")
        if df_tune is None or df_tune.empty:
            logger.warning(f"Trial {trial.number}: Failed to load data for {args.ticker}. Skipping trial.")
            return -np.inf # Indicate failure
        
        # Create environment
        env = make_vec_env(lambda: TradingEnv(df=df_tune), n_envs=1)
    except Exception as e:
        logger.error(f"Trial {trial.number}: Error creating environment: {e}. Skipping trial.", exc_info=True)
        return -np.inf # Indicate failure

    # --- 3. Create and Train Model ---
    try:
        model = sb3.PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            clip_range=clip_range,
            gamma=gamma,
            gae_lambda=gae_lambda,
            # policy_kwargs=dict(net_arch=net_arch), # If tuning net_arch
            verbose=0 # Suppress SB3 training logs during tuning
        )
        model.learn(total_timesteps=args.tune_timesteps)
        logger.info(f"Trial {trial.number}: Training complete.")
    except Exception as e:
        logger.error(f"Trial {trial.number}: Error during model training: {e}. Skipping trial.", exc_info=True)
        env.close() # Ensure env is closed even on error
        return -np.inf # Indicate failure

    # --- 4. Evaluate Model --- 
    # Simple evaluation: Run the model for a fixed number of steps and check total reward
    # More robust: Run a full backtest (might be slow for tuning)
    total_rewards = 0.0
    obs = env.reset() # Corrected: VecEnv usually returns only obs on reset
    n_eval_steps = 0
    max_eval_steps = len(df_tune) -1 # Evaluate on the whole dataset (or a portion)
    try:
        while n_eval_steps < max_eval_steps:
             action, _ = model.predict(obs, deterministic=True)
             # obs, reward, terminated, truncated, info = env.step(action) # Original line expecting 5 values
             # Corrected: VecEnv might return 4 values (obs, reward, dones, infos)
             obs, reward, dones, info = env.step(action) 
             total_rewards += reward[0] # Access the reward for the first (and only) env
             n_eval_steps += 1
             # Check termination/truncation using the combined 'dones' flag for the first env
             # if terminated[0] or truncated[0]: # Original check
             if dones[0]: # Corrected check
                 break
        eval_metric = total_rewards
        logger.info(f"Trial {trial.number}: Evaluation complete. Total Reward = {eval_metric:.4f}")
    except Exception as e:
        logger.error(f"Trial {trial.number}: Error during evaluation: {e}. Skipping trial.", exc_info=True)
        eval_metric = -np.inf # Indicate failure
    finally:
        env.close()

    return eval_metric

# --- Main Tuning Execution --- 
if __name__ == "__main__":
    logger.info("=======================================================")
    logger.info(f" Starting PPO Hyperparameter Tuning for {args.ticker} ")
    logger.info(f" > Number of Trials: {args.n_trials}")
    logger.info(f" > Timesteps per Trial: {args.tune_timesteps}")
    logger.info("=======================================================")

    if not SB3_AVAILABLE:
        logger.critical(f"Cannot start tuning. Stable-Baselines3 or TradingEnv not available: {IMPORT_ERROR}")
        sys.exit(1)
        
    # Create Optuna study
    # We want to maximize the evaluation metric (e.g., total reward)
    study = optuna.create_study(direction="maximize") 
    
    try:
        study.optimize(objective, n_trials=args.n_trials, timeout=None) # No timeout for now
    except KeyboardInterrupt:
         logger.warning("Tuning interrupted by user.")
    except Exception as e:
         logger.error(f"An error occurred during the optimization process: {e}", exc_info=True)

    # Print results
    logger.info("\n" + "="*50)
    logger.info(" Hyperparameter Tuning Results ")
    logger.info("="*50)
    try:
        logger.info(f"Number of finished trials: {len(study.trials)}")
        best_trial = study.best_trial
        logger.info(f"Best trial value (Max Reward): {best_trial.value:.4f}")
        logger.info("Best hyperparameters:")
        for key, value in best_trial.params.items():
            logger.info(f"  {key}: {value}")
    except ValueError:
         logger.warning("No completed trials found. Cannot determine best parameters.")
    except Exception as e:
         logger.error(f"Error retrieving study results: {e}")
         
    logger.info("Tuning process finished.") 