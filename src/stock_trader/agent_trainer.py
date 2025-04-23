import pandas as pd
import logging
import os
import warnings
# --- FinRL Imports ---
# Catch potential ImportError if finrl is not installed correctly
try:
    from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
    from finrl.agents.stablebaselines3.models import DRLAgent
    from finrl.config import INDICATORS, DATA_SAVE_DIR as FINRL_DATA_SAVE_DIR
    from finrl.main import check_and_make_directories
    from finrl.meta.data_processor import DataProcessor
    import stable_baselines3
    FINRL_INSTALLED = True
except ImportError as e:
    FINRL_INSTALLED = False
    FINRL_IMPORT_ERROR = e
    # Define dummy classes or variables if needed for placeholder execution
    class StockTradingEnv:
        def __init__(self, *args, **kwargs): pass
    class DRLAgent:
        def __init__(self, *args, **kwargs): pass
        def get_model(self, *args, **kwargs): return None
        def train_model(self, *args, **kwargs): return "dummy_trained_model"
    class DataProcessor:
        def __init__(self, *args, **kwargs): pass
        def download_data(self, *args, **kwargs): return pd.DataFrame({'date': [], 'close': []})
        def clean_data(self, df): return df
        def add_technical_indicator(self, df, indicators): return df.assign(dummy_indicator=0.0)
        def add_turbulence(self, df): return df
    INDICATORS = []
    FINRL_DATA_SAVE_DIR = 'data/'
    def check_and_make_directories(*args, **kwargs): pass

# Filter specific warnings from stable-baselines3 or dependencies
warnings.filterwarnings("ignore", category=FutureWarning, module='stable_baselines3')
warnings.filterwarnings("ignore", category=UserWarning, module='gymnasium')

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Constants --- (Moved project-specific paths here)
PROJECT_DATA_SAVE_DIR = 'data/' # Project's data directory
PROJECT_TRAINED_MODEL_DIR = 'models/finrl/' # Project's model directory
DEFAULT_TIMESTEPS = 20000 # Increased default for a slightly more meaningful train

# Technical indicators from FinRL config (can be customized)
TECHNICAL_INDICATORS_LIST = INDICATORS

# Model configurations (can be externalized to config file)
AGENT_KWARGS = {
    "ppo": {
        "n_steps": 2048,
        "ent_coef": 0.01,
        "learning_rate": 0.0003,
        "batch_size": 64,
    },
    "a2c": {"n_steps": 5, "ent_coef": 0.01, "learning_rate": 0.0007},
    "ddpg": {"batch_size": 128, "buffer_size": 50000, "learning_rate": 0.001},
    # Add other agents like sac, td3 if needed
}

def train_finrl_agent(ticker: str,
                      start_date: str = "2020-01-01",
                      end_date: str = "2023-12-31",
                      agent_type: str = "ppo",
                      total_timesteps: int = DEFAULT_TIMESTEPS,
                      model_save_path: str | None = None,
                      # Define environment config parameters
                      initial_amount: float = 100000.0,
                      hmax: int = 100, # Max shares to trade per transaction
                      transaction_cost_pct: float = 0.001, # Commission fee
                      reward_scaling: float = 1e-4,
                      tech_indicator_list: list[str] = TECHNICAL_INDICATORS_LIST,
                      ) -> str | None:
    """
    Trains a FinRL Deep Reinforcement Learning agent for a given stock ticker.

    Args:
        ticker (str): The stock ticker symbol (e.g., "AAPL").
        start_date (str): Training start date ("YYYY-MM-DD").
        end_date (str): Training end date ("YYYY-MM-DD").
        agent_type (str): Type of DRL agent ('ppo', 'a2c', 'ddpg').
        total_timesteps (int): Total timesteps for training.
        model_save_path (str | None): Path to save the trained model.
                                      If None, generates a default path under models/finrl/.
        initial_amount (float): Starting portfolio value for the environment.
        hmax (int): Max number of shares to trade per step.
        transaction_cost_pct (float): Transaction cost percentage.
        reward_scaling (float): Scaling factor for the reward function.
        tech_indicator_list (list[str]): List of technical indicators to use.

    Returns:
        str | None: The path where the trained model is saved, or None if training failed.
    """
    if not FINRL_INSTALLED:
        logger.error(f"FinRL library not installed or import failed: {FINRL_IMPORT_ERROR}. Cannot train agent.")
        return None
        
    logger.info(f"Starting FinRL agent training for {ticker} ({start_date} to {end_date}). Agent: {agent_type}, Timesteps: {total_timesteps}")

    # --- Define Paths and Create Directories ---
    # Use project-specific directories, ensure they exist
    os.makedirs(PROJECT_DATA_SAVE_DIR, exist_ok=True)
    os.makedirs(PROJECT_TRAINED_MODEL_DIR, exist_ok=True)
    # FinRL might also create its own internal directories
    check_and_make_directories([FINRL_DATA_SAVE_DIR]) # Ensure FinRL's data dir exists

    if model_save_path is None:
        model_save_path = os.path.join(PROJECT_TRAINED_MODEL_DIR, f"{agent_type}_{ticker}_{start_date}_{end_date}.zip")

    try:
        # --- 1. Data Fetching & Preprocessing ---
        logger.info("Step 1: Fetching and processing data...")
        dp = DataProcessor(data_source='yahoofinance', start_date=start_date, end_date=end_date, time_interval='1D') # Assuming daily interval
        price_array, tech_array, turbulence_array = dp.run(ticker_list=[ticker], 
                                                           technical_indicator_list=tech_indicator_list,
                                                           if_vix=True, # Include VIX turbulence index
                                                           cache=True) # Cache processed data
        # Create DataFrame from processed data (FinRL format)
        data_config = {
            'price_array': price_array,
            'tech_array': tech_array,
            'turbulence_array': turbulence_array
        }
        processed_df = dp.dataframe
        logger.info(f"Data processed. Shape: {processed_df.shape}")
        # logger.debug(f"Processed data head:\n{processed_df.head()}") # Can be verbose
        
        if processed_df.empty:
            logger.error("Data processing resulted in an empty DataFrame. Check data source or date range.")
            return None

        # --- 2. Environment Setup ---
        logger.info("Step 2: Setting up training environment...")
        # state_space = 1 + 2*len(tech_indicator_list) + 2 # Own shares + balance + indicators + VIX
        # Assuming state space is handled internally by FinRL env based on df columns
        stock_dimension = len(processed_df.tic.unique())
        state_space = 1 + 2*stock_dimension + len(tech_indicator_list)*stock_dimension
        logger.info(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")
        
        env_kwargs = {
            "hmax": hmax,
            "initial_amount": initial_amount,
            "transaction_cost_pct": transaction_cost_pct,
            "state_space": state_space,
            "stock_dim": stock_dimension,
            "tech_indicator_list": tech_indicator_list,
            "action_space": stock_dimension,
            "reward_scaling": reward_scaling,
            # "mode": "train" # Implicitly train mode
        }
        # Need to ensure processed_df has the correct format for the env
        # Usually requires columns like 'date', 'tic', 'close', and indicator columns
        train_env = StockTradingEnv(df=processed_df, **env_kwargs)
        # You might need to wrap the environment for stable-baselines3 if not done by DRLAgent
        # train_env = DummyVecEnv([lambda: train_env])

        # --- 3. Agent Initialization ---
        logger.info(f"Step 3: Initializing {agent_type.upper()} agent...")
        agent = DRLAgent(env=train_env) # DRLAgent wraps the env
        model_kwargs = AGENT_KWARGS.get(agent_type, {})
        # Use default policy from FinRL/stable-baselines3
        model = agent.get_model(agent_type, model_kwargs=model_kwargs, verbose=0) # Set verbose=0 to reduce SB3 logs
        logger.info(f"Agent {agent_type.upper()} initialized.")

        # --- 4. Training ---
        logger.info(f"Step 4: Training agent for {total_timesteps} timesteps...")
        # The DRLAgent's train_model handles the training loop
        trained_model = agent.train_model(model=model,
                                          tb_log_name=f'{agent_type}_{ticker}',
                                          total_timesteps=total_timesteps)
        logger.info("Agent training complete.")

        # --- 5. Model Saving ---
        logger.info(f"Step 5: Saving trained model to {model_save_path}")
        # trained_model object here is likely the stable_baselines3 model itself
        trained_model.save(model_save_path)
        logger.info(f"Model saved successfully to {model_save_path}")
        return model_save_path

    except Exception as e:
        logger.error(f"An error occurred during FinRL agent training: {e}", exc_info=True)
        return None


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger.info("Running FinRL Agent Trainer...")

    if not FINRL_INSTALLED:
        print(f"FinRL installation not found or import failed: {FINRL_IMPORT_ERROR}")
        print("Cannot run training example.")
    else:
        # Example usage:
        save_path = train_finrl_agent(
            ticker="AAPL",
            start_date="2021-01-01", # Shorter period for faster example run
            end_date="2023-01-01",
            agent_type="ppo",
            total_timesteps=10000 # Reduced timesteps for example
        )
        if save_path:
            print(f"FinRL training complete. Model saved at: {save_path}")
        else:
            print("FinRL training failed. Check logs.") 