import pandas as pd
import logging
import os
import warnings
# --- FinRL Imports ---
try:
    # DRLAgent might be used for loading model state or prediction wrapper
    from finrl.agents.stablebaselines3.models import DRLAgent
    # Plotting and stats functions
    from finrl.plot import backtest_stats, backtest_plot, get_daily_return
    # Environment and Data Processor
    from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
    from finrl.meta.data_processor import DataProcessor
    from finrl.config import INDICATORS, DATA_SAVE_DIR as FINRL_DATA_SAVE_DIR
    # Need agent classes for loading
    from stable_baselines3 import PPO, A2C, DDPG # Add SAC, TD3 etc. if needed
    from stable_baselines3.common.logger import configure as sb3_configure_logger
    from stable_baselines3.common.vec_env import DummyVecEnv
    FINRL_INSTALLED = True
except ImportError as e:
    FINRL_INSTALLED = False
    FINRL_IMPORT_ERROR = e
    # Dummy classes/functions for placeholder execution
    class StockTradingEnv:
        def __init__(self, *args, **kwargs): pass
    class DRLAgent:
        @staticmethod
        def DRL_prediction(*args, **kwargs): return pd.DataFrame({'account_value': []}), pd.DataFrame({'actions': []})
        @staticmethod
        def load(*args, **kwargs): return None
    def backtest_stats(*args, **kwargs): return pd.DataFrame({'Metric': [], 'Value': []})
    def backtest_plot(*args, **kwargs): pass
    def get_daily_return(*args, **kwargs): return pd.Series([])
    class DataProcessor:
        def __init__(self, *args, **kwargs): pass
        def run(self, *args, **kwargs): return None, None, None
        def download_data(self, *args, **kwargs): return pd.DataFrame({'date': [], 'close': []})
        def clean_data(self, df): return df
        def add_technical_indicator(self, df, indicators): return df.assign(dummy_indicator=0.0)
        def add_turbulence(self, df): return df
    INDICATORS = []
    FINRL_DATA_SAVE_DIR = 'data/'
    # Fix dummy class definitions for stable-baselines3 models
    class DummySB3Model:
        @staticmethod
        def load(*args, **kwargs):
            logger.warning("Using dummy SB3 model load.")
            return "dummy_model_object"
    PPO = A2C = DDPG = DummySB3Model # Assign the dummy class
    def sb3_configure_logger(*args, **kwargs): pass
    class DummyVecEnv: 
        def __init__(self, *args, **kwargs): pass

# Filter specific warnings
warnings.filterwarnings("ignore", category=FutureWarning, module='stable_baselines3')
warnings.filterwarnings("ignore", category=UserWarning, module='gymnasium')

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Constants ---
PROJECT_DATA_SAVE_DIR = 'data/' # Project's data directory
PROJECT_RESULTS_SAVE_DIR = 'results/finrl/' # Project's results directory
PROJECT_TRAINED_MODEL_DIR = 'models/finrl/' # Project's model directory
DEFAULT_AGENT_TYPE = 'ppo'
TECHNICAL_INDICATORS_LIST = INDICATORS

def run_backtest_finrl(ticker: str,
                       agent_path: str,
                       start_date: str = "2024-01-01",
                       end_date: str = "2024-04-01",
                       agent_type: str = DEFAULT_AGENT_TYPE,
                       # Pass env config consistent with training
                       initial_amount: float = 100000.0,
                       hmax: int = 100,
                       transaction_cost_pct: float = 0.001,
                       reward_scaling: float = 1e-4,
                       tech_indicator_list: list[str] = TECHNICAL_INDICATORS_LIST,
                       plot: bool = True, # Add option to generate plots
                       results_save_dir: str = PROJECT_RESULTS_SAVE_DIR,
                       baseline_ticker: str = '^GSPC' # Baseline for plotting
                       ) -> pd.DataFrame | None:
    """
    Runs a backtest using a pre-trained FinRL DRL agent, saves stats and plot.
    Enhanced error handling and clarity.
    """
    if not FINRL_INSTALLED:
        logger.error(f"FinRL library not installed or import failed: {FINRL_IMPORT_ERROR}. Cannot run backtest.")
        return None

    logger.info(f"Starting FinRL backtest for {ticker} using agent: {agent_path} ({start_date} to {end_date})")

    if not os.path.exists(agent_path):
        logger.error(f"Agent model file not found at: {agent_path}")
        return None

    # --- Define Paths and Create Directories ---
    os.makedirs(PROJECT_DATA_SAVE_DIR, exist_ok=True)
    os.makedirs(results_save_dir, exist_ok=True)
    # Configure stable-baselines3 logger (optional, redirects logs)
    # sb3_logger = sb3_configure_logger(results_save_dir, ["stdout", "csv", "tensorboard"])

    try:
        # --- 1. Load Backtest Data ---
        logger.info("Step 1: Fetching and processing backtest data...")
        # DataProcessor requires start/end dates that cover the period needed for indicator calculation
        # A buffer might be needed before the actual backtest start_date depending on indicators
        # FinRL's internal logic might handle this, but be mindful.
        dp = DataProcessor(data_source='yahoofinance', start_date=start_date, end_date=end_date, time_interval='1D')
        price_array, tech_array, turbulence_array = dp.run(ticker_list=[ticker], 
                                                           technical_indicator_list=tech_indicator_list,
                                                           if_vix=True,
                                                           cache=True)
        trade_df = dp.dataframe

        if trade_df.empty or len(trade_df) < 2: # Need at least 2 data points for backtest
            logger.error("Backtest data processing resulted in an empty or insufficient DataFrame.")
            return None
        logger.info(f"Backtest data processed. Shape: {trade_df.shape}, Date range: {trade_df['date'].min()} to {trade_df['date'].max()}")

        # --- 2. Setup Backtest Environment ---
        logger.info("Step 2: Setting up backtest environment...")
        stock_dimension = len(trade_df.tic.unique())
        state_space = 1 + 2*stock_dimension + len(tech_indicator_list)*stock_dimension
        env_kwargs = {
            "hmax": hmax,
            "initial_amount": initial_amount,
            "transaction_cost_pct": transaction_cost_pct,
            "state_space": state_space,
            "stock_dim": stock_dimension,
            "tech_indicator_list": tech_indicator_list,
            "action_space": stock_dimension,
            "reward_scaling": reward_scaling,
            "mode": "trade" # Explicitly set trade mode
        }
        backtest_env = StockTradingEnv(df=trade_df, **env_kwargs)
        backtest_env_instance = DummyVecEnv([lambda: backtest_env]) # SB3 needs a VecEnv

        # --- 3. Load Trained Agent ---
        logger.info(f"Step 3: Loading trained {agent_type.upper()} agent from {agent_path}...")
        if agent_type.lower() == 'ppo': model_class = PPO
        elif agent_type.lower() == 'a2c': model_class = A2C
        elif agent_type.lower() == 'ddpg': model_class = DDPG
        else: raise ValueError(f"Unsupported agent type: {agent_type}")
        
        # Pass the VecEnv instance for environment sanity checks during loading
        trained_model = model_class.load(agent_path, env=backtest_env_instance)
        logger.info("Agent loaded successfully.")

        # --- 4. Run Backtest Simulation ---
        logger.info("Step 4: Running backtest simulation...")
        # Use the standard SB3 predict method within a loop
        obs, _ = backtest_env_instance.reset() # Get initial observation
        account_values = [initial_amount] # Track portfolio value
        # DRLAgent.DRL_prediction is complex; simulating the loop directly might be clearer
        # for _ in range(len(trade_df.index.unique())-1): # Iterate through time steps
        #     action, _ = trained_model.predict(obs, deterministic=True)
        #     obs, rewards, dones, _, infos = backtest_env_instance.step(action)
        #     if dones:
        #         # Handle episode end if necessary (usually not in trade mode)
        #         # obs, _ = backtest_env_instance.reset()
        #         pass
        #     # Extract account value from infos dict (structure depends on FinRL Env)
        #     if isinstance(infos, list) and infos and 'account_value' in infos[0]:
        #         account_values.append(infos[0]['account_value'])
        #     else: # Fallback if structure changes
        #         account_values.append(account_values[-1]) # Assume no change if info missing

        # Use DRLAgent.DRL_prediction if it works reliably with loaded SB3 model
        # Note: DRL_prediction might internally reset the env, use with caution
        account_value_df, _ = DRLAgent.DRL_prediction(
             model=trained_model,
             environment=backtest_env # Pass the original env instance here
         )
        logger.info("Backtest simulation complete using DRL_prediction.")

        # --- 5. Calculate & Return Stats ---
        logger.info("Step 5: Calculating backtest statistics...")
        if not isinstance(account_value_df, pd.DataFrame) or 'account_value' not in account_value_df.columns:
             logger.error(f"DRL_prediction returned unexpected format: {type(account_value_df)}. Cannot calculate stats.")
             return None
        if account_value_df.empty:
             logger.error("DRL_prediction returned an empty DataFrame. Cannot calculate stats.")
             return None
             
        perf_stats_all = backtest_stats(account_value=account_value_df, value_col_name = 'account_value')
        perf_stats_df = perf_stats_all.reset_index()
        perf_stats_df.columns = ['Metric', 'Value']
        logger.info(f"Backtest stats calculated:\n{perf_stats_df.to_string()}")

        # --- 6. Save Results (Stats & Plots) ---
        result_prefix = f"{ticker}_{agent_type}_{start_date}_{end_date}"
        stats_filename = os.path.join(results_save_dir, f"stats_{result_prefix}.csv")
        perf_stats_df.to_csv(stats_filename, index=False)
        logger.info(f"Backtest statistics saved to: {stats_filename}")

        if plot:
            plot_filename = os.path.join(results_save_dir, f"plot_{result_prefix}.png")
            logger.info(f"Generating backtest plot to: {plot_filename}")
            # Ensure baseline dates match backtest period
            backtest_plot(account_value_df, 
                          baseline_ticker=baseline_ticker,
                          baseline_start=account_value_df['date'].min().strftime('%Y-%m-%d'), # Align baseline dates
                          baseline_end=account_value_df['date'].max().strftime('%Y-%m-%d'),
                          value_col_name = 'account_value',
                          filename=plot_filename)
            logger.info("Backtest plot saved.")

        return perf_stats_df

    except ValueError as ve:
         logger.error(f"Value error during backtest (often related to data shape or env setup): {ve}", exc_info=True)
         return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during FinRL backtest: {e}", exc_info=True)
        return None


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger.info("Running FinRL Backtester...")

    if not FINRL_INSTALLED:
        print(f"FinRL installation not found or import failed: {FINRL_IMPORT_ERROR}")
        print("Cannot run backtest example.")
    else:
        # Ensure a dummy model file exists from the trainer example
        # Use the path generated by the trainer
        TICKER = "AAPL"
        AGENT_TYPE = "ppo"
        TRAIN_START = "2021-01-01"
        TRAIN_END = "2023-01-01"
        DUMMY_MODEL_PATH = os.path.join(PROJECT_TRAINED_MODEL_DIR, f"{AGENT_TYPE}_{TICKER}_{TRAIN_START}_{TRAIN_END}.zip")

        if not os.path.exists(DUMMY_MODEL_PATH):
            logger.warning(f"Trained model not found at {DUMMY_MODEL_PATH}. ")
            logger.warning("Attempting to run trainer first...")
            # Import and run trainer (ensure it saves to the expected path)
            try:
                from src.stock_trader.agent_trainer import train_finrl_agent
                train_finrl_agent(
                    ticker=TICKER,
                    start_date=TRAIN_START,
                    end_date=TRAIN_END,
                    agent_type=AGENT_TYPE,
                    total_timesteps=5000, # Short train for demo
                    model_save_path=DUMMY_MODEL_PATH
                )
            except ImportError:
                 logger.error("Could not import train_finrl_agent to create dummy model.")
            except Exception as train_e:
                 logger.error(f"Error running trainer to create dummy model: {train_e}")
                 
        if os.path.exists(DUMMY_MODEL_PATH):
            logger.info(f"Using model: {DUMMY_MODEL_PATH}")
            # Example usage:
            backtest_results = run_backtest_finrl(
                ticker=TICKER,
                agent_path=DUMMY_MODEL_PATH,
                agent_type=AGENT_TYPE,
                start_date="2023-01-01", # Use a period after training
                end_date="2023-06-01",
                plot=True
            )
            if backtest_results is not None:
                print("\n--- FinRL Backtest Results --- ")
                print(backtest_results)
            else:
                print("\n--- FinRL backtesting failed. Check logs. ---")
        else:
            print(f"\n--- Could not find or create the model file {DUMMY_MODEL_PATH}. Backtest skipped. ---") 