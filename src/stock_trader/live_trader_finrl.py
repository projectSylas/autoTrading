import logging
import os
import pandas as pd
import numpy as np
# --- Technical Analysis Import ---
try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False
    logging.getLogger(__name__).warning("pandas-ta library not found. Technical indicator calculation will be limited or disabled.")

# --- Alpaca/FinRL Imports ---
try:
    from alpaca_trade_api.rest import REST, TimeFrame, APIError
    ALPACA_SDK_AVAILABLE = True
except ImportError:
    ALPACA_SDK_AVAILABLE = False
    logging.getLogger(__name__).warning("alpaca-trade-api library not found. Live trading functionality disabled.")
    class REST:
         def __init__(self, *args, **kwargs): raise NotImplementedError
         def get_account(self): raise NotImplementedError
         def get_position(self, *args, **kwargs): raise NotImplementedError
         def submit_order(self, *args, **kwargs): raise NotImplementedError
    class TimeFrame:
         Minute = "1Min"
    class APIError(Exception): pass

try:
    from finrl.agents.stablebaselines3.models import DRLAgent
    from stable_baselines3 import PPO, A2C, DDPG
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
    class DummySB3Model:
        @staticmethod
        def load(*args, **kwargs): # Correct indentation
            logger.warning("Using dummy SB3 model load.")
            return "dummy_model_object"
    PPO = A2C = DDPG = DummySB3Model
    def sb3_configure_logger(*args, **kwargs): pass
    class DummyVecEnv: 
        def __init__(self, *args, **kwargs): pass

# --- Config Import --- 
try:
    # Import the settings instance directly
    from src.config.settings import settings 
    if settings is None:
        raise ImportError("Settings object is None after import. Check settings.py validation.")
except ImportError:
    # Fallback settings object if import fails (basic functionality)
    settings = type('obj', (object,), {
        'ALPACA_API_KEY_ID': os.getenv("ALPACA_API_KEY_ID"),
        'ALPACA_SECRET_KEY': os.getenv("ALPACA_SECRET_KEY"),
        'ALPACA_BASE_URL': os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets/v2"),
        'FINRL_MODEL_DIR': 'models/finrl/', # Default path if settings fail
        # Need to add IS_CONFIG_VALID equivalent for fallback
        'IS_CONFIG_VALID': all([os.getenv("ALPACA_API_KEY_ID"), os.getenv("ALPACA_SECRET_KEY")])
    })()
    logging.getLogger(__name__).warning("Could not import settings instance. Using basic os.getenv fallbacks.")

# Setup logging
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Constants & Configuration ---
# Directly use the imported settings instance
TRAINED_MODEL_DIR = settings.FINRL_MODEL_DIR 

# --- Helper Function: Calculate Technical Indicators --- 
def calculate_technical_indicators(df: pd.DataFrame, indicator_list: list[str]) -> pd.DataFrame | None:
    """
    Calculates technical indicators using pandas_ta.
    IMPORTANT: The list and parameters MUST match the training environment.
    """
    if not PANDAS_TA_AVAILABLE:
        logger.error("pandas-ta is not available. Cannot calculate indicators.")
        return None
        
    logger.debug(f"Calculating indicators: {indicator_list}")
    df_ta = df.copy()
    try:
        # Apply indicators using pandas_ta strategy
        # Example strategy - customize based on actual indicators used
        custom_strategy = ta.Strategy(
            name="FinRL_Live_State",
            description="Calculate indicators needed for FinRL live agent state",
            ta=[
                {"kind": "sma", "length": 10, "col_names": "SMA_10"},
                {"kind": "sma", "length": 50, "col_names": "SMA_50"},
                {"kind": "rsi", "length": 14, "col_names": "RSI_14"},
                {"kind": "macd", "fast": 12, "slow": 26, "signal": 9, "col_names": ("MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9")},
                # Add other indicators from indicator_list here using their 'kind' and parameters
                # Example: {"kind": "cci", "length": 20, "col_names": "CCI_20"}
                # Example: {"kind": "adx", "length": 14, "col_names": ("ADX_14", "DMP_14", "DMN_14")}
            ]
        )
        df_ta.ta.strategy(custom_strategy)
        
        # Select only the necessary columns (based on the assumed indicator list)
        # Adjust this based on the actual columns generated by pandas-ta
        required_columns = [col for col in df_ta.columns if col.upper() in indicator_list or col in df.columns] # Keep original columns too for now
        
        # Filter out columns that might be fully NaN if calculation failed for some
        # df_ta = df_ta[required_columns].dropna(axis=1, how='all') 

        logger.debug(f"Calculated TA columns: {df_ta.columns.tolist()}")
        return df_ta
        
    except Exception as e:
        logger.error(f"Error calculating technical indicators with pandas-ta: {e}", exc_info=True)
        return None

# --- Alpaca API Connection (Using Settings) ---
def get_alpaca_api():
    """Initializes and returns an Alpaca API client instance using settings."""
    if not ALPACA_SDK_AVAILABLE:
        logger.error("Alpaca SDK not available. Cannot connect.")
        return None
    # Use settings instance attributes
    if not settings.ALPACA_API_KEY_ID or not settings.ALPACA_SECRET_KEY:
        logger.error("Alpaca API Key ID or Secret Key is missing in settings/environment.")
        return None
        
    logger.info(f"Connecting to Alpaca API at {settings.ALPACA_BASE_URL}...") # Use settings.ALPACA_BASE_URL
    try:
        api = REST(key_id=settings.ALPACA_API_KEY_ID, # Use settings attribute
                   secret_key=settings.ALPACA_SECRET_KEY, # Use settings attribute
                   base_url=settings.ALPACA_BASE_URL, # Use settings attribute
                   api_version='v2')
        account_info = api.get_account()
        logger.info(f"Alpaca connection successful. Account Status: {account_info.status}")
        return api
    except APIError as ape:
         logger.error(f"Alpaca API Error: {ape.status_code} - {ape}", exc_info=True)
         return None
    except Exception as e:
        logger.error(f"Failed to connect to Alpaca API: {e}", exc_info=True)
        return None

# --- Live Trading Logic (Updated with State Calculation) ---
def run_live_trading_finrl(ticker: str,
                           agent_path: str,
                           agent_type: str = "ppo",
                           trade_quantity: int = 10,
                           # Define the indicators needed for the state - MUST MATCH TRAINING!
                           tech_indicator_list: list[str] = ["SMA_10", "SMA_50", "RSI_14", "MACD_12_26_9", "MACDH_12_26_9", "MACDS_12_26_9"] 
                           ):
    """
    Runs live trading using FinRL agent and Alpaca API.
    Constructs state vector based on real market data and technical indicators.
    """
    logger.info(f"Starting FinRL live trading for {ticker} with agent {agent_path}...")
    logger.info(f" > State indicators assumed: {tech_indicator_list}") # Log assumed indicators

    if not FINRL_INSTALLED:
        logger.error("FinRL not installed. Cannot run live trading.")
        return
    if not ALPACA_SDK_AVAILABLE:
        logger.error("Alpaca SDK not installed. Cannot run live trading.")
        return

    # 1. Connect to Alpaca API
    api = get_alpaca_api()
    if api is None:
        logger.error("Cannot start live trading: Alpaca API connection failed.")
        return

    # 2. Load Trained Agent
    if not os.path.exists(agent_path):
        logger.error(f"Agent model file not found: {agent_path}")
        return
    logger.info(f"Loading trained {agent_type.upper()} agent from {agent_path}...")
    try:
        if agent_type.lower() == 'ppo': model_class = PPO
        elif agent_type.lower() == 'a2c': model_class = A2C
        elif agent_type.lower() == 'ddpg': model_class = DDPG
        else: raise ValueError(f"Unsupported agent type: {agent_type}")
        
        # Loading might require a dummy env if env details were saved with model
        # For SB3 models saved directly, env is usually not needed for loading
        trained_model = model_class.load(agent_path)
        logger.info("Agent loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load agent model from {agent_path}: {e}", exc_info=True)
        return

    # 3. Live Trading Loop (Conceptual - Single Run Example)
    logger.info("Entering live trading loop (Conceptual - single execution)...")
    try:
        # a. Fetch latest market data
        logger.info(f"Fetching latest market data for {ticker}...")
        try:
             # Fetch enough data for the longest indicator period (e.g., SMA_50 needs 50 points)
             # Increase limit if longer indicators are used
             bars = api.get_bars(ticker, TimeFrame.Minute, limit=100).df
             if bars.empty or len(bars) < 50: # Check if enough data for example indicators
                  logger.error(f"Fetched insufficient bar data ({len(bars)} bars). Need at least 50 for example indicators. Cannot construct state.")
                  return
             logger.info(f"Fetched {len(bars)} bars for {ticker}. Last close: {bars.iloc[-1]['close']:.2f}")
        except APIError as ape:
             logger.error(f"Alpaca API Error fetching bars: {ape}")
             return
        except Exception as e:
             logger.error(f"Error fetching bars: {e}", exc_info=True)
             return

        # b. Construct environment state
        logger.info("Constructing environment state...")
        try:
            # Fetch current balance and position
            account_info = api.get_account()
            current_balance = float(account_info.cash)
            try:
                position = api.get_position(ticker)
                shares_owned = int(position.qty)
            except APIError as e:
                 if e.status_code == 404: shares_owned = 0
                 else: raise

            # --- Calculate Technical Indicators --- 
            logger.debug("Calculating technical indicators for state...")
            bars_with_ta = calculate_technical_indicators(bars, tech_indicator_list)
            if bars_with_ta is None:
                 logger.error("Failed to calculate technical indicators. Cannot construct state.")
                 return
                 
            # Get the latest row of indicators
            latest_data = bars_with_ta.iloc[-1]

            # --- Extract Features from Latest Data --- 
            # Ensure column names here EXACTLY match those generated by calculate_technical_indicators
            # And the ORDER MUST MATCH the training environment's observation space!
            state_features = []
            all_required_found = True
            for indicator_name in tech_indicator_list:
                if indicator_name in latest_data:
                    state_features.append(latest_data[indicator_name])
                else:
                    logger.error(f"Required indicator '{indicator_name}' not found in calculated data! Columns: {latest_data.index.tolist()}")
                    state_features.append(0.0) # Append dummy value on error
                    all_required_found = False
            
            if not all_required_found:
                logger.error("Could not find all required indicators. State vector may be incorrect!")
                # Depending on strictness, could return here

            # Handle potential NaNs in the latest features (MUST match training env handling)
            # Example: Forward fill then backfill (common), or fill with 0
            state_features_array = np.nan_to_num(np.array(state_features), nan=0.0) # Simple NaN -> 0 replacement
            logger.debug(f"Latest indicator features (NaN replaced): {state_features_array}")

            # --- Construct the Final State Vector --- 
            # Structure: [balance, shares_owned, indicator1, indicator2, ...] 
            # ** ORDER MATTERS! MUST MATCH TRAINING ENV **
            current_state_list = [current_balance, shares_owned] + state_features_array.tolist()
            current_state = np.array(current_state_list)

            logger.info(f"Constructed state vector (len={len(current_state)}). First 5: {current_state[:5]}")
            # Optional: Add check for expected state vector length
            # expected_len = 2 + len(tech_indicator_list)
            # if len(current_state) != expected_len:
            #     logger.error(f"State vector length mismatch! Expected {expected_len}, got {len(current_state)}")
            #     return 

        except APIError as ape:
            logger.error(f"API Error during state construction: {ape}")
            return
        except Exception as e:
            logger.error(f"Error constructing state: {e}", exc_info=True)
            return

        # c. Get action from the agent
        logger.info("Getting action from agent...")
        # Use the correctly shaped state
        # action, _ = trained_model.predict(current_state_reshaped, deterministic=True) # For VecEnv shape
        action, _ = trained_model.predict(current_state, deterministic=True) # Assumes non-VecEnv shape
        # action = action[0] # Unwrap if prediction returns array like [action]
        logger.info(f"Agent predicted action: {action} (0:Hold, 1:Buy, 2:Sell typical)")

        # d. Execute trade based on action
        # Check current position before trading (already fetched for state)
        try:
            position = api.get_position(ticker)
            current_qty = int(position.qty)
            logger.info(f"Current position in {ticker}: {current_qty} shares.")
        except APIError as e:
            if e.status_code == 404: # Position not found
                 current_qty = 0
                 logger.info(f"No existing position found for {ticker}.")
            else:
                 logger.error(f"API Error getting position: {e}")
                 return # Stop if we can't get position

        if action == 1 and current_qty == 0: # Buy signal and no position
            logger.info(f"Executing MARKET BUY order for {trade_quantity} shares of {ticker}...")
            try:
                order = api.submit_order(symbol=ticker, qty=trade_quantity, side='buy', type='market', time_in_force='day')
                logger.info(f"BUY order submitted: ID={order.id}, Status={order.status}")
                # --- Add Slack Notification (Placeholder) ---
                # message = f"🚀 FinRL BUY Order Submitted: {trade_quantity} {ticker} @ Market. Order ID: {order.id}"
                # send_slack_notification(message, channel="#trading-alerts") # Replace channel if needed
                # ------------------------------------------
            except APIError as ape:
                 logger.error(f"Alpaca API Error submitting BUY order: {ape}")
                 # send_slack_notification(f"🚨 FinRL BUY Order FAILED: {ticker}. Reason: {ape}", channel="#trading-errors")
            except Exception as e:
                logger.error(f"Failed to submit BUY order: {e}", exc_info=True)
                # send_slack_notification(f"🚨 FinRL BUY Order FAILED: {ticker}. Reason: {e}", channel="#trading-errors")
        elif action == 2 and current_qty > 0: # Sell signal and have position
            sell_qty = current_qty # Sell entire position
            logger.info(f"Executing MARKET SELL order for {sell_qty} shares of {ticker}...")
            try:
                order = api.submit_order(symbol=ticker, qty=sell_qty, side='sell', type='market', time_in_force='day')
                logger.info(f"SELL order submitted: ID={order.id}, Status={order.status}")
                # --- Add Slack Notification (Placeholder) ---
                # message = f"💰 FinRL SELL Order Submitted: {sell_qty} {ticker} @ Market. Order ID: {order.id}"
                # send_slack_notification(message, channel="#trading-alerts")
                # ------------------------------------------
            except APIError as ape:
                 logger.error(f"Alpaca API Error submitting SELL order: {ape}")
                 # send_slack_notification(f"🚨 FinRL SELL Order FAILED: {ticker}. Reason: {ape}", channel="#trading-errors")
            except Exception as e:
                logger.error(f"Failed to submit SELL order: {e}", exc_info=True)
                # send_slack_notification(f"🚨 FinRL SELL Order FAILED: {ticker}. Reason: {e}", channel="#trading-errors")
        else: # Hold signal or already in desired state
            logger.info(f"HOLD action or already in desired position state (Action: {action}, Qty: {current_qty}). No trade executed.")

        logger.info("Live trading iteration finished.")

    except Exception as e:
        logger.error(f"Critical error in live trading loop: {e}", exc_info=True)
        # Add critical Slack notification here
        # message = f"🔥 CRITICAL ERROR in FinRL Live Trading Loop for {ticker}: {e}"
        # send_slack_notification(message, channel="#trading-critical")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger.info("Running FinRL Live Trader Placeholder...")

    # Ensure required settings are available
    if not settings.IS_CONFIG_VALID:
         print("Essential API Keys missing in config/environment. Cannot run live trader example.")
    elif not FINRL_INSTALLED:
         print(f"FinRL not installed ({FINRL_IMPORT_ERROR}). Cannot run live trader example.")
    elif not ALPACA_SDK_AVAILABLE:
         print("Alpaca SDK not installed. Cannot run live trader example.")
    else:
        # --- Prepare a dummy model file if it doesn't exist ---
        TICKER = "AAPL"
        AGENT_TYPE = "ppo"
        # Adjust path to match trainer's default saving location
        MODEL_PATH = os.path.join(settings.FINRL_MODEL_DIR, f"{AGENT_TYPE}_{TICKER}_live_placeholder.zip")
        if not os.path.exists(MODEL_PATH):
            logger.warning(f"Live trading model not found at {MODEL_PATH}. Creating a dummy PPO model...")
            try:
                # Need a dummy env to initialize agent before saving
                env = DummyVecEnv([lambda: type('obj', (object,), {'action_space': type('space',(object,),{'shape':(1,)}), 'observation_space': type('space',(object,),{'shape':(10,)})})()])
                dummy_model = PPO("MlpPolicy", env)
                dummy_model.save(MODEL_PATH)
                logger.info(f"Created dummy agent file: {MODEL_PATH}")
            except Exception as e:
                 logger.error(f"Failed to create dummy agent file: {e}")
                 MODEL_PATH = None # Prevent running if dummy creation failed
                 
        # --- Run live trading example if model exists ---
        if MODEL_PATH and os.path.exists(MODEL_PATH):
            logger.info(f"Starting live trading example for {TICKER}...")
            run_live_trading_finrl(ticker=TICKER, 
                                   agent_path=MODEL_PATH,
                                   agent_type=AGENT_TYPE,
                                   trade_quantity=5 # Example quantity
                                   )
            logger.info("Live trading example finished.")
        else:
             logger.info("Skipping live trading example as model file could not be ensured.") 