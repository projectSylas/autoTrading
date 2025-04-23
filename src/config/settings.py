import os
from dotenv import load_dotenv
import logging
from pydantic_settings import BaseSettings
from pydantic import Field, ValidationError
from typing import List, Optional # Import List and Optional

# Setup logging
logger = logging.getLogger(__name__)
# Basic config, assuming a central logger might be set up elsewhere eventually
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Load Environment Variables --- 
dotenv_path = os.path.join(os.path.dirname(__file__), '../../.env') 
loaded = load_dotenv(dotenv_path=dotenv_path, override=True) # override=True ensures .env takes precedence

if loaded:
    logger.info(f".env file loaded successfully from: {dotenv_path}")
else:
    logger.warning(f".env file not found at expected location: {dotenv_path}. Relying on system environment variables or defaults.")

# --- Project Paths --- 
# Moved path definitions here for clarity before Settings class
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')) 
DEFAULT_MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "models")
DEFAULT_DATA_SAVE_DIR = os.path.join(PROJECT_ROOT, "data")
DEFAULT_LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
DEFAULT_MODEL_WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "model_weights")

# --- Settings Class using Pydantic BaseSettings --- 
class Settings(BaseSettings):
    # Pydantic will automatically look for environment variables (case-insensitive)
    # matching the field names, or load from .env if python-dotenv is used.

    # --- API Keys --- 
    # Field(...) can be used for validation, default values if env var is missing
    # Use default=None if the key is truly optional, otherwise it's required.
    NEWS_API_KEY: str = Field(..., validation_alias='NEWS_API_KEY') # ... means required
    ALPACA_API_KEY_ID: str = Field(..., validation_alias='ALPACA_API_KEY_ID')
    ALPACA_SECRET_KEY: str = Field(..., validation_alias='ALPACA_SECRET_KEY')
    # Default to paper trading URL if ALPACA_BASE_URL is not set in env
    ALPACA_BASE_URL: str = Field(default="https://paper-api.alpaca.markets", validation_alias='ALPACA_BASE_URL')
    # Optional API Keys - provide default empty string if not set
    BINANCE_API_KEY: str = Field(default="", validation_alias='BINANCE_API_KEY')
    BINANCE_SECRET_KEY: str = Field(default="", validation_alias='BINANCE_SECRET_KEY')
    SLACK_WEBHOOK_URL: str = Field(default="", validation_alias='SLACK_WEBHOOK_URL') # Default is empty string

    # --- Paths --- 
    # Use Field with default factory for dynamic paths
    MODEL_SAVE_DIR: str = Field(default=DEFAULT_MODEL_SAVE_DIR, validation_alias='MODEL_SAVE_DIR')
    DATA_SAVE_DIR: str = Field(default=DEFAULT_DATA_SAVE_DIR, validation_alias='DATA_SAVE_DIR')
    LOG_DIR: str = Field(default=DEFAULT_LOG_DIR, validation_alias='LOG_DIR')
    MODEL_WEIGHTS_DIR: str = Field(default=DEFAULT_MODEL_WEIGHTS_DIR, validation_alias='MODEL_WEIGHTS_DIR')
    # Derived paths based on the base directories
    @property
    def FINRL_MODEL_DIR(self) -> str:
        return os.path.join(self.MODEL_SAVE_DIR, "finrl")
    @property
    def FREQAI_MODEL_DIR(self) -> str:
        return os.path.join(self.MODEL_SAVE_DIR, "freqai")
    @property
    def SENTIMENT_CACHE_DIR(self) -> str:
        return os.path.join(self.DATA_SAVE_DIR, "sentiment_cache")

    # --- Training/Strategy Parameters --- 
    # Use Field for type hints and defaults
    FINRL_DEFAULT_TIMESTEPS: int = Field(default=20000, validation_alias='FINRL_DEFAULT_TIMESTEPS')
    FINRL_INITIAL_AMOUNT: float = Field(default=100000.0, validation_alias='FINRL_INITIAL_AMOUNT')
    FINRL_HMAX: int = Field(default=100, validation_alias='FINRL_HMAX')
    FINRL_TRANSACTION_COST: float = Field(default=0.001, validation_alias='FINRL_TRANSACTION_COST')
    FINRL_REWARD_SCALING: float = Field(default=1e-4, validation_alias='FINRL_REWARD_SCALING')
    
    FREQAI_DEFAULT_IDENTIFIER: str = Field(default="default_freqai_id", validation_alias='FREQAI_DEFAULT_IDENTIFIER')
    FREQAI_LABEL_PERIOD_CANDLES: int = Field(default=24, validation_alias='FREQAI_LABEL_PERIOD_CANDLES')
    
    # --- Strategy Parameters from Notepad --- 
    # Note: Use Optional[List[str]] for lists loaded from comma-separated env var
    CORE_ASSETS: List[str] = Field(default_factory=lambda: "SPY,QQQ,GLD,SGOV,XLP".split(','), validation_alias='CORE_ASSETS')
    CORE_RSI_THRESHOLD: int = Field(default=30, validation_alias='CORE_RSI_THRESHOLD')
    CORE_VIX_THRESHOLD: int = Field(default=25, validation_alias='CORE_VIX_THRESHOLD')
    CORE_REBALANCE_THRESHOLD: float = Field(default=0.15, validation_alias='CORE_REBALANCE_THRESHOLD')
    # Assuming CHALLENGE_SYMBOLS is a comma-separated string in env
    CHALLENGE_SYMBOLS: List[str] = Field(default_factory=lambda: "BTCUSDT,ETHUSDT".split(','), validation_alias='CHALLENGE_SYMBOLS') 
    CHALLENGE_LEVERAGE: int = Field(default=10, validation_alias='CHALLENGE_LEVERAGE')
    CHALLENGE_SEED_PERCENTAGE: float = Field(default=0.05, validation_alias='CHALLENGE_SEED_PERCENTAGE')
    CHALLENGE_TP_RATIO: float = Field(default=0.06, validation_alias='CHALLENGE_TP_RATIO')
    CHALLENGE_SL_RATIO: float = Field(default=0.03, validation_alias='CHALLENGE_SL_RATIO')
    CHALLENGE_SMA_PERIOD: int = Field(default=7, validation_alias='CHALLENGE_SMA_PERIOD')
    CHALLENGE_RSI_PERIOD: int = Field(default=14, validation_alias='CHALLENGE_RSI_PERIOD')
    CHALLENGE_RSI_THRESHOLD: int = Field(default=30, validation_alias='CHALLENGE_RSI_THRESHOLD')
    CHALLENGE_INTERVAL: str = Field(default="1h", validation_alias='CHALLENGE_INTERVAL')
    CHALLENGE_LOOKBACK: str = Field(default="30d", validation_alias='CHALLENGE_LOOKBACK')
    CHALLENGE_RISK_PER_TRADE: float = Field(default=0.01, validation_alias='CHALLENGE_RISK_PER_TRADE')
    CHALLENGE_VOLUME_AVG_PERIOD: int = Field(default=20, validation_alias='CHALLENGE_VOLUME_AVG_PERIOD')
    CHALLENGE_VOLUME_SURGE_RATIO: float = Field(default=2.0, validation_alias='CHALLENGE_VOLUME_SURGE_RATIO')
    CHALLENGE_DIVERGENCE_LOOKBACK: int = Field(default=28, validation_alias='CHALLENGE_DIVERGENCE_LOOKBACK')
    CHALLENGE_BREAKOUT_LOOKBACK: int = Field(default=40, validation_alias='CHALLENGE_BREAKOUT_LOOKBACK')
    CHALLENGE_PULLBACK_LOOKBACK: int = Field(default=10, validation_alias='CHALLENGE_PULLBACK_LOOKBACK')
    CHALLENGE_PULLBACK_THRESHOLD: float = Field(default=0.01, validation_alias='CHALLENGE_PULLBACK_THRESHOLD')
    CHALLENGE_POC_LOOKBACK: int = Field(default=50, validation_alias='CHALLENGE_POC_LOOKBACK')
    CHALLENGE_POC_THRESHOLD: float = Field(default=0.7, validation_alias='CHALLENGE_POC_THRESHOLD')
    LOG_OVERRIDES_TO_DB: bool = Field(default=False, validation_alias='LOG_OVERRIDES_TO_DB')
    CHALLENGE_SYMBOL_DELAY_SECONDS: float = Field(default=1.0, validation_alias='CHALLENGE_SYMBOL_DELAY_SECONDS')
    
    # --- Database --- 
    DB_HOST: str = Field(default="db", validation_alias='DB_HOST')
    DB_PORT: int = Field(default=5433, validation_alias='DB_PORT')
    DB_USER: str = Field(default="trading_user", validation_alias='DB_USER')
    DB_PASSWORD: str = Field(default="trading_password", validation_alias='DB_PASSWORD')
    DB_NAME: str = Field(default="trading_db", validation_alias='DB_NAME')

    # --- Log Level --- 
    LOG_LEVEL: str = Field(default="INFO", validation_alias='LOG_LEVEL')
    
    # --- Slack Channels (Optional defaults) ---
    SLACK_CHANNEL_DEFAULT: str = Field(default="#tradingalram", validation_alias='SLACK_CHANNEL_DEFAULT')
    SLACK_CHANNEL_STATUS: str = Field(default="#tradingalram", validation_alias='SLACK_CHANNEL_STATUS')
    SLACK_CHANNEL_ALERTS: str = Field(default="#tradingalram", validation_alias='SLACK_CHANNEL_ALERTS')
    SLACK_CHANNEL_ERRORS: str = Field(default="#tradingalram", validation_alias='SLACK_CHANNEL_ERRORS')
    SLACK_CHANNEL_CRITICAL: str = Field(default="#tradingalram", validation_alias='SLACK_CHANNEL_CRITICAL')
    SLACK_CHANNEL_REPORTS: str = Field(default="#tradingalram", validation_alias='SLACK_CHANNEL_REPORTS')

    # --- Feature Flags & Other --- 
    ENABLE_SENTIMENT_ANALYSIS: bool = Field(default=True, validation_alias='ENABLE_SENTIMENT_ANALYSIS')
    ENABLE_VOLATILITY_ALERT: bool = Field(default=False, validation_alias='ENABLE_VOLATILITY_ALERT')
    ENABLE_BACKTEST: bool = Field(default=False, validation_alias='ENABLE_BACKTEST')
    VOLATILITY_THRESHOLD: float = Field(default=5.0, validation_alias='VOLATILITY_THRESHOLD')
    # Add other flags/settings as needed

    # --- Pydantic Settings Config --- 
    class Config:
        env_file = dotenv_path
        env_file_encoding = 'utf-8'
        case_sensitive = False # Environment variable names are case-insensitive
        extra = 'ignore' # Ignore extra fields from env/dotenv

# --- Instantiate Settings --- 
try:
    settings = Settings()
    logger.info(f"[{__name__}] Settings instance created successfully.")
    # Log some values to verify
    logger.info(f" > Loaded NEWS_API_KEY: {'Yes' if settings.NEWS_API_KEY else 'No'}")
    logger.info(f" > Loaded ALPACA_API_KEY_ID: {'Yes' if settings.ALPACA_API_KEY_ID else 'No'}")
    logger.info(f" > Loaded SLACK_WEBHOOK_URL: {'Yes' if settings.SLACK_WEBHOOK_URL else 'No'}")
    logger.info(f" > Alpaca Base URL: {settings.ALPACA_BASE_URL}")
    logger.info(f" > Core Assets: {settings.CORE_ASSETS}")
    logger.info(f" > Log Level: {settings.LOG_LEVEL}")
except ValidationError as ve:
    logger.error(f"[{__name__}] Pydantic ValidationError during Settings instantiation: {ve}", exc_info=False) # Set exc_info=False for cleaner logs
    # Log specific validation errors
    for error in ve.errors():
        logger.error(f"  - Field: {error['loc'][0] if error['loc'] else 'Unknown'}, Error: {error['msg']}")
    settings = None # Set to None on validation error
except Exception as e:
    logger.error(f"[{__name__}] Unexpected error during Settings instantiation: {e}", exc_info=True)
    settings = None

# --- Create Directories & Validate Config (using instantiated settings) ---
IS_CONFIG_VALID = False
if settings:
    def create_dirs(settings_obj: Settings):
        dirs_to_create = [
            settings_obj.MODEL_SAVE_DIR,
            settings_obj.DATA_SAVE_DIR,
            settings_obj.LOG_DIR,
            settings_obj.MODEL_WEIGHTS_DIR,
            # Derived paths are handled by property access
            settings_obj.SENTIMENT_CACHE_DIR 
        ]
        for dir_path in dirs_to_create:
            try:
                os.makedirs(dir_path, exist_ok=True)
                logger.debug(f"Ensured directory exists: {dir_path}")
            except OSError as e:
                logger.error(f"Error creating directory {dir_path}: {e}")
    create_dirs(settings)

    def validate_config(settings_obj: Settings) -> bool:
        """Checks if essential configuration variables are set in the settings object."""
        required_keys = ["NEWS_API_KEY", "ALPACA_API_KEY_ID", "ALPACA_SECRET_KEY"]
        missing_keys = []
        if not settings_obj.NEWS_API_KEY: missing_keys.append("NEWS_API_KEY")
        if not settings_obj.ALPACA_API_KEY_ID: missing_keys.append("ALPACA_API_KEY_ID")
        if not settings_obj.ALPACA_SECRET_KEY: missing_keys.append("ALPACA_SECRET_KEY")
        # Add checks for other critical keys if needed (e.g., SLACK_WEBHOOK_URL for notifications)
        if not settings_obj.SLACK_WEBHOOK_URL:
             logger.warning("SLACK_WEBHOOK_URL is not set. Slack notifications will be disabled.")
             # Consider if this should be 'required' depending on system needs
             # missing_keys.append("SLACK_WEBHOOK_URL") 

        if missing_keys:
            logger.warning(f"Missing essential API keys/config in .env or environment: {missing_keys}. Some functionalities might fail.")
            return False
        logger.info("Essential configuration variables seem to be present.")
        return True
    IS_CONFIG_VALID = validate_config(settings)

else:
    logger.critical("Settings object could not be initialized due to validation errors. Cannot proceed with directory creation or further validation.")
    # Exit or raise error if settings are critical for the application to run
    # sys.exit(1)

# Make settings easily importable
# Example usage in other files:
# from src.config.settings import settings
# if settings and settings.IS_CONFIG_VALID:
#     print(settings.ALPACA_BASE_URL) 