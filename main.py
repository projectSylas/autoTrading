import logging
import schedule
import time
import sys
import os

# --- Project Structure Adjustment (if needed) ---
# Ensure the script can find project modules
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR) # If main.py is in the root
# If main.py is inside src/, adjust: PROJECT_ROOT = os.path.dirname(SRC_DIR)
if PROJECT_ROOT not in sys.path:
     sys.path.append(PROJECT_ROOT)

# --- Setup Logging --- 
try:
    # Assuming a central logging config exists
    from src.utils.logging_config import setup_logging
    setup_logging()
    logger = logging.getLogger(__name__)
except ImportError:
    # Basic fallback logging if config is missing
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.warning("Could not import central logging config. Using basic logging.")

# --- Load Settings --- 
try:
    # Import the settings instance directly
    from src.config.settings import settings 
    if settings is None:
        raise ImportError("Settings object is None after import. Check settings.py for validation errors.")
    logger.info("Successfully loaded application settings instance.")
    # Log some key settings for verification
    logger.info(f"Alpaca Base URL: {settings.ALPACA_BASE_URL}") 
    # logger.info(f"FreqAI Enabled in Config: {settings.FREQAI_ENABLED}") # Example if FREQAI_ENABLED is in settings
except ImportError:
    logger.critical("Failed to import settings module or settings object is None. Cannot proceed.")
    sys.exit(1)
except Exception as e:
    logger.critical(f"An error occurred loading settings: {e}", exc_info=True)
    sys.exit(1)

# --- Import Core Modules --- 
try:
    from src.stock_trader.live_trader_finrl import get_alpaca_api, run_live_trading_finrl
    from src.utils.notifier import send_slack_notification
    # Import other necessary modules as needed
    # from src.stock_trader.agent_trainer import train_finrl_agent
    # from src.reporting.performance_analyzer import analyze_results
except ImportError as ie:
    logger.critical(f"Failed to import core modules: {ie}. Check PYTHONPATH and file structure.")
    sys.exit(1)

# --- Global Variables & State --- 
ALPACA_API = None
SYSTEM_RUNNING = True

# --- Core Functions --- 
def initialize_system():
    """Initializes necessary components like API connections."""
    global ALPACA_API
    logger.info("Initializing system components...")
    
    # Initialize Alpaca API
    logger.info("Attempting to connect to Alpaca API...")
    ALPACA_API = get_alpaca_api() # Function from live_trader_finrl
    if ALPACA_API is None:
        logger.error("Failed to connect to Alpaca API. FinRL trading will be disabled.")
        # Use channel from settings
        send_slack_notification("🚨 CRITICAL: Alpaca API connection failed on startup!", channel=settings.SLACK_CHANNEL_CRITICAL) 
        # Depending on requirements, might exit or continue without Alpaca
        # sys.exit(1) 
    else:
        logger.info("Alpaca API connection successful.")
        # Use channel from settings
        send_slack_notification("✅ System Initialized: Alpaca API Connected.", channel=settings.SLACK_CHANNEL_STATUS)
        
    # Add initialization for other components (e.g., database connection)
    logger.info("System initialization complete.")

def run_finrl_trading_task():
    """Defines the scheduled task for running FinRL live trading."""
    logger.info("--- Starting FinRL Trading Task --- ")
    if ALPACA_API is None:
        logger.warning("Skipping FinRL trading task: Alpaca API not available.")
        return
        
    # --- Configuration for the FinRL Task ---
    # TODO: Get these from settings or a dynamic source
    ticker_to_trade = "AAPL" 
    agent_type = "ppo"
    # Ensure the model path uses the settings directory
    model_filename = f"{agent_type}_{ticker_to_trade}_live_placeholder.zip" # Needs actual trained model name
    agent_path = os.path.join(settings.FINRL_MODEL_DIR, model_filename)
    trade_quantity = settings.FINRL_HMAX // 10 if settings.FINRL_HMAX >=10 else 1 # Example quantity based on settings
    
    logger.info(f"Running FinRL live trading for {ticker_to_trade}...")
    logger.info(f" > Agent Path: {agent_path}")
    logger.info(f" > Trade Quantity: {trade_quantity}")
    
    if not os.path.exists(agent_path):
        logger.error(f"FinRL Agent model not found at {agent_path}. Cannot execute trading task.")
        # Use channel from settings
        send_slack_notification(f"🚨 ERROR: FinRL agent model not found for {ticker_to_trade}! Path: {agent_path}", channel=settings.SLACK_CHANNEL_ERRORS)
        return

    try:
        # Execute the live trading function from the imported module
        run_live_trading_finrl(ticker=ticker_to_trade,
                               agent_path=agent_path,
                               agent_type=agent_type,
                               trade_quantity=trade_quantity
                               # Pass tech_indicator_list if required by the state constructor
                               )
        logger.info(f"FinRL trading task completed for {ticker_to_trade}.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during FinRL trading task: {e}", exc_info=True)
        # Use channel from settings
        send_slack_notification(f"🔥 CRITICAL Error in FinRL Trading Task for {ticker_to_trade}: {e}", channel=settings.SLACK_CHANNEL_CRITICAL)
        
    logger.info("--- Finished FinRL Trading Task --- ")

def check_freqtrade_status():
    """Placeholder function to check if Freqtrade process is running (if applicable)."""
    # This would typically involve checking process lists or a status file/API endpoint
    logger.info("Checking Freqtrade status (Placeholder - requires implementation).")
    # Example: Check if a process named 'freqtrade' is running
    # Or: Check a heartbeat file updated by Freqtrade

def run_daily_report():
    """Placeholder for generating and sending a daily summary report."""
    logger.info("Generating daily report (Placeholder)...")
    # 1. Aggregate results from logs or database
    # 2. Format the report
    # 3. Send via Slack
    try:
        # Placeholder logic
        report_content = "Daily Trading Summary:\n- FinRL Trades: 2\n- FreqAI Trades: 5\n- P&L: +$123.45"
        # Use channel from settings
        send_slack_notification(f"📊 Daily Report ({time.strftime('%Y-%m-%d')})\n```\n{report_content}\n```", channel=settings.SLACK_CHANNEL_REPORTS)
        logger.info("Daily report sent.")
    except Exception as e:
        logger.error(f"Failed to generate or send daily report: {e}", exc_info=True)

# --- Scheduler Setup --- 
def setup_scheduler():
    """Configures the scheduled jobs."""
    logger.info("Setting up scheduled tasks...")
    
    # Example Schedule:
    # Run FinRL trading task every hour at 5 minutes past the hour
    schedule.every().hour.at(":05").do(run_finrl_trading_task)
    logger.info("Scheduled FinRL trading task hourly at HH:05.")
    
    # Run Freqtrade status check every 15 minutes
    # schedule.every(15).minutes.do(check_freqtrade_status)
    # logger.info("Scheduled Freqtrade status check every 15 minutes.")
    
    # Run daily report every day at 17:00
    schedule.every().day.at("17:00").do(run_daily_report)
    logger.info("Scheduled daily report task daily at 17:00.")
    
    logger.info("Scheduler setup complete.")

# --- Main Execution --- 
if __name__ == "__main__":
    logger.info("============================================")
    logger.info(" Starting Automated Trading System Main     ")
    logger.info("============================================")
    
    initialize_system()
    setup_scheduler()
    
    logger.info("System running. Waiting for scheduled tasks...")
    # Send startup confirmation - Use channel from settings
    send_slack_notification("🚀 Automated Trading System Started Successfully!", channel=settings.SLACK_CHANNEL_STATUS)

    # Keep the script running to execute scheduled jobs
    while SYSTEM_RUNNING:
        try:
            schedule.run_pending()
            time.sleep(1) # Sleep briefly to avoid high CPU usage
        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt received. Shutting down...")
            SYSTEM_RUNNING = False
        except Exception as e:
            logger.error(f"An error occurred in the main loop: {e}", exc_info=True)
            # Potentially send a critical alert here - Use channel from settings
            send_slack_notification(f"🔥 CRITICAL Error in Main Loop: {e}. System may be unstable.", channel=settings.SLACK_CHANNEL_CRITICAL)
            # Decide whether to attempt recovery or shut down
            # For now, continue running but log the error
            time.sleep(5) # Longer sleep after error

    logger.info("--------------------------------------------")
    logger.info(" Automated Trading System Shutting Down     ")
    logger.info("--------------------------------------------")
    # Send shutdown confirmation - Use channel from settings
    send_slack_notification("🛑 Automated Trading System Shutting Down.", channel=settings.SLACK_CHANNEL_STATUS) 