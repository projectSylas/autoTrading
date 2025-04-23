# src/backtesting/backtest_runner.py

import backtrader as bt
import backtrader.feeds as btfeeds
import pandas as pd
from datetime import datetime
import os
import logging
from typing import Optional
import yfinance as yf # Import yfinance

# Import the strategy class
from src.backtesting.strategies.challenge_strategy_backtest import ChallengeStrategyBacktest
# Import settings if needed for data paths or parameters
from src.config.settings import settings
# Import data fetching utility if needed
from src.utils.common import get_historical_data # Assuming this function exists
# Import logging setup function and logger name
from src.utils.logging_config import setup_logging, APP_LOGGER_NAME # Import logger name

# --- Setup Logging ---
# Initial setup - call once when module loads
# setup_logging() # <<< REMOVED from top level
# --- End Setup Logging ---

# --- Get the application-wide logger ---
logger = logging.getLogger(APP_LOGGER_NAME) # Get logger instance first

# --- Function to print logger handlers --- 
def print_logger_handlers(stage: str):
    current_logger = logging.getLogger(APP_LOGGER_NAME)
    handlers = current_logger.handlers
    print(f"--- Logger '{APP_LOGGER_NAME}' Handlers ({stage}) ---")
    if not handlers:
        print("  No handlers found.")
    else:
        for i, handler in enumerate(handlers):
            print(f"  Handler {i}: {type(handler).__name__}, Level: {logging.getLevelName(handler.level)}")
            if isinstance(handler, logging.FileHandler):
                print(f"    -> File: {handler.baseFilename}")
    print("--------------------------------------------")

# --- Minimal Strategy for Debugging --- 
# class MinimalStrategy(bt.Strategy):
#     def log(self, txt, dt=None):
#         dt = dt or self.datas[0].datetime.date(0)
#         print(f'{dt.isoformat()}, {txt}')
# 
#     def __init__(self):
#         self.dataclose = self.datas[0].close
#         print("MinimalStrategy Initialized")
# 
#     def next(self):
#         self.log(f'Close, {self.dataclose[0]:.2f}')
# --- End Minimal Strategy ---

# --- Debugging Settings Import ---
# print(f"DEBUG: Type of settings object: {type(settings)}")
# print(f"DEBUG: Attributes of settings object: {dir(settings)}")
# --- End Debugging ---

def run_backtest(symbol: str,
                 from_date: str,
                 to_date: str,
                 interval: str = '1h',
                 initial_cash: float = 10000.0,
                 commission: float = 0.001, # Example commission (0.1%)
                 stake: int = 10, # Example fixed stake per trade
                 data_source: str = 'yahoo', # 'yahoo', 'db', 'csv', etc.
                 csv_path: Optional[str] = None,
                 plot: bool = True,
                 optimize: bool = False,
                 printlog: bool = False,
                 **kwargs):
    """
    Runs a backtrader backtest or optimization for the given symbol and period.

    Args:
        symbol (str): The trading symbol (e.g., 'BTC-USD' for Yahoo).
        from_date (str): Start date in 'YYYY-MM-DD' format.
        to_date (str): End date in 'YYYY-MM-DD' format.
        interval (str): Data interval (e.g., '1h', '1d'). Needs compatibility with data source.
        initial_cash (float): Starting cash for the backtest.
        commission (float): Commission per trade (e.g., 0.001 for 0.1%).
        stake (int): Default size for orders.
        data_source (str): Where to get the data from ('yahoo', 'csv', 'db').
        csv_path (Optional[str]): Path to the CSV file if data_source is 'csv'.
        plot (bool): Whether to generate a plot (ignored during optimization).
        optimize (bool): Whether to run parameter optimization instead of a single backtest.
        printlog (bool): Whether to enable detailed strategy logging.
        **kwargs: Additional keyword arguments for strategy parameters.
    """
    # --- Use the application logger ---
    logger.info(f"Starting backtest for {symbol} from {from_date} to {to_date}")

    # --- Prepare Strategy Parameters from Settings ---
    strategy_params_from_settings = {
        'sma_long_period': settings.CHALLENGE_SMA_PERIOD,
        'sma_short_period': 7, # Assuming 7 is fixed or add CHALLENGE_SMA_SHORT_PERIOD to settings
        'rsi_period': settings.CHALLENGE_RSI_PERIOD,
        'rsi_threshold': settings.CHALLENGE_RSI_THRESHOLD,
        'volume_avg_period': settings.CHALLENGE_VOLUME_AVG_PERIOD,
        'volume_surge_ratio': settings.CHALLENGE_VOLUME_SURGE_RATIO,
        'divergence_lookback': settings.CHALLENGE_DIVERGENCE_LOOKBACK,
        'breakout_lookback': settings.CHALLENGE_BREAKOUT_LOOKBACK,
        'pullback_lookback': settings.CHALLENGE_PULLBACK_LOOKBACK,
        'poc_lookback': settings.CHALLENGE_POC_LOOKBACK,
        'poc_threshold': settings.CHALLENGE_POC_THRESHOLD,
        'sl_ratio': settings.CHALLENGE_SL_RATIO,
        'tp_ratio': settings.CHALLENGE_TP_RATIO,
        'printlog': printlog,
        'pullback_threshold': settings.CHALLENGE_PULLBACK_THRESHOLD,
    }
    # --- Use the application logger ---
    logger.info(f"Prepared base strategy params: {strategy_params_from_settings}")
    # --- End Prepare Strategy Parameters ---

    cerebro = bt.Cerebro()

    # --- Setup Logging AFTER Cerebro initialization --- 
    setup_logging() # <<< MOVED HERE
    logger.info("Logging setup called after Cerebro initialization.")
    # --- End Setup Logging ---

    # --- Add Strategy or OptStrategy ---
    if optimize:
        # --- Use the application logger ---
        logger.info("Setting up strategy optimization...")
        # Define optimization ranges
        opt_params = {
            'divergence_lookback': [10, 14, 20],
            'breakout_lookback': [20, 30, 40],
            'poc_threshold': [0.6, 0.7, 0.8]
        }
        # *** 수정: 기본 파라미터에서 최적화 대상 키 제거 ***
        base_params_for_opt = strategy_params_from_settings.copy()
        for key_to_opt in opt_params.keys():
            if key_to_opt in base_params_for_opt:
                del base_params_for_opt[key_to_opt]
        # ***********************************************

        # Add strategy for optimization
        cerebro.optstrategy(
            ChallengeStrategyBacktest,
            **base_params_for_opt, # 최적화 키가 제거된 기본 파라미터
            **opt_params # Add optimization ranges
        )
        # --- Use the application logger ---
        logger.info(f"Optimization parameters: {opt_params}")
    else:
        # --- Use the application logger ---
        logger.info("Setting up single strategy run...")
        single_run_params = strategy_params_from_settings.copy()
        single_run_params.update(kwargs) # 단일 실행 시 kwargs로 받은 값으로 덮어쓰기
        # --- Use the application logger ---
        logger.info(f"Single run parameters: {single_run_params}")
        cerebro.addstrategy(ChallengeStrategyBacktest, **single_run_params)

    # --- Add Data Feed ---
    data = None
    fromdate = datetime.strptime(from_date, '%Y-%m-%d')
    todate = datetime.strptime(to_date, '%Y-%m-%d')

    try:
        if data_source.lower() == 'yahoo':
            # --- Use the application logger ---
            logger.info(f"Fetching data for {symbol} from yfinance ({interval})...")
            # Use yfinance directly
            df = yf.download(symbol, start=from_date, end=todate, interval=interval)
            
            if df.empty:
                # --- Use the application logger ---
                logger.error(f"No data fetched from yfinance for {symbol} in the specified period.")
                return None

            # --- Flatten MultiIndex Columns if they exist ---
            if isinstance(df.columns, pd.MultiIndex):
                # Keep only the first level (e.g., 'Open', 'High') and drop the ticker level
                df.columns = df.columns.droplevel(1)
                # --- Use the application logger ---
                logger.info("Flattened MultiIndex columns.")

            # --- Ensure required columns and correct naming for backtrader --- 
            # Make column names lower case to match rename keys and PandasData mapping
            df.columns = df.columns.str.lower()
            df.rename(columns={
                # 'Open': 'open', # Already lowercase now
                # 'High': 'high',
                # 'Low': 'low',
                # 'Close': 'close',
                'adj close': 'adj_close', # Rename specifically if needed
                # 'Volume': 'volume'
            }, inplace=True)
            
            # Make sure standard OHLCV columns exist AFTER potential flattening/lowercasing
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                # --- Use the application logger ---
                logger.error(f"Fetched data is missing required columns: {required_cols}. Columns found: {df.columns}")
                return None
                
            # Use adj_close as close if preferred and available
            # if 'adj_close' in df.columns:
            #     df['close'] = df['adj_close']

            # Reset the name of the columns index if it exists
            if df.columns.name:
                 # --- Use the application logger ---
                 logger.debug(f"DEBUG: Resetting columns name from '{df.columns.name}' to None.")
                 df.columns.name = None

            # --- Explicitly print columns before PandasData ---
            # Ensure logging level is set to DEBUG to see these messages
            # --- Use the application logger ---
            logger.debug(f"DEBUG: Columns before PandasData: {df.columns}")
            logger.debug(f"DEBUG: DataFrame index type: {type(df.index)}")
            logger.debug(f"DEBUG: DataFrame head right before PandasData:\n{df.head()}")
            # --- End Explicit Debug ---

            # --- Explicitly print DataFrame info before feeding to backtrader ---
            # --- Use the application logger ---
            logger.debug(f"DEBUG: DataFrame info before PandasData:\n{df.info()}")
            logger.debug(f"DEBUG: DataFrame head before PandasData:\n{df.head()}")
            # <<< ADDED: Detailed DataFrame Check >>>
            logger.debug("--- Detailed DataFrame Check Before PandasData ---")
            logger.debug(f"Index Type: {type(df.index)}")
            logger.debug(f"Index is DatetimeIndex: {isinstance(df.index, pd.DatetimeIndex)}")
            logger.debug(f"Index Name: {df.index.name}")
            logger.debug(f"Column Names: {df.columns.tolist()}")
            logger.debug(f"Column Data Types:\n{df.dtypes}")
            # Check for NaNs
            nan_check = df[required_cols].isnull().sum()
            logger.debug(f"NaN count per required column:\n{nan_check}")
            if nan_check.sum() > 0:
                logger.warning("NaN values detected in required columns before creating PandasData feed!")
            logger.debug("--------------------------------------------------")
            # <<< END ADDED >>>

            # data = bt.feeds.PandasData(dataname=df) # Original call
            # Create PandasData feed with explicit column mapping
            data = bt.feeds.PandasData(
                dataname=df,
                datetime=None,  # Use the DataFrame index as datetime
                open='open',
                high='high',
                low='low',
                close='close',
                volume='volume',
                openinterest=-1 # Explicitly tell backtrader there is no open interest column
            )
            # print(f"DEBUG: PandasData object created from yfinance fetch: {data}") # Remove debug print

        elif data_source.lower() == 'csv':
            if csv_path and os.path.exists(csv_path):
                # --- Use the application logger ---
                logger.info(f"Loading data for {symbol} from CSV: {csv_path}")
                # Load dataframe first to check/rename columns
                temp_df = pd.read_csv(csv_path)
                # --- Check/Rename CSV columns --- 
                # Assuming CSV has columns like Date, Open, High, Low, Close, Volume
                # Adjust based on actual CSV format
                temp_df.rename(columns={
                    'Date': 'datetime', # Example, adjust as needed
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                }, inplace=True)
                
                # Convert datetime column if it's not the index
                if 'datetime' in temp_df.columns:
                   try:
                       temp_df['datetime'] = pd.to_datetime(temp_df['datetime'])
                       temp_df.set_index('datetime', inplace=True)
                   except Exception as e:
                       # --- Use the application logger ---
                       logger.error(f"Error processing datetime column in CSV: {e}")
                       return None
                else: # Assume index is datetime
                    try:
                       temp_df.index = pd.to_datetime(temp_df.index)
                    except Exception as e:
                       # --- Use the application logger ---
                       logger.error(f"Error processing datetime index in CSV: {e}")
                       return None

                # Filter date range AFTER loading
                temp_df = temp_df[(temp_df.index >= fromdate) & (temp_df.index <= todate)]

                required_cols_csv = ['open', 'high', 'low', 'close', 'volume']
                if not all(col in temp_df.columns for col in required_cols_csv):
                    # --- Use the application logger ---
                    logger.error(f"CSV data is missing required columns: {required_cols_csv}. Columns found: {temp_df.columns}")
                    return None
                    
                if temp_df.empty:
                    # --- Use the application logger ---
                    logger.error(f"No data found in CSV for the specified date range.")
                    return None
                    
                # data = bt.feeds.PandasData(dataname=temp_df) # Original CSV call
                data = bt.feeds.PandasData(
                    dataname=temp_df,
                    datetime=None,
                    open='open',
                    high='high',
                    low='low',
                    close='close',
                    volume='volume',
                    openinterest=-1
                )
                # print(f"DEBUG: PandasData object created from CSV: {data}") # Remove debug print
            else:
                # --- Use the application logger ---
                logger.error(f"CSV path not provided or file does not exist: {csv_path}")
                return None
        elif data_source.lower() == 'db':
            # --- Use the application logger ---
            logger.info(f"Fetching data for {symbol} from database...")
            df_db = get_historical_data(symbol, interval, start_date=from_date, end_date=to_date)
            if df_db is not None and not df_db.empty:
                # --- Ensure correct columns from DB --- 
                df_db.rename(columns={
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                }, inplace=True)
                # Ensure index is datetime
                try:
                    df_db.index = pd.to_datetime(df_db.index)
                except Exception as e:
                    # --- Use the application logger ---
                    logger.error(f"Error processing index as datetime from DB: {e}")
                    return None
                      
                required_cols_db = ['open', 'high', 'low', 'close', 'volume']
                if not all(col in df_db.columns for col in required_cols_db):
                    # --- Use the application logger ---
                    logger.error(f"DB data is missing required columns: {required_cols_db}. Columns found: {df_db.columns}")
                    return None
                       
                # data = bt.feeds.PandasData(dataname=df_db) # Original DB call
                data = bt.feeds.PandasData(
                    dataname=df_db,
                    datetime=None,
                    open='open',
                    high='high',
                    low='low',
                    close='close',
                    volume='volume',
                    openinterest=None
                )
                # print(f"DEBUG: PandasData object created from DB: {data}") # Remove debug print
            else:
                # --- Use the application logger ---
                logger.error(f"Failed to fetch data from database/internal function for {symbol}.")
                return None
        else:
            # --- Use the application logger ---
            logger.error(f"Unsupported data source: {data_source}")
            return None

        # Replace 'if data:' with explicit None check
        if data is not None:
            # --- Use the application logger ---
            logger.debug(f"DEBUG: Data object created ({type(data)}), adding to cerebro.")
            cerebro.adddata(data)
            # --- Use the application logger ---
            logger.info("Data feed added to Cerebro.")
        else:
             # Log error before potentially returning
             # --- Use the application logger ---
             logger.error("Failed to load data feed (data object is None).")
             return None # Return None if data object is None after creation attempt

    except Exception as e:
        # --- Use the application logger ---
        logger.error(f"Error loading data feed: {e}", exc_info=True)
        return None

    # --- Configure Cerebro ---
    cerebro.broker.setcash(initial_cash)
    # Set commission
    cerebro.broker.setcommission(commission=commission)
    # Add stake size
    cerebro.addsizer(bt.sizers.FixedSize, stake=stake)
    # --- Use the application logger ---
    logger.info(f"Cerebro configured: Initial Cash=${initial_cash:,.2f}, Commission={commission*100:.3f}%, Stake={stake}")

    # --- Add Analyzers --- 
    # Add TradeAnalyzer regardless of mode to get trade counts in optimization
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')
    # Add Returns analyzer to calculate final value from total return
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns', timeframe=bt.TimeFrame.Days)

    if not optimize: # Add other analyzers only for single runs
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio', timeframe=bt.TimeFrame.Days)
        # cerebro.addanalyzer(bt.analyzers.Returns, _name='returns', timeframe=bt.TimeFrame.Days) # Already added above
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

    # --- Run Backtest/Optimization ---
    run_results = None # <<< Initialize outside try
    strategy_outputs = None # <<< Initialize outside try
    try:
        print_logger_handlers("Before cerebro.run()") # Debug: Print handlers before running
        # --- Use the application logger ---
        logger.info("Running Cerebro...")
        if optimize:
            strategy_outputs = cerebro.run(maxcpus=1) # Use maxcpus=1 for simpler debugging if needed
        else:
            run_results = cerebro.run()
        # --- Use the application logger ---
        logger.info("Cerebro run completed.")

    finally:
        # --- Ensure logging is potentially restored even if run fails ---
        print_logger_handlers("After try/finally (Before Re-Setup)") # Debug: Print handlers
        # Call setup_logging again, forcing reset if handlers might have been removed
        setup_logging(force_reset=True) # <<< Force reset
        logger.info("Logging setup potentially re-applied after Cerebro run.")
        print_logger_handlers("After Re-Setup") # Debug: Print handlers after reset attempt

    # --- Process Results --- 
    if optimize:
        # --- Use the application logger ---
        logger.info("Processing optimization results...")
        # Print Optimization Results
        # ... (Optimization result processing logic needed) ...
        print("Optimization results:", strategy_outputs) # Placeholder
        # Return optimization results directly
        return strategy_outputs
    elif run_results and isinstance(run_results, list) and len(run_results) > 0 and isinstance(run_results[0], bt.Strategy):
        # --- Added check for valid run_results list containing strategy instance ---
        # --- Use the application logger ---
        logger.info("Processing single run results...")
        # --- Get Final Portfolio Value --- 
        try:
            final_value = cerebro.broker.getvalue()
            initial_value = cerebro.broker.startingcash
            pnl = final_value - initial_value
            pnl_percent = (pnl / initial_value) * 100
            print('\n--- Backtest Results ---')
            print(f'Starting Portfolio Value: {initial_value:,.2f}')
            print(f'Final Portfolio Value:    {final_value:,.2f}')
            print(f'Net Profit/Loss:        {pnl:,.2f}')
            print(f'Net Profit/Loss (%):    {pnl_percent:.2f}%')
            print('------------------------')
        except Exception as e:
            logger.error(f"Error calculating final portfolio value: {e}", exc_info=True)
        # --- End Final Portfolio Value --- 

        # --- Plotting --- 
        if plot and not optimize:
            # --- Use the application logger ---
            logger.info("Generating plot...")
            try:
                # Increase figure size for better readability
                # Ensure plot is only called if cerebro ran successfully
                cerebro.plot(style='candlestick', barup='green', bardown='red', iplot=True, volume=True, figscale=1.5)
            except AttributeError as ae:
                logger.error(f"Plotting failed due to AttributeError (check if backtest ran correctly): {ae}", exc_info=True)
            except Exception as e:
                logger.error(f"Error during plotting: {e}", exc_info=True)
        # --- End Plotting ---

        # Return the first strategy instance's results (if any)
        return run_results[0]
    else:
        # --- Use the application logger ---
        logger.error(f"Backtest run did not produce valid results or strategy instance. run_results: {run_results}")
        return None

# --- Main Execution Block --- 
if __name__ == "__main__":
    # --- Setup Logging at the very start of the main block ---
    # Ensure logs capture setup issues if run_backtest isn't called
    setup_logging()
    logger = logging.getLogger(APP_LOGGER_NAME) # Re-get logger after setup
    logger.info("=== Backtest Runner Script Started ===")
    # --- End Setup Logging ---

    # --- Example Usage ---
    symbol = 'BTC-USD'
    start_date = '2023-10-01' # <<< 변경: 데이터 조회 가능 기간으로 수정
    end_date = '2023-12-30' # Use a recent date for testing
    # interval = '1d' # Changed to daily for faster testing
    # interval = '1h' # Switch back to hourly if needed
    # interval = settings.CHALLENGE_INTERVAL # Use interval from settings
    
    # Determine interval based on settings or default
    try:
        interval_setting = settings.CHALLENGE_INTERVAL
        logger.info(f"Using interval from settings: {interval_setting}")
    except AttributeError:
        interval_setting = '1h' # Default if not in settings
        logger.warning(f"CHALLENGE_INTERVAL not found in settings, defaulting to '{interval_setting}'")
    interval = interval_setting

    data_source = 'yahoo' # or 'csv', 'db'

    # --- Use the application logger ---
    logger.info(f"Running backtest for: {symbol}")
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info(f"Interval: {interval}")
    logger.info(f"Data Source: {data_source}")

    # --- Run Backtest --- 
    results = run_backtest(
        symbol=symbol,
        from_date=start_date,
        to_date=end_date,
        interval=interval,
        initial_cash=10000.0,
        commission=0.001,
        stake=1, # Example stake size for BTC
        data_source=data_source,
        # csv_path='path/to/your/data.csv', # Uncomment if using CSV
        plot=True, # Enable plotting for single runs
        optimize=False, # Set to True to run optimization
        printlog=True # Enable detailed strategy logging (redundant with file logging?)
        # --- Pass additional strategy params via kwargs if needed ---
        # Example: Override specific parameters for this run
        # poc_threshold=0.8 
    )

    # --- Check Results --- 
    if results:
        # If optimize=True, results will be list of lists from optstrategy
        # If optimize=False, results might be the strategy instance (or None)
        if isinstance(results, bt.Strategy):
            logger.info("Backtest finished successfully. Strategy instance returned.")
            # You can access analyzers here if added, e.g., results.analyzers.sharpe.get_analysis()
        elif isinstance(results, list): # Check if it looks like optimization output
            logger.info("Optimization finished successfully. Results list returned.")
            # Process optimization results
        else:
            logger.warning(f"Backtest finished, but the returned result type is unexpected: {type(results)}")
    else:
        logger.error("Backtest failed or produced no results.")

    logger.info("=== Backtest Runner Script Finished ===")
