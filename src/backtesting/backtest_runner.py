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

# Setup logging
logging.basicConfig(level=settings.LOG_LEVEL.upper(), format='%(asctime)s - %(levelname)s - %(message)s')

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
        **kwargs: Additional keyword arguments for strategy parameters.
    """
    # --- Temporarily Force DEBUG Logging --- 
    # Get the root logger
    root_logger = logging.getLogger()
    # Store original level if needed
    original_level = root_logger.level 
    # Force DEBUG level for this function execution
    root_logger.setLevel(logging.DEBUG)
    logging.debug("DEBUG logging forced for run_backtest execution.")
    # --- End Temporary Force --- 

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
        'printlog': settings.LOG_LEVEL.upper() == 'DEBUG',
        'pullback_threshold': settings.CHALLENGE_PULLBACK_THRESHOLD,
    }
    logging.info(f"Prepared base strategy params: {strategy_params_from_settings}")
    # --- End Prepare Strategy Parameters ---

    cerebro = bt.Cerebro()

    # --- Add Strategy or OptStrategy ---
    if optimize:
        logging.info("Setting up strategy optimization...")
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
        logging.info(f"Optimization parameters: {opt_params}")
    else:
        logging.info("Setting up single strategy run...")
        single_run_params = strategy_params_from_settings.copy()
        single_run_params.update(kwargs) # 단일 실행 시 kwargs로 받은 값으로 덮어쓰기
        logging.info(f"Single run parameters: {single_run_params}")
        cerebro.addstrategy(ChallengeStrategyBacktest, **single_run_params)

    # --- Add Data Feed ---
    data = None
    fromdate = datetime.strptime(from_date, '%Y-%m-%d')
    todate = datetime.strptime(to_date, '%Y-%m-%d')

    try:
        if data_source.lower() == 'yahoo':
            logging.info(f"Fetching data for {symbol} from yfinance ({interval})...")
            # Use yfinance directly
            df = yf.download(symbol, start=from_date, end=todate, interval=interval)
            
            if df.empty:
                logging.error(f"No data fetched from yfinance for {symbol} in the specified period.")
                return None

            # --- Flatten MultiIndex Columns if they exist ---
            if isinstance(df.columns, pd.MultiIndex):
                # Keep only the first level (e.g., 'Open', 'High') and drop the ticker level
                df.columns = df.columns.droplevel(1)
                logging.info("Flattened MultiIndex columns.")

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
                logging.error(f"Fetched data is missing required columns: {required_cols}. Columns found: {df.columns}")
                return None
                
            # Use adj_close as close if preferred and available
            # if 'adj_close' in df.columns:
            #     df['close'] = df['adj_close']

            # Reset the name of the columns index if it exists
            if df.columns.name:
                 logging.debug(f"DEBUG: Resetting columns name from '{df.columns.name}' to None.")
                 df.columns.name = None

            # --- Debug: Print DataFrame info before feeding to backtrader ---
            # print("--- DataFrame Info before PandasData ---")
            # print(df.info())
            # print("--- DataFrame Head before PandasData ---")
            # print(df.head())
            # print("-----------------------------------------")
            # --- End Debug ---

            # --- Explicitly print columns before PandasData ---
            # Ensure logging level is set to DEBUG to see these messages
            logging.debug(f"DEBUG: Columns before PandasData: {df.columns}")
            logging.debug(f"DEBUG: DataFrame index type: {type(df.index)}")
            logging.debug(f"DEBUG: DataFrame head right before PandasData:\n{df.head()}")
            # --- End Explicit Debug ---

            # --- Explicitly print DataFrame info before feeding to backtrader ---
            logging.debug(f"DEBUG: DataFrame info before PandasData:\n{df.info()}")
            logging.debug(f"DEBUG: DataFrame head before PandasData:\n{df.head()}")

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
                openinterest=None # Explicitly tell backtrader there is no open interest column
            )
            # print(f"DEBUG: PandasData object created from yfinance fetch: {data}") # Remove debug print

        elif data_source.lower() == 'csv':
            if csv_path and os.path.exists(csv_path):
                logging.info(f"Loading data for {symbol} from CSV: {csv_path}")
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
                       logging.error(f"Error processing datetime column in CSV: {e}")
                       return None
                else: # Assume index is datetime
                    try:
                       temp_df.index = pd.to_datetime(temp_df.index)
                    except Exception as e:
                       logging.error(f"Error processing index as datetime in CSV: {e}")
                       return None

                # Filter date range AFTER loading
                temp_df = temp_df[(temp_df.index >= fromdate) & (temp_df.index <= todate)]

                required_cols_csv = ['open', 'high', 'low', 'close', 'volume']
                if not all(col in temp_df.columns for col in required_cols_csv):
                    logging.error(f"CSV data is missing required columns: {required_cols_csv}. Columns found: {temp_df.columns}")
                    return None
                    
                if temp_df.empty:
                    logging.error(f"No data found in CSV for the specified date range.")
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
                    openinterest=None
                )
                # print(f"DEBUG: PandasData object created from CSV: {data}") # Remove debug print
            else:
                 logging.error(f"CSV file not found or path not provided: {csv_path}")
                 return None
        elif data_source.lower() == 'db':
            logging.info(f"Fetching data for {symbol} from database/internal function...")
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
                    logging.error(f"Error processing index as datetime from DB: {e}")
                    return None
                      
                required_cols_db = ['open', 'high', 'low', 'close', 'volume']
                if not all(col in df_db.columns for col in required_cols_db):
                    logging.error(f"DB data is missing required columns: {required_cols_db}. Columns found: {df_db.columns}")
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
                logging.error(f"Failed to fetch data from database/internal function for {symbol}.")
                return None
        else:
            logging.error(f"Unsupported data source: {data_source}")
            return None

        # Replace 'if data:' with explicit None check
        if data is not None:
            logging.debug(f"DEBUG: Data object created ({type(data)}), adding to cerebro.")
            cerebro.adddata(data)
        else:
             # Log error before potentially returning
             logging.error("Failed to load data feed (data object is None).")
             # Restore logger level before returning
             root_logger.setLevel(original_level)
             logging.debug("Restored original logging level due to data loading failure.")
             return None # Return None if data object is None after creation attempt

    except Exception as e:
        logging.error(f"Error loading data feed: {e}", exc_info=True)
        # Restore logger level before returning on exception
        root_logger.setLevel(original_level)
        logging.debug("Restored original logging level due to exception during data loading.")
        return None

    # --- Configure Cerebro ---
    cerebro.broker.setcash(initial_cash)
    # Set commission
    cerebro.broker.setcommission(commission=commission)
    # Add stake size
    cerebro.addsizer(bt.sizers.FixedSize, stake=stake)

    # --- Add Analyzers --- 
    # Add TradeAnalyzer regardless of mode to get trade counts in optimization
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')
    # Add Returns analyzer to calculate final value from total return
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns', timeframe=bt.TimeFrame.Days)

    if not optimize: # Add other analyzers only for single runs
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio', timeframe=bt.TimeFrame.Days)
        # cerebro.addanalyzer(bt.analyzers.Returns, _name='returns', timeframe=bt.TimeFrame.Days) # Already added above
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

    # --- Run Backtest or Optimization --- 
    logging.info(f'Starting portfolio value: {cerebro.broker.getvalue():.2f}')
    if optimize:
        logging.info("Running optimization...")
        run_results = cerebro.run(maxcpus=1) # Use maxcpus=1 for simplicity, or None for auto
        logging.info("Optimization finished.")
        
        # --- Optimization Results Analysis (Corrected Yet Again++) --- 
        print("\n--- Optimization Results ---")
        param_names = list(opt_params.keys())
        results_list = []

        # +++ Add Debugging for run_results +++
        try:
            print(f"DEBUG: Type of run_results: {type(run_results)}")
            if isinstance(run_results, list):
                print(f"DEBUG: Length of run_results list: {len(run_results)}")
                # Print type of first few elements if list is not empty
                if run_results:
                    print(f"DEBUG: Type of run_results[0]: {type(run_results[0])}")
                    if len(run_results) > 1:
                        print(f"DEBUG: Type of run_results[1]: {type(run_results[1])}")
                    # Optionally check if elements are lists themselves
                    if isinstance(run_results[0], list):
                        print(f"DEBUG: Length of run_results[0]: {len(run_results[0])}")
                        if run_results[0]:
                            print(f"DEBUG: Type of run_results[0][0]: {type(run_results[0][0])}")
                        else:
                            print(f"DEBUG: run_results[0] is an empty list.")
                    elif run_results[0] is None:
                        print(f"DEBUG: run_results[0] is None.")
                else:
                    print(f"DEBUG: run_results list is empty.")
            else:
                print(f"DEBUG: run_results is not a list.")
        except Exception as e_debug:
            print(f"ERROR during run_results debugging: {e_debug}")
        # +++ End Debugging +++

        print(f"DEBUG: Starting optimization results processing loop. Total runs: {len(run_results)}")
        for i, run in enumerate(run_results):
            print(f"DEBUG: Processing run #{i}")
            # Check if 'run' itself is empty or None (handles IndexError)
            if not run:
                logging.warning(f"Skipping empty or None result for run #{i}")
                continue
            # Check if run[0] (strategy instance) exists and is not None
            if not run[0]:
                 logging.warning(f"Strategy instance (run[0]) is None or empty for run #{i}. Skipping.")
                 continue
            opt_result_obj = run[0]

            try:
                params = opt_result_obj.params
                param_values = tuple(getattr(params, p) for p in param_names)
                print(f"DEBUG: Run #{i} - Params: {dict(zip(param_names, param_values))}")

                # --- Safely Access Analyzers ---
                analyzers_collection = getattr(opt_result_obj, 'analyzers', None) # Get analyzers collection safely
                returns_analysis = {}
                trade_analysis = {}

                if analyzers_collection: # Check if analyzers collection exists and is not None
                    returns_analyzer = getattr(analyzers_collection, 'returns', None)
                    trade_analyzer = getattr(analyzers_collection, 'trade_analyzer', None)

                    # Process Returns Analyzer
                    if returns_analyzer:
                        analysis_result = returns_analyzer.get_analysis()
                        if analysis_result is not None:
                            returns_analysis = analysis_result
                            print(f"DEBUG: Run #{i} - Returns analysis obtained.")
                        else:
                            logging.warning(f"Returns analyzer get_analysis() returned None for run #{i}")
                            print(f"DEBUG: Run #{i} - Returns analysis is None.")
                    else:
                        logging.warning(f"Returns analyzer not found in collection for run #{i}")
                        print(f"DEBUG: Run #{i} - Returns analyzer not found in collection.")

                    # Process Trade Analyzer with additional TypeError handling
                    if trade_analyzer:
                         try:
                             analysis_result = trade_analyzer.get_analysis()
                             if analysis_result is not None:
                                 trade_analysis = analysis_result
                                 print(f"DEBUG: Run #{i} - Trade analysis obtained.")
                             else:
                                 # This case might happen if trades exist but analysis returns None
                                 logging.warning(f"TradeAnalyzer get_analysis() returned None explicitly for run #{i}. Treating as no trades.")
                                 print(f"DEBUG: Run #{i} - Trade analysis is None.")
                                 trade_analysis = {} # Ensure trade_analysis is a dict
                         except TypeError as te:
                             # Catch the "object of type 'NoneType' has no len()" error specifically
                             if "'NoneType' has no len()" in str(te):
                                 logging.warning(f"TradeAnalyzer encountered internal TypeError (likely no trades) for run #{i}. Error: {te}")
                                 print(f"DEBUG: Run #{i} - Trade analysis treated as empty due to TypeError.")
                                 trade_analysis = {} # Treat as no trades / empty analysis
                             else:
                                 # Re-raise other unexpected TypeErrors
                                 raise te
                         except Exception as e_trade:
                             # Catch any other unexpected errors during trade analysis
                             logging.error(f"Unexpected error getting Trade Analysis for run #{i}: {e_trade}", exc_info=True)
                             trade_analysis = {} # Default to empty on other errors too
                    else:
                         logging.warning("TradeAnalyzer not found in collection for run #{i}")
                         print(f"DEBUG: Run #{i} - TradeAnalyzer not found in collection.")
                         trade_analysis = {} # Ensure trade_analysis is a dict
                else:
                    logging.warning(f"Analyzers collection (opt_result_obj.analyzers) not found or is None for run #{i}")
                    print(f"DEBUG: Run #{i} - Analyzers collection not found.")
                    # Ensure dicts for safe .get() usage later
                    returns_analysis = {}
                    trade_analysis = {}
                # --- End Safely Access Analyzers ---

                # Calculate final value safely using .get()
                total_return_ratio = returns_analysis.get('rtot', 0.0)
                final_value = initial_cash * (1 + total_return_ratio)
                total_return_pct = total_return_ratio * 100
                print(f"DEBUG: Run #{i} - Total Return: {total_return_pct:.2f}%, Final Value: {final_value:.2f}")

                # Calculate trade stats safely using .get() for nested dicts
                total_trades_dict = trade_analysis.get('total', {}) # Use get with default {}
                won_trades_dict = trade_analysis.get('won', {})     # Use get with default {}
                
                total_trades = total_trades_dict.get('closed', 0) if total_trades_dict else 0 # Check dict exists
                won_trades = won_trades_dict.get('total', 0) if won_trades_dict else 0       # Check dict exists
                
                win_rate = (won_trades / total_trades * 100) if total_trades else 0.0 # Avoid division by zero
                print(f"DEBUG: Run #{i} - Trades: {total_trades}, Win Rate: {win_rate}%")
                
                results_list.append({
                     'params': dict(zip(param_names, param_values)),
                     'final_value': final_value,
                     'return_pct': total_return_pct,
                     'trades': total_trades,
                     'win_rate': win_rate
                 })
                print(f"DEBUG: Run #{i} - Results appended successfully.")
            except AttributeError as ae:
                print(f"ERROR: AttributeError during processing run #{i}: {ae}. Result object structure unexpected.")
                print(f"DEBUG: Object type: {type(opt_result_obj)}, Attributes: {dir(opt_result_obj)}")
                import traceback
                traceback.print_exc() # Print full traceback for AttributeError
            except Exception as inner_e:
                print(f"ERROR: Exception during processing run #{i}: {inner_e}")
                import traceback
                traceback.print_exc() # Print full traceback for any other exception
        
        print(f"DEBUG: Finished processing loop. Number of results collected: {len(results_list)}")
        
        # Sort results by final value or another metric
        results_list.sort(key=lambda x: x['final_value'], reverse=True)
        print(f"DEBUG: Results sorted.")
        
        # Print sorted results
        print("--- Final Sorted Optimization Results Table --- ")
        for res in results_list:
            print(f"Params: {res['params']}, Final Value: {res['final_value']:.2f}, Return: {res['return_pct']:.2f}%, Trades: {res['trades']}, Win Rate: {res['win_rate']:.2f}%")
        print("--- End of Optimization Results Table --- ")

    else: # Single Run
        logging.info("Running single backtest...")
        results = cerebro.run()
        final_value = cerebro.broker.getvalue()
        logging.info(f'Final portfolio value: {final_value:.2f}')

        # --- Print Analysis --- (단일 실행 시 기존 분석 출력)
        if results and len(results) > 0:
            try:
                strategy_instance = results[0]
                print("\n--- Backtest Analysis ---")
                try:
                     # Check if analyzer exists before accessing
                     if hasattr(strategy_instance.analyzers, 'sharpe_ratio'):
                         sharpe = strategy_instance.analyzers.sharpe_ratio.get_analysis()
                         print(f"Sharpe Ratio: {sharpe.get('sharperatio', 'N/A')}")
                     else:
                         print("Sharpe Ratio Analyzer not found in results.")
                except KeyError: print("Sharpe Ratio: N/A")
                except AttributeError as ae_inner: print(f"Sharpe Ratio: Error accessing analyzer results: {ae_inner}")

                try:
                     # Check if analyzer exists before accessing
                     if hasattr(strategy_instance.analyzers, 'returns'):
                         returns = strategy_instance.analyzers.returns.get_analysis()
                         print(f"Total Return: {returns.get('rtot', 'N/A'):.4f}")
                         print(f"Average Daily Return: {returns.get('ravg', 'N/A'):.4f}")
                     else:
                        print("Returns Analyzer not found in results.")
                except KeyError: print("Returns: N/A")
                except AttributeError as ae_inner: print(f"Returns: Error accessing analyzer results: {ae_inner}")

                try:
                     # Check if analyzer exists before accessing
                     if hasattr(strategy_instance.analyzers, 'drawdown'):
                         drawdown = strategy_instance.analyzers.drawdown.get_analysis()
                         print(f"Max Drawdown: {drawdown.max.drawdown:.2f}%")
                         print(f"Max Drawdown Money: {drawdown.max.moneydown:.2f}")
                     else:
                        print("Drawdown Analyzer not found in results.")
                except KeyError: print("Drawdown: N/A")
                except AttributeError as ae_inner: print(f"Drawdown: Error accessing analyzer results: {ae_inner}")

                try:
                    # Check if analyzer exists before accessing
                    if hasattr(strategy_instance.analyzers, 'trade_analyzer'):
                        trade_analysis = strategy_instance.analyzers.trade_analyzer.get_analysis()
                        print("\n--- Trade Analysis ---")
                        print(f"Total Trades: {trade_analysis.total.total}")
                        print(f"Total Open Trades: {trade_analysis.total.open}")
                        print(f"Total Closed Trades: {trade_analysis.total.closed}")
                        print(f"\nWinning Trades:")
                        print(f"  - Count: {trade_analysis.won.total}")
                        if trade_analysis.won.total > 0:
                             print(f"  - Total PnL: {trade_analysis.won.pnl.total:.2f}")
                             print(f"  - Average PnL: {trade_analysis.won.pnl.average:.2f}")
                             print(f"  - Max PnL: {trade_analysis.won.pnl.max:.2f}")
                        print(f"\nLosing Trades:")
                        print(f"  - Count: {trade_analysis.lost.total}")
                        if trade_analysis.lost.total > 0:
                             print(f"  - Total PnL: {trade_analysis.lost.pnl.total:.2f}")
                             print(f"  - Average PnL: {trade_analysis.lost.pnl.average:.2f}")
                             print(f"  - Max PnL: {trade_analysis.lost.pnl.max:.2f}") # Max loss is min PnL

                        print(f"\nWin Rate: {(trade_analysis.won.total / trade_analysis.total.closed * 100) if trade_analysis.total.closed else 0:.2f}%")
                    else:
                         print("Trade Analyzer not found in results.")
                except KeyError: print("Trade Analysis: N/A (likely no closed trades)")
                except AttributeError as ae_inner: print(f"Trade Analysis: Error accessing analyzer results: {ae_inner}")
                except Exception as e: print(f"Error analyzing trades: {e}")

            except Exception as e_outer:
                logging.error(f"Error accessing strategy instance results[0]: {e_outer}", exc_info=True)

        else:
            logging.warning("Backtest did not produce results or strategy instance.")
        
        # --- Plotting --- (단일 실행 시에만 플롯 생성)
        if plot:
            # Generate plot and save to file instead of showing blocking window
            plot_filename = f"backtest_results_{symbol}_{from_date}_to_{to_date}.png" # 파일명 지정
            logging.info(f"Generating plot and saving to {plot_filename}...")
            try:
                # Use iplot=False to prevent opening interactive window (in some environments)
                # Set numfigs=1 to potentially combine plots if multiple data feeds/strats exist
                cerebro.plot(style='candlestick', barup='green', bardown='red',
                             savefig=True, # Enable saving to file
                             figfilename=plot_filename, # Specify the filename
                             iplot=False, # Prevent interactive plot if possible
                             numfigs=1)
            except Exception as e:
                logging.error(f"Error occurred during plotting: {e}", exc_info=True)

    # --- Restore original logging level --- 
    root_logger.setLevel(original_level)
    logging.debug("Restored original logging level.") # This might not show if level restored to INFO

    # Optimization doesn't return strategy instances in the same way
    return run_results if optimize else results 

# --- Main Execution Guard --- 
if __name__ == '__main__':
    logging.info("Running backtest_runner.py directly...")

    # --- Configuration for the test run --- 
    test_symbol = 'BTC-USD'
    test_from_date = '2023-01-01'
    test_to_date = '2023-12-31'
    test_interval = '1d'
    test_cash = 100000.0
    test_commission = 0.001
    test_stake = 1
    run_optimization = True # <<< 최적화 모드로 변경

    # Run the backtest or optimization
    run_backtest(
        symbol=test_symbol,
        from_date=test_from_date,
        to_date=test_to_date,
        interval=test_interval,
        initial_cash=test_cash,
        commission=test_commission,
        stake=test_stake,
        data_source='yahoo',
        plot=not run_optimization, # plot=False for optimization
        optimize=run_optimization  # optimize=True for optimization
    )
