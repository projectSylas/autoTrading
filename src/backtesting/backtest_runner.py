# src/backtesting/backtest_runner.py

import backtrader as bt
import pandas as pd
from datetime import datetime
import os
import logging
import yfinance as yf
import sys
from typing import Type # For Strategy type hint

# --- Project Structure Adjustment ---
# Ensure the project root is in the Python path
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BACKTESTING_DIR = SRC_DIR
PROJECT_ROOT = os.path.dirname(os.path.dirname(BACKTESTING_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# --- Early Logging Setup ---
# Setup logging as early as possible
from src.utils.logging_config import setup_logging, APP_LOGGER_NAME
try:
    setup_logging()
    logger = logging.getLogger(APP_LOGGER_NAME)
    logger.info("Initial logging setup complete.")
except Exception as e:
    # Fallback basic logging if setup fails
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.error(f"Failed to configure logging using setup_logging: {e}. Using basic config.", exc_info=True)

# --- Import Configuration and Utilities ---
try:
from src.config.settings import settings
    logger.info("Settings loaded successfully.")
except ImportError:
    logger.error("Failed to import settings. Ensure config/settings.py exists and is valid.", exc_info=True)
    # Define fallback defaults if settings fail to load
    class FallbackSettings:
        BACKTEST_TICKER = 'BTC-USD'
        BACKTEST_PERIOD = '365d'
        BACKTEST_INTERVAL = '1h'
        DATA_SOURCE = 'yahoo'
        CSV_DATA_PATH = None
        BACKTEST_INITIAL_CASH = 10000.0
        BACKTEST_COMMISSION = 0.001
        BACKTEST_STAKE = 1
        BACKTEST_PLOT = True
        # Strategy defaults (add all required by ChallengeStrategyBacktest)
        CHALLENGE_SMA_PERIOD = 20
        CHALLENGE_RSI_PERIOD = 14
        CHALLENGE_RSI_THRESHOLD = 30
        CHALLENGE_VOLUME_AVG_PERIOD = 20
        CHALLENGE_VOLUME_SURGE_RATIO = 1.5
        CHALLENGE_DIVERGENCE_LOOKBACK = 14
        CHALLENGE_BREAKOUT_LOOKBACK = 20
        CHALLENGE_PULLBACK_LOOKBACK = 5
        CHALLENGE_POC_LOOKBACK = 50
        CHALLENGE_POC_THRESHOLD = 0.7
        CHALLENGE_SL_RATIO = 0.05
        CHALLENGE_TP_RATIO = 0.10
        CHALLENGE_PULLBACK_THRESHOLD = 0.02
    settings = FallbackSettings()
    logger.warning("Using fallback settings due to import error.")

try:
    from src.utils.common import get_historical_data
    logger.info("Data fetching utility loaded.")
except ImportError:
    logger.error("Failed to import get_historical_data. Database source might not work.", exc_info=True)
    def get_historical_data(*args, **kwargs): # Dummy function
        logger.error("get_historical_data is not available.")
        return None

# --- Import the Strategy ---
try:
    # Explicitly import the intended strategy
    from src.backtesting.strategies.challenge_strategy_backtest import ChallengeStrategyBacktest
    logger.info(f"Strategy class '{ChallengeStrategyBacktest.__name__}' loaded.")
    strategy_to_run: Type[bt.Strategy] = ChallengeStrategyBacktest # Type hint for clarity
except ImportError:
    logger.error("Failed to import ChallengeStrategyBacktest. Ensure src/backtesting/strategies/challenge_strategy_backtest.py exists.", exc_info=True)
    # Define a minimal fallback strategy if import fails
    class MinimalFallbackStrategy(bt.Strategy):
        def log(self, txt, dt=None):
            dt = dt or self.datas[0].datetime.date(0)
            logger.info(f'[MinimalStrat] {dt.isoformat()}, {txt}')
        def __init__(self):
            self.dataclose = self.datas[0].close
            self.log('Minimal Fallback Strategy Initialized.')
        def next(self):
            if len(self.dataclose) > 0:
                self.log(f'Close, {self.dataclose[0]:.2f}')
    else:
                self.log('Waiting for data...')
    strategy_to_run = MinimalFallbackStrategy
    logger.warning("Using MinimalFallbackStrategy due to import error.")


# --- Main Execution Block ---
if __name__ == "__main__":
    logger.info("=== Backtest Runner Script Started ===")

    # --- Cerebro Engine Setup ---
    cerebro = bt.Cerebro()
    logger.info("Backtrader Cerebro engine initialized.")

    # --- Strategy Configuration ---
    # Consolidate parameters for the strategy
    strategy_params = {
        # Match the parameter names defined in ChallengeStrategyBacktest.params
        'sma_long_period': settings.CHALLENGE_SMA_PERIOD,
        'sma_short_period': getattr(settings, 'CHALLENGE_SMA_SHORT_PERIOD', 7), # Add short period if needed
        'rsi_period': settings.CHALLENGE_RSI_PERIOD,
        'rsi_oversold': settings.CHALLENGE_RSI_THRESHOLD, # Use rsi_oversold (matching class param)
        'rsi_overbought': getattr(settings, 'CHALLENGE_RSI_OVERBOUGHT', 70), # Add overbought level
        'volume_avg_period': settings.CHALLENGE_VOLUME_AVG_PERIOD,
        'volume_surge_ratio': settings.CHALLENGE_VOLUME_SURGE_RATIO,
        'divergence_lookback': settings.CHALLENGE_DIVERGENCE_LOOKBACK,
        'breakout_lookback': settings.CHALLENGE_BREAKOUT_LOOKBACK,
        'pullback_lookback': settings.CHALLENGE_PULLBACK_LOOKBACK,
        # 'poc_lookback': settings.CHALLENGE_POC_LOOKBACK, # POC not used internally yet
        # 'poc_threshold': settings.CHALLENGE_POC_THRESHOLD,
        'sl_ratio': settings.CHALLENGE_SL_RATIO,
        'tp_ratio': settings.CHALLENGE_TP_RATIO,
        'pullback_threshold': settings.CHALLENGE_PULLBACK_THRESHOLD,
        'printlog': True, # Enable strategy logging via _write_log
    }
    logger.info(f"Adding strategy '{strategy_to_run.__name__}' with params: {strategy_params}")
    try:
        cerebro.addstrategy(strategy_to_run, **strategy_params)
    except Exception as e:
        logger.error(f"Failed to add strategy to Cerebro: {e}", exc_info=True)
        sys.exit(1)

    # --- Data Loading Configuration ---
    ticker = getattr(settings, 'BACKTEST_TICKER', 'BTC-USD')
    # Use period OR specific dates based on settings availability
    if hasattr(settings, 'BACKTEST_FROM_DATE') and hasattr(settings, 'BACKTEST_TO_DATE'):
        from_date_str = settings.BACKTEST_FROM_DATE
        to_date_str = settings.BACKTEST_TO_DATE
        period = None # Explicitly set period to None if using dates
        logger.info(f"Using specific date range: {from_date_str} to {to_date_str}")
    else:
        period = getattr(settings, 'BACKTEST_PERIOD', '365d')
        from_date_str = None
        to_date_str = None
        logger.info(f"Using period: {period}")

    interval = getattr(settings, 'BACKTEST_INTERVAL', '1h')
    data_source = getattr(settings, 'DATA_SOURCE', 'yahoo')
    csv_path = getattr(settings, 'CSV_DATA_PATH', None)

    logger.info(f"Data config: Ticker={ticker}, Interval={interval}, Source={data_source}")

    # --- Data Loading and Preparation ---
    df_data = None
    try:
        logger.info(f"Attempting to load data from source: {data_source}")
        if data_source.lower() == 'yahoo':
            # Determine start/end for yfinance download
            start_dt = from_date_str
            end_dt = to_date_str
            if period and not start_dt: # Use period if dates are not set
                # yfinance doesn't directly use period like '365d' with interval < 1d well, calculate dates
                from datetime import date, timedelta
                end_dt_obj = date.today()
                start_dt_obj = end_dt_obj - timedelta(days=int(period[:-1])) # Simple calculation
                start_dt = start_dt_obj.strftime('%Y-%m-%d')
                end_dt = end_dt_obj.strftime('%Y-%m-%d') # Use today as end date
                logger.info(f"Calculated date range for period '{period}': {start_dt} to {end_dt}")

            logger.info(f"Fetching from yfinance: Ticker={ticker}, Start={start_dt}, End={end_dt}, Interval={interval}")
            df_data = yf.download(ticker, start=start_dt, end=end_dt, interval=interval, progress=False)
            if df_data.empty:
                raise ValueError(f"yfinance returned no data for {ticker} in the specified range/interval.")

            # --- Handle Potential MultiIndex Columns from yfinance --- 
            if isinstance(df_data.columns, pd.MultiIndex):
                logger.info("Detected MultiIndex columns, flattening by dropping ticker level.")
                # Assumes the structure is (Metric, Ticker), e.g., ('Close', 'BTC-USD')
                # We want to keep only the metric part ('Close')
                df_data.columns = df_data.columns.get_level_values(0)
                logger.debug(f"Columns after flattening MultiIndex: {df_data.columns.tolist()}")
            # --- End MultiIndex Handling ---

        elif data_source.lower() == 'csv' and csv_path and os.path.exists(csv_path):
            logger.info(f"Loading from CSV: {csv_path}")
            df_data = pd.read_csv(csv_path)
            # --- Basic CSV Processing ---
            # Find a potential date/time column and set as index
            potential_dt_cols = [col for col in df_data.columns if 'date' in col.lower() or 'time' in col.lower()]
            if potential_dt_cols:
                try:
                    dt_col = potential_dt_cols[0] # Use the first likely column
                    logger.info(f"Attempting to use CSV column '{dt_col}' as datetime index.")
                    df_data[dt_col] = pd.to_datetime(df_data[dt_col])
                    df_data.set_index(dt_col, inplace=True)
                    logger.info(f"CSV index set to '{dt_col}'.")
                   except Exception as e:
                    logger.error(f"Failed to process datetime column '{potential_dt_cols[0]}' in CSV: {e}. Attempting to use index.", exc_info=True)
                    # Fallback: try converting the existing index
                    try:
                        df_data.index = pd.to_datetime(df_data.index)
                        logger.info("Successfully converted existing CSV index to datetime.")
                    except Exception as e_idx:
                       logger.error(f"Failed to convert CSV index to datetime: {e_idx}", exc_info=True)
                       raise ValueError("Could not establish a datetime index for CSV data.")
            else:
                 # If no obvious date column, try converting the index directly
                 try:
                     df_data.index = pd.to_datetime(df_data.index)
                     logger.info("Successfully converted existing CSV index to datetime (no date column found).")
                 except Exception as e_idx:
                    logger.error(f"Failed to convert CSV index to datetime: {e_idx}", exc_info=True)
                    raise ValueError("Could not establish a datetime index for CSV data (no date column found).")

            # Filter by date range if specified
            if from_date_str and to_date_str:
                fromdate = datetime.strptime(from_date_str, '%Y-%m-%d')
                todate = datetime.strptime(to_date_str, '%Y-%m-%d')
                df_data = df_data[(df_data.index >= fromdate) & (df_data.index <= todate)]
                logger.info(f"Filtered CSV data to range: {from_date_str} to {to_date_str}")

        elif data_source.lower() == 'db':
            logger.info(f"Fetching from database via get_historical_data: Symbol={ticker}, Interval={interval}, Start={from_date_str}, End={to_date_str}")
            df_data = get_historical_data(ticker, interval, start_date=from_date_str, end_date=to_date_str) # Pass dates if available
            if df_data is None or df_data.empty:
                 raise ValueError("get_historical_data returned no data from the database.")

        else:
            raise ValueError(f"Unsupported or misconfigured data source: {data_source}")

        # --- Universal Data Cleaning and Validation ---
        if df_data is None or df_data.empty:
            raise ValueError(f"No data loaded from source '{data_source}'.")

        # 1. Standardize Column Names (lowercase, underscore spaces)
        df_data.columns = [str(col).lower().replace(' ', '_') for col in df_data.columns]
        logger.debug(f"Columns after standardization: {df_data.columns.tolist()}")

        # 2. Rename 'adj_close' to 'close' if it exists and 'close' doesn't, or prefer adj_close
        if 'adj_close' in df_data.columns:
            df_data['close'] = df_data['adj_close']
            logger.info("Using 'adj_close' as 'close'.")

        # 3. Verify Required Columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df_data.columns]
        if missing_cols:
            raise ValueError(f"Data is missing required OHLCV columns after cleaning: {missing_cols}. Found: {df_data.columns.tolist()}")

        # 4. Ensure DatetimeIndex
        if not isinstance(df_data.index, pd.DatetimeIndex):
            logger.warning(f"Data index is not DatetimeIndex (Type: {type(df_data.index)}). Attempting conversion.")
                try:
                df_data.index = pd.to_datetime(df_data.index)
                logger.info("Successfully converted index to DatetimeIndex.")
                except Exception as e:
                logger.error(f"Failed to convert data index to DatetimeIndex: {e}", exc_info=True)
                raise ValueError("Invalid data index type, cannot convert to DatetimeIndex.")

        # 5. Remove Rows with NaNs in essential columns
        initial_rows = len(df_data)
        df_data.dropna(subset=required_cols, inplace=True)
        rows_removed = initial_rows - len(df_data)
        if rows_removed > 0:
            logger.warning(f"Removed {rows_removed} rows with NaN values in OHLCV columns.")

        # 6. Final Check for Empty DataFrame
        if df_data.empty:
            raise ValueError("DataFrame became empty after cleaning (NaN removal or initial load failure).")

        logger.info(f"Data successfully loaded and prepared. Shape: {df_data.shape}, Index Range: {df_data.index.min()} to {df_data.index.max()}")
        logger.debug(f"Cleaned DataFrame head: {df_data.head()}")

        # --- Create Backtrader Data Feed ---
        # Use explicit column mapping for robustness
        data_feed = bt.feeds.PandasData(
            dataname=df_data,
            datetime=None,  # Use the DataFrame's DatetimeIndex
                    open='open',
                    high='high',
                    low='low',
                    close='close',
                    volume='volume',
            openinterest=-1 # Indicate no open interest data is used
        )
        cerebro.adddata(data_feed)
        logger.info("Data feed successfully added to Cerebro.")

    except FileNotFoundError as fnf_error:
        logger.error(f"Data loading error: CSV file not found at path '{csv_path}'. {fnf_error}", exc_info=True)
        sys.exit(1)
    except ValueError as val_error:
        logger.error(f"Data loading/preparation error: {val_error}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during data loading: {e}", exc_info=True)
        sys.exit(1)

    # --- Broker Configuration ---
    initial_cash = float(getattr(settings, 'BACKTEST_INITIAL_CASH', 10000.0))
    commission = float(getattr(settings, 'BACKTEST_COMMISSION', 0.001))
    stake = int(getattr(settings, 'BACKTEST_STAKE', 1))

    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission)
    cerebro.addsizer(bt.sizers.FixedSize, stake=stake)
    logger.info(f"Broker configured: Initial Cash=${initial_cash:,.2f}, Commission={commission*100:.3f}%, Stake={stake}")

    # --- Analyzers ---
    # Add standard analyzers for single runs
    logger.info("Adding standard analyzers: TradeAnalyzer, SharpeRatio, DrawDown, Returns")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio', timeframe=bt.TimeFrame.Days, riskfreerate=0.0) # Added riskfreerate
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns', timeframe=bt.TimeFrame.Days)

    # --- Run Backtest ---
    logger.info("Starting Cerebro backtest run...")
    run_results = None
    try:
        # Run the backtest. It returns a list of strategy instances.
            run_results = cerebro.run()
        logger.info("Cerebro run completed.")
    except Exception as e:
        logger.error(f"Exception occurred during cerebro.run(): {e}", exc_info=True)
        # Optionally try to get broker value even if run fails partially
        try:
            final_val = cerebro.broker.getvalue()
            logger.info(f"Broker value after exception: {final_val:,.2f}")
        except Exception as be:
            logger.error(f"Could not get broker value after exception: {be}")
        sys.exit(1)

    # --- Process and Print Results ---
    if run_results and isinstance(run_results, list) and len(run_results) > 0:
        # Safely access the first strategy instance
        try:
            strategy_instance = run_results[0]
            logger.info("Processing analysis results from the strategy instance...")

            # --- Basic Portfolio Stats ---
            final_portfolio_value = cerebro.broker.getvalue()
            net_profit = final_portfolio_value - initial_cash
            net_profit_percent = (net_profit / initial_cash) * 100 if initial_cash else 0

            print("--- Backtest Summary ---")
            print(f"Initial Portfolio Value: {initial_cash:,.2f}")
            print(f"Final Portfolio Value:   {final_portfolio_value:,.2f}")
            print(f"Net Profit/Loss:         {net_profit:,.2f}")
            print(f"Net Profit/Loss [%]:     {net_profit_percent:.2f}%")

            # --- Analyzer Results ---
            print("--- Analyzer Results ---")
            analyzers = strategy_instance.analyzers
            try:
                trade_analysis = analyzers.trade_analyzer.get_analysis()
                total_closed = trade_analysis.get('total', {}).get('closed', 0)
                total_won = trade_analysis.get('won', {}).get('total', 0)
                total_lost = trade_analysis.get('lost', {}).get('total', 0)
                win_rate = (total_won / total_closed * 100) if total_closed > 0 else 0.0
                avg_win_pnl = trade_analysis.get('won', {}).get('pnl', {}).get('average', 0.0)
                avg_loss_pnl = trade_analysis.get('lost', {}).get('pnl', {}).get('average', 0.0)
                print(f"Total Closed Trades: {total_closed}")
                print(f"Winning Trades:      {total_won}")
                print(f"Losing Trades:       {total_lost}")
                print(f"Win Rate [%]:        {win_rate:.2f}%")
                print(f"Avg Winning Trade PnL: {avg_win_pnl:,.2f}")
                print(f"Avg Losing Trade PnL:  {avg_loss_pnl:,.2f}")
            except AttributeError:
                logger.warning("Could not retrieve TradeAnalyzer results. Was it added?")
        except Exception as e:
                logger.error(f"Error processing TradeAnalyzer results: {e}", exc_info=True)

            try:
                sharpe_analysis = analyzers.sharpe_ratio.get_analysis()
                sharpe_ratio = sharpe_analysis.get('sharperatio', 'N/A')
                # Handle case where Sharpe ratio is None (e.g., no trades or zero variance)
                print(f"Sharpe Ratio (Annualized): {sharpe_ratio if sharpe_ratio is not None else 'N/A'}")
            except AttributeError:
                 logger.warning("Could not retrieve SharpeRatio results. Was it added?")
            except Exception as e:
                logger.error(f"Error processing SharpeRatio results: {e}", exc_info=True)

            try:
                drawdown_analysis = analyzers.drawdown.get_analysis()
                max_drawdown = drawdown_analysis.max.get('drawdown', 0.0)
                max_drawdown_len = drawdown_analysis.max.get('len', 0)
                print(f"Max Drawdown [%]:    {max_drawdown:.2f}%")
                print(f"Max Drawdown Length: {max_drawdown_len} bars")
            except AttributeError:
                 logger.warning("Could not retrieve DrawDown results. Was it added?")
            except Exception as e:
                logger.error(f"Error processing DrawDown results: {e}", exc_info=True)

            try:
                returns_analysis = analyzers.returns.get_analysis()
                total_return = returns_analysis.get('rtot', 0.0) * 100 # Already calculated above, but good cross-check
                # avg_return = returns_analysis.get('ravg', 0.0) * 100
                print(f"Total Return [%] (Analyzer): {total_return:.2f}%")
                # print(f"Average Daily Return [%]: {avg_return:.4f}%") # Can be noisy
    except AttributeError:
                 logger.warning("Could not retrieve Returns results. Was it added?")
            except Exception as e:
                logger.error(f"Error processing Returns results: {e}", exc_info=True)

            print("-" * 25)

        except AttributeError as ae:
            logger.error(f"Error accessing strategy instance or analyzers (AttributeError): {ae}. Backtest might have failed internally.", exc_info=True)
        except IndexError as ie:
             logger.error(f"Error accessing run_results list (IndexError): {ie}. Cerebro might not have returned strategy instances.", exc_info=True)
        except Exception as e:
            logger.error(f"An unexpected error occurred during results processing: {e}", exc_info=True)

    else:
        logger.error("Backtest did not produce valid results (run_results is None or empty). Cannot analyze or plot.")

    # --- Plotting ---
    plot_enabled = getattr(settings, 'BACKTEST_PLOT', True)
    if run_results and plot_enabled:
        logger.info("Attempting to generate plot...")
        try:
            # Ensure matplotlib backend is suitable (consider 'Agg' for non-GUI server environments)
            # import matplotlib
            # matplotlib.use('Agg')
            import matplotlib.pyplot as plt # Import only if plotting

            # Define plot directory relative to project root
            plots_dir = os.path.join(PROJECT_ROOT, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            plot_filename = f'backtest_{ticker}_{interval}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            plot_filepath = os.path.join(plots_dir, plot_filename)

            # Plot using cerebro.plot()
            # iplot=False prevents blocking in non-interactive environments
            figure = cerebro.plot(style='candlestick', barup='green', bardown='red',
                                  volup='#4CAF50', voldown='#F44336', iplot=False)[0][0]

            figure.savefig(plot_filepath, dpi=300) # Save the figure
            logger.info(f"Plot saved successfully to: {plot_filepath}")
            # plt.show() # Generally avoid plt.show() in automated scripts

        except ImportError:
            logger.error("Plotting requires matplotlib. Please install it (`pip install matplotlib`).")
        except IndexError:
             logger.error("Plotting failed: Cerebro.plot() did not return the expected figure structure.", exc_info=True)
        except Exception as e:
            logger.error(f"An unexpected error occurred during plotting: {e}", exc_info=True)

    elif not plot_enabled:
        logger.info("Plotting is disabled in settings (BACKTEST_PLOT=False).")
    elif not run_results:
         logger.warning("Skipping plotting because the backtest did not produce valid results.")

    logger.info("=== Backtest Runner Script Finished ===")
