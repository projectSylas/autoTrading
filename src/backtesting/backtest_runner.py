# src/backtesting/backtest_runner.py

import backtrader as bt
import backtrader.feeds as btfeeds
import pandas as pd
from datetime import datetime
import os
import logging

# Import the strategy class
from src.backtesting.strategies.challenge_strategy_backtest import ChallengeStrategyBacktest
# Import settings if needed for data paths or parameters
from src.config import settings
# Import data fetching utility if needed
from src.utils.common import get_historical_data # Assuming this function exists

# Setup logging
logging.basicConfig(level=settings.LOG_LEVEL.upper(), format='%(asctime)s - %(levelname)s - %(message)s')

def run_backtest(symbol: str,
                 from_date: str,
                 to_date: str,
                 interval: str = '1h',
                 initial_cash: float = 10000.0,
                 commission: float = 0.001, # Example commission (0.1%)
                 stake: int = 10, # Example fixed stake per trade
                 strategy_params: dict = {},
                 data_source: str = 'yahoo', # 'yahoo', 'db', 'csv', etc.
                 csv_path: Optional[str] = None,
                 plot: bool = True):
    """
    Runs a backtrader backtest for the given symbol and period.

    Args:
        symbol (str): The trading symbol (e.g., 'BTC-USD' for Yahoo).
        from_date (str): Start date in 'YYYY-MM-DD' format.
        to_date (str): End date in 'YYYY-MM-DD' format.
        interval (str): Data interval (e.g., '1h', '1d'). Needs compatibility with data source.
        initial_cash (float): Starting cash for the backtest.
        commission (float): Commission per trade (e.g., 0.001 for 0.1%).
        stake (int): Default size for orders.
        strategy_params (dict): Dictionary of parameters to override strategy defaults.
        data_source (str): Where to get the data from ('yahoo', 'csv', 'db').
        csv_path (Optional[str]): Path to the CSV file if data_source is 'csv'.
        plot (bool): Whether to generate a plot of the results.
    """
    cerebro = bt.Cerebro()

    # --- Add Strategy ---
    # Pass strategy parameters that might differ from defaults in settings
    cerebro.addstrategy(ChallengeStrategyBacktest, **strategy_params)

    # --- Add Data Feed ---
    data = None
    fromdate = datetime.strptime(from_date, '%Y-%m-%d')
    todate = datetime.strptime(to_date, '%Y-%m-%d')

    try:
        if data_source.lower() == 'yahoo':
            # Note: Yahoo interval mapping might be needed (e.g., '1h')
            # Backtrader uses different interval codes sometimes
            # Check btfeeds.YahooFinanceCSVData documentation if issues arise
            logging.info(f"Fetching data for {symbol} from Yahoo Finance...")
            data = btfeeds.YahooFinanceData(
                dataname=symbol,
                fromdate=fromdate,
                todate=todate,
                # interval=interval # Backtrader might handle interval differently
                # timeframe=bt.TimeFrame.Minutes if 'm' in interval else bt.TimeFrame.Days if 'd' in interval else ..., # Needs mapping
                # compression=int(interval[:-1]) if interval[-1] in ['m','h'] else 1 # Needs mapping
            )
        elif data_source.lower() == 'csv':
            if csv_path and os.path.exists(csv_path):
                logging.info(f"Loading data for {symbol} from CSV: {csv_path}")
                # Ensure CSV has compatible format (Open, High, Low, Close, Volume, datetime)
                data = btfeeds.GenericCSVData(
                    dataname=csv_path,
                    fromdate=fromdate,
                    todate=todate,
                    dtformat=('%Y-%m-%d %H:%M:%S'), # Adjust format as needed
                    datetime=0, open=1, high=2, low=3, close=4, volume=5,
                    timeframe=bt.TimeFrame.Minutes, # Adjust as per data
                    compression=60 # Adjust as per data (e.g., 60 for 1h)
                )
            else:
                 logging.error(f"CSV file not found or path not provided: {csv_path}")
                 return None
        elif data_source.lower() == 'db':
             # Fetch data using get_historical_data (returns pandas DataFrame)
             logging.info(f"Fetching data for {symbol} from database/internal function...")
             df = get_historical_data(symbol, interval, start_date=from_date, end_date=to_date) # Adjust args as needed
             if df is not None and not df.empty:
                  # Convert pandas DataFrame to backtrader feed
                  data = bt.feeds.PandasData(dataname=df)
             else:
                  logging.error(f"Failed to fetch data from database/internal function for {symbol}.")
                  return None
        else:
            logging.error(f"Unsupported data source: {data_source}")
            return None

        if data:
            cerebro.adddata(data)
        else:
             logging.error("Failed to load data feed.")
             return None

    except Exception as e:
        logging.error(f"Error loading data feed: {e}", exc_info=True)
        return None

    # --- Configure Cerebro ---
    cerebro.broker.setcash(initial_cash)
    # Set commission
    cerebro.broker.setcommission(commission=commission)
    # Add stake size
    cerebro.addsizer(bt.sizers.FixedSize, stake=stake)

    # --- Add Analyzers ---
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio', timeframe=bt.TimeFrame.Days)
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns', timeframe=bt.TimeFrame.Days)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer') # Analyzes individual trades

    # --- Run Backtest ---
    logging.info(f'Starting portfolio value: {cerebro.broker.getvalue():.2f}')
    results = cerebro.run()
    final_value = cerebro.broker.getvalue()
    logging.info(f'Final portfolio value: {final_value:.2f}')

    # --- Print Analysis ---
    if results and results[0]:
        strategy_instance = results[0]
        print("\n--- Backtest Analysis ---")
        try:
            sharpe = strategy_instance.analyzers.sharpe_ratio.get_analysis()
            print(f"Sharpe Ratio: {sharpe.get('sharperatio', 'N/A')}")
        except KeyError: print("Sharpe Ratio: N/A")

        try:
             returns = strategy_instance.analyzers.returns.get_analysis()
             print(f"Total Return: {returns.get('rtot', 'N/A'):.4f}")
             print(f"Average Daily Return: {returns.get('ravg', 'N/A'):.4f}")
        except KeyError: print("Returns: N/A")

        try:
             drawdown = strategy_instance.analyzers.drawdown.get_analysis()
             print(f"Max Drawdown: {drawdown.max.drawdown:.2f}%")
             print(f"Max Drawdown Money: {drawdown.max.moneydown:.2f}")
        except KeyError: print("Drawdown: N/A")

        try:
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
            # print(f"Profit Factor: {abs(trade_analysis.won.pnl.total / trade_analysis.lost.pnl.total) if trade_analysis.lost.pnl.total != 0 else 'inf'}")

        except KeyError: print("Trade Analysis: N/A (likely no closed trades)")
        except Exception as e: print(f"Error analyzing trades: {e}")


    # --- Plot Results ---
    if plot:
        try:
            # Adjust plot style if desired
            # cerebro.plot(style='candlestick', barup='green', bardown='red')
            cerebro.plot(iplot=False) # Set iplot=False if not using interactive plots
        except Exception as e:
            logging.error(f"Error generating plot: {e}")

    return final_value

# --- Main Execution Guard ---
if __name__ == '__main__':
    logging.info("Running backtest_runner.py directly...")

    # --- Configuration for the test run ---
    test_symbol = 'BTC-USD' # Symbol compatible with Yahoo Finance
    test_from_date = '2023-01-01'
    test_to_date = '2023-12-31'
    test_interval = '1d' # Yahoo interval (adjust if needed)
    test_cash = 100000.0
    test_commission = 0.001 # 0.1%
    test_stake = 1 # Fixed size for crypto/stocks

    # Example: Override strategy parameters if needed
    strategy_overrides = {
        # 'sma_long_period': 50,
        # 'printlog': False
    }

    # Run the backtest
    run_backtest(
        symbol=test_symbol,
        from_date=test_from_date,
        to_date=test_to_date,
        interval=test_interval,
        initial_cash=test_cash,
        commission=test_commission,
        stake=test_stake,
        strategy_params=strategy_overrides,
        data_source='yahoo', # Use 'csv' or 'db' if preferred
        # csv_path='/path/to/your/data.csv', # Provide path if using CSV
        plot=True # Set to False to disable plotting
    )
