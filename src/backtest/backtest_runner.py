import backtrader as bt
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Import the strategy class
from src.backtest.strategies.flight_challenge_strategy import FlightBacktestStrategy
# Import data loading function (adjust path if needed)
from src.utils.common import get_historical_data 
# Import settings if needed for parameters
# from src.config.settings import Settings

# Configure Matplotlib for backtrader plots
plt.rcParams['figure.figsize'] = [10, 6] # Adjust figure size
plt.rcParams['figure.facecolor'] = 'white' # Set background color
# Consider using interactive plots if running in Jupyter
# %matplotlib inline 

if __name__ == '__main__':
    # --- 1. Create a Cerebro engine instance ---
    cerebro = bt.Cerebro()

    # --- 2. Add the strategy ---
    # You can pass parameters to the strategy here
    cerebro.addstrategy(
        FlightBacktestStrategy,
        # Example: Override default params if needed
        # rsi_period=21 
    )

    # --- 3. Load Data ---
    # Define parameters for data loading
    symbol = 'BTCUSDT' # Example symbol
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    interval = '1d' # Example interval

    print(f"Loading data for {symbol} from {start_date} to {end_date} ({interval})...")
    try:
        # Assuming get_historical_data returns a pandas DataFrame with OHLCV data
        # Make sure the DataFrame index is a datetime index
        dataframe = get_historical_data(symbol, interval, start_date, end_date)
        
        if dataframe is None or dataframe.empty:
            raise ValueError("Failed to load data or data is empty.")

        # Convert DataFrame to backtrader data feed
        # Ensure column names match backtrader's expectations (e.g., 'open', 'high', 'low', 'close', 'volume')
        # If column names differ, rename them before creating the data feed.
        # Example renaming: dataframe.rename(columns={'Open': 'open', 'High': 'high', ...}, inplace=True)
        data = bt.feeds.PandasData(dataname=dataframe)

        # Add the Data Feed to Cerebro
        cerebro.adddata(data)
        print("Data loaded successfully.")

    except Exception as e:
        print(f"Error loading data: {e}")
        exit() # Exit if data loading fails

    # --- 4. Set initial cash ---
    start_cash = 10000.0
    cerebro.broker.setcash(start_cash)
    print(f'Starting Portfolio Value: {start_cash:.2f}')

    # --- 5. Set commission ---
    # Example: 0.1% commission on trades
    cerebro.broker.setcommission(commission=0.001)

    # --- 6. Add Analyzers (Optional) ---
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')
    print("Analyzers added.")

    # --- 7. Run the backtest ---
    print("Running backtest...")
    results = cerebro.run()
    print("Backtest finished.")

    # --- 8. Print Analysis Results ---
    print("----- Analysis Results -----")
    final_value = cerebro.broker.getvalue()
    print(f'Final Portfolio Value: {final_value:.2f}')
    print(f'Total Return: {(final_value - start_cash) / start_cash * 100:.2f}%')

    # Access analyzer results
    strat = results[0] # Get the first strategy instance
    try:
        sharpe = strat.analyzers.sharpe_ratio.get_analysis()
        returns = strat.analyzers.returns.get_analysis()
        drawdown = strat.analyzers.drawdown.get_analysis()
        trade_analysis = strat.analyzers.trade_analyzer.get_analysis()

        print(f"Sharpe Ratio: {sharpe.get('sharperatio', 'N/A')}")
        print(f"Annualized Return: {returns.get('rnorm100', 'N/A'):.2f}%")
        print(f"Max Drawdown: {drawdown.max.drawdown:.2f}%")
        
        if trade_analysis and 'total' in trade_analysis and trade_analysis['total']['total'] > 0:
             print(f"Total Trades: {trade_analysis['total']['total']}")
             print(f"Winning Trades: {trade_analysis.won.total}")
             print(f"Losing Trades: {trade_analysis.lost.total}")
             print(f"Win Rate: {trade_analysis.won.total / trade_analysis['total']['total'] * 100:.2f}%")
        else:
            print("No trades executed.")
            
    except KeyError as e:
        print(f"Could not retrieve analyzer key: {e}")
    except Exception as e:
         print(f"Error printing analysis results: {e}")


    # --- 9. Plot the results (Optional) ---
    print("Plotting results...")
    try:
        # Set plot style options for better readability
        plot_config = {
            'style': 'candlestick', # Use candlestick plot
            'barup': 'green',
            'bardown': 'red',
            'volup': 'green',
            'voldown': 'red',
            'plotdat': True,      # Plot the main data
            'plotind': True,      # Plot indicators defined in strategy
            'plothlines': True,   # Plot horizontal lines if any
            'plotyticks': True,   # Plot y-axis ticks
            'plotymargin': 0.05,  # Margin for y-axis
            'plotvolume': True,   # Plot volume
            'subplot': True,      # Indicators in separate subplots
        }
        # Increase DPI for higher resolution plots
        # cerebro.plot(iplot=False, dpi=300, **plot_config)
        cerebro.plot(iplot=False, **plot_config) # Use iplot=False for static plots
        print("Plot generated. Check for the plot window or saved file if configured.")
    except Exception as e:
        print(f"Error during plotting: {e}") 