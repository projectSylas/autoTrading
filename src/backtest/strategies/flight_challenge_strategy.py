import backtrader as bt
import pandas as pd
from src.utils.common import get_historical_data # Assuming data loading function exists
# Placeholder for technical indicator functions if not in common utils
# from src.utils.technical_indicators import calculate_rsi, calculate_sma 

class FlightBacktestStrategy(bt.Strategy):
    params = (
        ('rsi_period', 14),
        ('sma_period', 7),
        ('target_profit_pct', 0.10), # 10% target profit
        ('stop_loss_pct', 0.05),     # 5% stop loss
        # Add other necessary parameters like VIX threshold, volume threshold etc.
    )

    def log(self, txt, dt=None):
        ''' Logging function for this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} {txt}') # Print to console for now

    def __init__(self):
        # Keep a reference to the primary data feed
        self.dataclose = self.datas[0].close

        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None

        # --- Define indicators ---
        # self.rsi = bt.indicators.RSI(self.datas[0], period=self.params.rsi_period)
        # self.sma = bt.indicators.SimpleMovingAverage(self.datas[0], period=self.params.sma_period)
        # Add other indicators like volume, trendlines, divergences later

        # Placeholder for entry conditions based on requirements
        # self.entry_signal = bt.And(...) 

        self.log('Strategy Initialized')

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm {order.executed.comm:.2f}'
                )
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            elif order.issell():
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm {order.executed.comm:.2f}')

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        # Write down: no pending order
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log(f'OPERATION PROFIT, GROSS {trade.pnl:.2f}, NET {trade.pnlcomm:.2f}')


    def next(self):
        # Simply log the closing price of the series from the reference
        # self.log(f'Close, {self.dataclose[0]:.2f}')

        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        # --- Check Entry Conditions ---
        # Placeholder logic - Replace with actual Flight Challenge rules
        # Example: if not self.position and self.rsi < 30 and self.dataclose > self.sma: 
        if not self.position: # If not in the market
             # Example entry logic - needs refinement based on detailed rules
             # if self.entry_signal: # Replace with actual combined signal
             if True: # TEMPORARY: Placeholder for actual entry condition
                self.log(f'BUY CREATE, {self.dataclose[0]:.2f}')
                # Keep track of the created order to avoid a 2nd order
                # Calculate size based on risk or fixed amount later
                size = self.broker.get_cash() * 0.1 / self.dataclose[0] # Example: 10% cash
                self.order = self.buy(size=size) 

        # --- Check Exit Conditions (Stop Loss / Target Profit) ---
        else: # If in the market
            # Calculate potential profit/loss
            current_profit = (self.dataclose[0] - self.buyprice) / self.buyprice

            # Stop Loss Check
            if current_profit < -self.params.stop_loss_pct:
                self.log(f'STOP LOSS HIT, SELL CREATE, {self.dataclose[0]:.2f}')
                self.order = self.sell(size=self.position.size)
            
            # Target Profit Check
            elif current_profit > self.params.target_profit_pct:
                 self.log(f'TARGET PROFIT HIT, SELL CREATE, {self.dataclose[0]:.2f}')
                 self.order = self.sell(size=self.position.size)

            # Add other exit conditions if needed (e.g., trailing stop, time-based exit)


    def stop(self):
        self.log('(MA Period %2d) Ending Value %.2f' %
                 (self.params.sma_period, self.broker.getvalue()), dt=self.datas[0].datetime.date(0))
        self.log('Strategy Stopped') 