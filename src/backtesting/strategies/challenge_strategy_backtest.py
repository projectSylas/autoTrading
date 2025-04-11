# src/backtesting/strategies/challenge_strategy_backtest.py

import backtrader as bt
import pandas as pd
import numpy as np # For NaN checks
import logging
from typing import Optional # Added for Optional type hint
from src.config import settings # 설정값 사용
from src.utils import strategy_utils # 기존 유틸 함수 사용
# pandas-ta import도 필요할 수 있음
import pandas_ta as ta

class ChallengeStrategyBacktest(bt.Strategy):
    """
    Backtrader strategy class implementing the core logic of the Flight Challenge strategy.
    Focuses on technical indicators and conditions. AI overrides are not included by default.
    """
    params = (
        ('sma_long_period', settings.CHALLENGE_SMA_PERIOD),
        ('sma_short_period', 7),
        ('rsi_period', settings.CHALLENGE_RSI_PERIOD),
        ('rsi_threshold', settings.CHALLENGE_RSI_THRESHOLD),
        ('volume_avg_period', settings.CHALLENGE_VOLUME_AVG_PERIOD),
        ('volume_surge_ratio', settings.CHALLENGE_VOLUME_SURGE_RATIO),
        ('divergence_lookback', settings.CHALLENGE_DIVERGENCE_LOOKBACK),
        ('breakout_lookback', settings.CHALLENGE_BREAKOUT_LOOKBACK),
        ('pullback_lookback', settings.CHALLENGE_PULLBACK_LOOKBACK),
        ('poc_lookback', settings.CHALLENGE_POC_LOOKBACK),
        ('poc_threshold', settings.CHALLENGE_POC_THRESHOLD),
        ('sl_ratio', settings.CHALLENGE_SL_RATIO),
        ('tp_ratio', settings.CHALLENGE_TP_RATIO),
        ('printlog', True), # 로그 출력 여부
    )

    def log(self, txt, dt=None, doprint=False):
        ''' Logging function for this strategy'''
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()}, {txt}')

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close
        self.datahigh = self.datas[0].high
        self.datalow = self.datas[0].low
        self.datavolume = self.datas[0].volume

        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.bar_executed = 0 # Track bar number when order was executed

        # --- Define indicators ---
        # Note: backtrader has built-in indicators which are often preferred for performance.
        # However, to maintain consistency with strategy.py, we might calculate using pandas on buffered data.
        # Or, use backtrader indicators that match strategy_utils logic.

        # Option 1: Use backtrader indicators
        self.sma_long = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.sma_long_period)
        self.sma_short = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.sma_short_period)
        self.rsi = bt.indicators.RelativeStrengthIndex(
            period=self.params.rsi_period)
        self.volume_sma = bt.indicators.SimpleMovingAverage(
             self.datavolume, period=self.params.volume_avg_period)
        # POC requires custom implementation or approximation within backtrader
        # Divergence and Breakout/Pullback also require custom logic within next()

        # Placeholder for custom indicators if needed
        self.poc = None # Will need calculation in next() or precalculation
        self.rsi_divergence = 'none'
        self.pullback_signal = 'none'

        # Example: Pre-calculate pandas_ta indicators if needed (less efficient)
        # self.df = pd.DataFrame({ # Needs careful handling of data buffering
        #     'Open': self.datas[0].open.get(size=len(self.datas[0])),
        #      ...
        # })
        # self.poc_series = strategy_utils.calculate_poc(self.df, ...) # Requires careful slicing

        logging.info("Challenge Strategy Backtest Initialized.")

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm {order.executed.comm:.2f}'
                )
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm {order.executed.comm:.2f}')

            self.bar_executed = len(self) # Bar number of execution

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'Order Canceled/Margin/Rejected - Status: {order.getstatusname()}')

        # Write down: no pending order
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log(f'OPERATION PROFIT, GROSS {trade.pnl:.2f}, NET {trade.pnlcomm:.2f}')

    def _get_buffered_df(self, min_required_len: int) -> Optional[pd.DataFrame]:
        """
        Helper function to get buffered data as a pandas DataFrame.
        Requires enough data points to be loaded in the buffer.
        """
        if len(self.datas[0]) < min_required_len:
             # self.log(f"Warning: Not enough data ({len(self.datas[0])}) for required lookback ({min_required_len})")
             return None # Not enough data in the buffer yet

        # Determine the actual size to get (max needed)
        size = max(min_required_len, len(self.datas[0])) # Get all available if buffer is full

        # Get data lines as lists/arrays
        opens = self.datas[0].open.get(size=size)
        highs = self.datas[0].high.get(size=size)
        lows = self.datas[0].low.get(size=size)
        closes = self.datas[0].close.get(size=size)
        volumes = self.datas[0].volume.get(size=size)
        datetimes = self.datas[0].datetime.get(size=size) # Get datetime objects
        # Get RSI values from the indicator buffer
        rsi_values = self.rsi.get(size=size) # Make sure RSI indicator is calculated first

        # Convert datetimes to pandas Timestamps
        timestamps = [bt.num2date(dt) for dt in datetimes]

        # Create DataFrame
        df = pd.DataFrame({
            'Open': opens,
            'High': highs,
            'Low': lows,
            'Close': closes,
            'Volume': volumes,
            'RSI': rsi_values # Add RSI column
        }, index=pd.DatetimeIndex(timestamps)) # Use timestamps as index

        # Drop initial rows where RSI might be NaN if needed by utils functions
        # df.dropna(subset=['RSI'], inplace=True)
        # if len(df) < min_required_len: return None # Check again after dropna

        return df

    def next(self):
        # Log the closing price of the series from the latest bar
        # self.log('Close, %.2f' % self.dataclose[0])

        # --- Calculate Custom Indicators using Buffered Data ---
        # Determine the maximum lookback needed by all custom indicators
        max_lookback_needed = max(
            self.params.poc_lookback,
            self.params.divergence_lookback * 2, # divergence needs two periods
            self.params.breakout_lookback + self.params.pullback_lookback
        ) + 5 # Add a small buffer

        # Get buffered data as DataFrame
        df_buffer = self._get_buffered_df(max_lookback_needed)

        # Calculate indicators only if we have enough data
        if df_buffer is not None:
            # 1. Calculate POC
            try:
                self.poc = strategy_utils.calculate_poc(df_buffer, lookback=self.params.poc_lookback)
            except Exception as e:
                self.log(f"Error calculating POC: {e}")
                self.poc = None

            # 2. Detect Divergence
            try:
                self.rsi_divergence = strategy_utils.detect_rsi_divergence(
                    df_buffer,
                    lookback=self.params.divergence_lookback
                )
            except Exception as e:
                 self.log(f"Error detecting RSI Divergence: {e}")
                 self.rsi_divergence = 'none'

            # 3. Detect Breakout/Pullback
            try:
                self.pullback_signal = strategy_utils.detect_trendline_breakout_pullback(
                    df_buffer,
                    lookback_breakout=self.params.breakout_lookback,
                    lookback_pullback=self.params.pullback_lookback
                )
            except Exception as e:
                 self.log(f"Error detecting Breakout/Pullback: {e}")
                 self.pullback_signal = 'none'
        else:
             # Not enough data yet, reset custom indicators
             self.poc = None
             self.rsi_divergence = 'none'
             self.pullback_signal = 'none'

        # Access standard indicator values (calculated by backtrader)
        current_close = self.dataclose[0]
        current_volume = self.datavolume[0]
        sma_long_val = self.sma_long[0]
        sma_short_val = self.sma_short[0]
        rsi_val = self.rsi[0]
        avg_volume_val = self.volume_sma[0]

        # --- Condition Check ---
        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        # Check if we are in the market
        if not self.position:
            # --- Entry Logic ---
            decision = 'hold'
            reason = "No signal (or insufficient data)"
            side = None

            # Ensure indicators are valid before checking conditions
            if df_buffer is None or pd.isna(rsi_val) or pd.isna(sma_long_val) or pd.isna(sma_short_val):
                 self.log("Waiting for indicators to warm up...")
                 return # Skip condition checks if indicators are not ready

            # Check volume surge
            volume_surge_detected = False
            if not pd.isna(avg_volume_val) and avg_volume_val > 0 and not pd.isna(current_volume):
                if current_volume > avg_volume_val * self.params.volume_surge_ratio:
                     volume_surge_detected = True

            # Apply condition hierarchy (using self.poc, self.rsi_divergence, self.pullback_signal)
            if self.rsi_divergence == 'bearish':
                decision = 'hold'
                reason = f"Hold: Bearish RSI Divergence"
            elif self.pullback_signal == 'bearish_pullback':
                 decision = 'hold'
                 reason = f"Hold: Bearish Breakout/Pullback"
            elif current_close < sma_short_val:
                decision = 'hold'
                reason = f"Hold: Close < SMA{self.params.sma_short_period}"
            # Check for potential rejection near POC before bullish signals
            elif self.poc is not None and abs(current_close - self.poc) / self.poc < self.params.poc_threshold:
                 # Simple check: if price touched near POC but closed below it
                 if self.datahigh[0] >= self.poc * (1 - self.params.poc_threshold*0.5) and current_close < self.poc:
                     decision = 'hold'
                     reason = f"Hold: Potential Rejection near POC ({self.poc:.4f})"
                 # else: Near POC but no clear rejection, continue checks

            # Bullish Signals (only if decision is still 'hold')
            elif decision == 'hold':
                 if self.pullback_signal == 'bullish_pullback':
                     buy_reason_base = f"Bullish Breakout/Pullback"
                     confirmation = []
                     if volume_surge_detected: confirmation.append("Volume Surge")
                     if self.poc is not None and abs(current_close - self.poc) / self.poc < self.params.poc_threshold and current_close > self.poc:
                         confirmation.append(f"near POC Support ({self.poc:.4f})")

                     if confirmation:
                          decision = 'buy'
                          reason = f"Buy: {buy_reason_base} w/ {' & '.join(confirmation)}"
                          side = 'buy'
                     else:
                          reason = f"Potential Buy: {buy_reason_base}, awaiting confirmation"

                 elif self.poc is not None and self.datalow[0] < self.poc * (1 + self.params.poc_threshold*0.5) and current_close > self.poc and volume_surge_detected:
                      decision = 'buy'
                      reason = f"Buy: Bounce from POC Support ({self.poc:.4f}) w/ Volume"
                      side = 'buy'

                 elif rsi_val < self.params.rsi_threshold and current_close > sma_long_val and volume_surge_detected:
                      buy_reason = f"Buy: RSI Low + SMA Cross + Vol Surge"
                      if self.rsi_divergence == 'bullish': buy_reason += " + Bullish Div"
                      if self.poc is not None and current_close > self.poc:
                           buy_reason += f" > POC ({self.poc:.4f})"
                           decision = 'buy' # Only buy if above POC for this signal
                           reason = buy_reason
                           side = 'buy'
                      else:
                           reason = f"Hold: RSI/SMA Buy Signal but below POC ({self.poc:.4f if self.poc else 'N/A'})"


            # Place Buy Order if conditions met
            if decision == 'buy':
                self.log(f'BUY CREATE, {current_close:.2f} - Reason: {reason}')
                # TODO: Implement risk-based sizing
                # Example fixed size:
                stake_size = self.broker.getsizer().p.stake # Get stake from sizer
                self.order = self.buy(size=stake_size)


        else: # We are in the market
            # --- Exit Logic (Stop Loss / Take Profit) ---
            exit_reason = None
            if self.position.size > 0 and self.buyprice is not None: # Long position
                sl_price = self.buyprice * (1.0 - self.params.sl_ratio)
                tp_price = self.buyprice * (1.0 + self.params.tp_ratio)
                if current_close <= sl_price:
                    exit_reason = f"Stop Loss (Long) at {sl_price:.2f}"
                elif current_close >= tp_price:
                    exit_reason = f"Take Profit (Long) at {tp_price:.2f}"
            elif self.position.size < 0: # Short position
                # Implement Short Exit Logic if needed
                pass

            if exit_reason:
                self.log(f'CLOSE CREATE (Exit), {current_close:.2f} - Reason: {exit_reason}')
                self.order = self.close() # Close position

    # Optional: Implement stop() method for end-of-run cleanup or analysis
    # def stop(self):
    #     self.log(f'(SMA Long {self.params.sma_long_period:d}, SMA Short {self.params.sma_short_period:d}, RSI {self.params.rsi_period:d}) Ending Value {self.broker.getvalue():.2f}', doprint=True)
