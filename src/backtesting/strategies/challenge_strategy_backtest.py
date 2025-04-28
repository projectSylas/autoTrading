# src/backtesting/strategies/challenge_strategy_backtest.py

import backtrader as bt
import pandas as pd
import numpy as np
from typing import Optional
import datetime
import os
import math # Keep math for isnan checks if needed

# --- Import Sentiment Analysis --- 
from src.analysis.sentiment_analysis import get_market_sentiment

# --- REMOVED External Dependencies ---
# from src.utils import strategy_utils
# import pandas_ta as ta


class ChallengeStrategyBacktest(bt.Strategy):
    """
    Refactored strategy using only Backtrader internal indicators and logic.
    Integrates sentiment analysis as an additional filter for long entries.
    Uses direct file writing for detailed internal logging.
    """
    params = (
        # Parameters expected to be passed from the runner
        ('sma_long_period', 20),
        ('sma_short_period', 7),
        ('rsi_period', 14),
        ('rsi_oversold', 30), # Renamed from rsi_threshold for clarity
        ('rsi_overbought', 70),
        ('volume_avg_period', 20),
        ('volume_surge_ratio', 2.0),
        ('sl_ratio', 0.03),
        ('tp_ratio', 0.06),
        ('printlog', True), # Controls logging via self.log (if used) OR enables _write_log
        # --- Parameters for internal logic (simplified/placeholders) ---
        # These might not be directly used in the simplified version
        ('divergence_lookback', 28),
        ('breakout_lookback', 40),
        ('pullback_lookback', 10),
        ('pullback_threshold', 0.01),
        # POC parameters might be removed or implemented internally if needed
        # ('poc_lookback', 50),
        # ('poc_threshold', 0.005),
        # --- Sentiment Analysis Params ---
        ('sentiment_keyword', 'Bitcoin'), # Keyword for sentiment analysis
        ('enable_sentiment_filter', True), # Toggle sentiment filter on/off
    )

    def __init__(self):
        # --- Direct File Logging Setup ---
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        # Add strategy name to log file for clarity
        strategy_name = self.__class__.__name__
        self.log_file_path = os.path.join(log_dir, f"{strategy_name}_debug_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        self.log_file = None
        try:
            self.log_file = open(self.log_file_path, 'a', encoding='utf-8')
            print(f"[Strategy INFO] {strategy_name} debug log file created: {self.log_file_path}")
            self._write_log("Strategy __init__ Starting")
        except Exception as e:
            print(f"[Strategy ERROR] Failed to open strategy log file {self.log_file_path}: {e}")

        # --- Standard Data Lines ---
        self._write_log("Strategy __init__: Referencing data lines...")
        self.dataclose = self.datas[0].close
        self.datahigh = self.datas[0].high
        self.datalow = self.datas[0].low
        self.datavolume = self.datas[0].volume
        # Store data name for potential use in sentiment keyword
        self.dataname = self.datas[0]._name if hasattr(self.datas[0], '_name') else 'default_data'
        self._write_log(f"Strategy __init__: Data lines referenced for {self.dataname}.")

        # --- Order Tracking ---
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.bar_executed = 0
        self.active_sl_price = None
        self.active_tp_price = None

        # --- Backtrader Indicators --- 
        try:
            self._write_log("Strategy __init__: Initializing Backtrader indicators...")
            self.sma_long = bt.indicators.SimpleMovingAverage(self.dataclose, period=self.params.sma_long_period)
            self.sma_short = bt.indicators.SimpleMovingAverage(self.dataclose, period=self.params.sma_short_period)
            self.rsi = bt.indicators.RelativeStrengthIndex(self.dataclose, period=self.params.rsi_period)
            self.volume_sma = bt.indicators.SimpleMovingAverage(self.datavolume, period=self.params.volume_avg_period)

            # --- Add indicators for internal Breakout/Pullback/Divergence checks ---
            self.highest_high = bt.indicators.Highest(self.datahigh, period=self.params.breakout_lookback)
            self.lowest_low = bt.indicators.Lowest(self.datalow, period=self.params.breakout_lookback)
            self.rsi_highest = bt.indicators.Highest(self.rsi, period=self.params.divergence_lookback)
            self.rsi_lowest = bt.indicators.Lowest(self.rsi, period=self.params.divergence_lookback)

            self._write_log("Strategy __init__: Backtrader indicators initialized.")
        except Exception as e:
             self._write_log(f"Strategy __init__: CRITICAL ERROR initializing indicators: {e}")
             print(f"[Strategy CRITICAL ERROR] Initializing indicators: {e}")
             raise

        # --- Sentiment State ---
        self.last_sentiment_check_bar = -1
        self.current_market_sentiment = "neutral" # Initialize sentiment

        self._write_log(f"Strategy Parameters: {self.params.__dict__}")
        self._write_log("Strategy __init__ Completed")

    def _write_log(self, txt):
        """Writes a log message directly to the strategy's dedicated log file."""
        if not self.log_file:
            return
        try:
            # Try to get backtrader timestamp first
            timestamp = self.datas[0].datetime.datetime(0)
        except Exception:
            # Fallback to system time if backtrader time is not available (e.g., during __init__)
            timestamp = datetime.datetime.now()
        try:
            log_line = f"{timestamp.isoformat()} - {txt}\n"
            self.log_file.write(log_line)
            self.log_file.flush()
        except Exception as e:
            print(f"[Strategy ERROR] Failed to write to strategy log file: {e}")

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            self._write_log(f"Order Submitted/Accepted: Ref={order.ref}, Type={'Buy' if order.isbuy() else 'Sell'}")
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self._write_log(f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm {order.executed.comm:.2f}')
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
                self.active_sl_price = self.buyprice * (1.0 - self.params.sl_ratio)
                self.active_tp_price = self.buyprice * (1.0 + self.params.tp_ratio)
                self._write_log(f"SL Price set to: {self.active_sl_price:.2f}, TP Price set to: {self.active_tp_price:.2f}")
            else: # Sell (closing a long position)
                self._write_log(f'SELL EXECUTED (Close Long), Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm {order.executed.comm:.2f}')
                self.active_sl_price = None # Deactivate SL/TP on close
                self.active_tp_price = None
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self._write_log(f'Order Failed: {order.getstatusname()} - Ref={order.ref}')
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self._write_log(f'TRADE CLOSED: PNL Gross {trade.pnl:.2f}, PNL Net {trade.pnlcomm:.2f}')
        self.active_sl_price = None # Ensure SL/TP are deactivated after trade closes
        self.active_tp_price = None

    def next(self):
        current_bar_index = len(self)
        # --- Check Indicator Readiness ---
        indicators_ready = all(
            len(ind.lines[0]) > max(self.params.sma_long_period, self.params.sma_short_period, self.params.rsi_period, self.params.volume_avg_period, self.params.divergence_lookback, self.params.breakout_lookback) 
            and not math.isnan(ind[-1]) 
            for ind in [self.sma_long, self.sma_short, self.rsi, self.volume_sma, self.highest_high, self.lowest_low, self.rsi_highest, self.rsi_lowest]
        )

        if not indicators_ready:
            # self._write_log(f"Bar {current_bar_index}: Indicators not ready yet...") # Debug log
            return

        current_close = self.dataclose[0]
        current_rsi = self.rsi[0]
        current_volume = self.datavolume[0]

        # --- Check if an order is pending ---
        if self.order:
            return

        # --- Position Management (Check SL/TP if in market) ---
        if self.position:
            if self.active_tp_price is not None and current_close >= self.active_tp_price:
                self._write_log(f"Bar {current_bar_index}: TP Hit: Close={current_close:.2f} >= TP={self.active_tp_price:.2f}. Closing position.")
                self.order = self.close()
                return
            elif self.active_sl_price is not None and current_close <= self.active_sl_price:
                self._write_log(f"Bar {current_bar_index}: SL Hit: Close={current_close:.2f} <= SL={self.active_sl_price:.2f}. Closing position.")
                self.order = self.close()
                return

        # --- Sentiment Analysis Check (Perform periodically, e.g., once per day or N bars) ---
        # Avoid calling API on every single bar in backtesting for performance
        # Example: Check sentiment every 24 bars (assuming hourly data)
        check_interval = 24
        if self.params.enable_sentiment_filter and (current_bar_index - self.last_sentiment_check_bar >= check_interval):
            try:
                self._write_log(f"Bar {current_bar_index}: Checking market sentiment for '{self.params.sentiment_keyword}'...")
                # In a real scenario, you might want to dynamically determine the keyword
                # keyword_to_check = self.dataname # Or derive from self.dataname
                self.current_market_sentiment = get_market_sentiment(keyword=self.params.sentiment_keyword)
                self.last_sentiment_check_bar = current_bar_index
                self._write_log(f"Bar {current_bar_index}: Market sentiment updated to: {self.current_market_sentiment}")
            except Exception as e:
                 self._write_log(f"Bar {current_bar_index}: Error getting market sentiment: {e}")
                 # Decide how to handle sentiment error: default to neutral or skip trades?
                 self.current_market_sentiment = "neutral" # Default to neutral on error

        # --- Entry Signal Logic (Only if not in market) ---
        if not self.position:
            # --- Basic Conditions using Backtrader Indicators ---
            sma_cross_bullish = self.sma_short[0] > self.sma_long[0] and self.sma_short[-1] <= self.sma_long[-1]
            rsi_oversold_cond = current_rsi < self.params.rsi_oversold
            volume_avg = self.volume_sma[0]
            volume_surge_cond = False
            if volume_avg is not None and not math.isnan(volume_avg) and volume_avg > 0:
                 volume_surge_cond = current_volume > volume_avg * self.params.volume_surge_ratio

            # --- Advanced Conditions (Simplified Internal Implementation) ---
            bullish_divergence_cond = False
            if len(self.rsi) > self.params.divergence_lookback:
                 lookback = self.params.divergence_lookback
                 price_low_idx = np.argmin(self.datalow.get(ago=-1, size=lookback))
                 rsi_low_idx = np.argmin(self.rsi.get(ago=-1, size=lookback))
                 if (self.datalow[0] < self.datalow[-price_low_idx-1] and
                     self.rsi[0] > self.rsi[-rsi_low_idx-1]):
                     bullish_divergence_cond = True

            breakout_cond = False
            pullback_cond = False
            if len(self.highest_high) > 1:
                 if current_close > self.highest_high[-1]:
                     breakout_cond = True
                 elif self.dataclose[-1] > self.highest_high[-2] and current_close < self.dataclose[-1]:
                      if (self.dataclose[-1] - current_close) / self.dataclose[-1] < self.params.pullback_threshold:
                           pullback_cond = True

            # --- Combine Conditions for Long Entry (Example) ---
            long_signal = False
            condition1 = sma_cross_bullish or (breakout_cond and pullback_cond)
            condition2 = rsi_oversold_cond or bullish_divergence_cond
            condition3 = volume_surge_cond
            if condition1 and condition2 and condition3:
                 long_signal = True
                 self._write_log(f"Bar {current_bar_index}: LONG SIGNAL DETECTED (Pre-Sentiment): Close={current_close:.2f}, RSI={current_rsi:.1f}, "
                                 f"VolSurge={volume_surge_cond}, SMACross={sma_cross_bullish}, BrkOut={breakout_cond}, "
                                 f"Pullb={pullback_cond}, BullDiv={bullish_divergence_cond}")

            # --- Apply Sentiment Filter --- 
            sentiment_allows_buy = True # Default to allow if filter disabled or sentiment is not bearish
            if self.params.enable_sentiment_filter:
                if self.current_market_sentiment == "bearish":
                    sentiment_allows_buy = False
                    self._write_log(f"Bar {current_bar_index}: BUY SIGNAL BLOCKED by BEARISH sentiment ('{self.params.sentiment_keyword}').")
                else:
                    self._write_log(f"Bar {current_bar_index}: Sentiment check passed ({self.current_market_sentiment}).")

            # --- Execute Long Trade (If signal and sentiment allow) --- 
            if long_signal and sentiment_allows_buy:
                 cash = self.broker.get_cash()
                 size_to_buy = int((cash / current_close) * 0.95)
                 if size_to_buy > 0:
                     self._write_log(f"Bar {current_bar_index}: Placing BUY order: Size={size_to_buy}, Price ~{current_close:.2f}")
                     self.order = self.buy(size=size_to_buy)
                 else:
                     self._write_log(f"Bar {current_bar_index}: Skipping BUY: Calculated size {size_to_buy} <= 0 (Cash: {cash:.2f})")
            elif long_signal and not sentiment_allows_buy:
                 # Log that the signal occurred but was filtered
                 pass # Already logged above

            # --- Add Short Signal Logic Here if needed ---
            # Note: Sentiment filter usually applies to long entries, but could be adapted for shorts

    def stop(self):
        # Strategy cleanup actions when backtest ends
        self._write_log("Strategy stop() called.")
        if self.log_file:
            try:
                self.log_file.close()
                print(f"[Strategy INFO] Strategy debug log file closed: {self.log_file_path}")
            except Exception as e:
                print(f"[Strategy ERROR] Failed to close strategy log file: {e}")
        self.log_file = None
