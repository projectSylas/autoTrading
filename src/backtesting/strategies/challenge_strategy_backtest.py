# src/backtesting/strategies/challenge_strategy_backtest.py

import backtrader as bt
import pandas as pd
import numpy as np # For NaN checks
import logging
from typing import Optional # Added for Optional type hint
from src.utils import strategy_utils # 기존 유틸 함수 사용
# pandas-ta import도 필요할 수 있음
import pandas_ta as ta
import math
import sys # Import sys for module checking

class ChallengeStrategyBacktest(bt.Strategy):
    """
    Backtrader strategy class implementing the core logic of the Flight Challenge strategy.
    Focuses on technical indicators and conditions. AI overrides are not included by default.
    """
    # Initialize params with placeholder values first
    params = (
        ('sma_long_period', 20), # Placeholder, will be set in __init__
        ('sma_short_period', 7),
        ('rsi_period', 14),      # Placeholder
        ('rsi_threshold', 30),   # Placeholder
        ('volume_avg_period', 20), # Placeholder
        ('volume_surge_ratio', 2.0), # Placeholder
        ('divergence_lookback', 28), # Placeholder
        ('breakout_lookback', 40), # Placeholder
        ('pullback_lookback', 10), # Placeholder
        ('poc_lookback', 50),     # Placeholder
        ('poc_threshold', 0.005), # Placeholder
        ('pullback_threshold', 0.01), # Add default for optimization target
        ('sl_ratio', 0.05),      # Placeholder
        ('tp_ratio', 0.10),      # Placeholder
        ('printlog', True),
    )

    def log(self, txt, dt=None, doprint=False):
        ''' Logging function for this strategy'''
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()}, {txt}')

    def __init__(self):
        # from src.config import settings # <<< REMOVE THIS LINE

        # --- Enhanced Diagnostic Print (Inside __init__) --- 
        # <<< REMOVE THIS ENTIRE BLOCK >>>
        # print(f"\n--- Diagnostics inside {self.__class__.__name__}.__init__ ---")
        # try:
        #    ...
        # print("--- End Diagnostics ---\n")
        # --- End Enhanced Diagnostic Print ---

        # Override params with values from settings <-- This comment is now incorrect
        # Parameters are now directly populated by backtrader from kwargs passed in runner.
        # self.params.sma_long_period = settings.CHALLENGE_SMA_PERIOD # <<< No longer needed
        # self.params.rsi_period = settings.CHALLENGE_RSI_PERIOD       # <<< No longer needed
        # ... and so on for all lines accessing settings directly ...
        # Keep the lines assigning default values in params tuple at the class level.
        # Backtrader uses those defaults if not overridden by kwargs.

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

        # Option 1: Use backtrader indicators (Re-enabled)
        self.sma_long = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.sma_long_period)
        self.sma_short = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.sma_short_period)
        self.rsi = bt.indicators.RelativeStrengthIndex(
            period=self.params.rsi_period)
        self.volume_sma = bt.indicators.SimpleMovingAverage(
             self.datavolume, period=self.params.volume_avg_period)
        
        # Assign None temporarily so later code doesn't break immediately
        # self.sma_long = None # Removed temporary assignment
        # self.sma_short = None
        # self.rsi = None
        # self.volume_sma = None
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
        # --- __init__ 완료 확인 로그 (선택 사항, 일단 유지) ---
        self.log("--- __init__ completed successfully ---", doprint=True)

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
        try:
            # --- 디버깅용 print 로그 제거 ---
            # print(f"DEBUG PRINT: --- _get_buffered_df called. Buffer: {len(self.datas[0])}, Required: {min_required_len} ---")

            if len(self.datas[0]) < min_required_len:
                 # --- 디버깅용 print 로그 제거 ---
                 # print(f"DEBUG PRINT: --- _get_buffered_df returning None (data < required) ---")
                 return None

            # --- 디버깅용 print 로그 제거 ---
            size = max(min_required_len, len(self.datas[0]))
            # print(f"DEBUG PRINT: --- _get_buffered_df requesting size: {size} ---")

            opens = self.datas[0].open.get(size=size)
            highs = self.datas[0].high.get(size=size)
            lows = self.datas[0].low.get(size=size)
            closes = self.datas[0].close.get(size=size)
            volumes = self.datas[0].volume.get(size=size)
            datetimes = self.datas[0].datetime.get(size=size)
            rsi_values = self.rsi.get(size=size)
            # print(f"DEBUG PRINT: --- _get_buffered_df data retrieved successfully ---")

            timestamps = [bt.num2date(dt) for dt in datetimes]
            # print(f"DEBUG PRINT: --- _get_buffered_df timestamps converted ---")

            df = pd.DataFrame({
                'Open': opens, 'High': highs, 'Low': lows, 'Close': closes,
                'Volume': volumes, 'RSI': rsi_values
            }, index=pd.DatetimeIndex(timestamps))
            # print(f"DEBUG PRINT: --- _get_buffered_df DataFrame created successfully ---")

            return df

        except Exception as e:
            # --- 예외 발생 시 표준 로깅 사용 ---
            # print(...) 대신 logging.error 사용
            logging.error(f"EXCEPTION CAUGHT in _get_buffered_df: Type={type(e).__name__}, Args={e.args}", exc_info=True)
            # import traceback # traceback import 제거
            # print(traceback.format_exc()) # print 대신 logging 사용
            # print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            return None # 예외 발생 시 None 반환

    def next(self):
        # --- next() 메소드 호출 확인 로그 (선택 사항, 일단 주석 처리) ---
        # self.log("--- next() method called --- ", doprint=True)
        # --- 디버깅용 print 로그 제거 ---
        # print("DEBUG PRINT: --- Starting max_lookback calculation ---")

        # --- Calculate Custom Indicators using Buffered Data ---
        max_lookback_needed = max(
            self.params.poc_lookback,
            self.params.divergence_lookback * 2,
            self.params.breakout_lookback + self.params.pullback_lookback
        ) + 5
        # --- 디버깅용 print 로그 제거 ---
        # print(f"DEBUG PRINT: --- Calculated max_lookback_needed: {max_lookback_needed} ---")

        # --- 디버깅용 print 로그 제거 ---
        # print("DEBUG PRINT: --- About to call _get_buffered_df --- ")
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
        
        # <<< Logging for Custom Indicators (self.log 복원) >>>
        poc_str = f"{self.poc:.4f}" if self.poc is not None else "N/A"
        self.log(f"Custom Indicators: POC={poc_str}, RSI Div={self.rsi_divergence}, Pullback={self.pullback_signal}")

        # Access standard indicator values (calculated by backtrader)
        current_close = self.dataclose[0]
        current_volume = self.datavolume[0]
        sma_long_val = self.sma_long[0] # Ensure sma_long is defined if used
        sma_short_val = self.sma_short[0]
        rsi_val = self.rsi[0]
        avg_volume_val = self.volume_sma[0]

        # --- Condition Check ---
        if self.order:
             # --- 디버깅용 print 로그 제거 ---
             # print(f"DEBUG PRINT: next() returning early because self.order is not None: {self.order}")
             return

        if not self.position:
            decision = 'hold'
            reason = "No signal (or insufficient data)"
            side = None

            indicators_valid = all([
                not math.isnan(current_close),
                not math.isnan(current_volume),
                not math.isnan(sma_short_val),
                not math.isnan(rsi_val),
                avg_volume_val is not None and not math.isnan(avg_volume_val) and avg_volume_val > 0,
                self.pullback_signal != 'none'
            ])
            # <<< Logging for indicator validity (self.log 복원) >>>
            self.log(f"Indicators Valid Check: {indicators_valid}")

            poc_valid_and_above = self.poc is not None and not math.isnan(self.poc) and current_close > self.poc

            if indicators_valid:
                # Bullish Entry Conditions
                is_bullish_pullback = self.pullback_signal == 'bullish_pullback'
                is_volume_surge = current_volume > (avg_volume_val * self.params.volume_surge_ratio)
                is_above_sma7 = current_close > sma_short_val
                is_bullish_divergence = self.rsi_divergence == 'bullish'

                # <<< Conditional Logging for Individual Conditions (self.log 유지) >>>
                if is_bullish_pullback or is_volume_surge:
                    self.log(f"CONDITIONS MET CHECK: is_bullish_pullback={is_bullish_pullback}, is_volume_surge={is_volume_surge}, is_above_sma7={is_above_sma7}, poc_valid_and_above={poc_valid_and_above}, is_bullish_divergence={is_bullish_divergence}")

                # Combine conditions for Buy (is_bullish_pullback 복원됨)
                buy_condition = (
                    is_bullish_pullback and
                    is_volume_surge and
                    is_above_sma7 and
                    (self.poc is None or poc_valid_and_above)
                    # and is_bullish_divergence
                )
                
                # <<< Conditional Logging for Final Buy Condition (self.log 유지) >>>
                if buy_condition or (is_bullish_pullback or is_volume_surge):
                    self.log(f"Final Buy Condition Check: {buy_condition}")

                if buy_condition:
                    decision = 'buy'
                    reason = f"Bullish Pullback + Vol Surge + >SMA7"
                    if poc_valid_and_above: reason += " + >POC"
                    # if is_bullish_divergence: reason += " + Bullish Div" # 다이버전스 조건은 아직 미사용
                    side = 'buy'
            else:
                 reason = "Insufficient data or invalid indicators"
                 self.log(f"HOLD: {reason}")

            if decision == 'buy':
                self.log(f'BUY CREATE, {current_close:.2f} - Reason: {reason}')
                self.order = self.buy() # Sizer 사용
        else: # We are in the market
            # --- Exit Logic (Restored) ---
            exit_reason = None
            indicators_valid_exit = all([
                not math.isnan(current_close),
                not math.isnan(sma_short_val),
                self.rsi_divergence != 'none', # Need divergence signal
                self.pullback_signal != 'none' # Need pullback/breakdown signal
            ])

            # Check Take Profit
            if self.position.size > 0 and self.buyprice is not None and current_close >= self.buyprice * (1.0 + self.params.tp_ratio):
                exit_reason = f"Take Profit (Long) at {current_close:.2f}"
            # Check Stop Loss
            elif self.position.size > 0 and self.buyprice is not None and current_close <= self.buyprice * (1.0 - self.params.sl_ratio):
                exit_reason = f"Stop Loss (Long) at {current_close:.2f}"
            # Check Signal-based Exit
            elif self.position.size > 0 and indicators_valid_exit and exit_reason is None:
                is_bearish_divergence = self.rsi_divergence == 'bearish'
                is_bearish_breakdown = self.pullback_signal == 'bearish_breakdown'
                is_below_sma7 = current_close < sma_short_val

                if is_bearish_divergence or is_bearish_breakdown or is_below_sma7:
                    reason_detail = []
                    if is_bearish_divergence: reason_detail.append("Bearish Div")
                    if is_bearish_breakdown: reason_detail.append("Bearish Breakdown")
                    if is_below_sma7: reason_detail.append("<SMA7")
                    exit_reason = f"Signal Exit: { ' & '.join(reason_detail)}"

            if exit_reason:
                self.log(f'CLOSE CREATE (Exit), {current_close:.2f} - Reason: {exit_reason}')
                self.order = self.close()

    # Optional: Implement stop() method for end-of-run cleanup or analysis
    # def stop(self):
    #     self.log(f'(SMA Long {self.params.sma_long_period:d}, SMA Short {self.params.sma_short_period:d}, RSI {self.params.rsi_period:d}) Ending Value {self.broker.getvalue():.2f}', doprint=True)
