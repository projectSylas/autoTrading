# src/backtesting/strategies/challenge_strategy_backtest.py

import backtrader as bt
import pandas as pd
import numpy as np # For NaN checks
# import logging # <<< Removed logging import for strategy internal logging
from typing import Optional # Added for Optional type hint
from src.utils import strategy_utils # <<< RE-ADDED
# pandas-ta import도 필요할 수 있음
import pandas_ta as ta # <<< RE-ADDED
import math # <<< RE-ADDED
# import sys # Import sys for module checking <-- Removed, not used
import datetime # <<< Added for timestamping in direct file logging
import os # <<< Added for creating logs directory if needed

# --- Get Logger Name --- 
# from src.utils.logging_config import APP_LOGGER_NAME # Removed dependency on central logger config for internal logging
# STRATEGY_LOGGER_NAME = APP_LOGGER_NAME + '.strategy' # Removed

class ChallengeStrategyBacktest(bt.Strategy):
    """
    Restored version of the strategy with RSI filter.
    Focuses on technical indicators and conditions. Uses direct file writing for detailed logs.
    """
    params = (
        # --- RESTORED PARAMS ---
        ('sma_long_period', 20), # Placeholder, will be set by runner
        ('sma_short_period', 7), # Placeholder
        ('rsi_period', 14),      # Placeholder
        ('rsi_threshold', 30),   # Placeholder
        ('volume_avg_period', 20), # Placeholder
        ('volume_surge_ratio', 2.0), # Placeholder
        ('divergence_lookback', 28), # Placeholder
        ('breakout_lookback', 40), # Placeholder
        ('pullback_lookback', 10), # Placeholder
        ('poc_lookback', 50),     # Placeholder
        ('poc_threshold', 0.005), # Placeholder - Runner should set this too?
        ('pullback_threshold', 0.01), # Placeholder
        ('sl_ratio', 0.03),      # Set by runner
        ('tp_ratio', 0.06),      # Set by runner
        ('printlog', True),
    )

    def __init__(self):
        # --- Direct File Logging Setup ---
        log_dir = 'logs' # Define log directory
        os.makedirs(log_dir, exist_ok=True) # Ensure log directory exists
        # Use a consistent naming scheme, maybe add run timestamp later if needed
        self.log_file_path = os.path.join(log_dir, f"strategy_debug_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log") 
        self.log_file = None # Initialize file handle
        try:
            self.log_file = open(self.log_file_path, 'a', encoding='utf-8')
            print(f"[Strategy INFO] Strategy debug log will be written to: {self.log_file_path}") # Console info
            self._write_log("Strategy __init__ Starting") # Log point 1
        except Exception as e:
            print(f"[Strategy ERROR] Failed to open strategy log file {self.log_file_path}: {e}")

        # --- Standard Data Lines ---
        self._write_log("Strategy __init__: Referencing data lines...") # Log point 2a
        self.dataclose = self.datas[0].close
        self.datahigh = self.datas[0].high
        self.datalow = self.datas[0].low
        self.datavolume = self.datas[0].volume
        self._write_log("Strategy __init__: Data lines referenced.") # Log point 2b

        # --- Order Tracking ---
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.bar_executed = 0
        self.active_sl_price = None # Track active stop loss price
        self.active_tp_price = None # Track active take profit price

        # --- Backtrader Indicators --- 
        # Add try-except block for early error detection during indicator initialization
        try:
            self._write_log("Strategy __init__: Initializing standard indicators...") # Log point 3
            # Pass the specific data line (e.g., self.dataclose) instead of self.datas[0]
            self.sma_long = bt.indicators.SimpleMovingAverage(
                self.dataclose, period=self.params.sma_long_period) # Use self.dataclose
            self.sma_short = bt.indicators.SimpleMovingAverage(
                self.dataclose, period=self.params.sma_short_period) # Use self.dataclose
            self.rsi = bt.indicators.RelativeStrengthIndex(
                 self.dataclose, period=self.params.rsi_period) # Use self.dataclose
            self.volume_sma = bt.indicators.SimpleMovingAverage(
                 self.datavolume, period=self.params.volume_avg_period) # Use self.datavolume is correct
            self._write_log("Strategy __init__: Standard indicators initialized.") # Log point 4
        except Exception as e:
             # Log critical error if indicator initialization fails
             self._write_log(f"Strategy __init__: CRITICAL ERROR initializing standard indicators: {e}")
             print(f"[Strategy CRITICAL ERROR] Initializing standard indicators: {e}")
             # Re-raise the exception to potentially get a clearer traceback earlier
             raise

        # --- Custom Indicators (Calculated in next()) ---
        self.poc = None
        self.rsi_divergence = 'none'
        self.pullback_signal = 'none'

        # Use direct file write for logging
        self._write_log(f"Strategy Parameters: {self.params.__dict__}")
        self._write_log("Strategy __init__ Completed")

    def _write_log(self, txt):
        """Writes a log message directly to the strategy's dedicated log file."""
        if not self.log_file: # Do nothing if file wasn't opened successfully
            return
        try:
            timestamp = self.datas[0].datetime.datetime(0)
        except IndexError:
            timestamp = datetime.datetime.now()
        except Exception:
            timestamp = datetime.datetime.now()
        try:
            log_line = f"{timestamp.isoformat()} - {txt}\n"
            self.log_file.write(log_line)
            self.log_file.flush()
        except Exception as e:
            print(f"[Strategy ERROR] Failed to write to strategy log file: {e}")

    # --- RESTORED notify_order, notify_trade ---
    def notify_order(self, order):
        # Use _write_log instead of self.log
        if order.status in [order.Submitted, order.Accepted]:
            self._write_log(f"Order Submitted/Accepted: Ref={order.ref}, Type={'Buy' if order.isbuy() else 'Sell'}")
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self._write_log(
                    f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm {order.executed.comm:.2f}'
                )
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
                self.active_sl_price = self.buyprice * (1.0 - self.params.sl_ratio)
                self.active_tp_price = self.buyprice * (1.0 + self.params.tp_ratio)
                self._write_log(f"SL Price set to: {self.active_sl_price:.2f}, TP Price set to: {self.active_tp_price:.2f}")
            else:  # Sell
                self._write_log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm {order.executed.comm:.2f}')
                self.active_sl_price = None
                self.active_tp_price = None

            self.bar_executed = len(self) # Bar number of execution

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self._write_log(f'Order Failed: {order.getstatusname()} - Ref={order.ref}')

        self.order = None # Reset pending order tracker

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        # Use _write_log
        self._write_log(f'OPERATION PROFIT, GROSS {trade.pnl:.2f}, NET {trade.pnlcomm:.2f}')

    # --- RESTORED _get_buffered_df ---
    def _get_buffered_df(self, min_required_len: int) -> Optional[pd.DataFrame]:
        # Use _write_log for errors/debug if needed, but keep it minimal here
        # Standard logging is removed, rely on print for critical errors here if any
        try:
            if len(self.datas[0]) < min_required_len:
                 # Optional: self._write_log(f"Debug: Not enough data in buffer ({len(self.datas[0])} < {min_required_len})")
                 return None

            size = len(self.datas[0])
            opens = self.datas[0].open.get(size=size)
            highs = self.datas[0].high.get(size=size)
            lows = self.datas[0].low.get(size=size)
            closes = self.datas[0].close.get(size=size)
            volumes = self.datas[0].volume.get(size=size)
            datetimes = self.datas[0].datetime.get(size=size)

            # Check if RSI is available before getting its values
            if hasattr(self, 'rsi') and self.rsi:
                rsi_values = self.rsi.get(size=size)
                if len(rsi_values) > 0 and math.isnan(rsi_values[-1]):
                     # Optional: self._write_log("Debug: RSI still producing NaN, returning None from buffer.")
                     return None
            else:
                # RSI not initialized or available, return None or handle as needed
                self._write_log("Warning: RSI indicator not available in _get_buffered_df")
                # Return df without RSI or return None depending on downstream needs
                # For now, returning None to avoid errors in functions expecting RSI
                return None

            timestamps = [bt.num2date(dt) for dt in datetimes]

            df = pd.DataFrame({
                'Open': opens, 'High': highs, 'Low': lows, 'Close': closes,
                'Volume': volumes, 'RSI': rsi_values
            }, index=pd.DatetimeIndex(timestamps))

            df.dropna(subset=['RSI'], inplace=True)

            if len(df) < min_required_len:
                 # Optional: self._write_log(f"Debug: Not enough non-NaN RSI data ({len(df)} < {min_required_len})")
                 return None

            return df

        except Exception as e:
            # Print critical errors for buffer creation
            print(f"[Strategy CRITICAL ERROR] in _get_buffered_df: {e}")
            self._write_log(f"CRITICAL ERROR in _get_buffered_df: {e}") # Also log if possible
            return None

    def next(self):
        # Use _write_log for all strategy internal logging
        # self._write_log(f"--- next() called for bar {len(self)} ---") # Optional verbose log

        # --- Calculate Custom Indicators ---
        max_lookback_needed = max(
            getattr(self.params, 'poc_lookback', 50), # Use getattr for safety
            getattr(self.params, 'divergence_lookback', 28) * 2,
            getattr(self.params, 'breakout_lookback', 40) + getattr(self.params, 'pullback_lookback', 10)
        ) + 10 # Add a bit more buffer

        df_buffer = self._get_buffered_df(max_lookback_needed)

        if df_buffer is not None:
            try:
                self.poc = strategy_utils.calculate_poc(df_buffer, lookback=self.params.poc_lookback)
            except Exception as e:
                self._write_log(f"Error calculating POC: {e}")
                self.poc = None
            try:
                self.rsi_divergence = strategy_utils.detect_rsi_divergence(
                    df_buffer, lookback=self.params.divergence_lookback
                )
            except Exception as e:
                 self._write_log(f"Error detecting RSI Divergence: {e}")
                 self.rsi_divergence = 'none'
            try:
                self.pullback_signal = strategy_utils.detect_trendline_breakout_pullback(
                    df_buffer,
                    lookback_breakout=self.params.breakout_lookback,
                    lookback_pullback=self.params.pullback_lookback,
                    pullback_threshold=self.params.pullback_threshold
                )
            except Exception as e:
                 self._write_log(f"Error detecting Breakout/Pullback: {e}")
                 self.pullback_signal = 'none'
        else:
             self.poc = None
             self.rsi_divergence = 'none'
             self.pullback_signal = 'none'
             # self._write_log("Not enough data for custom indicator calculation.") # Reduce noise
             return # Skip rest of logic if buffer not ready

        # --- Log Custom Indicator Values (Reduced frequency maybe?) ---
        # poc_val = f"{self.poc:.4f}" if self.poc is not None else "N/A"
        # self._write_log(f"Custom Indicators: POC={poc_val}, RSI Div={self.rsi_divergence}, Pullback={self.pullback_signal}")

        # --- Access Standard Indicator Values ---
        current_close = self.dataclose[0]
        current_volume = self.datavolume[0]
        try:
            sma_long_val = self.sma_long[0]
            sma_short_val = self.sma_short[0]
            rsi_val = self.rsi[0]
            volume_sma_val = self.volume_sma[0]
        except IndexError:
            # self._write_log("Standard indicators not ready yet.") # Reduce noise
            return

        # --- Log Standard Indicator Values (Reduced frequency maybe?) ---
        # self._write_log(f"Standard Indicators: Close={current_close:.2f}, Vol={current_volume:.0f}, SMA{self.params.sma_short_period}={sma_short_val:.2f}, SMA_Long={sma_long_val:.2f}, RSI={rsi_val:.1f}, VolSMA={volume_sma_val:.0f}")

        # --- Condition Checks ---
        volume_surge = current_volume > (volume_sma_val * self.params.volume_surge_ratio) if volume_sma_val and volume_sma_val != 0 else False
        close_below_sma_short = current_close < sma_short_val
        close_near_poc = abs(current_close - self.poc) / current_close < self.params.poc_threshold if self.poc is not None else False
        is_bullish_divergence = self.rsi_divergence == 'bullish'
        is_bearish_divergence = self.rsi_divergence == 'bearish'
        is_bullish_pullback = self.pullback_signal == 'bullish_pullback'
        is_bearish_breakdown = self.pullback_signal == 'bearish_breakdown'

        # --- Log Conditions (Reduced frequency maybe?) ---
        # self._write_log(f"Conditions: VolSurge={volume_surge}, Close<SMA{self.params.sma_short_period}={close_below_sma_short}, BearDiv={is_bearish_divergence}, BullDiv={is_bullish_divergence}, BearPull={is_bearish_breakdown}, BullPull={is_bullish_pullback}, CloseNearPOC={close_near_poc}")

        # --- Trading Logic ---
        if self.order:
            # self._write_log("Pending order exists, skipping new order.") # Reduce noise
            return

        if not self.position:
            # --- BUY Logic (with RSI Filter) ---
            final_buy_signal = is_bullish_pullback and (rsi_val > 40 and rsi_val < 60)
            self._write_log(f"BUY CHECK: BullPull={is_bullish_pullback}, RSI={rsi_val:.1f} (Filter: 40-60), VolSurge={volume_surge}, BullDiv={is_bullish_divergence} => FinalBuy={final_buy_signal}")

            if final_buy_signal:
                self._write_log(f'BUY CREATE, Close={current_close:.2f}')
                self.order = self.buy()
                self.active_sl_price = None
                self.active_tp_price = None

        else: # In market
            # --- SELL Logic ---
            sell_sl_hit = self.active_sl_price is not None and current_close < self.active_sl_price
            sell_tp_hit = self.active_tp_price is not None and current_close > self.active_tp_price
            sell_signal = is_bearish_breakdown
            final_sell_signal = sell_sl_hit or sell_tp_hit or sell_signal

            sl_str = f"{self.active_sl_price:.2f}" if self.active_sl_price is not None else "N/A"
            tp_str = f"{self.active_tp_price:.2f}" if self.active_tp_price is not None else "N/A"
            self._write_log(f"SELL CHECK: Close={current_close:.2f}, SL Price={sl_str}, TP Price={tp_str}, SL Hit={sell_sl_hit}, TP Hit={sell_tp_hit}, Signal={sell_signal} (BearBreak only) => FinalSell={final_sell_signal}")

            if final_sell_signal:
                self._write_log(f'SELL CREATE, Close={current_close:.2f}')
                self.order = self.sell()

    def stop(self):
        """Close the strategy log file when the backtest stops."""
        self._write_log("Strategy Stop called.")
        if self.log_file:
            try:
                self.log_file.close()
                print(f"[Strategy INFO] Strategy debug log file closed: {self.log_file_path}")
            except Exception as e:
                print(f"[Strategy ERROR] Failed to close strategy log file {self.log_file_path}: {e}")
