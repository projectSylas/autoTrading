import logging
import pandas as pd
from datetime import datetime, timedelta
import os
import time # 포지션 관리 루프용
import math
from typing import Dict, Any, Optional, Tuple

# --- Setup logger for this module --- 
# This assumes setup_logging() from logging_config has already been called by the entry point (e.g., main.py or runner)
logger = logging.getLogger(__name__)
# --- End Setup logger --- 

# 설정 및 유틸리티 모듈 로드
from src.config import settings
from src.utils import common as strategy_utils # Renamed for clarity
from src.utils import notifier

# --- [STEP 1-A START] AI/Analysis 모듈 로드 시도 --- #
try:
    from src.analysis.sentiment import get_market_sentiment
except ImportError:
    get_market_sentiment = None
    # Use module logger
    logger.warning("Sentiment analysis module (src.analysis.sentiment.py) not found or import error. Sentiment features disabled.")

try:
    from src.ai_models.price_predictor import run_price_prediction_flow
except ImportError:
    run_price_prediction_flow = None
    # Use module logger
    logger.warning("Price predictor module (src.ai_models.price_predictor.py) not found or import error. Prediction features disabled.")
# --- [STEP 1-A END] AI/Analysis 모듈 로드 시도 --- #

# DB 로깅 함수 임포트
try:
    from src.utils.database import log_trade_to_db
except ImportError:
    log_trade_to_db = None
    # Use module logger
    logger.warning("Database logging function (log_trade_to_db) not found. DB logging disabled for trades.")

# --- API 클라이언트 초기화 (Binance 예시) ---\
FUTURES_CLIENT_TYPE = None
FUTURES_CLIENT = None
BinanceAPIException = None
BinanceOrderException = None

if settings.BINANCE_API_KEY and settings.BINANCE_SECRET_KEY:
    # Use module logger
    logger.info("Binance API keys found. Attempting to initialize Futures client...")
    try:
        # from binance.client import Client # Not strictly needed if only using Futures
        from binance.futures import Futures
        from binance.exceptions import BinanceAPIException, BinanceOrderException
        # TODO: Add setting for testnet vs mainnet
        FUTURES_CLIENT = Futures(key=settings.BINANCE_API_KEY, secret=settings.BINANCE_SECRET_KEY)
        FUTURES_CLIENT.ping()
        server_time = FUTURES_CLIENT.time()
        # Use module logger
        logger.info(f"✅ Binance Futures client initialized successfully. Server time: {datetime.fromtimestamp(server_time['serverTime']/1000)}")
        FUTURES_CLIENT_TYPE = "Binance"
    except ImportError:
        # Use module logger
        logger.error("❌ Failed to initialize Binance Futures client: 'python-binance' library not installed.")
    except BinanceAPIException as bae:
         # Use module logger
         logger.error(f"❌ Binance Futures API connection error: {bae.status_code} - {bae.message}")
    except Exception as e:
        # Use module logger
        logger.error(f"❌ Unexpected error initializing Binance Futures client: {e}")
else:
    # Use module logger
    logger.warning("Binance Futures API keys not found in settings. Challenge strategy execution will be limited.")


# --- Helper Functions --- #
def get_current_position(symbol: str) -> Optional[Dict[str, Any]]:
    """Retrieves the current open position for the given symbol."""
    if not FUTURES_CLIENT or FUTURES_CLIENT_TYPE != "Binance": return None
    try:
        positions = FUTURES_CLIENT.futures_position_information(symbol=symbol)
        for pos in positions:
            # Check for non-zero position amount
            if float(pos['positionAmt']) != 0:
                # Use module logger
                logger.info(f"Found open position for {symbol}: Size={pos['positionAmt']}, Entry={pos['entryPrice']}, PNL={pos['unRealizedProfit']}")
                return pos # Return the first non-zero position found
        return None # No open position
    except BinanceAPIException as bae:
        # Use module logger
        logger.error(f"Error fetching position for {symbol}: {bae.status_code} - {bae.message}")
        return None
    except Exception as e:
        # Use module logger
        logger.error(f"Unexpected error fetching position for {symbol}: {e}")
        return None

def create_futures_order(symbol: str, side: str, quantity: float, order_type: str = 'MARKET', reduce_only: bool = False) -> Optional[Dict[str, Any]]:
    """Places a futures order (MARKET or LIMIT etc.)."""
    if not FUTURES_CLIENT or FUTURES_CLIENT_TYPE != "Binance": return None
    order_side = side.upper() # BUY or SELL
    order_log_info = f"Symbol={symbol}, Side={order_side}, Qty={quantity}, Type={order_type}{', ReduceOnly' if reduce_only else ''}"
    # Use module logger
    logger.info(f"Attempting to place order: {order_log_info}...")
    try:
        params = {
            'symbol': symbol,
            'side': order_side,
            'type': order_type.upper(),
            'quantity': quantity
        }
        if reduce_only:
            params['reduceOnly'] = 'true'

        # For LIMIT orders, add price parameter
        # if order_type.upper() == 'LIMIT':
        #     params['price'] = # Get price from args
        #     params['timeInForce'] = 'GTC' # Good 'Til Cancelled

        order_result = FUTURES_CLIENT.futures_create_order(**params)
        # Use module logger
        logger.info(f"✅ Order placed successfully: {order_result}")
        return order_result
    except BinanceOrderException as boe:
         # Use module logger
         logger.error(f"❌ Binance order creation failed: {boe.status_code} - {boe.message}. Request: {params}")
         notifier.send_slack_notification("Challenge Order Failed", f"{order_log_info}\nError: {boe.message}")
         return None
    except BinanceAPIException as bae:
         # Use module logger
         logger.error(f"❌ Binance API error during order placement: {bae.status_code} - {bae.message}")
         notifier.send_slack_notification("Challenge Order API Error", f"{order_log_info}\nError: {bae.message}")
         return None
    except Exception as e:
        # Use module logger
        logger.error(f"❌ Unexpected error placing order: {e}")
        notifier.send_slack_notification("Challenge Order Unexpected Error", f"{order_log_info}\nError: {e}")
        return None


# --- Market Data and Analysis --- #
def get_market_data(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    """Fetches market klines data."""
    if not FUTURES_CLIENT or FUTURES_CLIENT_TYPE != "Binance": return pd.DataFrame()
    # Use module logger
    logger.info(f"Fetching {limit} {interval} klines for {symbol}...")
    try:
        klines = FUTURES_CLIENT.futures_klines(symbol=symbol, interval=interval, limit=limit)
        if not klines:
            # Use module logger
            logger.warning(f"No klines data returned for {symbol}.")
            return pd.DataFrame()
        df = pd.DataFrame(klines, columns=[
            'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
            'Close time', 'Quote asset volume', 'Number of trades',
            'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
        ])
        df = df[['Open time', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col])
        df.set_index('Open time', inplace=True)
        # Use module logger
        # logger.info(f"Loaded {len(df)} klines for {symbol}.") # Reduce log verbosity
        return df
    except BinanceAPIException as bae:
        # Use module logger
        logger.error(f"API error fetching klines for {symbol}: {bae}")
        return pd.DataFrame()
    except Exception as e:
        # Use module logger
        logger.error(f"Unexpected error fetching klines for {symbol}: {e}")
        return pd.DataFrame()

def analyze_prediction_trend(predictions_df: Optional[pd.DataFrame]) -> str:
    """Analyzes the trend from the HF prediction DataFrame (Placeholder)."""
    if predictions_df is None or predictions_df.empty:
        return "no_prediction"
    try:
        if 'predicted_price' not in predictions_df.columns:
             # Use module logger
             logger.warning("Prediction DataFrame missing 'predicted_price' column.")
             return "no_prediction"
        # Ensure at least two rows to calculate trend
        if len(predictions_df) < 2:
            # Use module logger
            logger.warning("Prediction DataFrame has less than 2 rows, cannot determine trend.")
            return "no_prediction"
        first_pred = predictions_df['predicted_price'].iloc[0]
        last_pred = predictions_df['predicted_price'].iloc[-1]
        if pd.isna(first_pred) or pd.isna(last_pred):
            # Use module logger
            logger.warning("NaN values found in prediction DataFrame.")
            return "no_prediction"
        price_change_ratio = (last_pred - first_pred) / first_pred if first_pred != 0 else 0
        if price_change_ratio > 0.001: # Example threshold for upward trend
            return "predict_up"
        elif price_change_ratio < -0.001: # Example threshold for downward trend
            return "predict_down"
        else:
            return "predict_flat"
    except IndexError: # Should be caught by len check, but just in case
        # Use module logger
        logger.warning("Index error analyzing prediction trend.")
        return "no_prediction"
    except Exception as e:
        # Use module logger
        logger.warning(f"Could not analyze prediction trend from DataFrame: {e}")
        return "no_prediction"

# --- Entry/Exit Condition Check (STEP 1-B UPDATE & 1-C INTEGRATION) --- #
def check_strategy_conditions(symbol: str, interval: str, lookback_period: str) -> Dict[str, Any]:
    """Checks technical conditions, sentiment, and predictions.
       Combines technical signals with AI/Analysis results for a final decision.
    """
    decision_info = {
        "decision": "hold", # Default action: hold/do nothing
        "reason": "Initial state",
        "sentiment_label": "N/A",
        "sentiment_score": None,
        "prediction_trend": "N/A",
        "hf_prediction_df": None, # Store the raw prediction df if available
        "side": None,             # 'buy' or 'sell'
        "price": None,            # Current price for reference
        "indicators": {},         # Store calculated indicator values
        "override_reason": None   # Reason if decision was overridden by sentiment/prediction
    }

    # --- [STEP 1-B START] Fetch Sentiment & Prediction --- #
    if settings.ENABLE_HF_SENTIMENT and get_market_sentiment:
        try:
            # Use base asset (e.g., BTC from BTCUSDT) as keyword? Or full symbol? Adjust as needed.
            keyword = symbol.replace("USDT", "") if "USDT" in symbol else symbol
            # Use module logger
            logger.info(f"Fetching sentiment for keyword: {keyword}...")
            s_label, s_score, _ = get_market_sentiment(keyword=keyword) # Call the imported function
            decision_info["sentiment_label"] = s_label
            decision_info["sentiment_score"] = s_score
            # Use module logger
            logger.info(f"Sentiment result: {s_label} ({s_score if s_score is not None else 'N/A':.3f})")
        except Exception as e:
            # Use module logger
            logger.error(f"Error getting market sentiment for {keyword}: {e}")
            decision_info["sentiment_label"] = "error"
            decision_info["sentiment_score"] = None # Ensure score is None on error
    elif settings.ENABLE_HF_SENTIMENT:
        # Use module logger
        logger.warning("Sentiment analysis enabled but get_market_sentiment function is not available.")
    else:
        # Use module logger
        logger.info("Sentiment analysis disabled.")

    if settings.ENABLE_HF_TIMESERIES and run_price_prediction_flow:
        try:
            # Use module logger
            logger.info(f"Running price prediction flow for {symbol}...")
            # Call the imported function
            # The function returns: (primary_model_predictions_df, hf_model_predictions_df)
            _, hf_predictions_df = run_price_prediction_flow(train=False, predict=True, symbol=symbol, interval=interval) # Pass symbol/interval if needed
            decision_info["hf_prediction_df"] = hf_predictions_df # Store the raw DF

            # Analyze the trend from the returned DataFrame
            pred_trend = analyze_prediction_trend(hf_predictions_df) # Assumes this helper exists
            decision_info["prediction_trend"] = pred_trend
            # Use module logger
            logger.info(f"Price prediction trend analysis result: {pred_trend}")

        except Exception as e:
            # Use module logger
            logger.error(f"Error running price prediction flow for {symbol}: {e}", exc_info=True)
            decision_info["prediction_trend"] = "error" # Mark as error
            decision_info["hf_prediction_df"] = None # Ensure DF is None on error
    elif settings.ENABLE_HF_TIMESERIES:
        # Use module logger
        logger.warning("Time series prediction enabled but run_price_prediction_flow function is not available.")
    else:
        # Use module logger
        logger.info("Time series prediction disabled.")
    # --- [STEP 1-B END] Fetch Sentiment & Prediction --- #


    # --- Get Market Data & Calculate Indicators --- #
    # Calculate appropriate limit based on lookback and interval
    limit = 300 # Default, adjust based on longest indicator period + buffer
    if interval == '1h': limit = 24 * 14 + 50 # ~2 weeks + buffer for hourly
    elif interval == '4h': limit = 6 * 14 + 50 # ~2 weeks + buffer for 4-hourly
    elif interval == '1d': limit = 30 + 50 # ~1 month + buffer for daily
    # Add more interval logic if needed

    df = get_market_data(symbol, interval, limit=limit)
    if df.empty or len(df) < 50: # Need sufficient data
        # Use module logger
        logger.warning(f"Not enough data for {symbol} ({len(df)} rows) to perform analysis.")
        decision_info["reason"] = "Insufficient data"
        return decision_info

    current_price = df['Close'].iloc[-1]
    decision_info["price"] = current_price

    # --- Calculate Technical Indicators --- #
    try:
        # Use settings for periods
        df['SMA_long'] = strategy_utils.calculate_sma(df['Close'], settings.CHALLENGE_SMA_PERIOD)
        df['SMA_short'] = strategy_utils.calculate_sma(df['Close'], 7) # Fixed short SMA
        df['RSI'] = strategy_utils.calculate_rsi(df['Close'], settings.CHALLENGE_RSI_PERIOD)
        df['Volume_MA'] = strategy_utils.calculate_sma(df['Volume'], settings.CHALLENGE_VOLUME_AVG_PERIOD)

        # Store latest indicator values
        decision_info["indicators"] = {
            'SMA_long': df['SMA_long'].iloc[-1],
            'SMA_short': df['SMA_short'].iloc[-1],
            'RSI': df['RSI'].iloc[-1],
            'Volume': df['Volume'].iloc[-1],
            'Volume_MA': df['Volume_MA'].iloc[-1]
        }
    except Exception as e:
        # Use module logger
        logger.error(f"Error calculating indicators for {symbol}: {e}")
        decision_info["reason"] = "Indicator calculation error"
        return decision_info

    # --- Core Strategy Logic ("Flight Challenge" Inspired) --- #
    # Condition Flags
    close_price = df['Close'].iloc[-1]
    prev_close = df['Close'].iloc[-2]
    sma_long = decision_info["indicators"]['SMA_long']
    sma_short = decision_info["indicators"]['SMA_short']
    rsi_value = decision_info["indicators"]['RSI']
    volume = decision_info["indicators"]['Volume']
    volume_ma = decision_info["indicators"]['Volume_MA']

    if pd.isna(sma_long) or pd.isna(sma_short) or pd.isna(rsi_value) or pd.isna(volume_ma):
         # Use module logger
        logger.warning(f"Indicators contain NaN values for {symbol}. Skipping analysis.")
        decision_info["reason"] = "NaN in indicators"
        return decision_info

    # --- Detect Core Conditions --- #
    # 1. Bullish Pullback (BullPull): Price dipped below short SMA but closed above it
    below_sma_short = df['Low'].iloc[-1] < sma_short
    close_above_sma_short = close_price > sma_short
    is_bull_pull = below_sma_short and close_above_sma_short

    # 2. Volume Surge (VolSurge): Current volume > MA * Ratio
    is_vol_surge = volume > volume_ma * settings.CHALLENGE_VOLUME_SURGE_RATIO

    # 3. Bullish Divergence (BullDiv): Price LL, RSI HL (using helper)
    is_bull_div = strategy_utils.detect_rsi_divergence(
        df['Close'], df['RSI'], lookback=settings.CHALLENGE_DIVERGENCE_LOOKBACK, type='bullish'
    )

    # 4. Bearish Pullback (BearPull): Price spiked above short SMA but closed below it
    above_sma_short = df['High'].iloc[-1] > sma_short
    close_below_sma_short = close_price < sma_short
    is_bear_pull = above_sma_short and close_below_sma_short

    # 5. Bearish Divergence (BearDiv): Price HH, RSI LH (using helper)
    is_bear_div = strategy_utils.detect_rsi_divergence(
        df['Close'], df['RSI'], lookback=settings.CHALLENGE_DIVERGENCE_LOOKBACK, type='bearish'
    )

    # --- Combine Conditions for Entry/Exit Signals --- #
    buy_condition = False
    sell_condition = False
    reason = "No conditions met"

    # LONG Entry Conditions
    if is_bull_pull and is_vol_surge:
        buy_condition = True
        reason = "Bullish Pullback + Volume Surge"
    elif is_bull_div:
        buy_condition = True
        reason = "Bullish RSI Divergence"
    # Add more LONG conditions (e.g., breakout + pullback, POC support)

    # SHORT Entry Conditions
    if is_bear_pull and is_vol_surge:
        sell_condition = True
        reason = "Bearish Pullback + Volume Surge"
    elif is_bear_div:
        sell_condition = True
        reason = "Bearish RSI Divergence"
    # Add more SHORT conditions

    # --- [STEP 1-C START] AI/Sentiment Override Logic --- #
    original_decision = "hold"
    if buy_condition: original_decision = "buy"
    elif sell_condition: original_decision = "sell"
    final_decision = original_decision
    override_reason = None

    # Sentiment Override Examples
    if buy_condition and decision_info["sentiment_label"] == "bearish":
        final_decision = "hold"
        override_reason = "Buy signal overridden by Bearish Sentiment"
        # Use module logger
        logger.info(f"{symbol}: {override_reason}")
    if sell_condition and decision_info["sentiment_label"] == "bullish":
        final_decision = "hold"
        override_reason = "Sell signal overridden by Bullish Sentiment"
        # Use module logger
        logger.info(f"{symbol}: {override_reason}")

    # Prediction Trend Override Examples (More nuanced logic might be needed)
    # Example: Strong prediction overrides weaker technical signal
    if final_decision == "hold" and decision_info["prediction_trend"] == "predict_up":
        # Consider entry if prediction is strong, even without technical signal?
        # final_decision = "buy"
        # override_reason = "Hold overridden by Upward Prediction Trend"
        # logger.info(f"{symbol}: {override_reason}")
        pass # Be cautious with this

    # Example: Prediction contradicts technical signal
    if final_decision == "buy" and decision_info["prediction_trend"] == "predict_down":
        final_decision = "hold"
        override_reason = "Buy signal overridden by Downward Prediction Trend"
        # Use module logger
        logger.info(f"{symbol}: {override_reason}")
    if final_decision == "sell" and decision_info["prediction_trend"] == "predict_up":
        final_decision = "hold"
        override_reason = "Sell signal overridden by Upward Prediction Trend"
        # Use module logger
        logger.info(f"{symbol}: {override_reason}")

    # --- [STEP 1-C END] AI/Sentiment Override Logic --- #

    # --- Update Decision Info --- #
    decision_info["decision"] = final_decision
    decision_info["reason"] = reason if override_reason is None else f"{reason} -> {override_reason}"
    decision_info["side"] = final_decision # Match side with final decision
    decision_info["override_reason"] = override_reason

    # --- Log Detailed Condition Info (DEBUG Level) --- #
    # Use module logger
    logger.debug(f"--- Condition Check: {symbol} ---")
    logger.debug(f"  Price: {close_price:.2f}")
    logger.debug(f"  Indicators: RSI={rsi_value:.2f}, Vol={volume:.0f}, VolMA={volume_ma:.0f}, SMA_S={sma_short:.2f}, SMA_L={sma_long:.2f}")
    logger.debug(f"  Conditions: BullPull={is_bull_pull}, VolSurge={is_vol_surge}, BullDiv={is_bull_div}")
    logger.debug(f"  Conditions: BearPull={is_bear_pull}, BearDiv={is_bear_div}")
    logger.debug(f"  Sentiment: {decision_info['sentiment_label']} ({decision_info['sentiment_score']})")
    logger.debug(f"  Prediction Trend: {decision_info['prediction_trend']}")
    logger.debug(f"  Original Decision: {original_decision} (Reason: {reason}) ")
    logger.debug(f"  Final Decision: {decision_info['decision']} (Reason: {decision_info['reason']})")
    logger.debug(f"-----------------------------------")

    return decision_info

def calculate_position_size(entry_price: float, stop_loss_price: float, risk_per_trade: float = settings.CHALLENGE_RISK_PER_TRADE) -> Optional[float]:
    """Calculates position size based on risk percentage of account balance.

    Args:
        entry_price (float): Expected entry price.
        stop_loss_price (float): Price level for stop loss.
        risk_per_trade (float): Percentage of account balance to risk (e.g., 0.01 for 1%).

    Returns:
        Optional[float]: Calculated position quantity, or None if inputs are invalid or client unavailable.
    """
    if not FUTURES_CLIENT or FUTURES_CLIENT_TYPE != "Binance": return None
    if risk_per_trade <= 0 or risk_per_trade >= 1:
        # Use module logger
        logger.error(f"Invalid risk_per_trade value: {risk_per_trade}. Must be between 0 and 1.")
        return None
    if entry_price <= 0 or stop_loss_price <= 0:
         # Use module logger
        logger.error(f"Entry price ({entry_price}) and stop loss price ({stop_loss_price}) must be positive.")
        return None
    if entry_price == stop_loss_price:
         # Use module logger
        logger.error("Entry price and stop loss price cannot be the same.")
        return None

    try:
        # Get current USDT balance (or equivalent margin asset)
        balances = FUTURES_CLIENT.futures_account_balance()
        usdt_balance = 0.0
        for balance in balances:
            if balance['asset'] == 'USDT': # Or BUSD, etc.
                usdt_balance = float(balance['balance'])
                break
        if usdt_balance <= 0:
            # Use module logger
            logger.error("Could not find USDT balance or balance is zero.")
             return None

        # Use module logger
        logger.info(f"Current USDT Balance: {usdt_balance:.2f}")

        # Calculate risk amount in USDT
        risk_amount = usdt_balance * risk_per_trade
        # Use module logger
        logger.info(f"Risk Amount per Trade: {risk_amount:.2f} USDT ({risk_per_trade*100:.2f}% of balance)")

        # Calculate loss per unit (absolute difference)
        loss_per_unit = abs(entry_price - stop_loss_price)
        if loss_per_unit == 0: # Should be caught earlier, but safe check
            # Use module logger
            logger.error("Loss per unit is zero, cannot calculate position size.")
            return None

        # Calculate position size
        position_size = risk_amount / loss_per_unit

        # Adjust for minimum order size and precision (get from exchange info)
        try:
            exchange_info = FUTURES_CLIENT.futures_exchange_info()
            min_qty = 0.001 # Default, replace with actual value
            step_size = 0.001 # Default, replace with actual value
            for s_info in exchange_info['symbols']:
                if s_info['symbol'] == symbol:
                     for f in s_info['filters']:
                         if f['filterType'] == 'LOT_SIZE':
                             min_qty = float(f['minQty'])
                             step_size = float(f['stepSize'])
                             break
                     break
            
            # Use module logger
            logger.debug(f"Exchange Info for {symbol}: minQty={min_qty}, stepSize={step_size}")

            if position_size < min_qty:
                # Use module logger
                logger.warning(f"Calculated position size ({position_size}) is below minimum quantity ({min_qty}). Setting to minimum.")
                position_size = min_qty
            else:
                # Adjust for step size (round down to nearest step)
                precision = int(-math.log10(step_size)) if step_size > 0 else 0
                position_size = math.floor(position_size * (10**precision)) / (10**precision)
                 # Use module logger
                logger.debug(f"Adjusted position size for step size: {position_size}")

        except Exception as einfo_e:
            # Use module logger
            logger.error(f"Could not get or process exchange info for quantity adjustment: {einfo_e}. Using unadjusted size.")

        # Use module logger
        logger.info(f"Calculated Position Size for {symbol}: {position_size:.5f}")
        return position_size

    except BinanceAPIException as bae:
        # Use module logger
        logger.error(f"API error calculating position size: {bae}")
        return None
    except Exception as e:
        # Use module logger
        logger.error(f"Unexpected error calculating position size: {e}")
        return None

def manage_open_position(symbol: str, current_price: float, position_info: Dict[str, Any]):
    """Checks TP/SL conditions for an open position and closes if necessary."""
    if not position_info or not current_price:
        return

        entry_price = float(position_info['entryPrice'])
        position_amt = float(position_info['positionAmt'])
    is_long = position_amt > 0
    is_short = position_amt < 0

    if not is_long and not is_short: # Should not happen if position_info is valid
        # Use module logger
        logger.warning(f"Manage position called for {symbol}, but position amount is zero.")
            return

    # Calculate TP and SL prices based on entry and configured ratios
    tp_price = entry_price * (1 + settings.CHALLENGE_TP_RATIO) if is_long else entry_price * (1 - settings.CHALLENGE_TP_RATIO)
    sl_price = entry_price * (1 - settings.CHALLENGE_SL_RATIO) if is_long else entry_price * (1 + settings.CHALLENGE_SL_RATIO)

    # --- Log current position details --- #
    # Use module logger
    logger.debug(f"Managing Position: {symbol}")
    logger.debug(f"  Side: {'Long' if is_long else 'Short'}, Amount: {position_amt}")
    logger.debug(f"  Entry: {entry_price:.4f}, Current: {current_price:.4f}")
    logger.debug(f"  TP Target: {tp_price:.4f}, SL Target: {sl_price:.4f}")

    close_reason = None
    close_side = None

    # --- Check Take Profit --- #
    if is_long and current_price >= tp_price:
        close_reason = "Take Profit (Long)"
        close_side = 'SELL'
    elif is_short and current_price <= tp_price:
        close_reason = "Take Profit (Short)"
        close_side = 'BUY'

    # --- Check Stop Loss --- #
    if is_long and current_price <= sl_price:
        close_reason = "Stop Loss (Long)"
        close_side = 'SELL'
    elif is_short and current_price >= sl_price:
        close_reason = "Stop Loss (Short)"
        close_side = 'BUY'

    # --- Close Position if Triggered --- #
    if close_reason and close_side:
        quantity_to_close = abs(position_amt)
         # Use module logger
        logger.info(f"Closing position for {symbol} due to {close_reason}. Qty: {quantity_to_close}")
        close_order = create_futures_order(
            symbol=symbol, 
            side=close_side, 
            quantity=quantity_to_close, 
            order_type='MARKET', 
            reduce_only=True # Ensure it only reduces the existing position
        )
        
        if close_order:
            pnl = float(position_info.get('unRealizedProfit', 0.0)) # Use last known PnL
            exit_price = current_price # Approximate exit price
            # Use module logger
            logger.info(f"✅ Position for {symbol} closed successfully. Approx PnL: {pnl:.2f}")
            notifier.send_slack_notification(
                f"Challenge Position Closed: {symbol}",
                f"Reason: {close_reason}\nSide: {'Long' if is_long else 'Short'}\nEntry: {entry_price:.4f}\nExit: {exit_price:.4f}\nQuantity: {quantity_to_close}\nPnL: {pnl:.2f}"
            )
            
            # --- Log Trade to DB --- #
                if log_trade_to_db:
                    try:
                        log_trade_to_db(
                        symbol=symbol, 
                        side='Long' if is_long else 'Short', 
                        entry_price=entry_price, 
                        exit_price=exit_price, 
                        quantity=quantity_to_close, 
                        pnl=pnl, 
                        strategy='Challenge', 
                        reason=close_reason
                    )
                    # Use module logger
                    logger.info(f"Trade for {symbol} logged to database.")
                except Exception as db_e:
                    # Use module logger
                    logger.error(f"Error logging trade to database: {db_e}")
            # --- End Log Trade to DB --- #
            
            else:
            # Use module logger
            logger.error(f"❌ Failed to place closing order for {symbol} ({close_reason}). Position remains open.")
            notifier.send_slack_notification("Challenge Close Order Failed", f"Symbol: {symbol}\nReason: {close_reason}")
            # Consider retry logic or manual intervention alert
    
    # else: logger.debug(f"No TP/SL hit for {symbol}. Holding position.") # Verbose, uncomment if needed

# --- Main Strategy Execution Loop --- #
def run_challenge_strategy():
    """Main loop to check conditions, manage positions, and place orders."""
    # Use module logger
    logger.info("--- Starting Challenge Strategy Cycle ---")

    if not FUTURES_CLIENT:
        # Use module logger
        logger.error("Binance Futures client not initialized. Cannot run challenge strategy.")
        return

    symbols = settings.CHALLENGE_SYMBOLS.split(',')
    interval = settings.CHALLENGE_INTERVAL
    lookback = settings.CHALLENGE_LOOKBACK

    for symbol in symbols:
        # Use module logger
        logger.info(f"Processing symbol: {symbol}...")
        
        try:
            # --- Check Current Position --- #
            current_position = get_current_position(symbol)
            latest_price_df = get_market_data(symbol, interval, limit=2) # Get latest price
            if latest_price_df.empty:
                # Use module logger
                logger.warning(f"Could not get latest price for {symbol}. Skipping management.")
                continue
            current_price = latest_price_df['Close'].iloc[-1]

            if current_position:
                # --- Manage Existing Position --- #
                manage_open_position(symbol, current_price, current_position)
            else:
                # --- Check for New Entry --- #
                # Use module logger
                logger.info(f"No open position for {symbol}. Checking for entry conditions...")
                conditions = check_strategy_conditions(symbol, interval, lookback)
                
                if conditions["decision"] == "buy" or conditions["decision"] == "sell":
                    # Use module logger
                    logger.info(f"Entry condition met for {symbol}: {conditions['reason']} (Side: {conditions['decision']})")
                    
                    entry_price = conditions['price'] # Use current price as entry approx
                    side = conditions['decision'].upper()
                    stop_loss_price = 0.0
                    if side == 'BUY':
                        stop_loss_price = entry_price * (1 - settings.CHALLENGE_SL_RATIO)
                    else: # SELL
                        stop_loss_price = entry_price * (1 + settings.CHALLENGE_SL_RATIO)
                    
                    # --- Calculate Position Size --- #
                            quantity = calculate_position_size(entry_price, stop_loss_price)

            if quantity and quantity > 0:
                        # --- Place Entry Order --- #
                        entry_order = create_futures_order(
                            symbol=symbol, 
                            side=side, 
                            quantity=quantity, 
                            order_type='MARKET'
                        )
                        
                        if entry_order:
                            # Use module logger
                            logger.info(f"✅ Entry order for {symbol} ({side} {quantity}) placed successfully.")
                            notifier.send_slack_notification(
                                f"Challenge Position Opened: {symbol}",
                                f"Reason: {conditions['reason']}\nSide: {side}\nEntry Approx: {entry_price:.4f}\nQuantity: {quantity}\nSL Approx: {stop_loss_price:.4f}"
                            )
                            else:
                            # Use module logger
                            logger.error(f"❌ Failed to place entry order for {symbol}.")
                            # No Slack needed here, create_futures_order sends its own failure alert
                    else:
                        # Use module logger
                        logger.warning(f"Could not calculate valid position size for {symbol}. No entry order placed.")
                        
                else:
                    # Use module logger
                    logger.info(f"No entry conditions met for {symbol}. Reason: {conditions['reason']}")

        except Exception as e:
            # Use module logger
            logger.error(f"Error processing symbol {symbol}: {e}", exc_info=True)
            notifier.send_slack_notification("Challenge Strategy Error", f"Symbol: {symbol}\nError: {e}")

        # --- Add a small delay between symbols --- #
        time.sleep(1) # Avoid hitting API rate limits

    # Use module logger
    logger.info("--- Finished Challenge Strategy Cycle ---")


# --- Main Execution (for direct script run) --- #
if __name__ == '__main__':
    # Import setup_logging here if running directly for testing
    # from src.utils.logging_config import setup_logging 
    # setup_logging()
     # Use module logger
    logger.info("Running challenge_strategy.py directly for testing...")
    run_challenge_strategy()