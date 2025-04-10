import logging
import pandas as pd
from datetime import datetime, timedelta
import os
import time # 포지션 관리 루프용
import math
from typing import Dict, Any, Optional, Tuple

# 설정 및 유틸리티 모듈 로드
from src.config import settings
from src.utils import common as strategy_utils # Renamed for clarity
from src.utils import notifier

# --- [STEP 1-A START] AI/Analysis 모듈 로드 시도 --- #
try:
    from src.analysis.sentiment import get_market_sentiment
except ImportError:
    get_market_sentiment = None
    logging.warning("Sentiment analysis module (src.analysis.sentiment.py) not found or import error. Sentiment features disabled.")

try:
    from src.ai_models.price_predictor import run_price_prediction_flow
except ImportError:
    run_price_prediction_flow = None
    logging.warning("Price predictor module (src.ai_models.price_predictor.py) not found or import error. Prediction features disabled.")
# --- [STEP 1-A END] AI/Analysis 모듈 로드 시도 --- #

# DB 로깅 함수 임포트
try:
    from src.utils.database import log_trade_to_db
except ImportError:
    log_trade_to_db = None
    logging.warning("Database logging function (log_trade_to_db) not found. DB logging disabled for trades.")

# --- API 클라이언트 초기화 (Binance 예시) ---\
FUTURES_CLIENT_TYPE = None
FUTURES_CLIENT = None
BinanceAPIException = None
BinanceOrderException = None

if settings.BINANCE_API_KEY and settings.BINANCE_SECRET_KEY:
    logging.info("Binance API keys found. Attempting to initialize Futures client...")
    try:
        # from binance.client import Client # Not strictly needed if only using Futures
        from binance.futures import Futures
        from binance.exceptions import BinanceAPIException, BinanceOrderException
        # TODO: Add setting for testnet vs mainnet
        FUTURES_CLIENT = Futures(key=settings.BINANCE_API_KEY, secret=settings.BINANCE_SECRET_KEY)
        FUTURES_CLIENT.ping()
        server_time = FUTURES_CLIENT.time()
        logging.info(f"✅ Binance Futures client initialized successfully. Server time: {datetime.fromtimestamp(server_time['serverTime']/1000)}")
        FUTURES_CLIENT_TYPE = "Binance"
    except ImportError:
        logging.error("❌ Failed to initialize Binance Futures client: 'python-binance' library not installed.")
    except BinanceAPIException as bae:
         logging.error(f"❌ Binance Futures API connection error: {bae.status_code} - {bae.message}")
    except Exception as e:
        logging.error(f"❌ Unexpected error initializing Binance Futures client: {e}")
else:
    logging.warning("Binance Futures API keys not found in settings. Challenge strategy execution will be limited.")


# --- Helper Functions --- #
def get_current_position(symbol: str) -> Optional[Dict[str, Any]]:
    """Retrieves the current open position for the given symbol."""
    if not FUTURES_CLIENT or FUTURES_CLIENT_TYPE != "Binance": return None
    try:
        positions = FUTURES_CLIENT.futures_position_information(symbol=symbol)
        for pos in positions:
            # Check for non-zero position amount
            if float(pos['positionAmt']) != 0:
                logging.info(f"Found open position for {symbol}: Size={pos['positionAmt']}, Entry={pos['entryPrice']}, PNL={pos['unRealizedProfit']}")
                return pos # Return the first non-zero position found
        return None # No open position
    except BinanceAPIException as bae:
        logging.error(f"Error fetching position for {symbol}: {bae.status_code} - {bae.message}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error fetching position for {symbol}: {e}")
        return None

def create_futures_order(symbol: str, side: str, quantity: float, order_type: str = 'MARKET', reduce_only: bool = False) -> Optional[Dict[str, Any]]:
    """Places a futures order (MARKET or LIMIT etc.)."""
    if not FUTURES_CLIENT or FUTURES_CLIENT_TYPE != "Binance": return None
    order_side = side.upper() # BUY or SELL
    order_log_info = f"Symbol={symbol}, Side={order_side}, Qty={quantity}, Type={order_type}{', ReduceOnly' if reduce_only else ''}"
    logging.info(f"Attempting to place order: {order_log_info}...")
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
        logging.info(f"✅ Order placed successfully: {order_result}")
        return order_result
    except BinanceOrderException as boe:
         logging.error(f"❌ Binance order creation failed: {boe.status_code} - {boe.message}. Request: {params}")
         notifier.send_slack_notification("Challenge Order Failed", f"{order_log_info}\nError: {boe.message}")
         return None
    except BinanceAPIException as bae:
         logging.error(f"❌ Binance API error during order placement: {bae.status_code} - {bae.message}")
         notifier.send_slack_notification("Challenge Order API Error", f"{order_log_info}\nError: {bae.message}")
         return None
    except Exception as e:
        logging.error(f"❌ Unexpected error placing order: {e}")
        notifier.send_slack_notification("Challenge Order Unexpected Error", f"{order_log_info}\nError: {e}")
        return None


# --- Market Data and Analysis --- #
def get_market_data(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    """Fetches market klines data."""
    if not FUTURES_CLIENT or FUTURES_CLIENT_TYPE != "Binance": return pd.DataFrame()
    logging.info(f"Fetching {limit} {interval} klines for {symbol}...")
    try:
        klines = FUTURES_CLIENT.futures_klines(symbol=symbol, interval=interval, limit=limit)
        if not klines:
            logging.warning(f"No klines data returned for {symbol}.")
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
        # logging.info(f"Loaded {len(df)} klines for {symbol}.") # Reduce log verbosity
        return df
    except BinanceAPIException as bae:
        logging.error(f"API error fetching klines for {symbol}: {bae}")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Unexpected error fetching klines for {symbol}: {e}")
        return pd.DataFrame()

def analyze_prediction_trend(predictions_df: Optional[pd.DataFrame]) -> str:
    """Analyzes the trend from the HF prediction DataFrame (Placeholder)."""
    if predictions_df is None or predictions_df.empty:
        return "no_prediction"
    try:
        if 'predicted_price' not in predictions_df.columns:
             logging.warning("Prediction DataFrame missing 'predicted_price' column.")
             return "no_prediction"
        # Ensure at least two rows to calculate trend
        if len(predictions_df) < 2:
            logging.warning("Prediction DataFrame has less than 2 rows, cannot determine trend.")
            return "no_prediction"
        first_pred = predictions_df['predicted_price'].iloc[0]
        last_pred = predictions_df['predicted_price'].iloc[-1]
        if pd.isna(first_pred) or pd.isna(last_pred):
            logging.warning("NaN values found in prediction DataFrame.")
            return "no_prediction"
        price_change_ratio = (last_pred - first_pred) / first_pred if first_pred != 0 else 0
        if price_change_ratio > 0.001: # Example threshold for upward trend
            return "predict_up"
        elif price_change_ratio < -0.001: # Example threshold for downward trend
            return "predict_down"
        else:
            return "predict_flat"
    except IndexError: # Should be caught by len check, but just in case
        logging.warning("Index error analyzing prediction trend.")
        return "no_prediction"
    except Exception as e:
        logging.warning(f"Could not analyze prediction trend from DataFrame: {e}")
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
            logging.info(f"Fetching sentiment for keyword: {keyword}...")
            s_label, s_score, _ = get_market_sentiment(keyword=keyword) # Call the imported function
            decision_info["sentiment_label"] = s_label
            decision_info["sentiment_score"] = s_score
            logging.info(f"Sentiment result: {s_label} ({s_score if s_score is not None else 'N/A':.3f})")
        except Exception as e:
            logging.error(f"Error getting market sentiment for {keyword}: {e}")
            decision_info["sentiment_label"] = "error"
            decision_info["sentiment_score"] = None # Ensure score is None on error
    elif settings.ENABLE_HF_SENTIMENT:
        logging.warning("Sentiment analysis enabled but get_market_sentiment function is not available.")
    else:
        logging.info("Sentiment analysis disabled.")

    if settings.ENABLE_HF_TIMESERIES and run_price_prediction_flow:
        try:
            logging.info(f"Running price prediction flow for {symbol}...")
            # Call the imported function
            # The function returns: (primary_model_predictions_df, hf_model_predictions_df)
            _, hf_predictions_df = run_price_prediction_flow(train=False, predict=True, symbol=symbol, interval=interval) # Pass symbol/interval if needed
            decision_info["hf_prediction_df"] = hf_predictions_df # Store the raw DF

            # Analyze the trend from the returned DataFrame
            pred_trend = analyze_prediction_trend(hf_predictions_df) # Assumes this helper exists
            decision_info["prediction_trend"] = pred_trend
            logging.info(f"Price prediction trend analysis result: {pred_trend}")

        except Exception as e:
            logging.error(f"Error running price prediction flow for {symbol}: {e}", exc_info=True)
            decision_info["prediction_trend"] = "error" # Mark as error
            decision_info["hf_prediction_df"] = None # Ensure DF is None on error
    elif settings.ENABLE_HF_TIMESERIES:
        logging.warning("Time series prediction enabled but run_price_prediction_flow function is not available.")
    else:
        logging.info("Time series prediction disabled.")
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
        decision_info["reason"] = "Insufficient market data for analysis"
        logging.warning(f"Insufficient data ({len(df)}) for {symbol} at {interval} interval.")
        # 데이터 부족 시 더 이상 진행하지 않고 반환
        return decision_info

    # --- Calculate Technical Indicators --- #
    # Use strategy_utils which should contain indicator calculations
    try:
        # Ensure calculation functions handle potential errors and missing columns
        df = strategy_utils.calculate_rsi(df, window=settings.CHALLENGE_RSI_PERIOD)
        df = strategy_utils.calculate_sma(df, window=settings.CHALLENGE_SMA_PERIOD)
        df = strategy_utils.calculate_bollinger_bands(df, window=20, num_std_dev=2) # Example
        # Add other indicators as needed from strategy_utils

        # Store latest values
        decision_info['indicators']['rsi'] = df['RSI'].iloc[-1] if 'RSI' in df.columns and not pd.isna(df['RSI'].iloc[-1]) else None
        decision_info['indicators']['sma'] = df[f'SMA_{settings.CHALLENGE_SMA_PERIOD}'].iloc[-1] if f'SMA_{settings.CHALLENGE_SMA_PERIOD}' in df.columns and not pd.isna(df[f'SMA_{settings.CHALLENGE_SMA_PERIOD}'].iloc[-1]) else None
        decision_info['indicators']['close'] = df['Close'].iloc[-1] if 'Close' in df.columns and not pd.isna(df['Close'].iloc[-1]) else None
        decision_info['price'] = decision_info['indicators']['close'] # Current price

        # Log indicator calculation success
        logging.debug(f"[{symbol}] Indicators calculated: RSI={decision_info['indicators']['rsi']}, SMA={decision_info['indicators']['sma']}, Close={decision_info['price']}")

    except Exception as e:
        logging.error(f"[{symbol}] Error calculating technical indicators: {e}", exc_info=True)
        decision_info["reason"] = "Error during indicator calculation"
        # Return early if indicators are crucial and failed
        return decision_info

    # --- Technical Condition Check (Initial Decision) --- #
    # Placeholder: Replace with your actual technical entry/exit logic based on calculated indicators
    latest_rsi = decision_info['indicators'].get('rsi')
    latest_sma = decision_info['indicators'].get('sma')
    latest_close = decision_info['price']

    # Example Buy Condition (RSI low, above SMA)
    if latest_rsi is not None and latest_rsi < settings.CHALLENGE_RSI_THRESHOLD and \
       latest_close is not None and latest_sma is not None and latest_close > latest_sma:
        decision_info["decision"] = "buy"
        decision_info["side"] = "buy"
        decision_info["reason"] = f"Tech Signal: RSI ({latest_rsi:.2f} < {settings.CHALLENGE_RSI_THRESHOLD}) and Close > SMA ({latest_sma:.2f})"
    # Example Sell/Short Condition (RSI high, below SMA - adjust logic as needed)
    elif latest_rsi is not None and latest_rsi > (100 - settings.CHALLENGE_RSI_THRESHOLD) and \
         latest_close is not None and latest_sma is not None and latest_close < latest_sma:
        decision_info["decision"] = "sell"
        decision_info["side"] = "sell"
        decision_info["reason"] = f"Tech Signal: RSI ({latest_rsi:.2f} > {100 - settings.CHALLENGE_RSI_THRESHOLD}) and Close < SMA ({latest_sma:.2f})"
    else:
        decision_info["decision"] = "hold"
        decision_info["reason"] = "No strong technical signal met"


    # --- [STEP 1-C START] Apply Sentiment & Prediction Overrides --- #
    initial_decision = decision_info["decision"]
    initial_reason = decision_info["reason"]
    override_applied = False

    # 1. Sentiment Override Check
    if settings.ENABLE_HF_SENTIMENT and decision_info["sentiment_label"] not in ["N/A", "error"]:
        sentiment_label = decision_info["sentiment_label"]
        sentiment_score = decision_info["sentiment_score"]

        is_negative_sentiment = sentiment_label == 'negative'
        is_low_score = sentiment_score is not None and sentiment_score < settings.SENTIMENT_CUTOFF_SCORE

        if initial_decision == "buy" and (is_negative_sentiment or is_low_score):
            override_msg_part = f"Negative sentiment ('{sentiment_label}'"
            if sentiment_score is not None:
                 override_msg_part += f", score: {sentiment_score:.3f} < {settings.SENTIMENT_CUTOFF_SCORE})"
            else:
                 override_msg_part += ")"
            override_msg = f"Buy signal overridden by {override_msg_part}"
            decision_info["decision"] = "hold" # 매수 취소
            decision_info["side"] = None # side도 초기화
            decision_info["override_reason"] = override_msg
            logging.info(f"[{symbol}] {override_msg}")
            override_applied = True

        # Optional: Add logic for positive sentiment potentially overriding a sell signal?
        # elif initial_decision == "sell" and sentiment_label == 'positive' and (sentiment_score is None or sentiment_score > (1 - settings.SENTIMENT_CUTOFF_SCORE)):
        #     override_msg = f"Sell signal overridden by Positive sentiment ('{sentiment_label}', score: {sentiment_score:.3f})"
        #     decision_info["decision"] = "hold"
        #     decision_info["side"] = None
        #     decision_info["override_reason"] = override_msg
        #     logging.info(f"[{symbol}] {override_msg}")
        #     override_applied = True


    # 2. Prediction Trend Override Check (Run even if sentiment already caused hold, might add reason)
    if settings.ENABLE_HF_TIMESERIES and decision_info["prediction_trend"] not in ["N/A", "error", "no_prediction"]:
        prediction_trend = decision_info["prediction_trend"]
        # Use initial_decision here to check against the original tech signal before sentiment override
        current_decision = decision_info["decision"] # Decision after potential sentiment override

        override_msg = None
        # 예측 하락 시 매수 보류/취소 (Tech Buy + Predict Down)
        if initial_decision == "buy" and prediction_trend == "predict_down":
            override_msg = f"Price prediction 'predict_down' conflicts with Buy signal."
            if current_decision != "hold": # Only log/set hold if not already held by sentiment
                decision_info["decision"] = "hold"
                decision_info["side"] = None
                override_applied = True

        # 예측 상승 시 매도 보류/취소 (Tech Sell + Predict Up)
        elif initial_decision == "sell" and prediction_trend == "predict_up":
            override_msg = f"Price prediction 'predict_up' conflicts with Sell signal."
            if current_decision != "hold": # Only log/set hold if not already held
                decision_info["decision"] = "hold"
                decision_info["side"] = None
                override_applied = True

        if override_msg:
             # Append reason if already overridden by sentiment, otherwise set it
            if decision_info["override_reason"]:
                decision_info["override_reason"] += f"; {override_msg}"
            else:
                decision_info["override_reason"] = override_msg
            logging.info(f"[{symbol}] {override_msg} -> Decision: {decision_info['decision'].upper()}")


    # 3. Final Reason Update
    if override_applied:
        # If overridden, combine initial reason with the override reason
        decision_info["reason"] = f"{initial_reason}. Overridden: {decision_info['override_reason']}"
    elif decision_info["decision"] == "hold":
         # If hold decision remained (no signal or signal cancelled), use a generic hold reason or the override reason if it exists
         decision_info["reason"] = decision_info.get("override_reason", "Hold - No strong signal or condition not met")
    # If decision is buy/sell and not overridden, the initial_reason is already set correctly


    # --- [STEP 1-C END] Apply Sentiment & Prediction Overrides --- #

    # Final decision logging
    log_msg = (
        f"[{symbol}] Strategy Check Completed:\n"
        f"  Decision: {decision_info['decision'].upper()} ({decision_info['side'] if decision_info['side'] else 'N/A'})\n"
        f"  Reason: {decision_info['reason']}\n"
        f"  Sentiment: {decision_info['sentiment_label']} ({decision_info['sentiment_score']:.3f if decision_info['sentiment_score'] is not None else 'N/A'}) (Enabled: {settings.ENABLE_HF_SENTIMENT})\n"
        f"  Prediction Trend: {decision_info['prediction_trend']} (Enabled: {settings.ENABLE_HF_TIMESERIES})\n"
        f"  Current Price: {decision_info.get('price', 'N/A')}\n"
        # f"  Indicators: {decision_info.get('indicators', {})}" # Uncomment for detailed indicator logging
    )
    logging.info(log_msg)

    # Send Slack notification only for actionable decisions or explicit overrides
    if decision_info["decision"] != "hold" or override_applied:
        slack_title = f"Challenge Strategy Alert ({symbol}) - {decision_info['decision'].upper()}"
        # Use a slightly more concise message for Slack?
        slack_message = (
            f"Decision: *{decision_info['decision'].upper()}* ({decision_info['side'] if decision_info['side'] else 'N/A'}) for {symbol}\n"
            f"Reason: {decision_info['reason']}\n"
            f"Price: {decision_info.get('price', 'N/A')}\n"
            f"Sentiment: {decision_info['sentiment_label']} ({decision_info['sentiment_score']:.3f if decision_info['sentiment_score'] is not None else 'N/A'})\n"
            f"Prediction: {decision_info['prediction_trend']}"
        )
        notifier.send_slack_notification(slack_title, slack_message)

    return decision_info

def calculate_position_size(entry_price: float, stop_loss_price: float, risk_per_trade: float = settings.CHALLENGE_RISK_PER_TRADE) -> Optional[float]:
    """Calculates position size based on entry, stop loss, and risk percentage."""
    if not entry_price or not stop_loss_price or entry_price == stop_loss_price:
        logging.warning("Invalid prices for position size calculation.")
        return None
    if risk_per_trade <= 0 or risk_per_trade >= 1:
        logging.warning(f"Invalid risk_per_trade value: {risk_per_trade}. Must be between 0 and 1.")
        return None

    # Get account balance (requires API call or cached value)
    # Placeholder - replace with actual balance fetching
    try:
        # Example using Binance client (adjust based on actual client)
        # balance_info = FUTURES_CLIENT.futures_account_balance()
        # usdt_balance = next((item['balance'] for item in balance_info if item['asset'] == 'USDT'), None)
        usdt_balance = 1000.0 # FIXME: Replace with actual balance query
        if usdt_balance is None:
             logging.error("Could not retrieve USDT balance for position sizing.")
             return None
        account_balance = float(usdt_balance)
    except Exception as e:
        logging.error(f"Error fetching account balance for sizing: {e}")
        return None


    risk_amount = account_balance * risk_per_trade
    risk_per_contract = abs(entry_price - stop_loss_price)

    if risk_per_contract == 0:
        logging.warning("Risk per contract is zero (entry = stop loss). Cannot calculate size.")
        return None

    quantity = risk_amount / risk_per_contract

    # TODO: Add leverage consideration if applicable
    # TODO: Add minimum order size and precision handling based on exchange info

    # Example: Round to 3 decimal places for BTCUSDT
    precision = 3 # Get precision from exchange info for the symbol
    quantity = math.floor(quantity * (10**precision)) / (10**precision)

    logging.info(f"Calculated position size: Account={account_balance:.2f}, RiskAmt={risk_amount:.2f}, RiskPerContract={risk_per_contract:.5f}, Qty={quantity}")

    if quantity <= 0: # Ensure quantity is positive and non-zero
        logging.warning(f"Calculated quantity is zero or negative ({quantity}). Cannot place order.")
        return None

    return quantity

def manage_open_position(symbol: str, current_price: float, position_info: Dict[str, Any]):
    """Manages an open position, checking for Stop Loss and Take Profit."""
    if not position_info or not current_price:
        return

    try:
        entry_price = float(position_info['entryPrice'])
        position_amt = float(position_info['positionAmt'])
        # unrealized_pnl = float(position_info['unRealizedProfit']) # PNL can be used for trailing stops etc.

        if position_amt == 0: # Should not happen if called correctly, but double check
            return

        side = "buy" if position_amt > 0 else "sell"
        quantity_to_close = abs(position_amt)

        # --- Define SL and TP Prices ---
        sl_ratio = settings.CHALLENGE_SL_RATIO # e.g., 0.05 for 5%
        tp_ratio = settings.CHALLENGE_TP_RATIO # e.g., 0.10 for 10%

        stop_loss_price = None
        take_profit_price = None

        if side == "buy":
            stop_loss_price = entry_price * (1 - sl_ratio)
            take_profit_price = entry_price * (1 + tp_ratio)
        elif side == "sell":
            stop_loss_price = entry_price * (1 + sl_ratio)
            take_profit_price = entry_price * (1 - tp_ratio)

        # --- Check Exit Conditions ---
        exit_reason = None
        exit_side = None

        if side == "buy":
            if current_price <= stop_loss_price:
                exit_reason = f"Stop Loss triggered at {stop_loss_price:.5f} (Current: {current_price:.5f})"
                exit_side = "sell"
            elif current_price >= take_profit_price:
                exit_reason = f"Take Profit triggered at {take_profit_price:.5f} (Current: {current_price:.5f})"
                exit_side = "sell"
        elif side == "sell":
            if current_price >= stop_loss_price:
                exit_reason = f"Stop Loss triggered at {stop_loss_price:.5f} (Current: {current_price:.5f})"
                exit_side = "buy"
            elif current_price <= take_profit_price:
                exit_reason = f"Take Profit triggered at {take_profit_price:.5f} (Current: {current_price:.5f})"
                exit_side = "buy"

        # --- Execute Exit Order ---
        if exit_reason and exit_side:
            logging.info(f"[{symbol}] Attempting to close position ({side}). Reason: {exit_reason}")
            # Use reduce_only=True to ensure it only closes the position
            order_result = create_futures_order(symbol, exit_side, quantity_to_close, order_type='MARKET', reduce_only=True)

            if order_result:
                logging.info(f"[{symbol}] Position closed successfully via {exit_side} order.")
                # Log exit trade to DB
                if log_trade_to_db:
                    # Calculate PNL approximately (actual PNL comes from execution report or trade history)
                    pnl_estimate = (current_price - entry_price) * position_amt if side == "buy" else (entry_price - current_price) * abs(position_amt)
                    try:
                        log_trade_to_db(
                            timestamp=datetime.now(), symbol=symbol, order_type="exit", side=exit_side,
                            price=order_result.get('avgPrice', current_price), quantity=order_result.get('executedQty', quantity_to_close),
                            strategy="challenge", status=order_result.get('status'), pnl=pnl_estimate, # Use estimated PNL
                            notes=exit_reason # Add exit reason to notes
                        )
                    except Exception as db_err:
                        logging.error(f"Failed to log exit trade to DB: {db_err}")
                # Send Slack notification
                notifier.send_slack_notification(f"Challenge Position Closed ({symbol})", f"Closed {side} position for {symbol}.\nReason: {exit_reason}\nEst. PNL: {pnl_estimate:.2f}") # Add PNL estimate
            else:
                 logging.error(f"[{symbol}] Failed to place closing order ({exit_side}). Position might still be open.")
                 notifier.send_slack_notification(f"Challenge Position Close FAILED ({symbol})", f"Failed to close {side} position for {symbol} via {exit_side} order.\nReason: {exit_reason}")

    except KeyError as ke:
         logging.error(f"Missing key in position_info: {ke}. Cannot manage position.")
    except Exception as e:
        logging.error(f"Unexpected error managing position for {symbol}: {e}", exc_info=True)


# --- Main Strategy Execution Logic (STEP 1-D UPDATE) --- #
def run_challenge_strategy():
    """Main execution function for the challenge strategy."""
    logging.info(f"===== Starting Challenge Strategy Cycle =====")

    if not FUTURES_CLIENT or FUTURES_CLIENT_TYPE != "Binance":
        logging.warning("Binance Futures client not available. Skipping strategy execution.")
        notifier.send_slack_notification("Challenge Strategy Error", "Binance Futures client not available. Strategy skipped.")
        return

    # Get list of symbols from settings
    symbols_to_trade = settings.CHALLENGE_SYMBOLS
    if not symbols_to_trade:
        logging.warning("No symbols defined in settings.CHALLENGE_SYMBOLS. Skipping strategy execution.")
        return

    for symbol in symbols_to_trade:
        logging.info(f"--- Processing symbol: {symbol} ---")
        try: # Add try-except block per symbol to prevent one failure from stopping others
            # 1. Check for Existing Position
            current_position = get_current_position(symbol)
            # Get latest price using a small interval (e.g., 1m)
            current_price_df = get_market_data(symbol, '1m', 2) # Get last 2 points to ensure we have the latest close
            current_price = current_price_df['Close'].iloc[-1] if not current_price_df.empty and len(current_price_df) > 0 else None

            if current_price is None:
                logging.warning(f"[{symbol}] Could not fetch current price. Skipping management and entry checks for this symbol.")
                continue # Skip to next symbol if price is unavailable

            # 2. Manage Existing Position (if any)
            if current_position:
                logging.info(f"[{symbol}] Open position found. Managing SL/TP...")
                manage_open_position(symbol, current_price, current_position)
                # Decide if we should check for new entry signals even if position exists
                # If strategy allows multiple positions or adding to position, remove 'continue'
                # For now, if position exists, we only manage it.
                logging.info(f"[{symbol}] Position management complete. Moving to next symbol.")
                continue # Move to the next symbol

            # 3. Check for New Entry Opportunities (if no position exists)
            if not current_position:
                logging.info(f"[{symbol}] No open position. Checking for entry signals...")
                # Use appropriate interval and lookback from settings
                interval = settings.CHALLENGE_INTERVAL # e.g., '1h'
                lookback = settings.CHALLENGE_LOOKBACK # e.g., '14d' # lookback might be implicit in indicator periods

                # Call the updated conditions check function
                # [STEP 1-D: Use the enhanced check_strategy_conditions]
                decision_info = check_strategy_conditions(symbol, interval, lookback) # Pass lookback if needed by check func

                # 4. Execute Action Based on Decision
                if decision_info["decision"] in ["buy", "sell"]:
                    entry_price = decision_info.get("price") # Price at the time of signal generation
                    if not entry_price:
                        logging.warning(f"[{symbol}] {decision_info['decision'].upper()} signal, but could not determine signal price. Fetching current price again.")
                        # Fetch current price again just before order
                        current_price_df = get_market_data(symbol, '1m', 2)
                        entry_price = current_price_df['Close'].iloc[-1] if not current_price_df.empty else None

                    if entry_price:
                        # --- Calculate Stop Loss ---
                        sl_ratio = settings.CHALLENGE_SL_RATIO
                        stop_loss_price = None
                        if decision_info["decision"] == "buy":
                            stop_loss_price = entry_price * (1 - sl_ratio)
                        elif decision_info["decision"] == "sell":
                            stop_loss_price = entry_price * (1 + sl_ratio)

                        # --- Calculate Position Size ---
                        quantity = None
                        if stop_loss_price:
                            quantity = calculate_position_size(entry_price, stop_loss_price)
                        else:
                             logging.warning(f"[{symbol}] Cannot calculate position size without a valid stop loss price.")

                        if quantity and quantity > 0:
                            order_side = decision_info["decision"].upper()
                            logging.info(f"[{symbol}] Placing {order_side} order. Qty: {quantity}, Entry: ~{entry_price:.5f}, SL: {stop_loss_price:.5f}. Reason: {decision_info['reason']}")
                            order_result = create_futures_order(symbol, order_side, quantity)

                            if order_result:
                                # Log trade to DB if function is available
                                if log_trade_to_db:
                                    try:
                                        # Use actual executed price and quantity if available
                                        filled_price = float(order_result.get('avgPrice', entry_price))
                                        filled_qty = float(order_result.get('executedQty', quantity))
                                        log_trade_to_db(
                                            timestamp=datetime.now(),
                                            symbol=symbol,
                                            order_type="entry",
                                            side=decision_info["decision"],
                                            price=filled_price,
                                            quantity=filled_qty,
                                            strategy="challenge",
                                            status=order_result.get('status'),
                                            pnl=0, # Initial PNL is 0
                                            sentiment_label=decision_info.get('sentiment_label'),
                                            sentiment_score=decision_info.get('sentiment_score'),
                                            prediction_trend=decision_info.get('prediction_trend'),
                                            stop_loss=stop_loss_price, # Log intended SL
                                            # take_profit=take_profit_price, # Log intended TP if calculated here
                                            notes=f"Order ID: {order_result.get('orderId')}"
                                        )
                                    except Exception as db_err:
                                        logging.error(f"Failed to log trade to DB: {db_err}")
                                # Confirmation notification sent from check_strategy_conditions
                            else:
                                logging.error(f"[{symbol}] {order_side} order placement failed.")
                                # Failure notification sent from create_futures_order

                        elif quantity is None:
                            logging.warning(f"[{symbol}] {decision_info['decision'].upper()} signal, but position size calculation failed.")
                        else: # quantity == 0
                             logging.warning(f"[{symbol}] {decision_info['decision'].upper()} signal, but calculated quantity is zero ({quantity}). No order placed.")
                    else:
                         logging.warning(f"[{symbol}] {decision_info['decision'].upper()} signal, but could not determine entry price for order.")

                elif decision_info["decision"] == "hold":
                    logging.info(f"[{symbol}] Decision is HOLD. Reason: {decision_info['reason']}")
                    # Log overridden holds if desired
                    if decision_info["override_reason"] and log_trade_to_db and settings.LOG_OVERRIDES_TO_DB: # Add setting flag
                        try:
                             log_trade_to_db(
                                 timestamp=datetime.now(), symbol=symbol, order_type="hold_override",
                                 side="none", price=decision_info.get('price'), quantity=0, strategy="challenge",
                                 status="overridden", pnl=0, sentiment_label=decision_info.get('sentiment_label'),
                                 sentiment_score=decision_info.get('sentiment_score'), prediction_trend=decision_info.get('prediction_trend'),
                                 notes=decision_info["override_reason"]
                             )
                        except Exception as db_err:
                             logging.error(f"Failed to log overridden hold to DB: {db_err}")

        except BinanceAPIException as bae:
             logging.error(f"[{symbol}] Binance API Error processing symbol: {bae}", exc_info=True)
             notifier.send_slack_notification(f"Challenge Strategy API Error ({symbol})", f"Error: {bae.status_code} - {bae.message}")
        except BinanceOrderException as boe:
             logging.error(f"[{symbol}] Binance Order Error processing symbol: {boe}", exc_info=True)
             # Notification likely already sent by create_futures_order or manage_open_position
        except Exception as e:
            logging.error(f"[{symbol}] Unexpected error processing symbol: {e}", exc_info=True)
            notifier.send_slack_notification(f"Challenge Strategy Unexpected Error ({symbol})", f"Error: {e}")

        # Add a small delay between symbols to avoid rate limits
        time.sleep(settings.CHALLENGE_SYMBOL_DELAY_SECONDS) # e.g., 1 second

    logging.info(f"===== Finished Challenge Strategy Cycle =====")

# --- Main Execution Guard --- #
if __name__ == "__main__":
    # Setup logging (consider moving to a central logging config)
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

    logging.info("Running challenge strategy directly.")
    # Load environment variables using dotenv if not already handled
    # from dotenv import load_dotenv
    # load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '..', '.env')) # Example path adjustment

    # Run the strategy once for testing
    run_challenge_strategy()

    # --- Scheduling Example (Optional) ---
    # import schedule
    # import time
    #
    # def job():
    #     logging.info("Running scheduled challenge strategy job...")
    #     run_challenge_strategy()
    #     logging.info("Scheduled challenge strategy job finished.")
    #
    # # schedule.every().hour.at(":05").do(job) # Example: Every hour at 5 minutes past
    # # schedule.every(15).minutes.do(job) # Example: Every 15 minutes
    #
    # # logging.info("Scheduler started. Waiting for next run...")
    # # while True:
    # #     schedule.run_pending()
    # #     time.sleep(30) # Check every 30 seconds