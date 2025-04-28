import logging
import os
import sys
import pandas as pd
import time
from datetime import datetime
from typing import Optional, Tuple

# --- Project Structure Adjustment ---
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
if PROJECT_ROOT not in sys.path:
     sys.path.append(PROJECT_ROOT)

# --- Import Custom Modules ---
try:
    from src.utils.common import get_historical_data, setup_logging, append_log, save_log_to_csv
    from src.utils.strategy_utils import (
        calculate_rsi,
        calculate_sma,
        detect_rsi_divergence,
        detect_trendline_breakout,
        detect_volume_spike,
        calculate_poc
    )
    # Assuming notifier.py exists and has send_slack_alert AND new alert functions
    from src.utils.notifier import send_slack_notification, send_entry_alert, send_exit_alert, send_error_alert # Import new functions
    from src.config.settings import settings
except ImportError as e:
     print(f"Error importing modules: {e}")
     # Define dummy functions if modules are not found during initial setup
     def get_historical_data(*args, **kwargs): return pd.DataFrame()
     def setup_logging(*args, **kwargs): pass
     def append_log(*args, **kwargs): pass
     def calculate_rsi(df, *args, **kwargs): # Assume it modifies df inplace or returns it
        df['RSI'] = pd.Series(index=df.index) # Dummy RSI column
        return df
     def calculate_sma(df, *args, **kwargs): # Assume it modifies df inplace or returns it
         sma_col = f"SMA_{kwargs.get('window', 7)}"
         df[sma_col] = pd.Series(index=df.index) # Dummy SMA column
         return df
     def detect_rsi_divergence(*args, **kwargs): return 'none' # Return type correction
     def detect_trendline_breakout(*args, **kwargs): return 'none' # Placeholder return type
     def detect_volume_spike(*args, **kwargs): return False # Placeholder return type
     def calculate_poc(*args, **kwargs): return None # Placeholder return type
     # Correct dummy function name to match the actual one if notifier exists
     def send_slack_notification(*args, **kwargs): pass
     def send_entry_alert(*args, **kwargs): pass # Add dummy for new function
     def send_exit_alert(*args, **kwargs): pass # Add dummy for new function
     def send_error_alert(*args, **kwargs): pass # Add dummy for new function


# --- Configuration ---
TICKER = "BTC-USD" # Example ticker for challenge trading (adjust as needed)
INTERVAL = "15m"   # Time interval for data
DATA_PERIOD = "7d" # How much historical data to fetch

SMA_PERIOD = 7
RSI_PERIOD = 14
# Remove local constants for TP/SL as they should come from settings
# TAKE_PROFIT_RATIO = 0.10 # 10% take profit
# STOP_LOSS_RATIO = 0.05   # 5% stop loss

LOG_FILE = os.path.join(PROJECT_ROOT, "logs", "log_challenge.csv")
LOG_COLUMNS = ['timestamp', 'ticker', 'action', 'entry_price', 'exit_price', 'pnl_ratio']

# Setup logger for this module
logger = setup_logging(f'challenge_strategy_{TICKER}')

# --- State Variables ---
# In a real scenario, these would be managed more robustly (e.g., DB, state file)
current_positions = {}

# --- Core Strategy Logic ---
def detect_entry_opportunity(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """주어진 데이터프레임에서 챌린지 전략의 진입 기회(long/short) 및 이유를 감지합니다.

    Args:
        df (pd.DataFrame): OHLCV 데이터.

    Returns:
        Tuple[Optional[str], Optional[str]]: (진입 방향 ('long' 또는 'short'), 진입 이유 문자열) 또는 (None, None)
    """
    if df is None or df.empty or len(df) < max(settings.CHALLENGE_SMA_PERIOD, settings.CHALLENGE_RSI_PERIOD, settings.CHALLENGE_DIVERGENCE_LOOKBACK, settings.CHALLENGE_BREAKOUT_LOOKBACK, 50): # 최소 필요 데이터 길이 (POC 등 고려)
        logger.warning("진입 기회 감지 불가: 데이터 부족")
        return None, None

    # --- 지표 계산 (common.py 함수 사용) --- #
    # 계산 실패 시 None 반환하므로 후속 처리 필요
    rsi_series = calculate_rsi(dataframe=df, window=settings.CHALLENGE_RSI_PERIOD)
    sma_series = calculate_sma(dataframe=df, window=settings.CHALLENGE_SMA_PERIOD)
    # df['poc'] = calculate_poc(df, lookback=settings.CHALLENGE_POC_LOOKBACK)
    poc_price = calculate_poc(dataframe=df, lookback=settings.CHALLENGE_POC_LOOKBACK)

    if rsi_series is None or sma_series is None:
        logger.warning("진입 기회 감지 불가: RSI 또는 SMA 계산 실패")
        return None, None

    # 마지막 데이터 포인트 기준 값들
    last_close = df['Close'].iloc[-1]
    last_rsi = rsi_series.iloc[-1]
    last_sma = sma_series.iloc[-1]
    # poc_price는 마지막 값이 아닐 수 있음 (계산된 단일 값)

    # --- 조건별 신호 감지 --- #
    signals = {}
    reasons = {}

    # 1. 추세선 이탈 후 되돌림
    # detect_trendline_breakout 은 type: pullback_after_breakout_up/down 등을 반환한다고 가정
    trend_info = detect_trendline_breakout(df, window=settings.CHALLENGE_BREAKOUT_LOOKBACK, peak_distance=5)
    if trend_info:
        if trend_info.get('type') == 'pullback_after_breakout_up':
            signals['trend'] = 'long'
            reasons['trend'] = "상단 추세선 돌파 후 되돌림"
        elif trend_info.get('type') == 'pullback_after_breakdown_down':
            signals['trend'] = 'short'
            reasons['trend'] = "하단 추세선 이탈 후 되돌림"

    # 2. RSI 다이버전스
    divergence = detect_rsi_divergence(df['Close'], rsi_series, window=settings.CHALLENGE_DIVERGENCE_LOOKBACK)
    if divergence == 'bullish':
        signals['divergence'] = 'long'
        reasons['divergence'] = f"RSI 상승 다이버전스 ({settings.CHALLENGE_DIVERGENCE_LOOKBACK} 기간)"
    elif divergence == 'bearish':
        signals['divergence'] = 'short'
        reasons['divergence'] = f"RSI 하락 다이버전스 ({settings.CHALLENGE_DIVERGENCE_LOOKBACK} 기간)"

    # 3. 거래량 급증
    volume_spike = detect_volume_spike(df, window=settings.CHALLENGE_VOLUME_AVG_PERIOD, factor=settings.CHALLENGE_VOLUME_SURGE_RATIO)
    if volume_spike:
        signals['volume'] = True # 방향성 없는 신호
        reasons['volume'] = f"최근 거래량 급증 (평균 대비 {settings.CHALLENGE_VOLUME_SURGE_RATIO}배 이상)"

    # 4. 7일 이평선 기준
    if last_close > last_sma:
        signals['sma'] = 'above'
        reasons['sma'] = f"{settings.CHALLENGE_SMA_PERIOD}일 이평선 상회"
    elif last_close < last_sma:
        signals['sma'] = 'below'
        reasons['sma'] = f"{settings.CHALLENGE_SMA_PERIOD}일 이평선 하회"

    # 5. 매물대(POC) 기준
    if poc_price:
        reasons['poc_ref'] = f" (참고 POC: {poc_price:.4f})"
        if last_close > poc_price: # * (1 + threshold) 등 정교화 가능
            signals['poc'] = 'above'
            reasons['poc'] = "POC 상단 위치" + reasons.get('poc_ref','')
        elif last_close < poc_price:
            signals['poc'] = 'below'
            reasons['poc'] = "POC 하단 위치" + reasons.get('poc_ref','')

    # --- 최종 진입 신호 결정 로직 (개선) --- #
    final_signal = None
    final_reason = []

    # Long 진입 조건 조합 예시:
    # (추세선 상승 돌파/되돌림 AND 상승 다이버전스) OR (이평선 상회 AND 거래량 급증 AND POC 상단 지지)
    if (signals.get('trend') == 'long' and signals.get('divergence') == 'bullish') or \
       (signals.get('sma') == 'above' and signals.get('volume') and signals.get('poc') == 'above'):
        final_signal = 'long'
        # 관련된 이유들만 조합
        if signals.get('trend') == 'long': final_reason.append(reasons['trend'])
        if signals.get('divergence') == 'bullish': final_reason.append(reasons['divergence'])
        if signals.get('sma') == 'above': final_reason.append(reasons['sma'])
        if signals.get('volume'): final_reason.append(reasons['volume'])
        if signals.get('poc') == 'above': final_reason.append(reasons['poc'])

    # Short 진입 조건 조합 예시:
    # (추세선 하락 이탈/되돌림 AND 하락 다이버전스) OR (이평선 하회 AND 거래량 급증 AND POC 하단 저항)
    elif (signals.get('trend') == 'short' and signals.get('divergence') == 'bearish') or \
         (signals.get('sma') == 'below' and signals.get('volume') and signals.get('poc') == 'below'):
        final_signal = 'short'
        # 관련된 이유들만 조합
        if signals.get('trend') == 'short': final_reason.append(reasons['trend'])
        if signals.get('divergence') == 'bearish': final_reason.append(reasons['divergence'])
        if signals.get('sma') == 'below': final_reason.append(reasons['sma'])
        if signals.get('volume'): final_reason.append(reasons['volume'])
        if signals.get('poc') == 'below': final_reason.append(reasons['poc'])

    # 최종 결과 반환
    if final_signal:
        reason_str = ", ".join(final_reason)
        logger.info(f"진입 기회 감지: {final_signal.upper()} - 이유: {reason_str}")
        return final_signal, reason_str
    else:
        logger.debug("진입 기회 없음")
        return None, None

def manage_position(symbol: str, position: dict, current_price: float):
    """현재 보유 포지션에 대한 익절/손절 관리.

    Args:
        symbol (str): 심볼명.
        position (dict): 현재 포지션 정보 (예: {'side': 'long', 'entry_price': 100, 'quantity': 1}).
        current_price (float): 현재 가격.

    Returns:
        bool: 포지션을 종료해야 하면 True, 아니면 False.
    """
    if not position:
        return False

    entry_price = position.get('entry_price')
    side = position.get('side')
    quantity = position.get('quantity', 0) # Get quantity as well for alert
    if entry_price is None or side is None:
        logger.warning("포지션 관리 불가: 진입 가격 또는 방향 정보 없음.")
        return False

    # Calculate PnL ratio based on side
    if side == 'long':
        pnl_ratio = (current_price - entry_price) / entry_price
    elif side == 'short':
        # PnL for short is calculated differently
        pnl_ratio = (entry_price - current_price) / entry_price
    else:
        logger.warning(f"알 수 없는 포지션 방향: {side}")
        return False

    tp_reached = False
    sl_reached = False
    log_reason = ""

    # Use TP/SL ratios from settings
    take_profit = settings.CHALLENGE_TP_RATIO
    stop_loss = settings.CHALLENGE_SL_RATIO

    # Check TP/SL conditions (Note: SL ratio is positive, loss condition uses -stop_loss)
    if pnl_ratio >= take_profit:
        tp_reached = True
        log_reason = f"Take Profit reached (+{pnl_ratio:.2%})"
    elif pnl_ratio <= -stop_loss: # Loss condition
        sl_reached = True
        log_reason = f"Stop Loss reached ({pnl_ratio:.2%})"

    if tp_reached or sl_reached:
        logger.info(f"포지션 종료 조건 충족 ({symbol} {side}): {log_reason}")
        # --- 실제 포지션 종료 로직 호출 (API 연동 필요) ---
        # success = close_position(symbol, position['quantity'], side)
        success = True # 임시
        # --- 끝 ---

        if success:
            # 로그 기록 (CSV)
            log_entry = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'symbol': symbol,
                'side': side,
                'entry_price': entry_price,
                'exit_price': current_price,
                'pnl_ratio': pnl_ratio, # Use the calculated pnl_ratio directly
                'reason': log_reason
            }
            save_log_to_csv(log_entry, LOG_FILE)

            # Slack 알림 (개선된 함수 사용)
            send_exit_alert(
                symbol=symbol,
                side=side,
                entry_price=entry_price,
                exit_price=current_price,
                quantity=quantity, # Pass quantity
                pnl_ratio=pnl_ratio,
                reason=log_reason,
                timestamp=datetime.now()
            )
            return True # 포지션 종료됨
        else:
            logger.error(f"포지션 종료 실패 ({symbol})")
            send_error_alert(subject=f"{symbol} 포지션 종료 실패", error_details=f"포지션 종료 API 호출 실패 추정. Side: {side}, Qty: {quantity}")
            return False # 포지션 종료 실패

    return False # 종료 조건 미충족

def run_challenge_strategy():
    """챌린지 전략 실행 (스케줄러에서 주기적으로 호출)."""
    global current_positions
    logger.info("===== 💰 챌린지 전략 실행 시작 =====")

    for symbol in settings.CHALLENGE_SYMBOLS:
        logger.info(f"--- {symbol} 처리 시작 ---")
        try:
            # 1. 최신 데이터 로드
            # 필요한 데이터 기간 및 인터벌 설정 (예: 최근 100개 1시간봉)
            # lookback_period = f"{max(settings.CHALLENGE_DIVERGENCE_LOOKBACK, settings.CHALLENGE_BREAKOUT_LOOKBACK, 100)}h" # 시간 단위 예시
            lookback_period = "5d" # 일단 5일치 데이터 사용
            interval = settings.CHALLENGE_INTERVAL # 예: "1h"
            df = get_historical_data(symbol, period=lookback_period, interval=interval)

            if df is None or df.empty:
                logger.warning(f"{symbol}: 데이터 로드 실패, 건너뜁니다.")
                continue

            current_price = df['Close'].iloc[-1]
            logger.debug(f"{symbol} 현재 가격: {current_price:.4f}")

            # 2. 현재 포지션 관리 (익절/손절 체크)
            position_closed = False
            if symbol in current_positions:
                position_closed = manage_position(symbol, current_positions[symbol], current_price)
                if position_closed:
                    del current_positions[symbol] # 포지션 종료 후 상태 업데이트

            # 3. 신규 진입 기회 탐색 (현재 포지션 없고, 방금 종료되지 않았다면)
            if symbol not in current_positions and not position_closed:
                entry_signal, reason = detect_entry_opportunity(df)

                if entry_signal:
                    logger.info(f"신규 진입 실행 ({symbol} {entry_signal}) - 이유: {reason}")
                    # --- 실제 진입 주문 로직 호출 (API 연동 필요) ---
                    # risk_per_trade = settings.INITIAL_CAPITAL * settings.CHALLENGE_SEED_PERCENTAGE * settings.CHALLENGE_RISK_PER_TRADE
                    # position_size = calculate_position_size(current_price, stop_loss_price, risk_per_trade)
                    # order_result = place_order(symbol, entry_signal, position_size)
                    order_result = {'status': 'filled', 'price': current_price, 'qty': 0.01} # 임시 결과
                    # --- 끝 ---

                    if order_result and order_result.get('status') == 'filled':
                        # 포지션 상태 업데이트
                        current_positions[symbol] = {
                            'side': entry_signal,
                            'entry_price': order_result['price'],
                            'quantity': order_result['qty'],
                            'entry_time': datetime.now()
                        }
                        logger.info(f"진입 성공. 현재 포지션: {current_positions[symbol]}")

                        # 로그 기록 (CSV)
                        log_entry = {
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'symbol': symbol,
                            'side': entry_signal,
                            'entry_price': order_result['price'],
                            'exit_price': None,
                            'pnl_ratio': None,
                            'reason': f"Entry: {reason}"
                        }
                        save_log_to_csv(log_entry, LOG_FILE)

                        # Slack 알림 (개선된 함수 사용)
                        send_entry_alert(
                             symbol=symbol,
                             side=entry_signal,
                             entry_price=order_result['price'],
                             quantity=order_result['qty'],
                             reason=reason,
                             timestamp=datetime.now()
                        )
                        # logger.info(f"진입 성공. 현재 포지션: {current_positions[symbol]}") # Logged by alert func
                    else:
                        logger.error(f"진입 주문 실패 또는 미체결 ({symbol})")
                        send_error_alert(subject=f"{symbol} 진입 주문 실패", error_details=f"주문 실패 또는 미체결. Signal: {entry_signal}, Reason: {reason}", level='warning')

            # API 호출 제한 등 고려하여 잠시 대기
            time.sleep(settings.CHALLENGE_SYMBOL_DELAY_SECONDS)

        except Exception as e:
            logger.error(f"{symbol} 처리 중 오류 발생: {e}", exc_info=True)
            # 오류 발생 시 Slack 알림 (개선된 함수 사용)
            import traceback
            error_details = f"Error: {e}\\n{traceback.format_exc()}"
            send_error_alert(subject=f"챌린지 전략 오류 ({symbol})", error_details=error_details, level='critical')
            continue # 다음 심볼 처리

    logger.info("===== 💰 챌린지 전략 실행 종료 =====")

# 직접 실행 시 테스트
if __name__ == "__main__":
    logger.info("챌린지 전략 모듈 직접 실행 테스트...")
    # 설정 로드 확인
    print(f"Challenge Symbols: {settings.CHALLENGE_SYMBOLS}")
    print(f"Log File Path: {LOG_FILE}")
    # 테스트 실행
    run_challenge_strategy()
    # 현재 포지션 상태 출력 (테스트용)
    print(f"Current positions after test run: {current_positions}") 