import logging
import pandas as pd
from datetime import datetime, timedelta
import os
import time # 포지션 관리 루프용
import math

# config, strategy_utils, notifier 모듈 로드 (새로운 경로)
try:
    # import config -> from src.config import settings as config
    from src.config import settings as config
    # import strategy_utils -> from src.utils import common as strategy_utils
    from src.utils import common as strategy_utils
    # import notifier -> from src.utils import notifier
    from src.utils import notifier
except ImportError as e:
    logging.error(f"필수 모듈(src.config.settings, src.utils.common, src.utils.notifier) 로드 실패: {e}")
    raise

# --- MEXC/Binance API 클라이언트 초기화 (placeholder) ---
# TODO: 실제 API 클라이언트 라이브러리(python-binance, mexc-api)를 사용하여 구현 필요
FUTURES_CLIENT_TYPE = None
FUTURES_CLIENT = None # 실제 클라이언트 객체를 저장할 변수

if config.MEXC_API_KEY and config.MEXC_SECRET_KEY:
    logging.info("MEXC API 키 확인됨. 클라이언트 초기화 시도...")
    try:
        # from mexc_api.spot import Spot # 선물 API는 별도 확인 필요
        # FUTURES_CLIENT = Spot(api_key=config.MEXC_API_KEY, api_secret=config.MEXC_SECRET_KEY)
        logging.warning("MEXC 선물 API 클라이언트 초기화 로직 필요.") # 실제 초기화 코드로 대체
        FUTURES_CLIENT_TYPE = "MEXC"
    except Exception as e:
        logging.error(f"MEXC 클라이언트 초기화 실패: {e}")
elif config.BINANCE_API_KEY and config.BINANCE_SECRET_KEY:
    logging.info("Binance API 키 확인됨. 클라이언트 초기화 시도...")
    try:
        from binance.client import Client
        from binance.futures import Futures
        from binance.exceptions import BinanceAPIException, BinanceOrderException
        # TODO: 실계좌/테스트넷 설정 확인 필요 (config.py 또는 .env 에 추가 고려)
        FUTURES_CLIENT = Futures(key=config.BINANCE_API_KEY, secret=config.BINANCE_SECRET_KEY)
        # 선물 계정 정보 확인으로 연결 테스트
        FUTURES_CLIENT.ping()
        server_time = FUTURES_CLIENT.time()
        logging.info(f"✅ Binance Futures 클라이언트 초기화 및 연결 성공. 서버 시간: {datetime.fromtimestamp(server_time['serverTime']/1000)}")
        FUTURES_CLIENT_TYPE = "Binance"
    except ImportError:
        logging.error("❌ Binance Futures 클라이언트 초기화 실패: 'python-binance' 라이브러리가 설치되지 않았습니다.")
        FUTURES_CLIENT = None
    except BinanceAPIException as bae:
         logging.error(f"❌ Binance Futures API 연결 오류: {bae.status_code} - {bae.message}")
         FUTURES_CLIENT = None
    except Exception as e:
        logging.error(f"❌ Binance Futures 클라이언트 초기화 중 예상치 못한 오류: {e}")
        FUTURES_CLIENT = None
else:
    logging.warning("MEXC 또는 Binance Futures API 키가 .env 파일에 설정되지 않았습니다. 챌린지 전략 실행 불가.")


# --- 주요 기능 함수 (기본 틀) ---

def get_futures_account_balance():
    """선물 계정 잔고 정보를 조회합니다."""
    if not FUTURES_CLIENT: return None
    logging.info(f"{FUTURES_CLIENT_TYPE}: 선물 계정 잔고 조회 시도...")
    try:
        if FUTURES_CLIENT_TYPE == "Binance":
            balance_info = FUTURES_CLIENT.futures_account_balance()
            # 필요한 정보만 추출 (예: USDT 잔고)
            usdt_balance = next((item for item in balance_info if item['asset'] == 'USDT'), None)
            if usdt_balance:
                logging.info(f"Binance 선물 USDT 잔고: {usdt_balance}")
                # totalWalletBalance 와 유사한 개념을 찾아 반환 (예: balance)
                return {"totalWalletBalance": usdt_balance.get('balance'),
                        "availableBalance": usdt_balance.get('availableBalance')}
            else:
                 logging.warning("Binance 선물 계정에서 USDT 잔고를 찾을 수 없습니다.")
                 return None
        elif FUTURES_CLIENT_TYPE == "MEXC":
            logging.warning("MEXC 선물 잔고 조회 로직 구현 필요.")
            # 예시: return FUTURES_CLIENT.futures_account()
            return {"totalWalletBalance": "1000.0", "availableBalance": "500.0"} # 임시 데이터
        else:
            return None
    except Exception as e:
        logging.error(f"{FUTURES_CLIENT_TYPE} 선물 잔고 조회 중 오류: {e}")
        return None

def get_market_data(symbol: str, interval: str = '1h', limit: int = 200) -> pd.DataFrame:
    """지정된 심볼의 시장 데이터(캔들스틱)를 조회합니다."""
    if not FUTURES_CLIENT: return pd.DataFrame()
    logging.info(f"{FUTURES_CLIENT_TYPE}: {symbol} {interval} 데이터 조회 시도 (최근 {limit}개)...")
    try:
        if FUTURES_CLIENT_TYPE == "Binance":
            # Binance API는 klines에 limit 파라미터가 조금 다르게 동작할 수 있음, 필요시 추가 조회 로직 구현
            klines = FUTURES_CLIENT.futures_klines(symbol=symbol, interval=interval, limit=limit)
            if not klines:
                logging.warning(f"Binance에서 {symbol} 데이터를 가져오지 못했습니다.")
                return pd.DataFrame()

            # DataFrame 형식으로 변환 (컬럼명 지정)
            df = pd.DataFrame(klines, columns=[
                'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
                'Close time', 'Quote asset volume', 'Number of trades',
                'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
            ])
            # 필요한 컬럼만 선택 및 타입 변환
            df = df[['Open time', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
            df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = pd.to_numeric(df[col])
            df.set_index('Open time', inplace=True)
            logging.info(f"Binance {symbol} 데이터 로드 완료 ({len(df)}개 캔들)")
            return df

        elif FUTURES_CLIENT_TYPE == "MEXC":
            logging.warning("MEXC 시장 데이터 조회 로직 구현 필요.")
            # 예시: klines = FUTURES_CLIENT.klines(symbol, interval, limit=limit)
            # 이후 Binance와 유사하게 DataFrame 변환
            return pd.DataFrame() # 임시 빈 데이터프레임
        else:
            return pd.DataFrame()

    except Exception as e:
        logging.error(f"{FUTURES_CLIENT_TYPE} {symbol} 시장 데이터 조회 중 오류: {e}")
        return pd.DataFrame()

def check_entry_conditions(symbol: str) -> dict | None:
    """챌린지 전략의 진입 조건을 확인합니다.

    플라이트 챌린지 매매법 기반:
    1. 추세선 이탈 후 되돌림?
    2. RSI 다이버전스?
    3. 거래량 급증?
    4. 7일 이평선 하방 이탈?
    5. VPVR 매물대 지지/저항?
    """
    if not FUTURES_CLIENT: return None
    logging.info(f"--- {symbol}: 챌린지 전략 진입 조건 확인 시작 ---")

    # 1. 데이터 가져오기 (예: 1시간 봉, 충분한 기간 - 30일)
    df = get_market_data(symbol, interval='1h', limit=24*30) # 약 30일치 1시간 봉 데이터
    if df is None or df.empty or len(df) < max(config.CHALLENGE_SMA_PERIOD, 50): # 최소 분석 기간 확인
        logging.warning(f"{symbol}: 시장 데이터를 가져오거나 데이터 양이 부족하여 조건 확인 불가.")
        return None

    # 2. 지표 계산 (strategy_utils 사용)
    try:
        # RSI 계산
        rsi_series = strategy_utils.calculate_rsi(dataframe=df, window=14, column='Close')
        df['rsi'] = rsi_series # DataFrame에 추가하여 사용

        # SMA 계산 (7일 이평선 -> 1시간봉 기준 7*24=168개)
        sma_series = strategy_utils.calculate_sma(dataframe=df, window=config.CHALLENGE_SMA_PERIOD * 24, column='Close')
        df['sma'] = sma_series

        # RSI 다이버전스 감지 (예: 최근 7일)
        rsi_divergence = None
        if rsi_series is not None:
             rsi_divergence = strategy_utils.detect_rsi_divergence(df['Close'], rsi_series, window=24*7)

        # 추세선 이탈/되돌림 감지 (Placeholder)
        trend_info = strategy_utils.detect_trendline_breakout(df, window=24*14) # 예: 2주 추세선

        # 거래량 급증 감지 (예: 최근 1일 평균 대비 2배)
        volume_spike = strategy_utils.detect_volume_spike(df, window=24, factor=2.0)

        # 거래량 기반 지지/저항 구간 (VPVR 근사, 예: 최근 2주)
        sr_levels = strategy_utils.detect_support_resistance_by_volume(df, window=24*14, n_levels=5)

        # --- 조건 조합 로직 (TODO: 상세 구현 필요) ---
        entry_signal = None
        current_price = df['Close'].iloc[-1]
        current_sma = df['sma'].iloc[-1] if sma_series is not None and not sma_series.empty else None

        logging.info(f"{symbol} 지표: 가격={current_price:.2f}, SMA({config.CHALLENGE_SMA_PERIOD*24}h)={current_sma:.2f if current_sma else 'N/A'}, RSI={df['rsi'].iloc[-1]:.2f if rsi_series is not None else 'N/A'}")
        logging.info(f"{symbol} 추가 정보: RSI Div={rsi_divergence}, Trend={trend_info}, Vol Spike={volume_spike}, S/R Levels={sr_levels}")

        # 예시: 7일 이평선 하방 이탈 + 거래량 급증 시 숏 진입 고려?
        if current_sma is not None and current_price < current_sma and df['Close'].iloc[-2] >= df['sma'].iloc[-2] and volume_spike:
             reason = f"{config.CHALLENGE_SMA_PERIOD*24}h SMA 하방 이탈 + 거래량 급증"
             logging.info(f"{symbol} 진입 조건 만족 (예시): {reason}")
             entry_signal = {'symbol': symbol, 'side': 'sell', 'reason': reason}

        # TODO: 추세선 되돌림, RSI 다이버전스, 매물대 조건 등을 조합하여
        #       LONG / SHORT 진입 신호 ('buy' / 'sell') 생성 로직 구현
        # 예시:
        # if trend_info and trend_info['type'] == 'retest_support' and rsi_divergence == 'bullish':
        #     entry_signal = {'symbol': symbol, 'side': 'buy', 'reason': '추세선 지지 리테스트 + 상승 다이버전스'}
        # elif ...

    except Exception as e:
        logging.error(f"{symbol} 진입 조건 확인 중 오류 발생: {e}", exc_info=True)
        return None

    if entry_signal:
        logging.info(f"✅ {symbol}: 최종 진입 신호 = {entry_signal}")
        return entry_signal
    else:
        logging.info(f"--- {symbol}: 진입 조건 미충족 ---")
        return None


def calculate_position_size(total_equity: float, symbol_price: float) -> float:
    """챌린지 전략에 사용할 포지션 크기(수량)를 계산합니다."""
    if total_equity <= 0 or symbol_price <= 0:
        logging.warning("총 자산 또는 심볼 가격이 0 이하이므로 포지션 크기를 계산할 수 없습니다.")
        return 0.0

    seed_amount_usd = total_equity * config.CHALLENGE_SEED_PERCENTAGE
    position_size_usd = seed_amount_usd * config.CHALLENGE_LEVERAGE
    quantity = position_size_usd / symbol_price

    # TODO: Binance 최소 주문 수량 및 단위 확인 후 수량 조절 로직 필요
    # 예: BTCUSDT는 소수점 3자리까지 가능
    quantity = round(quantity, 3) # 임시로 소수점 3자리 반올림

    logging.info(f"총 자산: ${total_equity:.2f}, 시드 비율: {config.CHALLENGE_SEED_PERCENTAGE:.1%}, 레버리지: {config.CHALLENGE_LEVERAGE}x")
    logging.info(f"계산된 포지션 크기 (USD): ${position_size_usd:.2f}, 수량: {quantity}")
    if quantity == 0:
         logging.warning("계산된 주문 수량이 0입니다. 최소 주문 수량 미달 가능성.")

    return quantity


def log_challenge_trade(
    symbol: str, side: str, quantity: float, entry_price: float,
    exit_price: float | None = None, pnl_percent: float | None = None,
    status: str = 'open', reason: str | None = None
):
    """챌린지 전략의 상세 거래 내역을 데이터베이스에 기록합니다."""
    try:
        from src.utils.database import log_trade_to_db # Import DB logging function
        # DB 스키마에 맞게 pnl_percent 포맷 조정 (숫자형)
        pnl_percent_numeric = pnl_percent if pnl_percent is not None else None

        log_trade_to_db(
            strategy='challenge',
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl_percent=pnl_percent_numeric,
            status=status,
            reason=reason
            # order_id 는 필요 시 추가 (create_futures_order 등에서 반환값 받아 전달)
        )
        logging.info(f"[DB] Challenge 거래 로그 기록: {symbol} {side} Qty:{quantity} Status:{status}")
    except ImportError:
        logging.error("Database logging module (src.utils.database) not found.")
    except Exception as e:
        logging.error(f"Challenge 거래 로그 DB 기록 실패: {e}")


# --- 메인 실행 로직 ---
def run_challenge_strategy():
    """챌린지 전략을 실행합니다 (포지션 관리 + 신규 진입 탐색)."""
    logging.info("===== 🚀 챌린지 전략 시작 =====")

    if not FUTURES_CLIENT:
        logging.warning("선물 거래소 API 클라이언트가 초기화되지 않아 챌린지 전략을 실행할 수 없습니다.")
        logging.info("===== 🚀 챌린지 전략 종료 =====")
        return

    # 1. 현재 포지션 확인 및 관리 (TP/SL 체크)
    check_and_manage_positions()

    # 2. 신규 진입 기회 탐색
    logging.info("--- 신규 진입 기회 탐색 시작 ---")
    active_symbols = {pos.get('symbol') for pos in get_current_positions()} # 현재 포지션 있는 심볼 확인

    # 계정 잔고 확인 (포지션 크기 계산용)
    balance_info = get_futures_account_balance()
    total_equity = 0.0
    if balance_info and balance_info.get('totalWalletBalance') is not None:
        try:
            total_equity = float(balance_info['totalWalletBalance'])
            logging.info(f"현재 {FUTURES_CLIENT_TYPE} 선물 총 자산(USDT 추정): ${total_equity:.2f}")
        except (ValueError, TypeError) as e:
             logging.error(f"선물 계정 총 자산을 숫자로 변환 실패: {balance_info['totalWalletBalance']} - {e}")
             total_equity = 0.0
    else:
        logging.warning("선물 계정 잔고 정보를 가져올 수 없어 신규 진입 불가.")
        logging.info("===== 🚀 챌린지 전략 종료 =====")
        return # 잔고 조회 안되면 신규 진입 불가

    # 설정된 심볼들에 대해 진입 조건 확인
    for symbol_config in config.CHALLENGE_SYMBOLS:
        # yfinance 심볼 형식("BTC-USD")을 거래소 형식("BTCUSDT")으로 변환 (필요시)
        exchange_symbol = symbol_config.replace("-", "")

        # 이미 해당 심볼 포지션이 있으면 신규 진입 안 함
        if exchange_symbol in active_symbols:
            logging.info(f"{exchange_symbol}: 이미 포지션 보유 중, 신규 진입 건너뜀.")
            continue

        # 진입 조건 확인
        entry_signal = check_entry_conditions(exchange_symbol)

        if entry_signal:
            # 현재가 조회 (포지션 크기 계산용)
            ticker = get_symbol_ticker(exchange_symbol)
            if not ticker or 'price' not in ticker:
                 logging.warning(f"{exchange_symbol}: 현재가 조회 실패, 진입 불가.")
                 continue
            current_price = float(ticker['price'])

            # 포지션 크기(수량) 계산
            quantity = calculate_position_size(total_equity, current_price)
            if quantity <= 0:
                 logging.warning(f"{exchange_symbol}: 계산된 주문 수량 0 이하, 진입 불가.")
                 continue

            # 주문 생성 시도
            order_result = create_futures_order(exchange_symbol, entry_signal['side'].upper(), quantity)

            if order_result:
                # 거래 로그 기록 (신규 진입)
                log_challenge_trade(
                    symbol=exchange_symbol,
                    side=entry_signal['side'],
                    quantity=quantity,
                    entry_price=current_price, # 실제 체결가는 별도 확인 필요
                    status='open',
                    reason=entry_signal['reason']
                )
                # Slack 알림은 create_futures_order 내부에서 전송됨
            else:
                 # 주문 실패 시 로그 (이미 create_futures_order 에서 로깅/알림)
                 pass
            # 한 번에 하나의 신규 진입만 허용? 또는 여러 개 허용? -> 현재는 루프 계속 진행
            # time.sleep(5) # 연속 주문 방지용 딜레이

    logging.info("===== 🚀 챌린지 전략 종료 =====")


if __name__ == "__main__":
    # 모듈 단독 실행 시 테스트용
    # .env 파일에 Binance API 키 및 Slack Webhook URL 입력 필요
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    if FUTURES_CLIENT:
         run_challenge_strategy()
         # 개별 함수 테스트 (예시)
         # print(get_futures_account_balance())
         # print(get_current_positions())
         # print(get_symbol_ticker("BTCUSDT"))
         # check_and_manage_positions()
         # check_entry_conditions("BTCUSDT")
         # create_futures_order("BTCUSDT", "BUY", 0.001)
         # close_position("BTCUSDT", 0.001, "BUY")
    else:
         logging.warning("Binance Futures 클라이언트가 초기화되지 않아 테스트를 실행할 수 없습니다.")

# --- Binance API 연동 함수 (Placeholder) ---

def get_current_positions() -> list:
    """현재 보유 중인 선물 포지션 목록을 조회합니다. (Binance)"""
    if not FUTURES_CLIENT: return []
    try:
        positions = FUTURES_CLIENT.futures_position_information()
        # positionAmt가 0이 아닌 포지션만 필터링
        active_positions = [
            pos for pos in positions if float(pos.get('positionAmt', 0)) != 0
        ]
        logging.info(f"현재 보유 포지션 {len(active_positions)}개 조회 완료.")
        # logging.debug(f"Active positions: {active_positions}") # 디버그 시 포지션 상세 출력
        return active_positions
    except BinanceAPIException as bae:
         logging.error(f"포지션 조회 API 오류: {bae.status_code} - {bae.message}")
         notifier.send_slack_notification(f"🚨 [Challenge] 포지션 조회 API 오류: {bae.message}")
         return []
    except Exception as e:
        logging.error(f"포지션 조회 중 오류: {e}", exc_info=True)
        notifier.send_slack_notification(f"🚨 [Challenge] 포지션 조회 중 오류 발생: {e}")
        return []

def get_symbol_ticker(symbol: str) -> dict | None:
    """지정된 심볼의 현재 가격 정보를 조회합니다. (Binance)"""
    if not FUTURES_CLIENT: return None
    try:
        ticker = FUTURES_CLIENT.futures_symbol_ticker(symbol=symbol)
        # logging.debug(f"{symbol} 현재가 조회: {ticker}")
        return ticker # {'symbol': 'BTCUSDT', 'price': '65000.00'}
    except BinanceAPIException as bae:
        logging.error(f"{symbol} 가격 조회 API 오류: {bae.status_code} - {bae.message}")
        return None
    except Exception as e:
        logging.error(f"{symbol} 가격 조회 중 오류: {e}")
        return None

def create_futures_order(symbol: str, side: str, quantity: float, order_type: str = 'MARKET') -> dict | None:
    """선물 주문을 생성합니다. (Binance)

    Args:
        symbol (str): 주문할 심볼 (예: BTCUSDT).
        side (str): 'BUY' 또는 'SELL'.
        quantity (float): 주문 수량.
        order_type (str): 주문 유형 (기본값: 'MARKET').

    Returns:
        dict | None: API 응답 또는 오류 시 None.
    """
    if not FUTURES_CLIENT: return None
    logging.info(f"주문 생성 시도: {side} {quantity} {symbol} ({order_type})")
    try:
        order = FUTURES_CLIENT.futures_create_order(
            symbol=symbol,
            side=side,
            type=order_type,
            quantity=quantity,
            # reduceOnly=False # 신규 진입 시
        )
        logging.info(f"✅ 주문 생성 성공: {symbol} {side} {quantity}, OrderId: {order.get('orderId')}")
        notifier.send_slack_notification(f"🛒 [Challenge] 주문 생성: {side} {quantity:.4f} {symbol}")
        return order
    except BinanceAPIException as bae:
         logging.error(f"주문 생성 API 오류 ({symbol} {side} {quantity}): {bae.status_code} - {bae.message}")
         # 예: -2019 (Insufficient margin), -1111 (Precision issue), -4003 (Quantity less than zero)
         notifier.send_slack_notification(f"🚨 [Challenge] {symbol} 주문 생성 API 오류: {bae.message}")
         return None
    except Exception as e:
        logging.error(f"주문 생성 중 오류 ({symbol} {side} {quantity}): {e}", exc_info=True)
        notifier.send_slack_notification(f"🚨 [Challenge] {symbol} 주문 생성 중 오류: {e}")
        return None

def close_position(symbol: str, quantity: float, side_to_close: str) -> dict | None:
    """기존 포지션을 청산하는 주문을 생성합니다. (Binance - reduceOnly 사용)

    Args:
        symbol (str): 청산할 심볼.
        quantity (float): 청산할 수량 (포지션 크기).
        side_to_close (str): 청산할 포지션의 방향 ('BUY' 또는 'SELL').
             BUY 포지션 청산 -> SELL 주문 / SELL 포지션 청산 -> BUY 주문.

    Returns:
        dict | None: API 응답 또는 오류 시 None.
    """
    if not FUTURES_CLIENT: return None
    close_side = 'SELL' if side_to_close == 'BUY' else 'BUY'
    logging.info(f"포지션 청산 시도: {close_side} {quantity} {symbol} (reduceOnly)")
    try:
        # 주의: quantity는 항상 양수여야 함 (API 요구사항)
        order = FUTURES_CLIENT.futures_create_order(
            symbol=symbol,
            side=close_side,
            type='MARKET',
            quantity=abs(quantity),
            reduceOnly=True # 기존 포지션 청산 주문임을 명시
        )
        logging.info(f"✅ 포지션 청산 주문 성공: {symbol}, OrderId: {order.get('orderId')}")
        # 실제 청산 완료 및 PNL은 별도 확인 필요 (예: 웹소켓 또는 다음 조회 시)
        return order
    except BinanceAPIException as bae:
         logging.error(f"포지션 청산 API 오류 ({symbol} {close_side} {quantity}): {bae.status_code} - {bae.message}")
         notifier.send_slack_notification(f"🚨 [Challenge] {symbol} 포지션 청산 API 오류: {bae.message}")
         return None
    except Exception as e:
        logging.error(f"포지션 청산 중 오류 ({symbol} {close_side} {quantity}): {e}", exc_info=True)
        notifier.send_slack_notification(f"🚨 [Challenge] {symbol} 포지션 청산 중 오류: {e}")
        return None

# --- 손익 관리 로직 ---
def check_and_manage_positions():
    """현재 포지션의 손익을 확인하고 TP/SL 조건 충족 시 청산합니다."""
    if not FUTURES_CLIENT: return
    logging.info("--- 포지션 손익 관리 시작 ---")
    active_positions = get_current_positions()

    if not active_positions:
        logging.info("현재 보유 포지션 없음.")
        return

    for position in active_positions:
        try:
            symbol = position.get('symbol')
            entry_price = float(position.get('entryPrice', 0))
            position_amt = float(position.get('positionAmt', 0))
            side = 'BUY' if position_amt > 0 else 'SELL'
            # unrealizedProfit 값 사용 또는 현재가 조회하여 계산
            # PNL 계산 방식 확인 필요 (Binance API 문서 참고)
            unrealized_pnl = float(position.get('unRealizedProfit', 0))

            if entry_price == 0 or position_amt == 0:
                logging.warning(f"{symbol}: 유효하지 않은 포지션 정보, 건너뜀 ({position})")
                continue

            # 현재가 조회
            ticker = get_symbol_ticker(symbol)
            if not ticker or 'price' not in ticker:
                 logging.warning(f"{symbol}: 현재가 조회 실패, 손익 관리 건너뜀.")
                 continue
            current_price = float(ticker['price'])

            # PNL 비율 계산 (대략적인 추정, 정확한 계산식은 검증 필요)
            # 비용/수수료는 고려 안 함
            if side == 'BUY':
                pnl_percent = (current_price - entry_price) / entry_price
            else: # SELL
                pnl_percent = (entry_price - current_price) / entry_price

            logging.info(f"포지션 확인: {symbol} ({side}), 진입가: {entry_price}, 현재가: {current_price}, 수량: {position_amt}, 추정 PnL: {pnl_percent:.2%}")

            # TP 조건 확인
            if pnl_percent >= config.CHALLENGE_TP_PERCENT:
                logging.info(f"🎯 TP 조건 충족: {symbol} ({pnl_percent:.2%}) >= {config.CHALLENGE_TP_PERCENT:.1%}. 포지션 청산 시도.")
                close_reason = f"TP Hit ({pnl_percent:.2%})"
                close_order = close_position(symbol, abs(position_amt), side)
                if close_order:
                    # 로그 기록 (청산 성공 시)
                    log_challenge_trade(symbol, side, abs(position_amt), entry_price,
                                        exit_price=current_price, # 청산 가격은 실제 체결가 반영 필요
                                        pnl_percent=pnl_percent,
                                        status='closed_tp', reason=close_reason)
                    notifier.send_slack_notification(f"✅ [Challenge] TP 실행: {symbol} 청산 ({close_reason})")
                else:
                     log_challenge_trade(symbol, side, abs(position_amt), entry_price, status='error', reason=f"TP Close Fail")
                continue # TP/SL 중 하나만 처리

            # SL 조건 확인
            if pnl_percent <= -config.CHALLENGE_SL_PERCENT:
                logging.info(f"🛑 SL 조건 충족: {symbol} ({pnl_percent:.2%}) <= {-config.CHALLENGE_SL_PERCENT:.1%}. 포지션 청산 시도.")
                close_reason = f"SL Hit ({pnl_percent:.2%})"
                close_order = close_position(symbol, abs(position_amt), side)
                if close_order:
                     log_challenge_trade(symbol, side, abs(position_amt), entry_price,
                                         exit_price=current_price, pnl_percent=pnl_percent,
                                         status='closed_sl', reason=close_reason)
                     notifier.send_slack_notification(f"🛑 [Challenge] SL 실행: {symbol} 청산 ({close_reason})")
                else:
                     log_challenge_trade(symbol, side, abs(position_amt), entry_price, status='error', reason=f"SL Close Fail")
                continue

        except ValueError as ve:
            logging.error(f"포지션 정보 처리 오류 (ValueError): {position} - {ve}")
        except Exception as e:
            logging.error(f"포지션 관리 중 예상치 못한 오류 ({position.get('symbol')}): {e}", exc_info=True) 

def calculate_challenge_indicators(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict | None]:
    """챌린지 전략에 필요한 지표(RSI, SMA, 다이버전스, 거래량, 지지/저항, 추세선 등)를 계산합니다."""
    if df.empty:
        return pd.DataFrame(), {}, None # 오류 시 빈 객체들 반환
    try:
        # 기본 지표 계산
        df['RSI'] = strategy_utils.calculate_rsi(dataframe=df, window=14) # RSI 계산
        df['SMA7'] = strategy_utils.calculate_sma(dataframe=df, window=7)   # 7일 이평선
        df['SMA20'] = strategy_utils.calculate_sma(dataframe=df, window=20) # 20일 이평선
        df['Volume_SMA20'] = df['Volume'].rolling(window=20).mean()      # 거래량 20일 이평선

        # NaN 값 처리 (초기 지표 계산으로 발생)
        df_cleaned = df.dropna().copy() # 이후 계산 위해 복사본 사용
        if df_cleaned.empty:
             logging.warning("지표 계산 후 유효 데이터 없음.")
             return pd.DataFrame(), {}, None

        # 고급 지표 계산 (strategy_utils 활용)
        # RSI 다이버전스 (최근 N개 봉 기준, 예: 14개)
        df_cleaned['RSI_Divergence'] = strategy_utils.detect_rsi_divergence(df_cleaned['Close'], df_cleaned['RSI'], window=14)

        # 거래량 급증 여부 (최근 1봉 기준, 20봉 평균 대비 2배)
        df_cleaned['Volume_Spike'] = strategy_utils.detect_volume_spike(df_cleaned, window=20, factor=2.0)

        # 거래량 기반 지지/저항 (최근 N개 봉 기준, 예: 30개, 3개 레벨)
        # 이 함수는 dict를 반환하므로, 각 행에 적용하기 어려움. 최신 값만 계산하여 사용하거나 다른 방식 고려.
        # 여기서는 최신 지지/저항 레벨만 계산하여 로깅/조건 판단에 활용 (DataFrame에 추가 X)
        latest_sr_levels = strategy_utils.detect_support_resistance_by_volume(df_cleaned, window=30, n_levels=3)
        # logging.info(f"최신 지지/저항 레벨: {latest_sr_levels}") # 필요 시 로깅

        # 추세선 이탈/되돌림 판단 (strategy_utils 호출)
        trend_event = strategy_utils.detect_trendline_breakout(df_cleaned, window=30, peak_distance=5)
        # DataFrame에 컬럼으로 추가하기보다 최신 이벤트 정보만 전달

        logging.info("챌린지 지표 계산 완료: RSI, SMA, Vol SMA, Divergence, Spike, SR, Trend")
        return df_cleaned, latest_sr_levels, trend_event # DataFrame, SR dict, Trend dict 반환

    except Exception as e:
        logging.error(f"챌린지 지표 계산 중 오류: {e}", exc_info=True)
        return pd.DataFrame(), {}, None # 오류 시 빈 객체들 반환

# --- 진입/청산 조건 판단 --- #
def detect_entry_opportunity(df: pd.DataFrame, sr_levels: dict, trend_event: dict | None) -> tuple[str | None, str | None]:
    """주어진 데이터, SR 레벨, 추세선 이벤트를 분석하여 진입 신호(long/short)를 판단합니다."""
    if df.empty:
        return None, None
    latest = df.iloc[-1]
    current_price = latest['Close']
    support = sr_levels.get('support', [])
    resistance = sr_levels.get('resistance', [])

    # --- Long 진입 조건 --- #
    long_conditions_met = []
    if latest.get('RSI_Divergence') == 'bullish': long_conditions_met.append("RSI Bullish Divergence")
    if support and abs(current_price - support[0]) / support[0] < 0.01 and current_price > df.iloc[-2]['Close'] and latest.get('Volume_Spike') == True:
        long_conditions_met.append(f"Support Bounce ({support[0]:.2f}) + Volume Spike")
    if current_price > latest['SMA7'] and latest['SMA7'] > df.iloc[-2]['SMA7']: long_conditions_met.append("Price > SMA7 (Rising)")

    # 조건 4: 하단 추세선 지지 확인 (되돌림)
    if trend_event and trend_event.get('type') == 'retest_support' and trend_event.get('trendline') == 'lower':
        long_conditions_met.append(f"Lower Trendline Retest Support (~{trend_event.get('price'):.2f})")

    # 최종 Long 결정 (다이버전스 또는 다른 조건 2개 이상)
    if "RSI Bullish Divergence" in long_conditions_met or len(long_conditions_met) >= 2:
        reason_str = ", ".join(long_conditions_met)
        logging.info(f"Long 진입 신호 감지. 충족 조건: [{reason_str}]")
        return 'long', reason_str

    # --- Short 진입 조건 --- #
    short_conditions_met = []
    if latest.get('RSI_Divergence') == 'bearish': short_conditions_met.append("RSI Bearish Divergence")
    if resistance and abs(current_price - resistance[0]) / resistance[0] < 0.01 and current_price < df.iloc[-2]['Close'] and latest.get('Volume_Spike') == True:
        short_conditions_met.append(f"Resistance Reject ({resistance[0]:.2f}) + Volume Spike")
    if current_price < latest['SMA7'] and latest['SMA7'] < df.iloc[-2]['SMA7']: short_conditions_met.append("Price < SMA7 (Falling)")

    # 조건 4: 상단 추세선 저항 확인 (되돌림)
    if trend_event and trend_event.get('type') == 'retest_resistance' and trend_event.get('trendline') == 'upper':
        short_conditions_met.append(f"Upper Trendline Retest Resistance (~{trend_event.get('price'):.2f})")

    # 최종 Short 결정 (다이버전스 또는 다른 조건 1개 이상)
    if "RSI Bearish Divergence" in short_conditions_met or len(short_conditions_met) >= 1:
        reason_str = ", ".join(short_conditions_met)
        logging.info(f"Short 진입 신호 감지. 충족 조건: [{reason_str}]")
        return 'short', reason_str

    # --- 진입 신호 없음 --- #
    logging.debug("진입 조건 미충족.")
    return None, None

# --- 메인 전략 실행 함수 수정 --- #
def run_challenge_strategy():
    # ... (초기화, 심볼 정의 등) ...
    client = initialize_binance_client()
    if not client: return
    symbol = config.CHALLENGE_SYMBOL
    current_pos = get_current_position(client, symbol)
    df_raw = get_binance_data(client, symbol, interval='1h', limit=200)
    if df_raw.empty: return

    # 지표 계산 시 trend_event도 함께 받음
    df_processed, latest_sr_levels, trend_event = calculate_challenge_indicators(df_raw)
    if df_processed.empty: return
    latest_price = df_processed.iloc[-1]['Close']

    if current_pos:
        manage_position(client, current_pos, latest_price)
        current_pos = get_current_position(client, symbol) # 포지션 상태 재확인

    if not current_pos:
        # 진입 판단 시 trend_event 전달
        entry_side, entry_reason = detect_entry_opportunity(df_processed, latest_sr_levels, trend_event)
        if entry_side:
            position_size = calculate_position_size(client, symbol, latest_price)
            if position_size and position_size > 0:
                create_market_order(client, symbol, entry_side.upper(), position_size)
            else:
                 logging.warning("계산된 포지션 크기 없음/0 이하. 진입 불가.")
        else:
            logging.info("신규 진입 기회 없음.")

    logging.info("===== 🔥 Challenge Trading 전략 실행 종료 ====")

# ... (if __name__ == "__main__") ... 