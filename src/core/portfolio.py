import alpaca_trade_api as tradeapi
import pandas as pd
import logging
import os # log_transaction 에서 사용
from datetime import datetime, date, timedelta # 월 1회 리밸런싱 체크용
import time
import yfinance as yf

# strategy_utils 및 notifier 임포트
import strategy_utils
import notifier

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# config 모듈에서 설정값 로드 (config.py가 같은 디렉토리에 있다고 가정)
try:
    import config
except ImportError:
    logging.error("config.py 파일을 찾을 수 없습니다. API 키 및 설정을 확인하세요.")
    raise

# --- Alpaca API 클라이언트 초기화 ---
try:
    api = tradeapi.REST(
        key_id=config.ALPACA_API_KEY,
        secret_key=config.ALPACA_SECRET_KEY,
        base_url=config.ALPACA_BASE_URL,
        api_version='v2'
    )
    # 계정 상태 확인
    account = api.get_account()
    logging.info(f"Alpaca 계정 연결 성공. 상태: {account.status}, 모드: {config.ALPACA_TRADING_MODE}")
except Exception as e:
    logging.error(f"Alpaca API 연결 실패: {e}")
    notifier.send_slack_notification(f"🚨 [Core] Alpaca API 연결 실패: {e}") # 연결 실패 시 알림
    raise

# --- 전역 변수 (월 1회 실행 제어용) ---
_last_rebalance_check_month = None

# --- 주요 기능 함수 (기본 틀) ---

def get_account_details():
    """Alpaca 계정 상세 정보를 조회합니다."""
    try:
        return api.get_account()
    except Exception as e:
        logging.error(f"계정 정보 조회 실패: {e}")
        notifier.send_slack_notification(f"🚨 [Core] 계정 정보 조회 실패: {e}")
        return None

def get_positions():
    """현재 보유 포지션 목록을 조회합니다."""
    try:
        return api.list_positions()
    except Exception as e:
        logging.error(f"포지션 조회 실패: {e}")
        notifier.send_slack_notification(f"🚨 [Core] 포지션 조회 실패: {e}")
        return []

def check_buy_conditions(symbol: str) -> bool:
    """주어진 종목에 대한 매수 조건을 확인합니다. (RSI, VIX, 금리)

    strategy_utils 모듈의 함수를 사용합니다.
    """
    logging.info(f"{symbol}: 매수 조건 확인 중 (RSI<{config.CORE_RSI_THRESHOLD}, VIX>{config.CORE_VIX_THRESHOLD})...")
    rsi_condition = False
    vix_condition = False
    interest_rate_condition = False # 금리 조건 변수 추가

    # 1. VIX 조건 확인 (strategy_utils 사용)
    current_vix = strategy_utils.get_current_vix()
    if current_vix is not None:
        vix_condition = current_vix > config.CORE_VIX_THRESHOLD
        logging.info(f"VIX 조건 확인 결과: 현재 VIX={current_vix:.2f}, 조건 충족={vix_condition}")
    else:
        logging.warning("VIX 조회 실패 시 매수 불가 처리 또는 다른 정책 적용 가능")

    # 2. RSI 조건 확인 (strategy_utils 사용)
    # RSI 계산 시 필요한 데이터 기간 조정 가능 (예: "3mo")
    rsi_series = strategy_utils.calculate_rsi(symbol=symbol, period="3mo", window=14)
    if rsi_series is not None and not rsi_series.empty:
        current_rsi = rsi_series.iloc[-1]
        rsi_condition = current_rsi < config.CORE_RSI_THRESHOLD
        logging.info(f"{symbol} RSI 조건 확인 결과: 현재 RSI={current_rsi:.2f}, 조건 충족={rsi_condition}")
    else:
        logging.warning(f"{symbol}: RSI 조건 확인 불가 (데이터 조회/계산 실패 등)")

    # 3. 금리 조건 확인 (TODO)
    # TODO: 외부 API 또는 데이터를 사용하여 현재 금리 상태(예: FOMC 발표)를 확인하는 로직 구현 필요
    # 예시: 금리 동결 또는 인하 상태일 때 True 설정
    interest_rate_condition = True # 임시로 항상 True 설정
    logging.info(f"금리 조건 확인 결과: {interest_rate_condition} (TODO: 실제 구현 필요)")

    # 최종 결과 반환 (모든 조건 충족 시 True)
    final_decision = rsi_condition and vix_condition and interest_rate_condition
    logging.info(f"{symbol}: 최종 매수 조건 결과 = {final_decision} (RSI:{rsi_condition}, VIX:{vix_condition}, 금리:{interest_rate_condition})")
    return final_decision

def calculate_target_allocations(symbols: list) -> dict:
    """목표 포트폴리오 비중을 계산합니다. (예: 동일 비중)"""
    if not symbols:
        return {}
    num_symbols = len(symbols)
    if num_symbols == 0:
        return {}
    allocation = 1.0 / num_symbols
    return {symbol: allocation for symbol in symbols}

def needs_rebalance(current_positions: list, target_allocations: dict) -> bool:
    """리밸런싱 필요 여부를 판단합니다. (현재 비중과 목표 비중 비교)"""
    account_details = get_account_details()
    if not account_details:
        logging.warning("리밸런싱 필요 여부 판단 불가: 계정 정보 없음.")
        return False

    try:
        total_equity = float(account_details.equity)
        if total_equity <= 0:
             logging.warning("리밸런싱 필요 여부 판단 불가: 총 자산 0 이하.")
             return False
    except ValueError:
        logging.error(f"리밸런싱 필요 여부 판단 불가: 총 자산 값 오류 ({account_details.equity})")
        return False

    current_allocations = {}
    position_value = 0.0

    for position in current_positions:
        symbol = position.symbol
        try:
            market_value = float(position.market_value)
            current_allocations[symbol] = market_value / total_equity
            position_value += market_value
        except Exception as e:
            logging.warning(f"{symbol} 포지션 가치 처리 중 오류: {e}")
            # 오류 발생 시 해당 포지션은 비중 계산에서 제외될 수 있음

    logging.info(f"현재 총 자산: ${total_equity:,.2f}, 포지션 가치: ${position_value:,.2f}")
    log_target = {k: f"{v:.1%}" for k, v in target_allocations.items()}
    log_current = {k: f"{v:.1%}" for k, v in current_allocations.items()}
    logging.info(f"목표 비중: {log_target}")
    logging.info(f"현재 비중: {log_current}")

    max_diff = 0.0
    symbol_max_diff = None
    for symbol, target_alloc in target_allocations.items():
        current_alloc = current_allocations.get(symbol, 0.0)
        diff = abs(current_alloc - target_alloc)
        if diff > max_diff:
            max_diff = diff
            symbol_max_diff = symbol
        if diff > config.CORE_REBALANCE_THRESHOLD:
            logging.info(f"⚠️ 리밸런싱 필요: {symbol} 비중 차이({diff:.2%}) > 임계값({config.CORE_REBALANCE_THRESHOLD:.1%})")
            notifier.send_slack_notification(f"⚖️ [Core] 리밸런싱 필요 감지: {symbol} 비중 차이 {diff:.2%}")
            return True

    logging.info(f"리밸런싱 필요 없음 (최대 비중 차이: {symbol_max_diff} {max_diff:.2%}).")
    return False

def execute_buy_order(symbol: str, amount_usd: float):
    """지정된 금액만큼 종목을 매수합니다."""
    try:
        # 최소 주문 금액 확인 (Alpaca는 $1)
        if amount_usd < 1.0:
            logging.warning(f"{symbol} 매수 주문 건너뜀: 주문 금액(${amount_usd:.2f})이 너무 작습니다.")
            return

        logging.info(f"매수 주문 시도: {symbol}, 금액: ${amount_usd:.2f}")
        order = api.submit_order(
            symbol=symbol,
            notional=amount_usd, # 달러 금액으로 주문
            side='buy',
            type='market',
            time_in_force='day' # 당일 유효 주문
        )
        logging.info(f"✅ 매수 주문 제출 성공: {symbol}, 주문 ID: {order.id}, 상태: {order.status}")
        log_transaction(symbol, 'buy', amount_usd, order.id, order.status)
        notifier.send_slack_notification(f"🛒 [Core] 매수 주문 제출: {symbol} ${amount_usd:.2f}")
    except tradeapi.rest.APIError as api_err:
         # Alpaca API 관련 에러 처리
         logging.error(f"❌ {symbol} 매수 주문 API 오류: {api_err.status_code} - {api_err}")
         # 특정 에러 코드에 따른 처리 가능 (예: 403 insufficient_balance, 403 market_closed 등)
         log_transaction(symbol, 'buy', amount_usd, 'N/A', f'APIError_{api_err.status_code}')
         notifier.send_slack_notification(f"🚨 [Core] {symbol} 매수 주문 API 오류: {api_err}")
    except Exception as e:
        logging.error(f"❌ {symbol} 매수 주문 실패: {e}", exc_info=True)
        log_transaction(symbol, 'buy', amount_usd, 'N/A', f'Exception')
        notifier.send_slack_notification(f"🚨 [Core] {symbol} 매수 주문 실패: {e}")

def execute_rebalance(target_allocations: dict):
    """목표 비중에 맞춰 포트폴리오를 리밸런싱합니다."""
    account_details = get_account_details()
    if not account_details:
        logging.error("리밸런싱 중단: 계정 정보를 가져올 수 없습니다.")
        return

    total_equity = float(account_details.equity)
    current_positions_list = get_positions()
    current_positions = {pos.symbol: pos for pos in current_positions_list}
    logging.info(f"리밸런싱 시작. 총 자산: ${total_equity:,.2f}")

    orders_to_submit = []
    # 1. 매도 주문 먼저 생성 (자금 확보)
    for symbol, position in current_positions.items():
        target_alloc = target_allocations.get(symbol, 0.0) # 목표 비중에 없으면 0
        target_value = total_equity * target_alloc
        try:
            current_value = float(position.market_value)
            current_qty = float(position.qty)
        except Exception as e:
             logging.warning(f"{symbol} 리밸런싱 중 포지션 정보 처리 오류: {e}")
             continue

        diff = target_value - current_value
        # 매도 조건: 목표 금액보다 현재 금액이 $1 이상 많고, 보유 수량이 0보다 클 때
        if diff < -1.0 and current_qty > 0:
            amount_usd_to_sell = abs(diff)
             # Alpaca notional 매도 사용
            logging.info(f"{symbol} 매도 주문 생성 (Notional): ${amount_usd_to_sell:.2f}")
            orders_to_submit.append({'symbol': symbol, 'notional': amount_usd_to_sell, 'side': 'sell'})

    # 2. 매수 주문 생성
    for symbol, target_alloc in target_allocations.items():
        target_value = total_equity * target_alloc
        current_value = 0.0
        if symbol in current_positions:
            try:
                current_value = float(current_positions[symbol].market_value)
            except Exception as e:
                 logging.warning(f"{symbol} 리밸런싱 중 포지션 정보 처리 오류: {e}")
                 # 매수 판단 시 현재 가치 0으로 간주

        diff = target_value - current_value
        # 매수 조건: 목표 금액보다 현재 금액이 $1 이상 적을 때
        if diff > 1.0:
            amount_usd_to_buy = diff
            logging.info(f"{symbol} 매수 주문 생성 (Notional): ${amount_usd_to_buy:.2f}")
            orders_to_submit.append({'symbol': symbol, 'notional': amount_usd_to_buy, 'side': 'buy'})

    # 생성된 주문 제출 (매도 먼저, 매수 나중)
    if not orders_to_submit:
        logging.info("리밸런싱 주문 없음.")
        return

    logging.info(f"리밸런싱 주문 목록 ({len(orders_to_submit)}개): {orders_to_submit}")
    submitted_orders = []
    failed_orders = []

    # 매도 주문 실행
    sell_orders = [o for o in orders_to_submit if o['side'] == 'sell']
    for order_data in sell_orders:
        if order_data['notional'] < 1.0:
             logging.warning(f"{order_data['symbol']} 리밸런싱 매도 주문 건너뜀: 금액 작음 (${order_data['notional']:.2f})")
             continue
        try:
            order = api.submit_order(
                symbol=order_data['symbol'],
                notional=order_data['notional'],
                side=order_data['side'], type='market', time_in_force='day'
            )
            logging.info(f"✅ 리밸런싱 매도 주문 제출: {order_data['symbol']}, 금액: ${order_data['notional']:.2f}, ID: {order.id}")
            submitted_orders.append(order)
            log_transaction(order_data['symbol'], order_data['side'], order_data['notional'], order.id, order.status)
        except tradeapi.rest.APIError as api_err:
             logging.error(f"❌ 리밸런싱 매도 주문 API 오류 ({order_data['symbol']}): {api_err.status_code} - {api_err}")
             failed_orders.append(order_data)
             log_transaction(order_data['symbol'], order_data['side'], order_data['notional'], 'N/A', f'APIError_{api_err.status_code}')
        except Exception as e:
            logging.error(f"❌ 리밸런싱 매도 주문 실패 ({order_data['symbol']}): {e}", exc_info=True)
            failed_orders.append(order_data)
            log_transaction(order_data['symbol'], order_data['side'], order_data['notional'], 'N/A', f'Exception')

    # TODO: 매도 주문 완료 대기 로직 추가 가능 (필요 시)
    # time.sleep(5) # 예시: 잠시 대기 후 매수 진행

    # 매수 주문 실행
    buy_orders = [o for o in orders_to_submit if o['side'] == 'buy']
    for order_data in buy_orders:
        if order_data['notional'] < 1.0:
             logging.warning(f"{order_data['symbol']} 리밸런싱 매수 주문 건너뜀: 금액 작음 (${order_data['notional']:.2f})")
             continue
        try:
            # 매수 전 사용 가능 금액 재확인 가능
            # account = get_account_details()
            # if float(account.buying_power) < order_data['notional']:
            #      logging.warning(...) continue

            order = api.submit_order(
                symbol=order_data['symbol'],
                notional=order_data['notional'],
                side=order_data['side'], type='market', time_in_force='day'
            )
            logging.info(f"✅ 리밸런싱 매수 주문 제출: {order_data['symbol']}, 금액: ${order_data['notional']:.2f}, ID: {order.id}")
            submitted_orders.append(order)
            log_transaction(order_data['symbol'], order_data['side'], order_data['notional'], order.id, order.status)
        except tradeapi.rest.APIError as api_err:
             logging.error(f"❌ 리밸런싱 매수 주문 API 오류 ({order_data['symbol']}): {api_err.status_code} - {api_err}")
             failed_orders.append(order_data)
             log_transaction(order_data['symbol'], order_data['side'], order_data['notional'], 'N/A', f'APIError_{api_err.status_code}')
        except Exception as e:
            logging.error(f"❌ 리밸런싱 매수 주문 실패 ({order_data['symbol']}): {e}", exc_info=True)
            failed_orders.append(order_data)
            log_transaction(order_data['symbol'], order_data['side'], order_data['notional'], 'N/A', f'Exception')

    # 최종 결과 알림
    success_count = len(submitted_orders)
    fail_count = len(failed_orders)
    if fail_count > 0:
         notifier.send_slack_notification(f"🚨 [Core] 리밸런싱 중 {fail_count}개 주문 실패. 로그 확인 필요.")
    elif success_count > 0:
         notifier.send_slack_notification(f"✅ [Core] 리밸런싱 주문 {success_count}개 제출 완료.")
    else:
         logging.info("리밸런싱 주문 제출 내역 없음.") # 모두 금액 작아서 건너뛴 경우 등

def log_transaction(symbol: str, side: str, amount_usd: float, order_id: str, status: str):
    """거래 내역을 CSV 파일에 기록합니다."""
    try:
        log_entry = pd.DataFrame({
            'timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")], # ISO 형식으로 저장
            'symbol': [symbol],
            'side': [side],
            'amount_usd': [round(amount_usd, 2)],
            'order_id': [order_id],
            'status': [status] # 주문 상태 (submitted, filled, canceled, rejected 등)
        })
        # 파일이 없으면 헤더와 함께 새로 쓰고, 있으면 이어서 쓴다
        log_file = config.LOG_CORE_FILE
        header = not os.path.exists(log_file)
        log_entry.to_csv(log_file, mode='a', header=header, index=False)
        logging.info(f"거래 로그 기록: {symbol} {side} ${amount_usd:.2f}, ID: {order_id}, 상태: {status}")
    except Exception as e:
        logging.error(f"거래 로그 기록 실패: {e}")

# --- 월 1회 실행 여부 체크 함수 ---
def is_first_run_of_month() -> bool:
    """현재 실행이 해당 월의 첫 실행인지 확인합니다."""
    global _last_rebalance_check_month
    current_month = date.today().month
    if _last_rebalance_check_month != current_month:
        logging.info(f"{current_month}월 첫 리밸런싱 체크 실행.")
        _last_rebalance_check_month = current_month
        return True
    return False

# --- 메인 실행 로직 ---
def run_core_portfolio_strategy():
    """안정형 포트폴리오 전략을 실행합니다."""
    logging.info("===== 💰 안정형 포트폴리오 전략 시작 =====")

    # 1. 리밸런싱 (매월 첫 실행 시에만 체크)
    if is_first_run_of_month():
        logging.info("--- 월간 리밸런싱 체크 --- ")
        current_positions_list = get_positions()
        if current_positions_list is not None: # get_positions 실패 시 None 반환 가능
            target_allocations = calculate_target_allocations(config.CORE_PORTFOLIO_SYMBOLS)
            if needs_rebalance(current_positions_list, target_allocations):
                execute_rebalance(target_allocations)
            else:
                logging.info("월간 리밸런싱 조건 미충족.")
        else:
            logging.warning("리밸런싱 체크 건너뜀: 포지션 조회 실패.")
    else:
        logging.info("이번 달 리밸런싱 체크 이미 수행됨.")

    # 2. 신규 매수 조건 확인 및 실행 (매일 실행)
    logging.info("--- 일일 신규 매수 체크 --- ")
    account_details = get_account_details()
    if not account_details:
         logging.warning("신규 매수 로직 중단: 계정 정보를 가져올 수 없음.")
         logging.info("===== 💰 안정형 포트폴리오 전략 종료 =====")
         return

    try:
        # 사용 가능 현금 (Alpaca에서는 non_marginable_buying_power 또는 cash 사용)
        # trading_blocked=True 이면 매수 불가
        if account_details.trading_blocked:
             logging.warning("신규 매수 중단: 계정이 거래 제한 상태입니다.")
             return
        cash_available = float(account_details.non_marginable_buying_power)
        logging.info(f"사용 가능 현금 (Non-marginable): ${cash_available:.2f}")
    except Exception as e:
         logging.error(f"사용 가능 현금 확인 중 오류: {e}")
         return

    # TODO: 매수 금액 결정 로직 개선 (예: 가용 현금의 일정 비율, 최대 투입 금액 제한 등)
    buy_amount_per_symbol = 100 # 임시 매수 금액 ($100)
    required_cash = buy_amount_per_symbol * len(config.CORE_PORTFOLIO_SYMBOLS) # 대략적인 필요 금액

    if cash_available > buy_amount_per_symbol : # 최소 1개 종목 매수 가능 금액 확인
        current_symbols = {p.symbol for p in get_positions()} # 현재 보유 종목 set
        symbols_to_check = [s for s in config.CORE_PORTFOLIO_SYMBOLS if s not in current_symbols]

        if not symbols_to_check:
            logging.info("모든 목표 종목을 이미 보유 중입니다. 신규 매수 건너뜀.")
        else:
            logging.info(f"매수 조건 확인할 미보유 종목: {symbols_to_check}")
            available_cash_for_new_buys = cash_available # 가용 현금 전체 사용 가능? 또는 비율 설정?
            symbols_bought_count = 0
            # TODO: 우선순위 부여 가능 (RSI가 더 낮은 종목 먼저 매수 등)
            for symbol in symbols_to_check:
                if available_cash_for_new_buys < buy_amount_per_symbol:
                     logging.info("남은 가용 현금 부족으로 추가 매수 중단.")
                     break
                if check_buy_conditions(symbol):
                     logging.info(f"{symbol}: 매수 조건 충족. 매수 실행.")
                     execute_buy_order(symbol, buy_amount_per_symbol)
                     available_cash_for_new_buys -= buy_amount_per_symbol # 사용한 금액 차감
                     symbols_bought_count += 1
                     # time.sleep(1) # 연속 주문 시 API 제한 피하기 위해 잠시 대기
                else:
                     logging.info(f"{symbol}: 매수 조건 미충족.")
            if symbols_bought_count > 0:
                logging.info(f"신규 매수 {symbols_bought_count}건 실행 완료.")
            else:
                 logging.info("매수 조건 만족하는 신규 종목 없음.")

    else:
        logging.info(f"신규 매수를 위한 현금 부족 (가용: ${cash_available:.2f}, 최소 필요 추정: ${buy_amount_per_symbol:.2f})")

    logging.info("===== 💰 안정형 포트폴리오 전략 종료 =====")


if __name__ == "__main__":
    # 모듈 단독 실행 시 테스트용
    # config.py 에 실제 API 키 입력 필요
    if not config.ALPACA_API_KEY or config.ALPACA_API_KEY == "YOUR_ALPACA_API_KEY":
         logging.warning("'.env' 파일에 실제 Alpaca API 키를 입력해주세요.")
    else:
         # run_core_portfolio_strategy() # 전체 실행 테스트
         # 개별 함수 테스트 (예시)
         # print(get_account_details())
         # print(get_positions())
         # print(check_buy_conditions("SPY"))
         # target = calculate_target_allocations(config.CORE_PORTFOLIO_SYMBOLS)
         # print(needs_rebalance(get_positions(), target))
         # execute_buy_order("AAPL", 1.5) # $1.5 매수 테스트
         # execute_rebalance(target) # 리밸런싱 테스트
         pass 