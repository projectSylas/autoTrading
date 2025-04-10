import pandas as pd
from prophet import Prophet
from prophet.diagnostics import performance_metrics, cross_validation
import logging
from datetime import datetime, timedelta
import math # 괴리율 계산 시 사용

# notifier 모듈 로드 (Slack 알림용)
try:
    import notifier
except ImportError:
    logging.error("notifier.py 파일을 찾을 수 없습니다.")
    notifier = None

# strategy_utils 모듈 로드 (데이터 로드용, 선택 사항)
try:
    import strategy_utils
except ImportError:
    logging.error("strategy_utils.py 파일을 찾을 수 없습니다.")
    strategy_utils = None

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Prophet 로깅 레벨 조정 (너무 많은 정보 출력 방지)
# prophet_logger = logging.getLogger('prophet.models')
# prophet_logger.setLevel(logging.WARNING)
# cmdstanpy 로거 가져오기 (컴파일 메시지 등)
# cmdstanpy_logger = logging.getLogger('cmdstanpy')
# cmdstanpy_logger.setLevel(logging.WARNING)


def prepare_data_for_prophet(df: pd.DataFrame, price_col: str = 'Close') -> pd.DataFrame:
    """Prophet 모델 학습에 필요한 형식으로 DataFrame을 변환합니다.

    Args:
        df (pd.DataFrame): 시계열 데이터 (DatetimeIndex 포함).
        price_col (str): 사용할 가격 컬럼명.

    Returns:
        pd.DataFrame: 'ds' (날짜/시간)와 'y' (가격) 컬럼을 가진 DataFrame.
                      오류 발생 시 빈 DataFrame 반환.
    """
    if df is None or df.empty or price_col not in df.columns:
        logging.warning("Prophet 데이터 준비 불가: DataFrame 비어있거나 가격 컬럼 없음.")
        return pd.DataFrame()
    if not isinstance(df.index, pd.DatetimeIndex):
        logging.warning("Prophet 데이터 준비 불가: DataFrame 인덱스가 DatetimeIndex가 아님.")
        return pd.DataFrame()

    try:
        prophet_df = df.reset_index() # DatetimeIndex를 컬럼으로 변환
        # 컬럼 이름 변경: 날짜/시간 -> 'ds', 가격 -> 'y'
        prophet_df = prophet_df.rename(columns={df.index.name or 'index': 'ds', price_col: 'y'})
        # 필요한 컬럼만 선택
        prophet_df = prophet_df[['ds', 'y']]
        # 'y' 값에 NaN 이나 inf 가 없는지 확인 (Prophet 요구사항)
        prophet_df = prophet_df.dropna(subset=['y'])
        prophet_df = prophet_df[prophet_df['y'].apply(math.isfinite)]
        logging.info(f"Prophet 학습 데이터 준비 완료 ({len(prophet_df)} 행)")
        return prophet_df
    except Exception as e:
        logging.error(f"Prophet 데이터 준비 중 오류: {e}")
        return pd.DataFrame()

def forecast_price(df_prophet: pd.DataFrame, periods: int = 24, freq: str = 'H') -> pd.DataFrame | None:
    """Prophet 모델을 학습시키고 미래 가격을 예측합니다.

    Args:
        df_prophet (pd.DataFrame): 'ds', 'y' 컬럼을 가진 학습 데이터.
        periods (int): 예측할 기간 (단위: freq).
        freq (str): 예측할 기간의 빈도 (예: 'H'=시간, 'D'=일).

    Returns:
        pd.DataFrame | None: 예측 결과 DataFrame (ds, yhat, yhat_lower, yhat_upper 포함).
                            오류 발생 시 None 반환.
    """
    if df_prophet is None or df_prophet.empty:
        logging.warning("예측 불가: 학습 데이터 없음.")
        return None

    try:
        logging.info(f"Prophet 모델 학습 시작 (데이터 {len(df_prophet)}개)...")
        # Prophet 모델 초기화 (기본 설정 사용, 필요시 파라미터 조정)
        # 예: seasonality_mode='multiplicative', daily_seasonality=True 등
        model = Prophet()
        # 모델 학습
        model.fit(df_prophet)
        logging.info("Prophet 모델 학습 완료.")

        # 미래 예측용 DataFrame 생성
        future = model.make_future_dataframe(periods=periods, freq=freq)
        logging.info(f"미래 {periods}{freq} 예측 수행...")
        # 예측 수행
        forecast = model.predict(future)
        logging.info("예측 완료.")

        # 결과 확인 (마지막 예측값)
        logging.debug(f"최근 예측 결과:\n{forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()}")
        return forecast

    except Exception as e:
        # Prophet 또는 cmdstanpy 관련 특정 오류 처리 가능
        logging.error(f"Prophet 예측 중 오류 발생: {e}", exc_info=True)
        return None

def detect_anomaly(
    forecast: pd.DataFrame,
    actual_df: pd.DataFrame,
    price_col: str = 'Close',
    threshold_percent: float = 10.0 # 이상 감지 임계값 (%)
) -> tuple[bool, float, pd.Timestamp | None, float | None, float | None]:
    """예측값과 실제값의 괴리율을 계산하여 이상 현상을 감지합니다.

    Args:
        forecast (pd.DataFrame): Prophet 예측 결과 (ds, yhat 컬럼 포함).
        actual_df (pd.DataFrame): 실제 가격 데이터 (DatetimeIndex 포함).
        price_col (str): 실제 가격 컬럼명.
        threshold_percent (float): 이상 현상으로 판단할 괴리율 임계값 (%).

    Returns:
        tuple[bool, float, pd.Timestamp | None, float | None, float | None]: (이상 현상 여부, 현재 괴리율, 해당 시점, 실제 가격, 예측 가격)
                                                  오류 시 (False, 0.0, None, None, None) 반환.
    """
    if forecast is None or forecast.empty or actual_df is None or actual_df.empty or price_col not in actual_df.columns:
        logging.warning("이상 감지 불가: 예측 또는 실제 데이터 없음.")
        return False, 0.0, None, None, None

    try:
        # 가장 최근 실제 데이터 시점 및 가격 확인
        latest_actual_time = actual_df.index[-1]
        latest_actual_price = actual_df[price_col].iloc[-1]

        # 해당 시점의 예측값 찾기
        # forecast['ds']는 datetime 객체, actual_df.index는 Timestamp 객체일 수 있으므로 타입 일치 필요
        latest_forecast_row = forecast[forecast['ds'] == pd.to_datetime(latest_actual_time)]

        if latest_forecast_row.empty:
            logging.warning(f"이상 감지 불가: {latest_actual_time} 에 해당하는 예측값 없음.")
            return False, 0.0, None, None, None

        latest_predicted_price = latest_forecast_row['yhat'].iloc[0]
        yhat_lower = latest_forecast_row['yhat_lower'].iloc[0]
        yhat_upper = latest_forecast_row['yhat_upper'].iloc[0]

        # 괴리율 계산 (%)
        if latest_predicted_price == 0:
             deviation_percent = float('inf') if latest_actual_price != 0 else 0.0
        else:
             deviation_percent = ((latest_actual_price - latest_predicted_price) / latest_predicted_price) * 100

        logging.info(f"시간: {latest_actual_time}, 실제가: {latest_actual_price:.2f}, 예측가: {latest_predicted_price:.2f} ({yhat_lower:.2f}~{yhat_upper:.2f}), 괴리율: {deviation_percent:.2f}%")

        # 임계값 비교
        is_anomaly = abs(deviation_percent) > threshold_percent

        if is_anomaly:
            direction = "상승" if deviation_percent > 0 else "하락"
            logging.warning(f"🚨 이상 변동 감지! 실제 가격이 예측 범위를 {direction} 방향으로 {abs(deviation_percent):.2f}% 벗어남 (임계값: {threshold_percent}%)")
            # Slack 알림
            if notifier:
                message = f"🚨 [Volatility Alert] 이상 변동 감지!\n" \
                          f"시간: {latest_actual_time}\n" \
                          f"실제가: {latest_actual_price:.2f}\n" \
                          f"예측가: {latest_predicted_price:.2f} ({yhat_lower:.2f}~{yhat_upper:.2f})\n" \
                          f"괴리율: {deviation_percent:.2f}% (임계값: {threshold_percent}%) {direction}"
                notifier.send_slack_notification(message)
            return True, deviation_percent, latest_actual_time, latest_actual_price, latest_predicted_price
        else:
            logging.info("정상 범위 내 변동성.")
            return False, deviation_percent, latest_actual_time, latest_actual_price, latest_predicted_price

    except IndexError:
         logging.error("이상 감지 중 오류: 실제 데이터 또는 예측 데이터 접근 오류 (IndexError)")
         return False, 0.0, None, None, None
    except Exception as e:
        logging.error(f"이상 감지 중 오류 발생: {e}", exc_info=True)
        return False, 0.0, None, None, None

# --- 메인 함수 (실행 예시) --- 
def run_volatility_check(symbol: str, history_days: int = 90, forecast_hours: int = 24, interval: str = '1h', threshold: float = 10.0):
    """특정 심볼의 가격 데이터를 가져와 변동성 이상을 체크하고 DB에 로그를 기록합니다."""
    # DB 로깅 함수 임포트
    log_volatility_to_db_func = None
    try:
        from src.utils.database import log_volatility_to_db as log_volatility_to_db_func
    except ImportError:
        logging.warning("Database logging function (log_volatility_to_db) not found. DB logging disabled.")

    logging.info(f"===== 📈 {symbol} 변동성 체크 시작 =====")

    # 1. 데이터 로드 (strategy_utils 또는 직접 yfinance 사용)
    df_raw = None
    if strategy_utils:
        df_raw = strategy_utils.get_historical_data(symbol, period=f"{history_days}d", interval=interval)
    else:
         # strategy_utils 없으면 yfinance 직접 사용 (예시)
         import yfinance as yf
         try:
              df_raw = yf.download(symbol, start=(datetime.now() - timedelta(days=history_days)).strftime('%Y-%m-%d'), interval=interval)
              if not df_raw.empty:
                   df_raw.index = pd.to_datetime(df_raw.index).tz_localize(None)
                   logging.info(f"{symbol} 데이터 직접 로드 완료 ({len(df_raw)} 행)")
              else:
                   logging.warning(f"{symbol} 데이터 직접 로드 실패.")
                   df_raw = pd.DataFrame()
         except Exception as e:
              logging.error(f"{symbol} 데이터 직접 로드 중 오류: {e}")
              df_raw = pd.DataFrame()

    if df_raw is None or df_raw.empty:
        logging.error(f"{symbol} 데이터 로드 실패. 변동성 체크 중단.")
        # DB 로그 (선택 사항: 데이터 로드 실패 기록)
        # if log_volatility_to_db_func:
        #     log_volatility_to_db_func(symbol=symbol, is_anomaly=None, reason="Data load failed")
        logging.info(f"===== 📈 {symbol} 변동성 체크 종료 =====")
        return

    # 2. Prophet 데이터 준비
    df_prophet = prepare_data_for_prophet(df_raw, price_col='Close')
    if df_prophet.empty:
        logging.error("Prophet 데이터 준비 실패. 변동성 체크 중단.")
        logging.info(f"===== 📈 {symbol} 변동성 체크 종료 =====")
        return

    # 3. 가격 예측
    forecast_result = forecast_price(df_prophet, periods=forecast_hours, freq='H')
    if forecast_result is None:
        logging.error("Prophet 예측 실패. 변동성 체크 중단.")
        logging.info(f"===== 📈 {symbol} 변동성 체크 종료 =====")
        return

    # 4. 이상 감지
    is_anomaly, deviation, check_time, actual_price, predicted_price = detect_anomaly(
        forecast_result, df_raw, price_col='Close', threshold_percent=threshold
    )

    # 5. DB 로그 기록
    if log_volatility_to_db_func and check_time is not None:
        try:
            log_volatility_to_db_func(
                symbol=symbol,
                check_time=check_time,
                is_anomaly=is_anomaly,
                actual_price=actual_price,
                predicted_price=predicted_price,
                deviation_percent=deviation
            )
            logging.info(f"[DB] Volatility log saved for {symbol}.")
        except Exception as db_err:
            logging.error(f"Volatility DB logging failed for {symbol}: {db_err}")

    # TODO: 예측 결과 시각화 (선택 사항)
    # try:
    #     fig = model.plot(forecast_result)
    #     fig.savefig(f"{symbol}_prophet_forecast.png")
    #     fig2 = model.plot_components(forecast_result)
    #     fig2.savefig(f"{symbol}_prophet_components.png")
    # except Exception as plot_err:
    #     logging.warning(f"Prophet 예측 결과 시각화 실패: {plot_err}")

    logging.info(f"===== 📈 {symbol} 변동성 체크 종료 =====")


if __name__ == "__main__":
    # 테스트 실행
    # Bitcoin 1시간 봉 데이터로 90일 학습 후 24시간 예측, 5% 이상 괴리 시 알림
    run_volatility_check(symbol="BTC-USD", history_days=90, forecast_hours=24, interval='1h', threshold=5.0)

    # SPY 일봉 데이터로 365일 학습 후 30일 예측, 10% 이상 괴리 시 알림
    # run_volatility_check(symbol="SPY", history_days=365, forecast_hours=30*24, interval='1d', threshold=10.0) # freq='D' 필요? 