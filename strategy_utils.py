import yfinance as yf
import pandas as pd
import ta
import logging
import numpy as np # 다이버전스 등에 사용
from scipy.signal import find_peaks # RSI 다이버전스 감지용

# 로깅 설정 (다른 모듈에서도 사용 가능하도록)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_historical_data(symbol: str, period: str = "1mo", interval: str = "1d") -> pd.DataFrame:
    """yfinance를 사용하여 지정된 종목의 과거 데이터를 가져옵니다."""
    try:
        ticker = yf.Ticker(symbol)
        # 데이터 로드 시 auto_adjust=True 사용 고려 (수정 종가 기준)
        hist = ticker.history(period=period, interval=interval, auto_adjust=False) # auto_adjust=False로 OHLC 유지
        if hist.empty:
            logging.warning(f"{symbol}: yfinance에서 과거 데이터를 가져올 수 없습니다. (기간: {period}, 간격: {interval})")
            return pd.DataFrame() # 빈 DataFrame 반환
        # 타임존 정보 제거 (호환성 문제 방지)
        if isinstance(hist.index, pd.DatetimeIndex):
             hist.index = hist.index.tz_localize(None)
        logging.info(f"{symbol}: yfinance 과거 데이터 로드 완료 ({len(hist)} 행)")
        return hist
    except Exception as e:
        logging.error(f"{symbol} 과거 데이터 조회 중 오류 발생: {e}")
        return pd.DataFrame() # 오류 시 빈 DataFrame 반환

def get_current_vix(period: str = "5d") -> float | None:
    """VIX 지수(^VIX)의 현재 값을 가져옵니다."""
    hist = get_historical_data("^VIX", period=period)
    if not hist.empty and 'Close' in hist.columns:
        try:
            current_vix = hist['Close'].iloc[-1]
            logging.info(f"현재 VIX 지수: {current_vix:.2f}")
            return float(current_vix)
        except (IndexError, ValueError) as e:
             logging.error(f"VIX 데이터 처리 중 오류 발생: {e}")
             return None
    else:
        logging.warning("VIX 데이터를 가져오지 못했습니다.")
        return None

def calculate_rsi(
    symbol: str | None = None,
    dataframe: pd.DataFrame | None = None,
    period: str = "1mo",
    window: int = 14,
    column: str = 'Close'
) -> pd.Series | None: # 마지막 값 대신 전체 Series 반환하도록 변경 고려 -> 다이버전스 등에 활용 용이
    """주어진 DataFrame 또는 심볼의 RSI 값을 계산하여 Series로 반환합니다."""
    if dataframe is None:
        if symbol is None:
            raise ValueError("RSI 계산을 위해 symbol 또는 dataframe 중 하나는 제공되어야 합니다.")
        hist = get_historical_data(symbol, period=period)
    else:
        hist = dataframe.copy() # 원본 데이터프레임 변경 방지

    if not hist.empty and column in hist.columns:
        try:
            # NaN 값 처리 (fillna 또는 dropna) - RSI 계산 전 처리 필요
            hist_cleaned = hist.dropna(subset=[column])
            if hist_cleaned.empty:
                 logging.warning(f"{symbol or 'DataFrame'}: '{column}' 데이터가 없어 RSI 계산 불가.")
                 return None
            if len(hist_cleaned) < window:
                 logging.warning(f"{symbol or 'DataFrame'}: RSI 계산 불가 (데이터 포인트 {len(hist_cleaned)} < window {window})")
                 return None

            rsi_indicator = ta.momentum.RSIIndicator(hist_cleaned[column], window=window)
            rsi_series = rsi_indicator.rsi()

            if not rsi_series.empty:
                logging.info(f"{symbol or 'DataFrame'} RSI ({window} 기간) 계산 완료 (마지막 값: {rsi_series.iloc[-1]:.2f})")
                return rsi_series # 전체 Series 반환
            else:
                logging.warning(f"{symbol or 'DataFrame'}: RSI 계산 결과 없음.")
                return None
        except Exception as e:
            logging.error(f"{symbol or 'DataFrame'} RSI 계산 중 오류 발생: {e}")
            return None
    else:
        logging.warning(f"{symbol or 'DataFrame'}: RSI 계산을 위한 '{column}' 컬럼이 없거나 데이터가 비어있습니다.")
        return None

def calculate_sma(
    symbol: str | None = None,
    dataframe: pd.DataFrame | None = None,
    period: str = "1mo",
    interval: str = "1d",
    window: int = 7,
    column: str = 'Close'
) -> pd.Series | None: # 마지막 값 대신 전체 Series 반환하도록 변경 고려 -> 추세 비교 등에 활용 용이
    """주어진 DataFrame 또는 심볼의 SMA 값을 계산하여 Series로 반환합니다."""
    if dataframe is None:
        if symbol is None:
             raise ValueError("SMA 계산을 위해 symbol 또는 dataframe 중 하나는 제공되어야 합니다.")
        # yfinance 데이터 조회 시에는 interval 파라미터 사용
        hist = get_historical_data(symbol, period=period, interval=interval)
    else:
        hist = dataframe.copy() # 원본 데이터프레임 변경 방지

    if not hist.empty and column in hist.columns:
        try:
             # NaN 값 처리
            hist_cleaned = hist.dropna(subset=[column])
            if hist_cleaned.empty:
                 logging.warning(f"{symbol or 'DataFrame'}: '{column}' 데이터가 없어 SMA 계산 불가.")
                 return None
            # 데이터 포인트가 window 크기보다 작으면 SMA 계산 불가
            if len(hist_cleaned) < window:
                 logging.warning(f"{symbol or 'DataFrame'}: SMA 계산 불가 (데이터 포인트 {len(hist_cleaned)} < window {window})")
                 return None

            sma_indicator = ta.trend.SMAIndicator(hist_cleaned[column], window=window)
            sma_series = sma_indicator.sma_indicator()

            if not sma_series.empty:
                logging.info(f"{symbol or 'DataFrame'} SMA ({window} 기간) 계산 완료 (마지막 값: {sma_series.iloc[-1]:.4f})")
                return sma_series # 전체 Series 반환
            else:
                logging.warning(f"{symbol or 'DataFrame'}: SMA 계산 결과 없음.")
                return None
        except Exception as e:
            logging.error(f"{symbol or 'DataFrame'} SMA 계산 중 오류 발생: {e}")
            return None
    else:
        logging.warning(f"{symbol or 'DataFrame'}: SMA 계산을 위한 '{column}' 컬럼이 없거나 데이터가 비어있습니다.")
        return None

# --- RSI Divergence Detection ---
def detect_rsi_divergence(prices: pd.Series, rsi: pd.Series, window: int = 14) -> str | None:
    """주어진 가격과 RSI 시리즈에서 최근 N개 봉 기준 다이버전스를 감지합니다.

    Args:
        prices (pd.Series): 종가 시리즈.
        rsi (pd.Series): RSI 시리즈.
        window (int): 다이버전스를 확인할 최근 기간 (봉 개수).

    Returns:
        str | None: 'bullish', 'bearish', 또는 None (다이버전스 없음).
    """
    if prices.empty or rsi.empty or len(prices) < window or len(rsi) < window:
        logging.debug("다이버전스 감지 불가: 데이터 부족.")
        return None

    # 최근 window 기간 데이터만 사용
    prices_window = prices.tail(window)
    rsi_window = rsi.tail(window)

    # 가격과 RSI의 고점(peaks) 찾기 (scipy.signal.find_peaks 사용)
    # prominence: 봉우리 높이의 중요도 (주변 대비 얼마나 튀어나왔는지)
    # distance: 봉우리 간 최소 거리
    peak_prominence = (prices_window.max() - prices_window.min()) * 0.05 # 예: 가격 변동폭의 5% 이상인 피크만
    peak_distance = 3 # 예: 최소 3개 봉 간격

    price_peaks_indices, _ = find_peaks(prices_window, prominence=peak_prominence, distance=peak_distance)
    rsi_peaks_indices, _ = find_peaks(rsi_window, prominence=1, distance=peak_distance) # RSI는 1 이상 변화면 유의미하다고 가정

    # 가격과 RSI의 저점(troughs) 찾기 (음수로 변환하여 고점 찾기)
    trough_prominence = peak_prominence # 동일 기준 적용
    trough_distance = peak_distance

    price_troughs_indices, _ = find_peaks(-prices_window, prominence=trough_prominence, distance=trough_distance)
    rsi_troughs_indices, _ = find_peaks(-rsi_window, prominence=1, distance=trough_distance)

    # --- 하락형 다이버전스 (Bearish Divergence) 확인 --- #
    # 조건: 가격은 고점을 높이는데 (Higher High), RSI는 고점을 낮춤 (Lower High)
    # 최근 두 개의 유의미한 고점을 비교
    if len(price_peaks_indices) >= 2 and len(rsi_peaks_indices) >= 2:
        # 가장 최근 두 개의 가격 고점
        last_price_peak_idx = price_peaks_indices[-1]
        prev_price_peak_idx = price_peaks_indices[-2]
        # 해당 시점들의 RSI 값 (find_peaks는 원래 시리즈의 인덱스를 반환하지 않으므로, window 내 인덱스로 접근)
        # 실제로는 인덱스 매칭이 더 견고해야 함 (가장 가까운 RSI 피크 찾기 등)
        # 여기서는 단순화하여 동일 인덱스 가정
        if last_price_peak_idx in rsi_window.index and prev_price_peak_idx in rsi_window.index:
             last_rsi_at_price_peak = rsi_window.loc[prices_window.index[last_price_peak_idx]]
             prev_rsi_at_price_peak = rsi_window.loc[prices_window.index[prev_price_peak_idx]]

             # 가격 고점 높아짐 & RSI 고점 낮아짐
             if prices_window.iloc[last_price_peak_idx] > prices_window.iloc[prev_price_peak_idx] and \
                last_rsi_at_price_peak < prev_rsi_at_price_peak:
                 logging.debug(f"하락형 다이버전스 감지 가능성: 가격 HH ({prices_window.index[prev_price_peak_idx].date()} -> {prices_window.index[last_price_peak_idx].date()}), RSI LH")
                 # 추가 검증 로직 가능 (예: RSI 피크 인덱스와 가격 피크 인덱스 일치 여부 등)
                 # 가장 최근 봉이 마지막 피크 근처에 있을 때만 유효하다고 판단할 수도 있음
                 if window - last_price_peak_idx <= 3: # 예: 마지막 피크가 최근 3봉 이내
                      logging.info("🐻 하락형 다이버전스 (Bearish Divergence) 감지됨.")
                      return 'bearish'

    # --- 상승형 다이버전스 (Bullish Divergence) 확인 --- #
    # 조건: 가격은 저점을 낮추는데 (Lower Low), RSI는 저점을 높임 (Higher Low)
    # 최근 두 개의 유의미한 저점을 비교
    if len(price_troughs_indices) >= 2 and len(rsi_troughs_indices) >= 2:
        # 가장 최근 두 개의 가격 저점
        last_price_trough_idx = price_troughs_indices[-1]
        prev_price_trough_idx = price_troughs_indices[-2]
        # 해당 시점들의 RSI 값 (단순 인덱스 매칭 가정)
        if last_price_trough_idx in rsi_window.index and prev_price_trough_idx in rsi_window.index:
             last_rsi_at_price_trough = rsi_window.loc[prices_window.index[last_price_trough_idx]]
             prev_rsi_at_price_trough = rsi_window.loc[prices_window.index[prev_price_trough_idx]]

             # 가격 저점 낮아짐 & RSI 저점 높아짐
             if prices_window.iloc[last_price_trough_idx] < prices_window.iloc[prev_price_trough_idx] and \
                last_rsi_at_price_trough > prev_rsi_at_price_trough:
                 logging.debug(f"상승형 다이버전스 감지 가능성: 가격 LL ({prices_window.index[prev_price_trough_idx].date()} -> {prices_window.index[last_price_trough_idx].date()}), RSI HL")
                 # 추가 검증 로직
                 if window - last_price_trough_idx <= 3: # 예: 마지막 저점이 최근 3봉 이내
                     logging.info("🐂 상승형 다이버전스 (Bullish Divergence) 감지됨.")
                     return 'bullish'

    # 다이버전스 없음
    logging.debug("다이버전스 감지되지 않음.")
    return None

# --- Trendline Detection (Simplified Linear Regression Approach) ---
def detect_trendline_breakout(df: pd.DataFrame, window: int = 30, peak_distance: int = 5) -> dict | None:
    """가격 데이터에서 단순 추세선(상단/하단) 이탈 및 되돌림을 감지합니다.

    Args:
        df (pd.DataFrame): 가격 데이터 ('High', 'Low', 'Close' 포함).
        window (int): 추세선을 분석할 최근 기간 (봉 개수).
        peak_distance (int): 고점/저점 식별 시 최소 거리.

    Returns:
        dict | None: 감지된 이벤트 정보 (예: {'type': 'breakout_up', 'trendline': 'upper'}) 또는 None.
                     반환 타입 예시:
                     - breakout_up: 상단 추세선 상향 돌파
                     - breakout_down: 하단 추세선 하향 돌파
                     - retest_resistance: 상단 추세선 저항 확인 (돌파 실패)
                     - retest_support: 하단 추세선 지지 확인 (돌파 실패)
    """
    if df.empty or len(df) < window:
        logging.debug("추세선 분석 불가: 데이터 부족.")
        return None

    try:
        data = df.tail(window).copy()
        data['index_num'] = np.arange(len(data)) # 시간 대신 숫자 인덱스 사용

        # 고점/저점 찾기
        highs = data['High']
        lows = data['Low']
        # prominence는 가격 변동성에 따라 조절 필요
        prominence = (highs.max() - lows.min()) * 0.03 # 변동폭의 3% 정도

        peak_indices, _ = find_peaks(highs, distance=peak_distance, prominence=prominence)
        trough_indices, _ = find_peaks(-lows, distance=peak_distance, prominence=prominence)

        upper_trend = None
        lower_trend = None

        # 상단 추세선 (최소 2개 고점 필요)
        if len(peak_indices) >= 2:
            peak_data = data.iloc[peak_indices]
            # 선형 회귀: y = slope * x + intercept (y=High, x=index_num)
            slope, intercept = np.polyfit(peak_data['index_num'], peak_data['High'], 1)
            upper_trend = {'slope': slope, 'intercept': intercept}
            logging.debug(f"상단 추세선 추정: 기울기={slope:.4f}, 절편={intercept:.2f}")

        # 하단 추세선 (최소 2개 저점 필요)
        if len(trough_indices) >= 2:
            trough_data = data.iloc[trough_indices]
            slope, intercept = np.polyfit(trough_data['index_num'], trough_data['Low'], 1)
            lower_trend = {'slope': slope, 'intercept': intercept}
            logging.debug(f"하단 추세선 추정: 기울기={slope:.4f}, 절편={intercept:.2f}")

        # 현재 상태 판단
        current_index_num = data['index_num'].iloc[-1]
        current_close = data['Close'].iloc[-1]
        prev_close = data['Close'].iloc[-2]

        event = None

        # 상단 추세선 관련 이벤트 확인
        if upper_trend:
            current_trend_price = upper_trend['slope'] * current_index_num + upper_trend['intercept']
            prev_trend_price = upper_trend['slope'] * (current_index_num - 1) + upper_trend['intercept']
            logging.debug(f"현재 상단 추세선 가격: {current_trend_price:.2f}")

            # 상향 돌파 (이전 종가 <= 이전 추세선 가격, 현재 종가 > 현재 추세선 가격)
            if prev_close <= prev_trend_price and current_close > current_trend_price:
                logging.info("📈 상단 추세선 상향 돌파 감지.")
                event = {'type': 'breakout_up', 'trendline': 'upper', 'price': current_trend_price}
            # 저항 확인 (현재 고점 > 추세선 가격, 그러나 종가는 추세선 아래)
            elif data['High'].iloc[-1] > current_trend_price and current_close < current_trend_price:
                 logging.info("📉 상단 추세선 저항 확인 (돌파 실패) 감지.")
                 event = {'type': 'retest_resistance', 'trendline': 'upper', 'price': current_trend_price}

        # 하단 추세선 관련 이벤트 확인 (상단 이벤트 없을 때만)
        if not event and lower_trend:
            current_trend_price = lower_trend['slope'] * current_index_num + lower_trend['intercept']
            prev_trend_price = lower_trend['slope'] * (current_index_num - 1) + lower_trend['intercept']
            logging.debug(f"현재 하단 추세선 가격: {current_trend_price:.2f}")

            # 하향 돌파 (이전 종가 >= 이전 추세선 가격, 현재 종가 < 현재 추세선 가격)
            if prev_close >= prev_trend_price and current_close < current_trend_price:
                logging.info("📉 하단 추세선 하향 돌파 감지.")
                event = {'type': 'breakout_down', 'trendline': 'lower', 'price': current_trend_price}
            # 지지 확인 (현재 저점 < 추세선 가격, 그러나 종가는 추세선 위)
            elif data['Low'].iloc[-1] < current_trend_price and current_close > current_trend_price:
                 logging.info("📈 하단 추세선 지지 확인 (돌파 실패) 감지.")
                 event = {'type': 'retest_support', 'trendline': 'lower', 'price': current_trend_price}

        return event

    except Exception as e:
        logging.error(f"추세선 분석 중 오류: {e}", exc_info=True)
        return None

# --- Volume Spike Detection ---
def detect_volume_spike(df: pd.DataFrame, window: int = 20, factor: float = 2.0) -> bool:
    """최근 거래량이 평균 대비 급증했는지 확인합니다."""
    if df.empty or 'Volume' not in df.columns or len(df) < window + 1:
        return False
    try:
        avg_volume = df['Volume'].rolling(window=window).mean().iloc[-2] # 직전 평균
        current_volume = df['Volume'].iloc[-1]
        is_spike = current_volume > avg_volume * factor
        logging.debug(f"거래량 급증 확인: 현재={current_volume:.0f}, 평균({window}봉)={avg_volume:.0f}, 기준={factor}배, 결과={is_spike}")
        return is_spike
    except Exception as e:
        logging.error(f"거래량 급증 확인 중 오류: {e}")
        return False

# --- Support/Resistance by Volume (VPVR Approximation) ---
def detect_support_resistance_by_volume(df: pd.DataFrame, window: int = 30, n_levels: int = 5) -> dict:
    """지정된 기간 동안 거래량이 많은 가격대를 찾아 지지/저항 수준으로 반환합니다. (VPVR 근사)

    TODO: 보다 정교한 VPVR 계산 로직 적용 가능.
          현재는 가격 구간별 거래량 합산 방식 사용.
    """
    if df.empty or 'Close' not in df.columns or 'Volume' not in df.columns or len(df) < window:
        return {'support': [], 'resistance': []}

    try:
        window_data = df.tail(window).copy()
        # 가격 범위를 몇 개의 구간(bin)으로 나눌지 결정 (예: 100개)
        num_bins = 50
        price_min = window_data['Close'].min()
        price_max = window_data['Close'].max()
        if price_max == price_min: return {'support': [], 'resistance': []} # 가격 변동 없으면 계산 불가

        bins = pd.cut(window_data['Close'], bins=num_bins)
        volume_by_price = window_data.groupby(bins)['Volume'].sum()

        # 거래량이 많은 상위 N개 구간 찾기
        top_levels_series = volume_by_price.nlargest(n_levels)

        support_levels = []
        resistance_levels = []
        current_price = df['Close'].iloc[-1]

        # 각 레벨의 중간값을 지지/저항으로 분류
        for interval, volume in top_levels_series.items():
            level_price = interval.mid # 구간의 중간 가격
            if level_price < current_price:
                support_levels.append(round(level_price, 2))
            else:
                resistance_levels.append(round(level_price, 2))

        support_levels.sort(reverse=True) # 높은 가격부터
        resistance_levels.sort() # 낮은 가격부터

        logging.debug(f"거래량 기반 지지/저항 ({window}봉, {n_levels}개): 지지={support_levels}, 저항={resistance_levels}")
        return {'support': support_levels, 'resistance': resistance_levels}

    except Exception as e:
        logging.error(f"거래량 기반 지지/저항 계산 중 오류: {e}", exc_info=True)
        return {'support': [], 'resistance': []}

# --- Save Log to CSV --- (core_portfolio.py 에서 복사)
def save_log_to_csv(log_entry: dict, log_file: str):
    """딕셔너리 형태의 로그 항목을 CSV 파일에 추가합니다."""
    try:
        df_entry = pd.DataFrame([log_entry])
        # 파일 존재 및 비어있는지 여부 확인하여 헤더 추가 결정
        header = not os.path.exists(log_file) or os.path.getsize(log_file) == 0
        # mode='a'로 append, index=False로 인덱스 제외, encoding 명시
        df_entry.to_csv(log_file, mode='a', header=header, index=False, encoding='utf-8-sig')
        # logging.debug(f"로그 저장 완료: {log_file}, 내용: {log_entry}") # 로그량이 많을 수 있으므로 debug 레벨
    except Exception as e:
        logging.error(f"CSV 로그 파일 '{log_file}' 저장 중 오류 발생: {e}")


if __name__ == "__main__":
    # 모듈 단독 실행 시 테스트용
    logging.info("--- Strategy Utils 테스트 --- ")
    # VIX 테스트
    vix = get_current_vix()
    if vix is not None:
        logging.info(f"테스트 VIX 결과: {vix}")

    # RSI 테스트 (SPY 심볼 사용)
    spy_rsi_series = calculate_rsi(symbol="SPY", period="3mo", window=14)
    if spy_rsi_series is not None:
        logging.info(f"테스트 SPY RSI 결과 (symbol): 마지막 값 {spy_rsi_series.iloc[-1]:.2f}")

    # SMA 테스트 (BTC-USD 심볼 사용, 1시간 봉)
    btc_sma_series = calculate_sma(symbol="BTC-USD", period="7d", interval="1h", window=7)
    if btc_sma_series is not None:
        logging.info(f"테스트 BTC-USD SMA 결과 (symbol, 1h): 마지막 값 {btc_sma_series.iloc[-1]:.2f}")

    # DataFrame 직접 전달 테스트
    logging.info("--- DataFrame 전달 테스트 --- ")
    try:
        # BTC-USD 1시간 봉 데이터 가져오기
        btc_df = get_historical_data("BTC-USD", period="30d", interval="1h") # 데이터 기간 늘림
        if not btc_df.empty:
            # DataFrame으로 SMA 계산 (7일 = 168시간)
            btc_sma_series_df = calculate_sma(dataframe=btc_df, window=7*24)
            if btc_sma_series_df is not None:
                logging.info(f"테스트 BTC-USD 7일 SMA 결과 (DataFrame, 1h): 마지막 값 {btc_sma_series_df.iloc[-1]:.2f}")

            # DataFrame으로 RSI 계산
            btc_rsi_series_df = calculate_rsi(dataframe=btc_df, window=14)
            if btc_rsi_series_df is not None:
                 logging.info(f"테스트 BTC-USD 14 RSI 결과 (DataFrame, 1h): 마지막 값 {btc_rsi_series_df.iloc[-1]:.2f}")

            # --- 신규 함수 테스트 ---
            if btc_rsi_series_df is not None:
                 # RSI 다이버전스 테스트
                divergence = detect_rsi_divergence(btc_df['Close'], btc_rsi_series_df, window=24*7) # 최근 7일 데이터로 확인
                logging.info(f"테스트 BTC-USD RSI 다이버전스 결과 (7일, 단순): {divergence}")

            # 추세선 테스트 (Placeholder)
            trend_info = detect_trendline_breakout(btc_df, window=24*3) # 최근 3일
            logging.info(f"테스트 BTC-USD 추세선 결과 (3일, Placeholder): {trend_info}")

            # 거래량 급증 테스트
            volume_spike = detect_volume_spike(btc_df, window=24, factor=2.5) # 최근 1일 평균 대비 2.5배
            logging.info(f"테스트 BTC-USD 거래량 급증 결과 (1일 평균 대비 2.5배): {volume_spike}")

            # 지지/저항 테스트
            sr_levels = detect_support_resistance_by_volume(btc_df, window=24*14, n_levels=5) # 최근 2주 데이터, 5개 레벨
            logging.info(f"테스트 BTC-USD 거래량 기반 지지/저항 결과 (2주): {sr_levels}")

        else:
            logging.warning("테스트용 BTC-USD DataFrame 생성 실패")
    except Exception as e:
        logging.error(f"DataFrame 테스트 중 오류 발생: {e}")


    logging.info("--- 테스트 종료 --- ") 