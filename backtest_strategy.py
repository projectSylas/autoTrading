import backtrader as bt
import pandas as pd
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)

# 전략 유틸리티 임포트
try:
    import strategy_utils
except ImportError:
    logging.error("strategy_utils.py 를 찾을 수 없습니다.")
    strategy_utils = None

# config 임포트 (챌린지 전략 파라미터 사용)
try:
    import config
except ImportError:
     logging.error("config.py 모듈을 찾을 수 없습니다.")
     # 기본값 사용 또는 오류 발생
     class ConfigFallback:
         CHALLENGE_SMA_PERIOD = 7
         CHALLENGE_TP_PERCENT = 0.12
         CHALLENGE_SL_PERCENT = 0.05
     config = ConfigFallback()

class FlightBacktestStrategy(bt.Strategy):
    """플라이트 챌린지 매매법 기반 백테스트 전략 (backtrader)

    진입 조건 (예시):
    - RSI < 30 (Long)
    - 7일 SMA 하방 이탈 + 거래량 급증 (Short)
    손익 조건:
    - TP: +12%
    - SL: -5%
    """
    params = (
        # 지표 파라미터
        ('rsi_period', 14),
        ('sma_short_period', 7),
        ('sma_long_period', 20),
        ('volume_sma_period', 20),
        ('divergence_window', 14),
        ('trendline_window', 30),
        ('sr_window', 30),
        ('volume_spike_factor', 2.0),
        # 손익 관리 파라미터
        ('tp_ratio', 0.10), # 10% 익절
        ('sl_ratio', 0.05), # 5% 손절
        # 주문 관련 파라미터
        ('stake', 1), # 기본 주문 수량 (예: 1 계약)
        # ('cash_portion', 0.1) # 또는 가용 현금의 비율로 주문
    )

    def log(self, txt, dt=None):
        ''' 로깅 함수 '''
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} - {txt}')

    def __init__(self):
        # 데이터 라인 참조
        self.dataclose = self.datas[0].close
        self.datahigh = self.datas[0].high
        self.datalow = self.datas[0].low
        self.datavolume = self.datas[0].volume
        self.dataopen = self.datas[0].open

        # 주문 추적용 변수
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.bar_executed = 0 # 마지막 거래 발생 시점

        # 지표 저장용 변수 (next에서 계산 후 저장)
        self.rsi = None
        self.sma_short = None
        self.sma_long = None
        self.volume_sma = None
        self.rsi_divergence = None
        self.volume_spike = None
        self.sr_levels = {}
        self.trend_event = None

        # 로깅 간소화 (선택적)
        # self.log("Strategy Initialized")

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # 주문 접수/수락 상태 - 특별한 동작 없음
            return

        # 주문 완료 확인
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm {order.executed.comm:.2f}'
                )
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            elif order.issell():
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm {order.executed.comm:.2f}')

            self.bar_executed = len(self) # 거래 발생 시점 기록

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'Order Canceled/Margin/Rejected - Status: {order.getstatusname()}')

        # 주문 상태 초기화 (다른 주문 가능하도록)
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log(f'OPERATION PROFIT, GROSS {trade.pnl:.2f}, NET {trade.pnlcomm:.2f}')

    def get_data_as_dataframe(self, lookback) -> pd.DataFrame:
        """최근 lookback 기간의 데이터를 pandas DataFrame으로 변환"""
        dates = [self.datas[0].datetime.datetime(-i) for i in range(lookback)]
        closes = [self.dataclose[-i] for i in range(lookback)]
        highs = [self.datahigh[-i] for i in range(lookback)]
        lows = [self.datalow[-i] for i in range(lookback)]
        opens = [self.dataopen[-i] for i in range(lookback)]
        volumes = [self.datavolume[-i] for i in range(lookback)]

        df = pd.DataFrame({
            'Open': opens[::-1],
            'High': highs[::-1],
            'Low': lows[::-1],
            'Close': closes[::-1],
            'Volume': volumes[::-1]
        }, index=pd.to_datetime(dates[::-1]))
        return df

    def next(self):
        # 현재 날짜 로깅 (디버깅용)
        # self.log(f'Close, {self.dataclose[0]:.2f}')

        # 진행 중인 주문 있으면 대기
        if self.order:
            return

        # 지표 계산에 필요한 최소 데이터 확인
        required_bars = max(self.p.rsi_period, self.p.sma_long_period, self.p.volume_sma_period,
                            self.p.divergence_window, self.p.trendline_window, self.p.sr_window) + 1 # 최소 2개 봉은 필요
        if len(self) < required_bars:
            return

        # --- 지표 계산 --- #
        try:
            # 최근 데이터 DataFrame으로 변환 (계산에 필요한 충분한 기간 사용)
            df_indicators = self.get_data_as_dataframe(required_bars)
            if df_indicators.empty:
                 return # 데이터 변환 실패 시 중단

            # strategy_utils 함수 호출 (DataFrame 전달)
            self.rsi = strategy_utils.calculate_rsi(df_indicators, window=self.p.rsi_period).iloc[-1]
            self.sma_short = strategy_utils.calculate_sma(df_indicators, window=self.p.sma_short_period).iloc[-1]
            self.sma_long = strategy_utils.calculate_sma(df_indicators, window=self.p.sma_long_period).iloc[-1]
            self.volume_sma = df_indicators['Volume'].rolling(window=self.p.volume_sma_period).mean().iloc[-1]

            # 복합 지표 계산 (데이터프레임과 RSI Series 전달)
            self.rsi_divergence = strategy_utils.detect_rsi_divergence(df_indicators['Close'], strategy_utils.calculate_rsi(df_indicators, window=self.p.rsi_period), window=self.p.divergence_window)
            self.volume_spike = strategy_utils.detect_volume_spike(df_indicators, window=self.p.volume_sma_period, factor=self.p.volume_spike_factor)
            self.sr_levels = strategy_utils.detect_support_resistance_by_volume(df_indicators, window=self.p.sr_window)
            self.trend_event = strategy_utils.detect_trendline_breakout(df_indicators, window=self.p.trendline_window)

        except Exception as e:
            self.log(f"지표 계산 중 오류: {e}")
            return

        # 지표 값 확인 (디버깅용)
        # self.log(f"RSI: {self.rsi:.2f}, SMA7: {self.sma_short:.2f}, Div: {self.rsi_divergence}, Spike: {self.volume_spike}")
        # self.log(f"SR: {self.sr_levels}, Trend: {self.trend_event}")

        current_price = self.dataclose[0]
        support = self.sr_levels.get('support', [])
        resistance = self.sr_levels.get('resistance', [])

        # --- 진입/청산 로직 --- #

        # 현재 포지션 없음 -> 신규 진입 시도
        if not self.position:
            long_conditions_met = []
            short_conditions_met = []

            # Long 조건 확인
            if self.rsi_divergence == 'bullish': long_conditions_met.append("Div")
            if support and abs(current_price - support[0]) / support[0] < 0.01 and current_price > self.dataclose[-1] and self.volume_spike:
                long_conditions_met.append(f"Supp({support[0]:.0f})+Spike")
            if current_price > self.sma_short and self.sma_short > self.data.sma_short[-1]: long_conditions_met.append("SMA7>Rise") # sma_short는 내부 계산 필요
            if self.trend_event and self.trend_event.get('type') == 'retest_support': long_conditions_met.append(f"TrendSupp({self.trend_event.get('price'):.0f})")

            # Short 조건 확인
            if self.rsi_divergence == 'bearish': short_conditions_met.append("Div")
            if resistance and abs(current_price - resistance[0]) / resistance[0] < 0.01 and current_price < self.dataclose[-1] and self.volume_spike:
                short_conditions_met.append(f"Resist({resistance[0]:.0f})+Spike")
            if current_price < self.sma_short and self.sma_short < self.data.sma_short[-1]: short_conditions_met.append("SMA7<Fall")
            if self.trend_event and self.trend_event.get('type') == 'retest_resistance': short_conditions_met.append(f"TrendResist({self.trend_event.get('price'):.0f})")

            # Long 진입 결정
            if "Div" in long_conditions_met or len(long_conditions_met) >= 2:
                entry_side = 'long'
                entry_reason = ",".join(long_conditions_met)
            # Short 진입 결정
            elif "Div" in short_conditions_met or len(short_conditions_met) >= 1:
                 entry_side = 'short' # 여기가 elif 로 연결되어야 함
                 entry_reason = ",".join(short_conditions_met)

            # 진입 주문 실행
            if entry_side:
                self.log(f'{entry_side.upper()} Signal ({entry_reason}). Creating Order...')
                size = self.p.stake # 고정 수량 사용
                # TODO: 자금 비율 기반 수량 계산 로직 추가 가능
                # cash = self.broker.getcash()
                # size = (cash * self.p.cash_portion) / current_price ...

                if entry_side == 'long':
                    self.order = self.buy(size=size)
                    # TP/SL 주문 동시 제출 (OCO 미지원 시)
                    tp_price = current_price * (1 + self.p.tp_ratio)
                    sl_price = current_price * (1 - self.p.sl_ratio)
                    # self.sell(...) 호출 전 transmit 파라미터 확인 필요 (backtrader 버전따라)
                    self.sell(exectype=bt.Order.Limit, price=tp_price, size=size, transmit=False, parent=self.order) # TP, parent 지정
                    self.order = self.sell(exectype=bt.Order.Stop, price=sl_price, size=size, transmit=True, parent=self.order) # SL, parent 지정
                elif entry_side == 'short':
                     self.order = self.sell(size=size)
                     tp_price = current_price * (1 - self.p.tp_ratio)
                     sl_price = current_price * (1 + self.p.sl_ratio)
                     self.buy(exectype=bt.Order.Limit, price=tp_price, size=size, transmit=False, parent=self.order) # TP
                     self.order = self.buy(exectype=bt.Order.Stop, price=sl_price, size=size, transmit=True, parent=self.order) # SL

        # 이미 포지션 보유 중 -> 여기서는 별도 로직 없음 (진입 시 TP/SL 설정됨)
        # 또는 여기서 Trailing Stop 등 로직 추가 가능
        else:
             pass

    def notify_trade(self, trade):
        """거래(매수/매도 쌍) 완료 시 호출"""
        if not trade.isclosed:
            return

        self.log(f'OPERATION PROFIT, GROSS {trade.pnl:.2f}, NET {trade.pnlcomm:.2f}')
        # 숏 포지션 정보 초기화 (거래 완료 시)
        self.sellprice = None
        self.sellcomm = None

    def next(self):
        """매 봉마다 호출되는 메인 로직"""
        # 현재 로그 출력
        # self.log(f'Close: {self.dataclose[0]:.2f}, SMA: {self.sma[0]:.2f}, RSI: {self.rsi[0]:.2f}, Vol: {self.datavolume[0]:.0f}')

        # 진행 중인 주문이 있는지 확인
        if self.order:
            return

        # 포지션이 없는 경우 진입 조건 확인
        if not self.position:
            # 조건 1: RSI 과매도 -> Long 진입
            if self.rsi[0] < self.params.rsi_oversold:
                self.log(f'LONG ENTRY SIGNAL: RSI Oversold ({self.rsi[0]:.2f} < {self.params.rsi_oversold})')
                # TODO: 포지션 크기 계산 로직 추가 (레버리지 반영)
                size = 1 # 임시 고정 크기
                self.order = self.buy(size=size)
                return # 진입 주문 후 다음 봉 대기

            # 조건 2: 7일 SMA 하방 이탈 + 거래량 급증 -> Short 진입
            current_volume = self.datavolume[0]
            avg_volume = self.volume_avg[0]
            is_volume_spike = current_volume > avg_volume * self.params.volume_factor

            if self.dataclose[0] < self.sma[0] and self.dataclose[-1] >= self.sma[-1] and is_volume_spike:
                 self.log(f'SHORT ENTRY SIGNAL: SMA Cross Down ({self.dataclose[0]:.2f} < {self.sma[0]:.2f}) + Volume Spike ({current_volume:.0f} > {avg_volume:.0f}*{self.params.volume_factor})')
                 # TODO: 포지션 크기 계산 로직 추가 (레버리지 반영)
                 size = 1 # 임시 고정 크기
                 self.order = self.sell(size=size)
                 return # 진입 주문 후 다음 봉 대기

            # TODO: 추세선 이탈/되돌림, 다이버전스 등 다른 조건 추가

        # 포지션이 있는 경우 (TP/SL은 notify_order에서 처리됨)
        else:
             # 포지션 보유 시 로깅 또는 추가 로직 (예: 트레일링 스탑)
             pass

    def stop(self):
        """전략 종료 시 호출"""
        self.log(f'(SMA {self.params.sma_period}, RSI {self.params.rsi_period}) Ending Value {self.broker.getvalue():.2f}', doprint=True)

# backtrader 실행 및 결과 출력/시각화 로직은 backtest_runner.py 에서 구현 