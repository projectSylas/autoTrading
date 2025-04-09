import backtrader as bt
import yfinance as yf
import pandas as pd
import logging
from datetime import datetime

# 백테스트 전략 클래스 임포트
try:
    from backtest_strategy import FlightBacktestStrategy
except ImportError:
    logging.error("backtest_strategy.py 파일을 찾을 수 없습니다.")
    raise

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def run_backtest(
    symbol: str,
    start_date: str,
    end_date: str,
    timeframe: str = '1h',
    cash: float = 10000.0,
    commission: float = 0.001, # 거래 수수료 (예: 0.1%)
    leverage: int = 1, # 레버리지 (선물 거래 시)
    strategy_params: dict = None # 전략 파라미터 오버라이드
):
    """주어진 조건으로 backtrader 백테스트를 실행하고 결과를 출력/시각화합니다."""

    logging.info(f"--- 백테스트 시작: {symbol} ({start_date} ~ {end_date}), Timeframe: {timeframe} ---")
    cerebro = bt.Cerebro()

    # 데이터 로드 (yfinance 사용)
    logging.info(f"{symbol} 데이터 로드 중...")
    try:
        data_df = yf.download(symbol, start=start_date, end=end_date, interval=timeframe)
        if data_df.empty:
            logging.error(f"{symbol} 데이터를 다운로드할 수 없습니다.")
            return
        # backtrader 형식에 맞게 컬럼명 변경 (Open, High, Low, Close, Volume)
        data_df.rename(columns={
            'Open': 'open', 'High': 'high', 'Low': 'low',
            'Close': 'close', 'Adj Close': 'adj_close', 'Volume': 'volume'
        }, inplace=True)
        # 타임존 정보 제거
        data_df.index = pd.to_datetime(data_df.index).tz_localize(None)
        data = bt.feeds.PandasData(dataname=data_df)
        logging.info(f"데이터 로드 완료 ({len(data_df)} 행)")
    except Exception as e:
        logging.error(f"데이터 로드 중 오류 발생: {e}")
        return

    # 데이터 추가
    cerebro.adddata(data)

    # 전략 추가
    # 기본 파라미터에 오버라이드 파라미터 적용
    final_strategy_params = FlightBacktestStrategy.params.__dict__.copy()
    if strategy_params:
        final_strategy_params.update(strategy_params)
    cerebro.addstrategy(FlightBacktestStrategy, **final_strategy_params)

    # 초기 자본금 설정
    cerebro.broker.setcash(cash)

    # 수수료 설정
    cerebro.broker.setcommission(commission=commission)

    # 레버리지 설정 (선물 거래 시)
    if leverage > 1:
        # 주의: backtrader의 기본 브로커는 레버리지를 직접 지원하지 않을 수 있음.
        # 선물 거래를 정확히 시뮬레이션하려면 커스텀 브로커나 관련 라이브러리 확장 필요.
        # 여기서는 레버리지가 포지션 크기 계산에 반영된다고 가정 (strategy 내부에서 구현 필요)
        logging.info(f"레버리지 설정: {leverage}x (전략 내부에서 반영 필요)")
        # cerebro.broker.set_coc(True) # Cash-on-Cash (more realistic for margin)

    # 분석기 추가 (결과 분석용)
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='tradeanalyzer')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

    # 백테스트 실행
    logging.info("백테스트 실행 중...")
    initial_portfolio_value = cerebro.broker.getvalue()
    results = cerebro.run()
    final_portfolio_value = cerebro.broker.getvalue()
    logging.info("백테스트 완료.")

    # 결과 분석 및 출력
    logging.info("--- 백테스트 결과 --- ")
    logging.info(f"초기 자산: {initial_portfolio_value:,.2f}")
    logging.info(f"최종 자산: {final_portfolio_value:,.2f}")
    pnl = final_portfolio_value - initial_portfolio_value
    pnl_percent = (pnl / initial_portfolio_value) * 100
    logging.info(f"총 손익: {pnl:,.2f} ({pnl_percent:.2f}%)")

    # 분석기 결과 추출
    try:
        strategy_result = results[0]
        trade_analysis = strategy_result.analyzers.tradeanalyzer.get_analysis()
        sharpe_ratio = strategy_result.analyzers.sharpe.get_analysis()['sharperatio']
        max_drawdown = strategy_result.analyzers.drawdown.get_analysis()['max']['drawdown']
        total_return = strategy_result.analyzers.returns.get_analysis()['rtot'] * 100

        logging.info(f"총 거래 횟수: {trade_analysis.total.closed}")
        if trade_analysis.total.closed > 0:
             logging.info(f"승률: {trade_analysis.won.total / trade_analysis.total.closed:.2%}")
             logging.info(f"평균 거래 손익: {trade_analysis.pnl.net.average:.2f}")
             # 손익비 계산 필요
        logging.info(f"Sharpe Ratio: {sharpe_ratio:.3f}")
        logging.info(f"최대 낙폭 (MDD): {max_drawdown:.2f}%")
        logging.info(f"총 수익률 (로그 기준): {total_return:.2f}%")

    except KeyError as ke:
        logging.warning(f"분석기 결과 추출 중 오류 (KeyError): {ke}. 일부 분석기가 실행되지 않았을 수 있습니다.")
    except Exception as e:
        logging.error(f"분석기 결과 처리 중 오류: {e}")

    # 결과 시각화 (Plotly 사용 추천)
    logging.info("결과 시각화 생성 중...")
    try:
        # cerebro.plot() # 기본 Matplotlib 플롯 (별도 창)
        # Plotly로 더 보기 좋은 차트 생성 가능 (backtrader_plotly 라이브러리 등 활용)
        # 여기서는 기본 plot() 사용
        figure = cerebro.plot(style='candlestick', barup='green', bardown='red')[0][0]
        plot_filename = f"backtest_{symbol}_{start_date}_{end_date}.png"
        figure.savefig(plot_filename)
        logging.info(f"백테스트 결과 차트 저장: {plot_filename}")
    except Exception as e:
        logging.error(f"백테스트 결과 시각화 중 오류: {e}")


if __name__ == "__main__":
    # 백테스트 실행 예시
    run_backtest(
        symbol="BTC-USD",       # yfinance 심볼 (Binance는 BTCUSDT)
        start_date="2023-01-01",
        end_date="2023-12-31",
        timeframe='1h',         # 1시간 봉
        cash=10000.0,
        commission=0.0004,      # Binance 선물 수수료 예시 (시장가)
        leverage=10,            # 레버리지 예시
        strategy_params={       # 전략 파라미터 오버라이드 예시
            'tp_percent': 0.10,
            'sl_percent': 0.04,
            'rsi_oversold': 25
        }
    ) 