import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import logging
import os

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# config 모듈 로드 (새로운 경로) 및 로그 파일 경로 설정
try:
    # import config -> from src.config import settings as config
    from src.config import settings as config
except ImportError:
    logging.error("src.config.settings 모듈을 찾을 수 없습니다. 로그 파일 경로를 알 수 없습니다.")
    config = None
    # 로그 파일 기본 경로 수정 (루트의 logs/ 디렉토리)
    LOG_CORE_FILE = os.path.join("logs", "core.csv")
    LOG_CHALLENGE_FILE = os.path.join("logs", "challenge.csv")
else:
    # restructure.sh 에 따른 로그 파일 경로 수정
    LOG_CORE_FILE = os.path.join("logs", "core.csv")
    LOG_CHALLENGE_FILE = os.path.join("logs", "challenge.csv")
    # config에 로그 경로 변수가 정의되어 있다면 해당 값을 사용 (선택적)
    # 예: LOG_CORE_FILE = config.LOG_CORE_FILE_PATH if hasattr(config, 'LOG_CORE_FILE_PATH') else LOG_CORE_FILE

# Plotly 기본 템플릿 설정 (선택 사항)
pio.templates.default = "plotly_white"

# DB 유틸리티 임포트
try:
    from src.utils.database import get_log_data
except ImportError:
     logging.error("src.utils.database 모듈 로드 실패. 시각화 생성 불가.")
     get_log_data = None

def plot_core_summary(df: pd.DataFrame):
    """안정형 포트폴리오 로그 요약 및 시각화 (Matplotlib 예시)."""
    if df.empty:
        logging.info("Core 로그 데이터가 없어 시각화를 건너뜁니다.")
        return

    logging.info("--- 안정형 포트폴리오 요약 --- ")
    # 총 거래 횟수
    total_trades = len(df)
    logging.info(f"총 거래 제출 횟수: {total_trades}")
    # 매수/매도 횟수
    buy_trades = df[df['side'] == 'buy'].shape[0]
    sell_trades = df[df['side'] == 'sell'].shape[0]
    logging.info(f"매수: {buy_trades}, 매도: {sell_trades}")
    # 주문 상태별 횟수
    status_counts = df['status'].value_counts()
    logging.info(f"주문 상태별 횟수:\n{status_counts}")

    # 시각화 (예시: 주문 상태 파이 차트)
    try:
        plt.style.use('seaborn-v0_8-whitegrid') # 스타일 설정
        fig, ax = plt.subplots(figsize=(8, 6))
        status_counts.plot(kind='pie', ax=ax, autopct='%1.1f%%', startangle=90,
                           title='Core Portfolio Order Status Distribution')
        ax.set_ylabel('') # 파이 차트 y 라벨 제거
        plt.tight_layout()
        # 파일 저장 또는 화면 표시
        save_path = "core_status_summary.png"
        plt.savefig(save_path)
        logging.info(f"Core 포트폴리오 주문 상태 요약 차트 저장: {save_path}")
        # plt.show() # 로컬 실행 시 화면 표시
        plt.close(fig) # 메모리 해제
    except Exception as e:
        logging.error(f"Core 요약 차트 생성 중 오류: {e}")

def plot_challenge_summary(df: pd.DataFrame):
    """챌린지 전략 로그 요약 및 시각화 (Plotly 예시)."""
    if df.empty:
        logging.info("Challenge 로그 데이터가 없어 시각화를 건너뜁니다.")
        return

    logging.info("--- 챌린지 전략 요약 --- ")
    # 로그 타입 확인 (signal 로그인지 trade 로그인지)
    if 'entry_price' not in df.columns:
         logging.warning("Challenge 로그는 신호 기록만 포함하는 것 같습니다. 거래 요약 불가.")
         # 신호 로그 요약 (예: 심볼별, 방향별 신호 횟수)
         signal_counts = df.groupby(['symbol', 'side']).size().unstack(fill_value=0)
         logging.info(f"챌린지 신호 발생 횟수:\n{signal_counts}")
         return

    # 거래 로그 요약
    total_trades = len(df[df['status'].str.startswith('open')]) # 진입 횟수 기준
    closed_trades = df[df['status'].str.startswith('closed')]
    logging.info(f"총 진입 횟수: {total_trades}")
    logging.info(f"총 청산 횟수: {len(closed_trades)}")

    if not closed_trades.empty:
        # 수익률 계산 (pnl_percent 컬럼 사용)
        try:
             # '%' 제거하고 float 변환
             closed_trades['pnl_numeric'] = closed_trades['pnl_percent'].str.rstrip('%').astype(float) / 100.0
        except Exception as e:
             logging.warning(f"PNL 수치 변환 오류: {e}")
             closed_trades['pnl_numeric'] = 0.0

        # 승률 계산
        win_trades = closed_trades[closed_trades['pnl_numeric'] > 0]
        loss_trades = closed_trades[closed_trades['pnl_numeric'] <= 0]
        win_rate = len(win_trades) / len(closed_trades) if len(closed_trades) > 0 else 0
        logging.info(f"승률: {win_rate:.2%}")
        # 평균 손익비
        avg_profit = win_trades['pnl_numeric'].mean() if not win_trades.empty else 0
        avg_loss = loss_trades['pnl_numeric'].mean() if not loss_trades.empty else 0
        profit_factor = abs(avg_profit / avg_loss) if avg_loss != 0 else float('inf')
        logging.info(f"평균 수익: {avg_profit:.2%}, 평균 손실: {avg_loss:.2%}, 손익비: {profit_factor:.2f}")

        # 누적 수익률 시각화 (Plotly 예시)
        try:
            closed_trades = closed_trades.sort_values(by='timestamp')
            closed_trades['cumulative_pnl'] = (1 + closed_trades['pnl_numeric']).cumprod()

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=closed_trades['timestamp'],
                y=closed_trades['cumulative_pnl'],
                mode='lines',
                name='Cumulative PnL'
            ))
            fig.update_layout(
                title='Challenge Strategy Cumulative PnL',
                xaxis_title='Date',
                yaxis_title='Cumulative Return (1 = Start)',
                hovermode="x unified"
            )
            # 파일 저장 또는 HTML 생성
            save_path = "challenge_cumulative_pnl.html"
            fig.write_html(save_path)
            logging.info(f"Challenge 누적 수익률 차트 저장: {save_path}")
            # fig.show() # 로컬 실행 시 브라우저에서 보기
        except Exception as e:
            logging.error(f"Challenge 누적 수익률 차트 생성 중 오류: {e}")

# --- 메인 실행 로직 --- 
def run_visualizer(days: int = 30):
    """데이터베이스에서 로그 데이터를 로드하고 시각화를 생성합니다.

    Args:
        days (int): 조회할 최근 일수.
    """
    logging.info(f"===== 📊 시각화 시작 (최근 {days}일 데이터) =====")
    if not get_log_data:
        logging.error("데이터베이스 모듈 로드 실패로 시각화를 중단합니다.")
        return

    # 안정형 포트폴리오 시각화
    # df_core = load_log_data(LOG_CORE_FILE)
    try:
        # get_log_data 함수에 필터 기능 추가 필요 가정 (예: WHERE절 추가)
        # 실제 구현 시 database.py의 get_log_data 수정 필요
        df_core = get_log_data('trades', days=days, query_filter="strategy = 'core'")
        plot_core_summary(df_core)
    except Exception as e:
        logging.error(f"Core 시각화 데이터 로드 또는 처리 중 오류: {e}")

    # 챌린지 전략 시각화
    # df_challenge = load_log_data(LOG_CHALLENGE_FILE)
    try:
        df_challenge = get_log_data('trades', days=days, query_filter="strategy = 'challenge'")
        plot_challenge_summary(df_challenge)
    except Exception as e:
        logging.error(f"Challenge 시각화 데이터 로드 또는 처리 중 오류: {e}")

    logging.info("===== 📊 시각화 종료 =====")

if __name__ == "__main__":
    # 로그 파일이 존재하고 데이터가 있어야 의미있는 결과 확인 가능
    if config:
        run_visualizer()
    else:
         logging.warning("config 모듈 로드 실패로 시각화를 실행할 수 없습니다.") 