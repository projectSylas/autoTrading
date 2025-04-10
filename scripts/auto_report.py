import logging
import pandas as pd
from datetime import datetime, timedelta
import os # 로그 파일 경로 지정을 위해 추가

# 설정 및 알림 모듈 임포트 (새로운 경로)
# import config -> from src.config import settings as config
from src.config import settings as config
try:
    # from notifier import ... -> from src.utils.notifier import ...
    from src.utils.notifier import send_slack_notification, create_slack_block, format_dataframe_for_slack
except ImportError:
    logging.error("src.utils.notifier 로드 실패. 리포트 전송 불가.")
    send_slack_notification = None
    create_slack_block = None
    format_dataframe_for_slack = None

# DB 유틸리티 임포트
try:
    from src.utils.database import get_log_data
except ImportError:
     logging.error("src.utils.database 모듈 로드 실패. 리포트 생성 불가.")
     get_log_data = None

# 로그 파일 경로 직접 정의 (루트의 logs/ 디렉토리 기준)
# from main import LOG_BACKTEST_FILE, LOG_SENTIMENT_FILE, LOG_VOLATILITY_FILE -> 삭제
LOG_DIR = "logs" # 로그 디렉토리
LOG_BACKTEST_FILE = os.path.join(LOG_DIR, "backtest.csv")
LOG_SENTIMENT_FILE = os.path.join(LOG_DIR, "sentiment.csv")
LOG_VOLATILITY_FILE = os.path.join(LOG_DIR, "volatility.csv")
# restructure.sh 에서 로그 파일 이름도 변경됨 (log_*.csv -> *.csv)

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_report_blocks() -> list:
    """데이터베이스에서 로그를 읽어 Slack Block Kit 형식의 리포트 블록 리스트를 생성합니다."""
    if not get_log_data:
        return [create_slack_block("리포트 생성 오류", "데이터베이스 모듈 로드 실패")]

    report_blocks = []
    today_str = datetime.now().strftime("%Y-%m-%d")
    days_to_report = 1 # 최근 1일 데이터 기준

    # --- 1. 거래 요약 (Core + Challenge) --- #
    try:
        # 최근 1일 거래 내역 조회 (최대 10건)
        df_trades = get_log_data('trades', days=days_to_report, limit=10)
        if not df_trades.empty and format_dataframe_for_slack:
            # 필요한 컬럼만 선택하여 표시 (예시)
            trade_summary = format_dataframe_for_slack(
                df_trades[['timestamp', 'strategy', 'symbol', 'side', 'status', 'pnl_percent']].sort_values('timestamp')
            )
            report_blocks.append(create_slack_block(f"📊 최근 거래 요약 ({today_str})", trade_summary))
        else:
            report_blocks.append(create_slack_block(f"📊 최근 거래 요약 ({today_str})", f"최근 {days_to_report}일 데이터 없음."))
    except Exception as e:
         report_blocks.append(create_slack_block(f"📊 최근 거래 요약 ({today_str})", f"데이터 조회 오류: {e}"))
    report_blocks.append({"type": "divider"})


    # --- 2. 시장 감성 분석 요약 --- #
    if config.ENABLE_SENTIMENT:
        try:
            df_sentiment = get_log_data('sentiment_logs', days=days_to_report, limit=5)
            if not df_sentiment.empty and format_dataframe_for_slack:
                sentiment_summary_table = format_dataframe_for_slack(df_sentiment.sort_values('timestamp'))
                report_blocks.append(create_slack_block(f"📰 시장 감성 분석 ({today_str})", sentiment_summary_table))
                if 'negative' in df_sentiment['sentiment'].unique():
                    report_blocks.append(create_slack_block("주의", "부정적인 시장 감성이 포함되었습니다."))
            else:
                report_blocks.append(create_slack_block(f"📰 시장 감성 분석 ({today_str})", f"최근 {days_to_report}일 데이터 없음."))
        except Exception as e:
             report_blocks.append(create_slack_block(f"📰 시장 감성 분석 ({today_str})", f"데이터 조회 오류: {e}"))
        report_blocks.append({"type": "divider"})

    # --- 3. 변동성 감지 요약 --- #
    if config.ENABLE_VOLATILITY:
        try:
            df_volatility = get_log_data('volatility_logs', days=days_to_report)
            if not df_volatility.empty:
                anomalies = df_volatility[df_volatility['is_anomaly'] == True]
                if not anomalies.empty:
                     anomaly_summary = format_dataframe_for_slack(anomalies.sort_values('timestamp'), max_rows=3)
                     report_blocks.append(create_slack_block(f"📈 변동성 이상 감지 ({today_str})", f"**이상 변동 감지됨:**\n{anomaly_summary}"))
                else:
                     report_blocks.append(create_slack_block(f"📈 변동성 이상 감지 ({today_str})", f"최근 {days_to_report}일 내 이상 변동 감지 내역 없음."))
            else:
                report_blocks.append(create_slack_block(f"📈 변동성 이상 감지 ({today_str})", f"최근 {days_to_report}일 데이터 없음."))
        except Exception as e:
             report_blocks.append(create_slack_block(f"📈 변동성 이상 감지 ({today_str})", f"데이터 조회 오류: {e}"))

    # TODO: 백테스트 결과는 별도 테이블에서 가져오도록 수정 필요 (backtest_runner 수정 후)
    # if config.ENABLE_BACKTEST:
    #     ...

    return report_blocks

def send_summary_report():
    """데이터베이스 로그 기반 요약 리포트를 생성하고 Slack으로 전송합니다."""
    logging.info("--- 📑 자동 요약 리포트 생성 시작 (DB 기반) ---")
    if not send_slack_notification:
        logging.error("Notifier가 없어 리포트를 전송할 수 없습니다.")
        return

    report_blocks = generate_report_blocks()

    if not report_blocks:
        logging.warning("리포트 내용이 없습니다. 전송할 데이터 없음.")
        send_slack_notification("일일 자동 분석 요약", "분석 결과 데이터가 없습니다.", level="warning")
        return

    # 최종 리포트 전송
    success = send_slack_notification(
        f"일일 자동 분석 요약 ({datetime.now().strftime('%Y-%m-%d')})",
        blocks=report_blocks,
        level="info"
    )

    if success:
        logging.info("--- ✅ 자동 요약 리포트 전송 완료 ---")
    else:
        logging.error("--- ❌ 자동 요약 리포트 전송 실패 ---")


if __name__ == "__main__":
    # 스크립트 직접 실행 시 리포트 생성 및 전송 테스트
    send_summary_report() 