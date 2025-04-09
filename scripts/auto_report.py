import logging
import pandas as pd
from datetime import datetime, timedelta

# 설정 및 알림 모듈 임포트
import config
try:
    from notifier import send_slack_notification, create_slack_block, format_dataframe_for_slack
except ImportError:
    logging.error("notifier.py 로드 실패. 리포트 전송 불가.")
    send_slack_notification = None
    create_slack_block = None
    format_dataframe_for_slack = None

# 로그 파일 경로 (main.py와 동일하게 참조)
from main import LOG_BACKTEST_FILE, LOG_SENTIMENT_FILE, LOG_VOLATILITY_FILE

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_latest_log(log_file: str, days: int = 1) -> pd.DataFrame:
    """지정된 CSV 로그 파일에서 최근 N일치 데이터를 로드합니다."""
    try:
        df = pd.read_csv(log_file, parse_dates=['timestamp'], encoding='utf-8-sig')
        cutoff_date = datetime.now() - timedelta(days=days)
        # timestamp 컬럼 기준으로 필터링
        latest_df = df[df['timestamp'] >= cutoff_date]
        logging.info(f"로그 로드 완료: {log_file} (최근 {days}일, {len(latest_df)} 행)")
        return latest_df
    except FileNotFoundError:
        logging.warning(f"로그 파일 없음: {log_file}")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"로그 파일 '{log_file}' 로드 중 오류: {e}")
        return pd.DataFrame()

def generate_report_blocks() -> list:
    """각 로그 파일을 읽어 Slack Block Kit 형식의 리포트 블록 리스트를 생성합니다."""
    report_blocks = []
    today_str = datetime.now().strftime("%Y-%m-%d")

    # --- 1. 백테스트 결과 요약 --- #
    if config.ENABLE_BACKTEST:
        # 백테스트 로그는 보통 전체 결과가 저장되므로, 최근 실행 기준으로 로드
        # 여기서는 간단히 파일 존재 여부 및 내용 일부 표시 예시
        try:
            # 백테스트는 보통 실행 시점에 결과를 저장하므로, 파일 존재 유무나 마지막 수정 시간 등으로 판단
            # 여기서는 main.py에서 analysis_results를 저장하는 방식으로 변경해야 더 유용
            # 지금은 파일 로드 예시만 보여줌
            df_backtest = pd.read_csv(LOG_BACKTEST_FILE, encoding='utf-8-sig')
            if not df_backtest.empty:
                # 마지막 행(가장 최근 결과)을 요약으로 사용하거나, 특정 컬럼 값 표시
                # 예시: 마지막 행의 주요 지표 표시
                latest_result = df_backtest.iloc[-1].to_dict()
                summary = "\n".join([f"- {k}: {v}" for k, v in latest_result.items() if k != 'timestamp']) # timestamp 제외
                report_blocks.append(create_slack_block(f"📊 백테스트 결과 ({today_str})", summary))
            else:
                report_blocks.append(create_slack_block(f"📊 백테스트 결과 ({today_str})", "실행 기록 없음 또는 결과 비어있음."))
        except FileNotFoundError:
             report_blocks.append(create_slack_block(f"📊 백테스트 결과 ({today_str})", "로그 파일 없음."))
        except Exception as e:
             report_blocks.append(create_slack_block(f"📊 백테스트 결과 ({today_str})", f"로그 로드 오류: {e}"))
        report_blocks.append({"type": "divider"})

    # --- 2. 시장 감성 분석 요약 --- #
    if config.ENABLE_SENTIMENT:
        df_sentiment = load_latest_log(LOG_SENTIMENT_FILE, days=1)
        if not df_sentiment.empty and format_dataframe_for_slack:
            sentiment_summary_table = format_dataframe_for_slack(df_sentiment, max_rows=5)
            report_blocks.append(create_slack_block(f"📰 시장 감성 분석 ({today_str})", sentiment_summary_table))
            # 부정적 감성 하이라이트
            if 'negative' in df_sentiment['sentiment'].unique():
                report_blocks.append(create_slack_block("주의", "부정적인 시장 감성이 포함되었습니다."))
        else:
            report_blocks.append(create_slack_block(f"📰 시장 감성 분석 ({today_str})", "최근 1일 데이터 없음."))
        report_blocks.append({"type": "divider"})

    # --- 3. 변동성 감지 요약 --- #
    if config.ENABLE_VOLATILITY:
        df_volatility = load_latest_log(LOG_VOLATILITY_FILE, days=1)
        if not df_volatility.empty:
            # 이상 감지(Anomaly) 결과만 필터링하여 보여주거나, 최근 감지 결과 요약
            anomalies = df_volatility[df_volatility['is_anomaly'] == True]
            if not anomalies.empty:
                 anomaly_summary = format_dataframe_for_slack(anomalies, max_rows=3)
                 report_blocks.append(create_slack_block(f"📈 변동성 이상 감지 ({today_str})", f"**이상 변동 감지됨:**\n{anomaly_summary}"))
            else:
                 report_blocks.append(create_slack_block(f"📈 변동성 이상 감지 ({today_str})", "최근 1일 내 이상 변동 감지 내역 없음."))
        else:
            report_blocks.append(create_slack_block(f"📈 변동성 이상 감지 ({today_str})", "최근 1일 데이터 없음."))
        # 변동성 로그 형식이 is_anomaly 포함하도록 수정 필요 가정

    return report_blocks

def send_summary_report():
    """로그 파일들을 기반으로 요약 리포트를 생성하고 Slack으로 전송합니다."""
    logging.info("--- 📑 자동 요약 리포트 생성 시작 ---")
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