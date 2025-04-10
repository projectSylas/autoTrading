import schedule
import time
import logging
from datetime import datetime
import pandas as pd
import os

# 설정 로드
from src.config import settings as config

# 기능 모듈 임포트 (오류 발생 시에도 main 실행은 가능하도록 try-except 사용)
try:
    from src.backtest.runner import run_backtest
    # 백테스트 결과 분석/요약 함수가 있다면 임포트 (예: analyze_backtest_results)
except ImportError:
    logging.error("src.backtest.runner 로드 실패. 백테스트 기능 비활성화됨.")
    run_backtest = None

try:
    from src.analysis.sentiment import get_market_sentiment
except ImportError:
    logging.error("src.analysis.sentiment 로드 실패. 감성 분석 기능 비활성화됨.")
    get_market_sentiment = None

try:
    from src.analysis.volatility import run_volatility_check
    # 또는 detect_anomaly 함수만 직접 임포트할 수도 있음
except ImportError:
    logging.error("src.analysis.volatility 로드 실패. 변동성 감지 기능 비활성화됨.")
    run_volatility_check = None

# 알림 모듈 임포트
try:
    from src.utils.notifier import send_slack_notification, format_dataframe_for_slack, create_slack_block
except ImportError:
    logging.error("src.utils.notifier 로드 실패. Slack 알림 비활성화됨.")
    send_slack_notification = None
    format_dataframe_for_slack = None # 포맷팅 함수도 같이 처리
    create_slack_block = None

# 데이터베이스 초기화 시도 (애플리케이션 시작 시)
try:
    from src.utils.database import initialize_database
    initialize_database()
except ImportError:
    logging.error("src.utils.database 모듈을 찾을 수 없습니다. DB 초기화 불가.")
except Exception as e:
    logging.error(f"데이터베이스 초기화 중 오류 발생: {e}. 시스템 중단 가능성.")
    # 필요 시 여기서 시스템 종료 처리
    # raise

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 로그 파일 경로 정의
# LOG_BACKTEST_FILE = "log_backtest.csv"
# LOG_SENTIMENT_FILE = "log_sentiment.csv"
# LOG_VOLATILITY_FILE = "log_volatility.csv"

# save_log 함수 제거
# def save_log(data: dict | pd.DataFrame, log_file: str, append: bool = True):
#     ...

# --- 통합 실행 잡 정의 ---
def run_analysis_pipeline():
    """백테스트, 감성 분석, 변동성 감지를 순차적으로 실행하는 메인 파이프라인."""
    logging.info("===== 🚀 자동 분석 파이프라인 시작 ====")
    start_time = time.time()
    config.check_env_vars() # 실행 시점에 환경 변수 체크

    results_summary = {} # 요약 정보는 유지 (리포팅 등에 사용 가능)

    # --- 1. 백테스트 실행 --- #
    if config.ENABLE_BACKTEST and run_backtest:
        logging.info("--- 📊 백테스트 실행 시작 ---")
        try:
            # run_backtest 는 분석 결과(dict)만 반환하고, 상세 로그는 내부적으로 DB에 기록하도록 수정 필요 가정
            # backtest_result_df, analysis_results = run_backtest(symbol=config.CHALLENGE_SYMBOL)
            analysis_results = run_backtest(symbol=config.CHALLENGE_SYMBOL) # 수정된 반환값 가정

            # if backtest_result_df is not None: # -> 제거
            if analysis_results: # 분석 결과가 있으면
                # save_log(backtest_result_df, LOG_BACKTEST_FILE, append=False) # -> 제거
                results_summary['backtest'] = analysis_results # 요약 정보 저장

                # Slack 알림 (요약 정보만)
                if send_slack_notification:
                     body = "\n".join([f"- {k}: {v}" for k, v in analysis_results.items()])
                     send_slack_notification("백테스트 완료", message_body=body, level="success")
            else:
                logging.warning("백테스트 실행되었으나 요약 결과 없음.")
                results_summary['backtest'] = "결과 없음"

        except Exception as e:
            logging.error(f"백테스트 실행 중 오류 발생: {e}", exc_info=True)
            if send_slack_notification:
                send_slack_notification("백테스트 오류", f"백테스트 실행 중 오류 발생: {e}", level="error")
            results_summary['backtest'] = f"오류: {e}"
        logging.info("--- 📊 백테스트 실행 종료 ---")
    else:
        logging.info("백테스트 기능 비활성화됨.")

    # --- 2. 시장 감성 분석 --- #
    if config.ENABLE_SENTIMENT and get_market_sentiment:
        logging.info("--- 📰 시장 감성 분석 시작 ---")
        try:
            # get_market_sentiment 가 내부적으로 DB에 로그를 기록하도록 수정 필요 가정
            # sentiment, score, articles_df = get_market_sentiment(keyword, days_ago=1)

            keywords = [config.CHALLENGE_SYMBOL, "Bitcoin", "Federal Reserve", "Economy"]
            all_sentiment_results = [] # Slack 보고용 결과는 유지
            for keyword in keywords:
                # get_market_sentiment 가 분석 결과 dict를 반환한다고 가정
                sentiment_result = get_market_sentiment(keyword, days_ago=1)
                if sentiment_result:
                     all_sentiment_results.append(sentiment_result)
                     logging.info(f"'{keyword}' 감성 분석 완료: {sentiment_result}")
                else:
                    logging.warning(f"'{keyword}' 감성 분석 결과 없음.")
                time.sleep(1) # API 호출 제한 고려

            if all_sentiment_results:
                sentiment_df_for_report = pd.DataFrame(all_sentiment_results)
                # save_log(sentiment_df, LOG_SENTIMENT_FILE, append=True) # -> 제거
                results_summary['sentiment'] = sentiment_df_for_report # Slack용 요약 결과 저장

                # Slack 알림 (요약 테이블)
                if send_slack_notification and format_dataframe_for_slack and create_slack_block:
                    blocks = [
                        create_slack_block("분석 결과 요약", format_dataframe_for_slack(sentiment_df_for_report))
                    ]
                    if 'negative' in sentiment_df_for_report['sentiment'].values:
                         blocks.append(create_slack_block("주의", "부정적인 시장 감성이 감지되었습니다."))
                    send_slack_notification("시장 감성 분석 완료", blocks=blocks, level="info")
            else:
                 logging.warning("감성 분석 실행되었으나 결과 없음.")
                 results_summary['sentiment'] = "결과 없음"

        except Exception as e:
            logging.error(f"시장 감성 분석 중 오류 발생: {e}", exc_info=True)
            if send_slack_notification:
                send_slack_notification("감성 분석 오류", f"시장 감성 분석 중 오류 발생: {e}", level="error")
            results_summary['sentiment'] = f"오류: {e}"
        logging.info("--- 📰 시장 감성 분석 종료 ---")
    else:
        logging.info("시장 감성 분석 기능 비활성화됨.")

    # --- 3. 변동성 감지 --- #
    if config.ENABLE_VOLATILITY and run_volatility_check:
        logging.info("--- 📈 변동성 감지 시작 ---")
        try:
            # run_volatility_check 함수가 내부적으로 DB에 로그 기록 및 알림 수행하도록 수정 필요 가정
            # anomaly_log = run_volatility_check(...)
            run_volatility_check(symbol=config.CHALLENGE_SYMBOL, threshold=config.VOLATILITY_THRESHOLD)

            # save_log(anomaly_log, LOG_VOLATILITY_FILE, append=True) # -> 제거
            # results_summary['volatility'] = anomaly_log # 요약 필요시 run_volatility_check가 반환하도록

            logging.info("변동성 감지 작업 완료.") # 성공 시 로깅
            results_summary['volatility'] = "실행 완료" # 단순 완료 표시

        except Exception as e:
            logging.error(f"변동성 감지 중 오류 발생: {e}", exc_info=True)
            if send_slack_notification:
                send_slack_notification("변동성 감지 오류", f"변동성 감지 중 오류 발생: {e}", level="error")
            results_summary['volatility'] = f"오류: {e}"
        logging.info("--- 📈 변동성 감지 종료 ---")
    else:
        logging.info("변동성 감지 기능 비활성화됨.")

    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"===== ✅ 자동 분석 파이프라인 종료 (총 소요 시간: {elapsed_time:.2f}초) ====")

    # 최종 결과 요약 리포트 전송 (선택 사항 - results_summary 사용 가능)
    # from scripts.auto_report import send_summary_report_from_data
    # send_summary_report_from_data(results_summary)


# --- 스케줄링 설정 (예: 매일 오전 9시에 실행) ---
# schedule.every().day.at("09:00").do(run_analysis_pipeline)
# schedule.every(1).minutes.do(run_analysis_pipeline) # 테스트용: 1분마다 실행


if __name__ == "__main__":
    logging.info("자동 분석 시스템 시작...")
    # 즉시 1회 실행
    run_analysis_pipeline()

    # 스케줄러 실행 (주석 해제 시 사용)
    # logging.info("스케줄러 시작. 대기 중...")
    # while True:
    #     schedule.run_pending()
    #     time.sleep(60) # 1분마다 체크

    logging.info("자동 분석 시스템 종료.") 