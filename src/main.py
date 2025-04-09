import schedule
import time
import logging
from datetime import datetime
import pandas as pd
import os

# 설정 로드
import config

# 기능 모듈 임포트 (오류 발생 시에도 main 실행은 가능하도록 try-except 사용)
try:
    from backtest_runner import run_backtest
    # 백테스트 결과 분석/요약 함수가 있다면 임포트 (예: analyze_backtest_results)
except ImportError:
    logging.error("backtest_runner.py 로드 실패. 백테스트 기능 비활성화됨.")
    run_backtest = None

try:
    from sentiment_analysis import get_market_sentiment
except ImportError:
    logging.error("sentiment_analysis.py 로드 실패. 감성 분석 기능 비활성화됨.")
    get_market_sentiment = None

try:
    from volatility_alert import run_volatility_check
    # 또는 detect_anomaly 함수만 직접 임포트할 수도 있음
except ImportError:
    logging.error("volatility_alert.py 로드 실패. 변동성 감지 기능 비활성화됨.")
    run_volatility_check = None

# 알림 모듈 임포트
try:
    from notifier import send_slack_notification, format_dataframe_for_slack, create_slack_block
except ImportError:
    logging.error("notifier.py 로드 실패. Slack 알림 비활성화됨.")
    send_slack_notification = None
    format_dataframe_for_slack = None # 포맷팅 함수도 같이 처리
    create_slack_block = None

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 로그 파일 경로 정의
LOG_BACKTEST_FILE = "log_backtest.csv"
LOG_SENTIMENT_FILE = "log_sentiment.csv"
LOG_VOLATILITY_FILE = "log_volatility.csv"

def save_log(data: dict | pd.DataFrame, log_file: str, append: bool = True):
    """데이터를 CSV 로그 파일에 저장합니다.

    Args:
        data (dict | pd.DataFrame): 저장할 데이터.
        log_file (str): 저장할 로그 파일 경로.
        append (bool): 기존 파일에 추가할지 여부 (False면 덮어쓰기).
    """
    try:
        if isinstance(data, dict):
            df_to_save = pd.DataFrame([data])
        elif isinstance(data, pd.DataFrame):
            df_to_save = data
        else:
            logging.warning(f"지원하지 않는 데이터 타입({type(data)})으로 로그 저장 불가: {log_file}")
            return

        if 'timestamp' not in df_to_save.columns:
             df_to_save['timestamp'] = datetime.now()

        mode = 'a' if append else 'w'
        header = not (append and os.path.exists(log_file))

        df_to_save.to_csv(log_file, mode=mode, header=header, index=False, encoding='utf-8-sig')
        logging.info(f"로그 저장 완료: {log_file} ({len(df_to_save)} 행)")

    except Exception as e:
        logging.error(f"로그 파일 '{log_file}' 저장 중 오류 발생: {e}")


# --- 통합 실행 잡 정의 ---
def run_analysis_pipeline():
    """백테스트, 감성 분석, 변동성 감지를 순차적으로 실행하는 메인 파이프라인."""
    logging.info("===== 🚀 자동 분석 파이프라인 시작 ====")
    start_time = time.time()
    config.check_env_vars() # 실행 시점에 환경 변수 체크

    results_summary = {}

    # --- 1. 백테스트 실행 --- #
    if config.ENABLE_BACKTEST and run_backtest:
        logging.info("--- 📊 백테스트 실행 시작 ---")
        try:
            # run_backtest 함수는 결과 DataFrame 또는 dict를 반환한다고 가정
            # 필요한 파라미터 전달 (예: 심볼, 기간 등은 config 또는 기본값 사용)
            backtest_result_df, analysis_results = run_backtest(symbol=config.CHALLENGE_SYMBOL)

            if backtest_result_df is not None:
                save_log(backtest_result_df, LOG_BACKTEST_FILE, append=False) # 백테스트 로그는 보통 덮어쓰기
                results_summary['backtest'] = analysis_results # 요약 정보 저장

                # Slack 알림 (요약 정보만)
                if send_slack_notification and analysis_results:
                     body = "\n".join([f"- {k}: {v}" for k, v in analysis_results.items()])
                     send_slack_notification("백테스트 완료", message_body=body, level="success")
            else:
                logging.warning("백테스트 실행되었으나 결과 없음.")
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
            # 분석할 주요 키워드 (config 또는 직접 지정)
            keywords = [config.CHALLENGE_SYMBOL, "Bitcoin", "Federal Reserve", "Economy"]
            sentiment_results = []
            for keyword in keywords:
                sentiment, score, articles_df = get_market_sentiment(keyword, days_ago=1) # 최근 1일 뉴스 분석
                result = {
                    'keyword': keyword,
                    'sentiment': sentiment,
                    'score': score,
                    'article_count': len(articles_df)
                }
                sentiment_results.append(result)
                logging.info(f"'{keyword}' 감성 분석 결과: {sentiment} (점수: {score:.2f}, 기사 수: {len(articles_df)})")
                time.sleep(1) # API 호출 제한 고려

            if sentiment_results:
                sentiment_df = pd.DataFrame(sentiment_results)
                save_log(sentiment_df, LOG_SENTIMENT_FILE, append=True) # 감성 로그는 누적
                results_summary['sentiment'] = sentiment_df # 전체 결과 저장

                # Slack 알림 (요약 테이블)
                if send_slack_notification and format_dataframe_for_slack and create_slack_block:
                    blocks = [
                        create_slack_block("분석 결과 요약", format_dataframe_for_slack(sentiment_df))
                    ]
                    # 부정적 감성 강조
                    if 'negative' in sentiment_df['sentiment'].values:
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
            # run_volatility_check 함수는 내부적으로 anomaly 감지 및 로깅/알림을 수행한다고 가정
            # 결과로 anomaly 여부, 괴리율 등을 반환받아 로깅할 수도 있음
            # 여기서는 예시로 run_volatility_check를 직접 호출
            # 실제로는 anomaly 결과만 받아오는 함수를 호출하는 것이 더 좋을 수 있음
            run_volatility_check(symbol=config.CHALLENGE_SYMBOL, threshold=config.VOLATILITY_THRESHOLD)
            # run_volatility_check 에서 반환하는 값이 있다면 받아서 로그 기록 및 요약
            # 예: is_anomaly, deviation, timestamp = run_volatility_check(...)
            # anomaly_log = {'symbol': config.CHALLENGE_SYMBOL, 'is_anomaly': is_anomaly, 'deviation': deviation, 'check_time': timestamp}
            # save_log(anomaly_log, LOG_VOLATILITY_FILE, append=True)
            # results_summary['volatility'] = anomaly_log # 요약 정보 업데이트

            # volatility_alert.py 내부에서 이미 알림을 보내므로 여기서는 추가 알림 불필요할 수 있음
            # 만약 run_volatility_check가 결과만 반환한다면 여기서 알림 로직 추가
            logging.info("변동성 감지 작업 완료.") # run_volatility_check가 성공적으로 끝나면 로깅
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

    # 최종 결과 요약 리포트 전송 (선택 사항)
    # send_summary_report(results_summary)


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