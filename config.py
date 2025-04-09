import os
from dotenv import load_dotenv
import logging

# .env 파일 로드 (파일이 없어도 오류 발생 안 함)
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- API Keys ---
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
# 'True' 문자열을 boolean True로 변환, 기본값은 True (페이퍼 트레이딩)
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "True").lower() == 'true'

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")

# --- Strategy Parameters ---
# .env 파일에 없으면 기본값 사용
CORE_ASSETS = os.getenv("CORE_ASSETS", "SPY,QQQ,GLD,SGOV,XLP").split(',')
CORE_REBALANCE_THRESHOLD = float(os.getenv("CORE_REBALANCE_THRESHOLD", "0.15"))

CHALLENGE_SYMBOL = os.getenv("CHALLENGE_SYMBOL", "BTCUSDT")
CHALLENGE_LEVERAGE = int(os.getenv("CHALLENGE_LEVERAGE", "10"))
CHALLENGE_TP_RATIO = float(os.getenv("CHALLENGE_TP_RATIO", "0.10"))
CHALLENGE_SL_RATIO = float(os.getenv("CHALLENGE_SL_RATIO", "0.05"))

VOLATILITY_THRESHOLD = float(os.getenv("VOLATILITY_THRESHOLD", "5.0"))

# --- Module Execution Flags ---
# .env 파일에 없으면 기본값 True 사용
ENABLE_BACKTEST = os.getenv("ENABLE_BACKTEST", "True").lower() == 'true'
ENABLE_SENTIMENT = os.getenv("ENABLE_SENTIMENT", "True").lower() == 'true'
ENABLE_VOLATILITY = os.getenv("ENABLE_VOLATILITY", "True").lower() == 'true'


# --- Environment Variable Check ---
def check_env_vars():
    """필수 환경 변수가 설정되었는지 확인하고 경고를 로깅합니다."""
    required_core = ["ALPACA_API_KEY", "ALPACA_SECRET_KEY", "SLACK_WEBHOOK_URL"]
    required_challenge = ["BINANCE_API_KEY", "BINANCE_SECRET_KEY", "SLACK_WEBHOOK_URL"] # 챌린지 전략이 Binance를 쓴다고 가정
    required_sentiment = ["NEWS_API_KEY", "SLACK_WEBHOOK_URL"] # Slack 알림 포함 가정

    warnings = []

    # Core Portfolio 관련 변수 확인 (Alpaca)
    if not all(os.getenv(var) for var in required_core):
        missing = [var for var in required_core if not os.getenv(var)]
        warnings.append(f"Core Portfolio 실행에 필요한 환경 변수 누락: {', '.join(missing)}. 일부 기능이 제한될 수 있습니다.")

    # Challenge Trading 관련 변수 확인 (Binance 가정)
    if not all(os.getenv(var) for var in required_challenge):
        missing = [var for var in required_challenge if not os.getenv(var)]
        warnings.append(f"Challenge Trading 실행에 필요한 환경 변수 누락: {', '.join(missing)}. 일부 기능이 제한될 수 있습니다.")

    # Sentiment Analysis 관련 변수 확인
    if ENABLE_SENTIMENT and not all(os.getenv(var) for var in required_sentiment):
        missing = [var for var in required_sentiment if not os.getenv(var)]
        warnings.append(f"Sentiment Analysis 실행에 필요한 환경 변수 누락: {', '.join(missing)}. 감성 분석 기능이 비활성화될 수 있습니다.")
        # 필요시 여기서 ENABLE_SENTIMENT = False 처리 가능

    # Volatility Alert는 외부 API 키가 직접 필요하지 않음 (데이터는 yfinance 등에서 온다고 가정)
    # 그러나 Slack 알림은 필요할 수 있음
    if ENABLE_VOLATILITY and not os.getenv("SLACK_WEBHOOK_URL"):
         warnings.append("Volatility Alert 실행 시 Slack 알림에 필요한 SLACK_WEBHOOK_URL 환경 변수가 누락되었습니다.")

    if warnings:
        logging.warning("환경 변수 설정 경고:")
        for warning in warnings:
            logging.warning(f"- {warning}")
    else:
        logging.info("필요한 환경 변수들이 잘 설정되었습니다.")

# 스크립트 로드 시 환경 변수 체크 실행
# check_env_vars() # 주석 처리: main.py 등 실제 실행 시점에서 호출하는 것이 더 적합할 수 있음 