import os
from dotenv import load_dotenv
import logging
from pydantic_settings import BaseSettings

# .env 파일 로드 (파일이 없어도 오류 발생 안 함)
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Settings(BaseSettings):
    # --- API Keys ---
    ALPACA_API_KEY: str = os.getenv("ALPACA_API_KEY", "")
    ALPACA_SECRET_KEY: str = os.getenv("ALPACA_SECRET_KEY", "")
    ALPACA_PAPER: bool = os.getenv("ALPACA_PAPER", "True").lower() == "true"
    BINANCE_API_KEY: str = os.getenv("BINANCE_API_KEY", "")
    BINANCE_SECRET_KEY: str = os.getenv("BINANCE_SECRET_KEY", "")
    NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")
    SLACK_WEBHOOK_URL: str = os.getenv("SLACK_WEBHOOK_URL", "")

    # --- Database ---
    DB_HOST: str = os.getenv("DB_HOST", "db")
    DB_PORT: int = int(os.getenv("DB_PORT", "5433"))
    DB_USER: str = os.getenv("DB_USER", "trading_user")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "trading_password")
    DB_NAME: str = os.getenv("DB_NAME", "trading_db")

    # --- AI Model Hyperparameters (Existing) ---
    PREDICTOR_SYMBOL: str = os.getenv("PREDICTOR_SYMBOL", "BTCUSDT")
    SEQ_LEN: int = int(os.getenv("SEQ_LEN", "60"))
    HIDDEN_DIM: int = int(os.getenv("HIDDEN_DIM", "128"))
    N_LAYERS: int = int(os.getenv("N_LAYERS", "2"))
    DROPOUT: float = float(os.getenv("DROPOUT", "0.2"))
    LEARNING_RATE: float = float(os.getenv("LEARNING_RATE", "0.001"))
    EPOCHS: int = int(os.getenv("EPOCHS", "50"))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "32"))
    MODEL_TYPE: str = os.getenv("MODEL_TYPE", "LSTM") # LSTM or Transformer
    MODEL_SAVE_PATH: str = os.getenv("MODEL_SAVE_PATH", "./model_weights/price_predictor.pt")

    RL_ENV_SYMBOL: str = os.getenv("RL_ENV_SYMBOL", "BTCUSDT")
    RL_TIMESTEPS: int = int(os.getenv("RL_TIMESTEPS", "100000"))
    RL_ALGORITHM: str = os.getenv("RL_ALGORITHM", "PPO")
    RL_POLICY: str = os.getenv("RL_POLICY", "MlpPolicy")
    RL_LEARNING_RATE: float = float(os.getenv("RL_LEARNING_RATE", "0.0003"))
    RL_BATCH_SIZE: int = int(os.getenv("RL_BATCH_SIZE", "64"))
    RL_MODEL_SAVE_PATH: str = os.getenv("RL_MODEL_SAVE_PATH", "./model_weights/rl_model.zip")
    RL_NORMALIZATION_REFERENCE_PRICE: float | None = os.getenv("RL_NORMALIZATION_REFERENCE_PRICE")
    if RL_NORMALIZATION_REFERENCE_PRICE is not None:
        RL_NORMALIZATION_REFERENCE_PRICE = float(RL_NORMALIZATION_REFERENCE_PRICE)

    # --- Hugging Face Models --- New Settings
    HF_SENTIMENT_MODEL_NAME: str = os.getenv("HF_SENTIMENT_MODEL_NAME", "ProsusAI/finbert")
    HF_TIMESERIES_MODEL_NAME: str = os.getenv("HF_TIMESERIES_MODEL_NAME", "google/timesfm-1.0-200m")
    MODEL_WEIGHTS_DIR: str = os.getenv("MODEL_WEIGHTS_DIR", "./model_weights/")
    ENABLE_HF_SENTIMENT: bool = os.getenv("ENABLE_HF_SENTIMENT", "True").lower() == "true"
    ENABLE_HF_TIMESERIES: bool = os.getenv("ENABLE_HF_TIMESERIES", "False").lower() == "true"
    SENTIMENT_NEUTRAL_THRESHOLD_LOW: float = float(os.getenv("SENTIMENT_NEUTRAL_THRESHOLD_LOW", "0.4"))
    SENTIMENT_NEUTRAL_THRESHOLD_HIGH: float = float(os.getenv("SENTIMENT_NEUTRAL_THRESHOLD_HIGH", "0.6"))
    SENTIMENT_CUTOFF_SCORE: float = float(os.getenv("SENTIMENT_CUTOFF_SCORE", "0.35"))
    HF_MAX_SEQ_LENGTH: int = int(os.getenv("HF_MAX_SEQ_LENGTH", "512"))
    HF_DEVICE: str = os.getenv("HF_DEVICE", "auto") # 'auto', 'cuda', 'cpu'

    # --- Strategy Parameters ---
    # .env 파일에 없으면 기본값 사용
    CORE_ASSETS: list = os.getenv("CORE_ASSETS", "SPY,QQQ,GLD,SGOV,XLP").split(',')
    CORE_RSI_THRESHOLD: int = int(os.getenv("CORE_RSI_THRESHOLD", "30"))
    CORE_VIX_THRESHOLD: int = int(os.getenv("CORE_VIX_THRESHOLD", "25"))
    CORE_REBALANCE_THRESHOLD: float = float(os.getenv("CORE_REBALANCE_THRESHOLD", "0.15"))
    CHALLENGE_SYMBOLS: list = os.getenv("CHALLENGE_SYMBOLS", "BTCUSDT,ETHUSDT").split(',')
    CHALLENGE_LEVERAGE: int = int(os.getenv("CHALLENGE_LEVERAGE", "10"))
    CHALLENGE_SEED_PERCENTAGE: float = float(os.getenv("CHALLENGE_SEED_PERCENTAGE", "0.05"))
    CHALLENGE_TP_RATIO: float = float(os.getenv("CHALLENGE_TP_RATIO", "0.10"))
    CHALLENGE_SL_RATIO: float = float(os.getenv("CHALLENGE_SL_RATIO", "0.05"))
    CHALLENGE_SMA_PERIOD: int = int(os.getenv("CHALLENGE_SMA_PERIOD", "20"))
    CHALLENGE_RSI_PERIOD: int = int(os.getenv("CHALLENGE_RSI_PERIOD", "14"))
    CHALLENGE_RSI_THRESHOLD: int = int(os.getenv("CHALLENGE_RSI_THRESHOLD", "30"))
    CHALLENGE_INTERVAL: str = os.getenv("CHALLENGE_INTERVAL", "1h")
    CHALLENGE_LOOKBACK: str = os.getenv("CHALLENGE_LOOKBACK", "30d")
    CHALLENGE_RISK_PER_TRADE: float = float(os.getenv("CHALLENGE_RISK_PER_TRADE", "0.01"))

    # --- Newly Added Challenge Parameters ---
    CHALLENGE_VOLUME_AVG_PERIOD: int = int(os.getenv("CHALLENGE_VOLUME_AVG_PERIOD", "20"))
    CHALLENGE_VOLUME_SURGE_RATIO: float = float(os.getenv("CHALLENGE_VOLUME_SURGE_RATIO", "2.0"))
    CHALLENGE_DIVERGENCE_LOOKBACK: int = int(os.getenv("CHALLENGE_DIVERGENCE_LOOKBACK", "28"))
    CHALLENGE_BREAKOUT_LOOKBACK: int = int(os.getenv("CHALLENGE_BREAKOUT_LOOKBACK", "40"))
    CHALLENGE_PULLBACK_LOOKBACK: int = int(os.getenv("CHALLENGE_PULLBACK_LOOKBACK", "10"))
    CHALLENGE_POC_LOOKBACK: int = int(os.getenv("CHALLENGE_POC_LOOKBACK", "50"))
    CHALLENGE_POC_THRESHOLD: float = float(os.getenv("CHALLENGE_POC_THRESHOLD", "0.005"))
    LOG_OVERRIDES_TO_DB: bool = os.getenv("LOG_OVERRIDES_TO_DB", "False").lower() == "true"
    CHALLENGE_SYMBOL_DELAY_SECONDS: float = float(os.getenv("CHALLENGE_SYMBOL_DELAY_SECONDS", "1.0"))

    # --- Log Level --- 
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

    # --- Volatility Alert Settings ---
    ENABLE_VOLATILITY: bool = os.getenv("ENABLE_VOLATILITY", "False").lower() == "true"
    VOLATILITY_THRESHOLD: float = float(os.getenv("VOLATILITY_THRESHOLD", "5.0"))

    # --- Feature Flags ---
    ENABLE_BACKTEST: bool = os.getenv("ENABLE_BACKTEST", "False").lower() == "true"
    ENABLE_SENTIMENT_ANALYSIS: bool = os.getenv("ENABLE_SENTIMENT_ANALYSIS", "True").lower() == "true"
    ENABLE_VOLATILITY_ALERT: bool = os.getenv("ENABLE_VOLATILITY_ALERT", "False").lower() == "true"

# 설정 인스턴스 생성 (다른 모듈에서 import하여 사용)
settings = Settings()

# 사용 예시:
# from src.config.settings import settings
# print(settings.ALPACA_API_KEY)
# print(settings.HF_SENTIMENT_MODEL_NAME)

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
    if ENABLE_SENTIMENT_ANALYSIS and not all(os.getenv(var) for var in required_sentiment):
        missing = [var for var in required_sentiment if not os.getenv(var)]
        warnings.append(f"Sentiment Analysis 실행에 필요한 환경 변수 누락: {', '.join(missing)}. 감성 분석 기능이 비활성화될 수 있습니다.")
        # 필요시 여기서 ENABLE_SENTIMENT_ANALYSIS = False 처리 가능

    # Volatility Alert는 외부 API 키가 직접 필요하지 않음 (데이터는 yfinance 등에서 온다고 가정)
    # 그러나 Slack 알림은 필요할 수 있음
    if ENABLE_VOLATILITY_ALERT and not os.getenv("SLACK_WEBHOOK_URL"):
         warnings.append("Volatility Alert 실행 시 Slack 알림에 필요한 SLACK_WEBHOOK_URL 환경 변수가 누락되었습니다.")

    # --- Added: Check for RL Reference Price ---
    # Check if RL is being used (e.g., based on a flag or existence of RL settings)
    # For simplicity, we check if the model path likely exists based on algorithm setting.
    potential_rl_usage = bool(RL_ALGORITHM and RL_POLICY)
    if potential_rl_usage and RL_NORMALIZATION_REFERENCE_PRICE <= 0.0:
         warnings.append(
             f"RL 전략 사용 가능성이 있으나 정규화 기준 가격(RL_NORMALIZATION_REFERENCE_PRICE)이 .env에 설정되지 않았거나 유효하지 않습니다 ({RL_NORMALIZATION_REFERENCE_PRICE}). "
             f"RL 에이전트가 올바르게 작동하지 않을 수 있습니다. 모델 훈련 후 해당 값을 .env에 설정하세요."
         )
    # --- End Added ---

    if warnings:
        logging.warning("환경 변수 설정 경고:")
        for warning in warnings:
            logging.warning(f"- {warning}")
    else:
        logging.info("필요한 환경 변수들이 잘 설정되었습니다.")

# 스크립트 로드 시 환경 변수 체크 실행
# check_env_vars() # 주석 처리: main.py 등 실제 실행 시점에서 호출하는 것이 더 적합할 수 있음 