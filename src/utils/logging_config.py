import logging
import os
import sys
from logging.handlers import RotatingFileHandler
# from src.config.settings import settings # 설정 로드 제거, 기본값 사용 또는 환경 변수 사용

# --- 기본 설정 값 (환경 변수 등으로 대체 가능) ---
LOG_DIR = os.getenv('LOG_DIR', 'logs')
# --- DEBUG 레벨 고정 --- 
LOG_LEVEL = 'DEBUG' # 모든 로그를 기록하도록 DEBUG로 고정

# 로그 디렉토리 생성 (없으면)
os.makedirs(LOG_DIR, exist_ok=True)

# --- 로거 이름 설정 --- 
APP_LOGGER_NAME = 'trading_app'

def setup_logging(force_reset: bool = False):
    """
    애플리케이션 전반의 로깅 설정을 구성합니다.
    모든 로그(DEBUG 이상)를 콘솔과 app.log 파일에 기록합니다.
    """
    logger = logging.getLogger(APP_LOGGER_NAME)

    # --- 핸들러 중복 추가 방지 --- 
    # 이미 설정된 경우 (예: 여러 모듈에서 호출) 또는 force_reset이 아닐 경우 건너뛰기
    if logger.hasHandlers() and not force_reset:
        # print(f"[{__name__}] Logger '{APP_LOGGER_NAME}' already configured.")
        return logger

    # --- 강제 리셋 또는 초기 설정 시 기존 핸들러 제거 --- 
    if force_reset:
        print(f"[{__name__}] Force reset: Removing existing handlers for '{APP_LOGGER_NAME}'.")
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()

    # --- 로거 레벨 설정 (DEBUG 고정) --- 
    logger.setLevel(logging.DEBUG)
    logger.propagate = False # 루트 로거로 전파 방지 (중복 출력 방지)

    # --- 포매터 생성 --- 
    log_format = '%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)

    # --- 콘솔 핸들러 (DEBUG 레벨) --- 
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG) # 콘솔에도 DEBUG 레벨 출력
    logger.addHandler(console_handler)

    # --- 파일 핸들러 (app.log, DEBUG 레벨) --- 
    log_file_path = os.path.join(LOG_DIR, 'app.log')
    file_handler = RotatingFileHandler(
        log_file_path,
        maxBytes=10*1024*1024, # 10 MB
        backupCount=5,
        encoding='utf-8'
        # delay=False # 명시적으로 버퍼 지연 비활성화 시도 (선택 사항)
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG) # 파일에도 DEBUG 레벨 기록
    logger.addHandler(file_handler)

    # --- 오류 파일 핸들러 (error.log, ERROR 레벨 이상) --- 
    error_log_file_path = os.path.join(LOG_DIR, 'error.log')
    error_file_handler = RotatingFileHandler(
        error_log_file_path,
        maxBytes=5*1024*1024, # 5 MB
        backupCount=3,
        encoding='utf-8'
    )
    error_file_handler.setFormatter(formatter)
    error_file_handler.setLevel(logging.ERROR)
    logger.addHandler(error_file_handler)

    print(f"[{__name__}] Logging setup complete for '{APP_LOGGER_NAME}'. Level: DEBUG (Fixed), Handlers: {len(logger.handlers)}")
    return logger

# --- Example Usage (if run directly) ---
if __name__ == '__main__':
    logger = setup_logging()
    logger.debug("This is a debug message.")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.critical("This is a critical message.")

    # Example of logging from a specific module
    # 다른 모듈에서는 아래와 같이 로거를 가져와 사용
    module_logger = logging.getLogger(APP_LOGGER_NAME + '.MyModule')
    module_logger.info("Message from MyModule logger.") 