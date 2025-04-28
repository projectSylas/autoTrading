# 프로젝트 환경 설정

## 1. 가상 환경 (Poetry)

이 프로젝트는 `Poetry`를 사용하여 파이썬 패키지 및 가상 환경을 관리합니다.

### Poetry 설치 (최초 1회)

```bash
curl -sSL https://install.python-poetry.org | python3 -
```
(설치 방법은 Poetry 공식 문서를 참고하세요.)

### 프로젝트 설정

1.  **프로젝트 클론 또는 생성**: `git clone <repository_url>` 또는 `poetry new <project_name>`
2.  **프로젝트 디렉토리로 이동**:
    ```bash
    cd /Users/hjkim/Dev/Hjkim/Trading
    ```
3.  **가상 환경 생성 위치 설정 (권장)**:
    프로젝트 내부에 `.venv` 디렉토리를 생성하도록 설정하면 관리가 용이합니다.
    ```bash
    poetry config virtualenvs.in-project true --local
    ```
4.  **의존성 설치 및 가상 환경 활성화**:
    `pyproject.toml` 파일에 정의된 의존성을 설치하고 가상 환경을 활성화합니다.
    ```bash
    poetry install
    ```
5.  **가상 환경 활성화 (셸 실행)**:
    Poetry가 관리하는 가상 환경 내에서 셸을 실행합니다.
    ```bash
    poetry shell
    ```
    이제 프롬프트 앞에 `(.venv)` 와 유사한 표시가 나타나며, 이 셸 안에서 실행되는 모든 파이썬 관련 명령은 가상 환경 내에서 동작합니다.

### 가상 환경 비활성화

`poetry shell`로 활성화된 환경에서는 셸을 종료하면 됩니다:

```bash
exit
```

만약 `source .venv/bin/activate` 방식으로 활성화했다면 다음 명령을 사용합니다:

```bash
deactivate
```

## 2. 환경 변수 (`.env`)

API 키, Slack 토큰 등 민감한 정보나 설정값은 프로젝트 루트 디렉토리에 `.env` 파일을 생성하여 관리합니다. 이 파일은 `.gitignore`에 추가하여 Git 저장소에 포함되지 않도록 합니다.

`.env` 파일 예시:

```dotenv
# API Keys
BINANCE_API_KEY="YOUR_BINANCE_API_KEY"
BINANCE_SECRET_KEY="YOUR_BINANCE_SECRET_KEY"
ALPACA_API_KEY="YOUR_ALPACA_API_KEY"
ALPACA_SECRET_KEY="YOUR_ALPACA_SECRET_KEY"
NEWS_API_KEY="YOUR_NEWS_API_KEY"

# Slack
SLACK_BOT_TOKEN="YOUR_SLACK_BOT_TOKEN"
SLACK_CHANNEL="#your-trading-alerts-channel"

# Database (예정)
# DB_HOST="localhost"
# DB_PORT="5432"
# DB_NAME="trading_db"
# DB_USER="trading_user"
# DB_PASSWORD="your_db_password"

# Model/Log Paths
MODEL_SAVE_DIR="./model_weights/"
LOG_DIR="./logs/"

# Strategy Parameters
CHALLENGE_SMA_PERIOD=7
CHALLENGE_SL_RATIO=0.05
CHALLENGE_TP_RATIO=0.1
```

Python 코드에서는 `python-dotenv` 라이브러리를 사용하여 이 환경 변수를 로드할 수 있습니다.

```python
import os
from dotenv import load_dotenv

load_dotenv() # .env 파일 로드

api_key = os.getenv("BINANCE_API_KEY")
slack_token = os.getenv("SLACK_BOT_TOKEN")
```

## 3. 필수 외부 라이브러리 설치 (TA-Lib)

TA-Lib 라이브러리는 C 라이브러리 의존성이 있어 설치가 까다로울 수 있습니다. 시스템에 맞게 설치해야 합니다.

**macOS (Homebrew 사용 시):**

```bash
ta-lib # TA-Lib C 라이브러리 설치
poetry add TA-Lib # Python 래퍼 설치
```

다른 운영체제는 TA-Lib 공식 문서를 참고하여 C 라이브러리를 먼저 설치한 후 `poetry add TA-Lib`를 실행하세요. 