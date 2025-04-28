# 프로젝트 환경 설정 및 문제 해결 가이드

## 1. Poetry 가상 환경 설정

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
3.  **가상 환경 생성 위치 설정 (필수 권장)**:
    프로젝트 내부에 `.venv` 디렉토리를 생성하도록 설정하면 다른 프로젝트 환경과의 충돌을 방지하고 관리가 매우 용이해집니다.
    ```bash
    poetry config virtualenvs.in-project true --local
    ```
    **반드시** 이 설정을 프로젝트 루트에서 실행하여 가상 환경이 프로젝트 폴더 내(`.venv`)에 생성되도록 합니다.
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
    이제 프롬프트 앞에 `(.venv)` 와 유사한 표시(또는 설정에 따라 프로젝트명-해시값 형태)가 나타나며, 이 셸 안에서 실행되는 모든 파이썬 관련 명령은 해당 프로젝트의 가상 환경 내에서 동작합니다.

### 가상 환경 비활성화

-   `poetry shell`로 활성화된 환경: 셸을 종료하면 자동으로 비활성화됩니다 (`exit` 또는 터미널 창 닫기).
-   `source .venv/bin/activate` 방식으로 활성화한 경우 (권장하지 않음): `deactivate` 명령을 사용합니다.

## 2. Poetry 가상 환경 문제 해결 (Troubleshooting)

Poetry 사용 중 가상 환경 관련 문제가 발생했을 때의 진단 및 해결 과정입니다.

### 문제 상황: 다른 프로젝트의 가상 환경이 활성화되는 경우

**증상:**

-   Trading 프로젝트 디렉토리로 이동(`cd /Users/hjkim/Dev/Hjkim/Trading`)했음에도 불구하고, 터미널 프롬프트에 다른 프로젝트 이름(예: `cleddit-api-py3.11hjkim@MacBook-Pro-2`)이 계속 표시됩니다.
-   `poetry env info` 명령 실행 시, 현재 프로젝트(`Trading`)의 가상 환경 경로(`Path: /Users/hjkim/Dev/Hjkim/Trading/.venv`)가 아닌, 다른 프로젝트(예: `Cleddit-Api`)의 가상 환경 경로가 표시됩니다.
-   `poetry shell` 명령 실행 시 `The command "shell" does not exist.` 와 같은 오류가 발생합니다.

**원인 분석:**

1.  **잘못된 가상 환경 활성화 상태 유지:** 현재 터미널 세션이 이전에 다른 프로젝트(`Cleddit-Api`)에서 `source .venv/bin/activate` 방식 등으로 활성화된 가상 환경 상태를 유지하고 있습니다. `cd` 명령으로 디렉토리만 변경해도, 셸 자체는 이전 가상 환경에 묶여 있습니다.
2.  **Poetry 명령 인식 불가:** 활성화된 다른 가상 환경 내에는 `poetry` 실행 파일이 설치되어 있지 않거나 경로 설정이 꼬여 있어 `poetry shell` 과 같은 명령을 인식하지 못합니다.
3.  **프롬프트 표시:** 터미널 프롬프트에 표시되는 `cleddit-api-py3.11hjkim` 등은 `activate` 스크립트에 의해 자동으로 생성된 활성화된 가상 환경의 이름입니다. 이를 통해 현재 어떤 가상 환경에 있는지 시각적으로 확인할 수 있습니다.

**해결 단계:**

🔥 **1단계: 현재 활성화된 가상 환경 확실히 비활성화**

   다른 프로젝트의 가상 환경이 활성화된 터미널에서 다음 명령을 실행하여 환경을 비활성화합니다.

   ```bash
   deactivate
   ```

   만약 `deactivate` 명령이 없거나 작동하지 않으면, 해당 터미널 창/탭을 완전히 종료(`exit` 또는 창 닫기)하는 것이 가장 확실합니다.

🔥 **2단계: 터미널 완전 재시작**

   -   기존 터미널 창/탭을 닫습니다.
   -   **완전히 새로운 터미널 창/탭**을 엽니다.
   -   *절대로* 문제가 발생했던 이전 셸 세션에서 작업을 이어서 하지 마십시오. 특히 `source .../activate`를 사용했던 이력이 있다면 반드시 새로 시작해야 합니다.

✅ **3단계: 정상 상태 확인 및 Poetry 환경 진입**

   새 터미널에서 다시 Trading 프로젝트 디렉토리로 이동합니다.

   ```bash
   cd /Users/hjkim/Dev/Hjkim/Trading
   ```

   가상 환경 정보를 확인하여 올바른 경로가 표시되는지 확인합니다.

   ```bash
   poetry env info
   ```

   **기대 결과:** `Path:` 항목에 `/Users/hjkim/Dev/Hjkim/Trading/.venv` 와 같이 현재 프로젝트 내의 `.venv` 경로가 표시되어야 합니다. (만약 `virtualenvs.in-project true` 설정이 안 되어 있다면 시스템 경로 어딘가에 있는 해시값 형태의 경로가 나올 수 있습니다.)

   정상 경로가 확인되면, `poetry shell` 명령으로 가상 환경을 활성화합니다.

   ```bash
   poetry shell
   ```

   이제 터미널 프롬프트가 Trading 프로젝트의 가상 환경으로 변경되고, `python`, `pip` 등 모든 관련 명령이 이 환경 내에서 실행됩니다.

💡 **핵심:** 문제는 Poetry 자체가 아니라, 터미널 세션이 의도치 않게 다른 가상 환경에 묶여 있었던 것입니다. `deactivate` 또는 터미널 재시작으로 연결을 끊고, `poetry shell`을 통해 올바른 환경으로 진입하는 것이 중요합니다.

## 3. 환경 변수 (`.env`) 관리

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
# ... 기타 AI 모델 파라미터 등
LSTM_WINDOW_SIZE=60
LSTM_EPOCHS=30
```

Python 코드에서는 `python-dotenv` 라이브러리를 사용하여 이 환경 변수를 로드할 수 있습니다.

```python
# 예: src/config/settings.py 또는 main.py 상단
import os
from dotenv import load_dotenv

# .env 파일 로드 (프로젝트 루트에 있는 .env 파일을 찾음)
load_dotenv()

# 환경 변수 사용
binance_api_key = os.getenv("BINANCE_API_KEY")
slack_token = os.getenv("SLACK_BOT_TOKEN")
sma_period = int(os.getenv("CHALLENGE_SMA_PERIOD", 7)) # 기본값 설정 가능
```

## 4. 필수 외부 라이브러리 설치 (TA-Lib)

TA-Lib 라이브러리는 C 라이브러리 의존성이 있어 운영체제별로 설치 방법이 다릅니다.

**macOS (Homebrew 사용 시):**

1.  TA-Lib C 라이브러리 설치:
    ```bash
    brew install ta-lib
    ```
2.  Python 래퍼 설치 (Poetry 사용):
    ```bash
    poetry add TA-Lib
    ```

**다른 운영체제:**

TA-Lib 공식 문서 또는 관련 커뮤니티 가이드를 참조하여 해당 운영체제에 맞는 C 라이브러리를 먼저 설치한 후, `poetry add TA-Lib` 또는 `pip install TA-Lib`를 실행하세요.

--- 