# 자동매매 시스템 프로젝트

## 🎯 프로젝트 목표

안정적인 장기 투자(Core Portfolio)와 단기 고수익 추구(Challenge Trading) 전략을 결합하고, AI 기반 예측 모델(가격 예측, 감성 분석) 및 기타 분석 도구(변동성 감지 등)를 통합하여 자동화된 매매 시스템을 구축 및 운영하는 것을 목표로 합니다.

## ✨ 주요 기능 및 현황

*   **Core Portfolio (Alpaca)**: 미국 주요 ETF 기반 자산 배분 및 리밸런싱 전략. (`src/core/portfolio.py`, 구현 예정)
    *   요구사항: RSI < 30, VIX > 25 등 조건 매수, 월별 리밸런싱.
*   **Challenge Trading (Binance/MEXC)**: '플라이트 챌린지' 매매법 기반 고배율 트레이딩 전략. (`src/challenge/strategy.py`, 핵심 로직 구현 중)
    *   핵심 조건(추세선 이탈+되돌림, RSI 다이버전스, 거래량 급증, 7일 이평선, VPVR 근사)은 `src/utils/strategy_utils.py`에 구현 완료.
    *   전략 실행 흐름 및 AI 연동 로직 구현됨 (`src/challenge/strategy.py`).
    *   **주의:** 실제 주문 연동 로직은 미구현.
*   **AI 가격 예측**: TimesFM/LSTM/Transformer 기반 가격 예측. (`src/ai_models/price_predictor.py`, 모델 로딩/예측/저장 흐름 및 전략 연동 완료)
*   **AI 뉴스 감성 분석**: FinBERT 모델 기반 시장 심리 분석. (`src/ai_models/sentiment.py`, 구현 및 전략 연동 완료)
*   **백테스팅 (`backtrader`)**: 챌린지 전략 검증 및 파라미터 최적화. (`src/backtesting/`, 설정 및 실행 환경 구축 완료, 최적화 결과 처리 오류 해결)
    *   **이슈:** 현재 모든 파라미터 조합에서 거래가 발생하지 않는 문제 발생. (TODO 1순위)
*   **데이터베이스 로깅 (PostgreSQL)**: 거래, AI 결과 등 로그 기록. (`src/utils/database.py`, DB 초기화 및 테이블 생성, 일부 로깅 함수 정의 존재. 상세 로직 구현 필요)
*   **Slack 알림**: 매매 신호, 오류, 분석 결과 등 알림. (`src/utils/notifier.py`, 기본 함수 구조 존재, 상세 구현 필요)
*   **변동성 이상 감지 (Prophet)**: 가격 예측 기반 이상 변동성 감지 및 알림. (`src/analysis/volatility.py`, `main.py`에서 호출되나 상세 로직 구현 필요)
*   **설정 관리**: API 키, 전략/모델 파라미터 중앙 관리. (`src/config/settings.py`, `.env`, 구현 완료)
*   **유틸리티**: 공통 함수, 기술 지표 계산 등. (`src/utils/common.py`, `src/utils/strategy_utils.py`, 구현 완료)
*   **실행 제어 (`main.py`)**: 분석 파이프라인(백테스트, 감성, 변동성) 호출 구조 존재. 실제 전략 스케줄링 및 실행 로직 미구현.
*   **데이터 시각화**: DB 데이터 기반 시각화. (`src/analysis/visualizer.py`, 구현 예정)
*   **자동 리포팅**: 일일 요약 리포트 Slack 전송. (`scripts/auto_report.py`, 구현 예정)
*   **RL 기반 전략**: 강화학습 에이전트 트레이딩 자동화. (관련 파일 존재 가능성 있으나, 현재 주요 개발 범위 아님)
*   **Docker 지원**: PostgreSQL 및 앱 실행 환경 구성. (`docker/`, 설정 제공, 선택적 사용)
*   **유닛 테스트**: `pytest` 기반 테스트. (`tests/`, 일부 구현, 커버리지 확대 필요)

## 🔧 기술 스택

*   Python 3.11+
*   **데이터베이스**: PostgreSQL 15+
*   **컨테이너**: Docker, Docker Compose
*   **테스트**: `pytest`
*   **주요 라이브러리**:
    *   `pandas`, `numpy`, `pandas-ta`, `scipy`
    *   `alpaca-trade-api`, `python-binance` (또는 다른 거래소 라이브러리)
    *   `yfinance`
    *   `schedule`, `python-dotenv`, `logging`
    *   `matplotlib`, `seaborn`, `plotly` (시각화, 백테스팅)
    *   `requests` (Slack)
    *   `backtrader`
    *   `newsapi-python`, `transformers`, `torch`, `scikit-learn` (AI/Sentiment/Prediction)
    *   `prophet` (변동성 감지)
    *   `psycopg2-binary` (PostgreSQL)
    # RL 관련 라이브러리 (stable-baselines3 등)는 필요시 추가

## 📁 프로젝트 구조 (주요 디렉토리)

```
/Trading/
├── src/
│   ├── main.py                  # 메인 실행 스크립트
│   ├── core/                    # 안정형 포트폴리오 전략
│   ├── challenge/               # 챌린지 전략 (규칙 기반)
│   ├── backtesting/             # 백테스팅 모듈 (backtrader)
│   ├── ai_models/               # AI 모델 (예측, 감성분석)
│   ├── analysis/                # 데이터 분석 (변동성, 시각화)
│   ├── utils/                   # 공통 유틸리티 (DB, 알림, 지표 계산 등)
│   └── config/                  # 설정 관리
├── scripts/                   # 보조 스크립트 (자동 리포팅 등)
├── docker/                    # Docker 설정
├── notebooks/                 # Jupyter 노트북 (분석/실험용)
├── tests/                     # 유닛 테스트 코드
├── model_weights/             # 학습된 모델 가중치 (Git 무시)
├── logs/                      # 로그 파일 저장 (필요시)
├── .env                       # 환경 변수 (Git 무시)
├── .gitignore
├── README.md                  # 프로젝트 설명 (본 파일)
└── pyproject.toml             # Poetry 의존성 관리
```

## ⚙️ 설정 방법

1.  **Clone Repository**
2.  **Poetry 설치 및 의존성 설치**: Python 3.11+ 및 Poetry가 설치되어 있어야 합니다.
    ```bash
    poetry install
    ```
3.  **.env 파일 생성**: `.env.example` 또는 아래 예시를 참고하여 API 키, DB 접속 정보, 전략/모델 파라미터 등을 입력합니다.
    ```dotenv
    # --- API Keys ---
    ALPACA_API_KEY="YOUR_ALPACA_KEY"
    ALPACA_SECRET_KEY="YOUR_ALPACA_SECRET"
    ALPACA_PAPER="True" # True: Paper Trading, False: Live Trading
    BINANCE_API_KEY="YOUR_BINANCE_KEY" # Or other exchange
    BINANCE_SECRET_KEY="YOUR_BINANCE_SECRET"
    NEWS_API_KEY="YOUR_NEWSAPI_KEY"
    SLACK_WEBHOOK_URL="YOUR_SLACK_WEBHOOK_URL"

    # --- Database (PostgreSQL) ---
    # Docker 사용 시:
    # DB_HOST=db
    # DB_PORT=5433
    # 로컬/외부 DB 사용 시:
    DB_HOST="localhost" # or your DB host
    DB_PORT=5432 # or your DB port
    DB_USER="trading_user"
    DB_PASSWORD="trading_password"
    DB_NAME="trading_db"

    # --- AI Model Settings ---
    HF_SENTIMENT_MODEL_NAME="ProsusAI/finbert"
    HF_TIMESERIES_MODEL_NAME="google/timesfm-1.0-200m"
    MODEL_WEIGHTS_DIR="./model_weights/"
    ENABLE_HF_SENTIMENT="True"
    ENABLE_HF_TIMESERIES="False"
    # ... (Sentiment thresholds, HF device etc.) ...

    # --- Strategy Parameters (Challenge) ---
    CHALLENGE_SYMBOLS="BTCUSDT,ETHUSDT"
    CHALLENGE_INTERVAL="1h"
    CHALLENGE_LOOKBACK="30d"
    CHALLENGE_LEVERAGE=10
    # ... (Challenge 파라미터: TP/SL Ratio, SMA, RSI, Volume, Divergence, Breakout, Pullback, POC 등) ...

    # --- Volatility Settings ---
    ENABLE_VOLATILITY="True" # 예시
    VOLATILITY_THRESHOLD=5.0 # 예시

    # --- Log Level ---
    LOG_LEVEL="INFO" # DEBUG, INFO, WARNING, ERROR, CRITICAL

    # --- Feature Flags ---
    # (Strategy, Volatility 등 모듈별 활성화 플래그)
    ```
4.  **(Optional) Docker 환경 실행**:
    ```bash
    # DB만 실행
    docker-compose -f docker/docker-compose.yml up -d db
    ```

## ▶️ 실행 방법

1.  **메인 스케줄러 실행 (구현 후)**:
    ```bash
    poetry run python src/main.py
    ```
2.  **백테스트 실행**:
    *   `src/backtesting/backtest_runner.py` 파일 하단의 `if __name__ == '__main__':` 블록에서 설정 조정.
    *   실행:
        ```bash
        poetry run python src/backtesting/backtest_runner.py
        ```
3.  **챌린지 전략 직접 실행 (실제 주문 연동 후)**:
    ```bash
    # 예시: poetry run python src/challenge/strategy.py --symbol BTCUSDT
    ```
4.  **AI 모델 직접 실행/테스트**:
    ```bash
    poetry run python src/ai_models/price_predictor.py --symbol BTCUSDT --interval 1h --train=False --predict=True
    poetry run python src/ai_models/sentiment.py
    ```
5.  **(Optional) 유닛 테스트 실행**:
    ```bash
    poetry run pytest tests/
    ```

## 🧠 주요 로직 및 특징

*   **듀얼 전략**: 안정형(Core) + 고수익 추구(Challenge) 전략 병행 설계.
*   **챌린지 전략**: 기술적 지표(추세선, 되돌림, 다이버전스, 거래량, 이평선, 매물대) 기반 매매 로직.
*   **AI 통합**: 감성 점수, 가격 예측 추세를 기술적 분석 결정의 보조/오버라이드 요소로 활용.
*   **백테스팅**: `backtrader` 활용 과거 데이터 기반 전략 검증 및 파라미터 최적화 지원.
*   **모듈화 및 설정 관리**: 기능별 코드 분리 및 `.env`/`settings.py` 기반 설정 관리.

## ⚠️ 주의 사항

*   **투자 위험**: 모든 투자는 원금 손실의 위험이 있습니다. 본인의 판단과 책임 하에 투자 결정을 내리십시오.
*   **API/DB 보안**: `.env` 파일의 민감 정보는 안전하게 관리하고 외부에 노출하지 마십시오.
*   **테스트 중요성**: 실제 자금 투입 전 반드시 페이퍼 트레이딩/테스트넷 및 충분한 백테스팅을 수행하십시오.
*   **백테스트 한계**: 과거 성과가 미래 수익을 보장하지 않습니다.
*   **AI 모델 한계**: AI 결과는 참고 자료이며, 맹신은 금물입니다.

## 🚀 향후 개선 방향 (TODO)

1.  **챌린지 전략 개선 (최우선)**: 백테스트에서 거래가 발생하지 않는 문제 해결 (진입/청산 조건 로직 및 파라미터 재검토).
2.  **실제 주문 연동**: Challenge 전략 및 Core 전략의 실제 거래소 API 연동 및 주문 실행 로직 구현.
3.  **Core Portfolio 전략 구현**: `src/core/portfolio.py`에 안정형 포트폴리오 로직 상세 구현.
4.  **DB 로깅 완성**: `utils/database.py`의 모든 로깅 함수 상세 구현 및 실제 데이터 저장 확인.
5.  **Slack 알림 완성**: `utils/notifier.py` 상세 구현 및 다양한 상황별 알림 연동.
6.  **메인 스케줄러 완성**: `src/main.py`에 실제 전략 실행 스케줄링 로직 구현.
7.  **변동성 감지 로직 구현**: `src/analysis/volatility.py`에 Prophet 기반 상세 로직 구현.
8.  **시각화 모듈 구현**: `src/analysis/visualizer.py` 구현.
9.  **자동 리포팅 구현**: `scripts/auto_report.py` 구현.
10. **테스트 커버리지 확대**: 주요 기능에 대한 유닛/통합 테스트 추가.
11. **오류 처리 강화**: API 오류, 데이터 오류 등 예외 처리 및 안정성 강화.
12. **(Optional) RL 기반 전략 구체화 및 통합**.

---
*README 최종 업데이트: 2024-08-01*