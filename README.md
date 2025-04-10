# 자동매매 시스템 프로젝트

## 🎯 프로젝트 목표

안정적인 장기 투자(Core Portfolio)와 단기 고수익 추구(Challenge Trading) 전략을 결합하고, AI 기반 예측 및 의사결정 모델을 통합한 자동매매 시스템 구축 및 운영.

## ✨ 주요 기능

*   **Core Portfolio (Alpaca)**: 미국 주요 ETF 기반의 자산 배분 및 리밸런싱 자동화 (`src/core/portfolio.py`)
*   **Challenge Trading (Rules-based)**: 규칙 기반 고배율 선물 트레이딩 자동화 (`src/challenge/strategy.py`)
*   **Challenge Trading (RL-based)**: 강화학습 에이전트 기반 선물 트레이딩 자동화 (`src/challenge/rl_strategy.py`)
    *   **AI Fallback**: RL 모델 추론 실패 시 규칙 기반 전략으로 자동 전환 (`src/challenge/rl_strategy.py`)
*   **AI 가격 예측**: LSTM/Transformer 기반 주가 예측 모델 및 결과 DB 로깅 (`src/ai_models/price_predictor.py`)
*   **AI 강화학습**: RL 환경 정의, 에이전트 학습 및 상태 정규화 관리 (`src/ai_models/rl_environment.py`, `rl_trainer.py`)
*   **백테스팅**: `backtrader`를 이용한 규칙 기반 전략 성과 검증 (`src/backtest/`)
*   **뉴스 감성 분석**: News API 및 FinBERT 모델 활용 시장 심리 분석 (`src/analysis/sentiment.py`)
*   **변동성 이상 감지**: `prophet` 라이브러리 활용 가격 예측 및 이상 변동성 알림 (`src/analysis/volatility.py`)
*   **데이터 시각화**: DB 로그 데이터 기반 요약 및 시각화 (`src/analysis/visualizer.py`)
*   **유틸리티**: 기술적 지표 계산, 데이터 로딩, 공통 함수 (`src/utils/common.py`)
*   **알림**: Slack을 통한 매매 신호, 오류, Fallback 등 실시간 알림 (`src/utils/notifier.py`)
*   **설정 관리**: API 키, 전략/모델 파라미터 등 환경 변수 관리 (`src/config/settings.py`, `.env`)
*   **실행 제어**: `schedule` 라이브러리 이용 주기적 작업 실행 관리 (`src/main.py`)
*   **자동 리포팅**: DB 로그 기반 일일 요약 리포트 Slack 전송 (`scripts/auto_report.py`)
*   **데이터베이스 로깅**: PostgreSQL 이용 거래, 감성, 변동성, **AI 예측** 등 로그 기록 (`src/utils/database.py`)
*   **Docker 지원**: Docker Compose 이용 PostgreSQL 및 앱 실행 환경 구성 (`docker/`)
*   **유닛 테스트**: `pytest` 기반 주요 모듈 단위 테스트 (`tests/`)

## 🔧 기술 스택

*   Python 3.11+
*   **데이터베이스**: PostgreSQL 15+
*   **컨테이너**: Docker, Docker Compose
*   **테스트**: `pytest`
*   **주요 라이브러리**:
    *   `pandas`, `numpy`
    *   `alpaca-trade-api`, `python-binance` (또는 MEXC API)
    *   `yfinance`, `ta`, `scipy`
    *   `schedule`, `python-dotenv`, `logging`
    *   `matplotlib`, `plotly` (시각화)
    *   `requests` (Slack)
    *   `backtrader`
    *   `newsapi-python`, `transformers` (Sentiment)
    *   `prophet` (Volatility)
    *   `psycopg2-binary` (PostgreSQL)
    *   `torch`, `scikit-learn` (AI Prediction)
    *   `stable-baselines3[extra]`, `gymnasium` (Reinforcement Learning)
    *   `unittest.mock` (Testing)

## 📁 프로젝트 구조

```
/Trading/
├── src/
│   ├── __init__.py
│   ├── main.py                  # 메인 실행 스크립트
│   ├── core/                    # 안정형 포트폴리오 전략
│   ├── challenge/               # 챌린지 전략 (규칙/RL 기반)
│   ├── analysis/                # 데이터 분석 (감성, 변동성, 시각화)
│   ├── backtest/                # 백테스팅 모듈
│   ├── utils/                   # 공통 유틸리티 (DB, 알림, 지표 계산 등)
│   ├── config/                  # 설정 관리
│   └── ai_models/               # AI 모델 (예측, RL 환경/학습)
├── scripts/                   # 보조 스크립트 (리포팅 등)
├── docker/                    # Docker 설정
├── notebooks/                 # Jupyter 노트북 (분석/실험용)
├── tests/                     # 유닛 테스트 코드
│   ├── __init__.py
│   ├── test_settings.py       # 설정 모듈 테스트
│   ├── test_common.py         # 공통 유틸리티 테스트
│   ├── test_database.py       # 데이터베이스 테스트
│   ├── test_price_predictor.py # 가격 예측 모델 테스트
│   └── test_rl_environment.py # RL 환경 테스트
├── model_weights/             # 학습된 모델 가중치 (Git 무시)
├── logs/                      # 로그 파일 저장 (필요시)
├── data/                      # 데이터 저장 (필요시)
├── .env                       # 환경 변수 (Git 무시)
├── .gitignore
├── README.md                  # 프로젝트 설명 (본 파일)
├── requirements.txt           # (또는 pyproject.toml) 의존성 목록
└── ... (기타 설정 파일)
```

## ⚙️ 설정 방법

1.  **Clone Repository**
2.  **.env 파일 생성**: `.env.example` 또는 아래 예시를 참고하여 API 키, DB 접속 정보, 전략/모델 파라미터 등을 입력합니다.
    ```dotenv
    # --- API Keys ---
    ALPACA_API_KEY="..."
    ALPACA_SECRET_KEY="..."
    ALPACA_PAPER="True" # True: Paper Trading, False: Live Trading
    BINANCE_API_KEY="..." # Or MEXC
    BINANCE_SECRET_KEY="..."
    NEWS_API_KEY="..."
    SLACK_WEBHOOK_URL="..."

    # --- Database (Docker Compose 기준) ---
    DB_HOST=db
    DB_PORT=5433
    DB_USER=trading_user
    DB_PASSWORD=trading_password
    DB_NAME=trading_db

    # --- AI Model Hyperparameters ---
    PREDICTOR_SYMBOL="BTCUSDT"
    # ... (Predictor params: SEQ_LEN, HIDDEN_DIM, LAYERS, LR, EPOCHS, BATCH_SIZE, MODEL_TYPE) ...
    RL_ENV_SYMBOL="BTCUSDT"
    # ... (RL params: TIMESTEPS, ALGORITHM, POLICY, LR, BATCH_SIZE) ...

    # --- Strategy Parameters ---
    # ... (Core, Challenge, Volatility params: ASSETS, THRESHOLDS, RATIOS, PERIODS, LEVERAGE) ...

    # --- Feature Flags ---
    ENABLE_BACKTEST="True"
    ENABLE_SENTIMENT="True"
    ENABLE_VOLATILITY="True"
    # ...
    ```
3.  **Docker 환경 실행 (권장)**:
    ```bash
    docker-compose -f docker/docker-compose.yml up --build -d
    ```
4.  **(선택적) 로컬 환경 의존성 설치**:
    ```bash
    pip install -r requirements.txt
    # Testing dependencies
    pip install pytest pytest-mock
    ```

## ▶️ 실행 방법

1.  **Docker 환경 실행**: `docker-compose up` 명령으로 실행 후 로그 확인.
    ```bash
    docker-compose -f docker/docker-compose.yml logs -f app
    ```
2.  **로컬 환경 실행**:
    ```bash
    python src/main.py
    ```
3.  **AI 모델 학습**:
    ```bash
    # RL 모델 학습
    python src/ai_models/rl_trainer.py
    # 가격 예측 모델 학습
    python src/ai_models/price_predictor.py --train --predict=False
    ```
4.  **유닛 테스트 실행**:
    ```bash
    pytest tests/
    # 또는 특정 파일: pytest tests/test_common.py
    ```
5.  **개별 모듈/스크립트 실행**: 개발/테스트 목적.

## 🧠 주요 로직 및 특징

*   **Fallback Mechanism**: RL 전략(`rl_strategy.py`) 실행 중 모델 로드 실패, 데이터 관측 실패, 예측 오류 발생 시, 자동으로 규칙 기반 챌린지 전략(`strategy.py`)의 진입 조건을 확인하고 실행을 시도합니다. Fallback 실행 시 Slack으로 알림이 전송됩니다.
*   **설정 관리**: 모든 주요 파라미터는 `.env` 파일을 통해 주입되며, `src/config/settings.py`에서 로드 및 관리됩니다. 타입 변환 및 기본값 처리가 포함됩니다.
*   **데이터 로깅**: 거래 내역, 감성 분석 결과, 변동성 감지 로그, AI 예측 결과 등 주요 이벤트는 PostgreSQL 데이터베이스에 기록됩니다 (`utils/database.py`).
*   **모듈화**: 기능별로 코드가 모듈화되어 있으며 (`core`, `challenge`, `ai_models`, `utils` 등), 의존성을 관리합니다.
*   **테스트**: 주요 설정 및 유틸리티 함수에 대한 유닛 테스트가 `tests/` 디렉토리에 포함되어 있으며, `pytest`로 실행 가능합니다.

## ⚠️ 주의 사항

*   **투자 위험**: 모든 투자는 원금 손실의 위험이 있습니다. 실제 투자 결정은 본인의 책임 하에 신중하게 진행해야 합니다.
*   **API/DB 보안**: `.env` 파일의 민감 정보는 안전하게 관리하고 외부에 노출하지 마세요.
*   **테스트**: 실제 자금 투입 전 페이퍼 트레이딩/테스트넷 및 소액 테스트를 충분히 수행하세요.
*   **AI 모델 검증**: AI 모델의 성능은 데이터와 파라미터에 크게 의존합니다. 충분한 검증 없이 맹신하지 마세요.

## 🚀 향후 개선 방향 (TODO)

*   **유닛/통합 테스트 커버리지 확대**: DB 상호작용, AI 모델 로직, 전략 실행 등 테스트 추가.
*   **DB 함수 구현 완료**: `log_prediction_to_db` 등 Placeholder 함수 실제 구현.
*   **RL 상태 정규화 일치**: `rl_environment.py`와 `rl_strategy.py` 간 상태 정규화 로직 통일 및 검증.
*   **실제 주문 연동**: Placeholder 주문 로직을 실제 거래소 API와 완벽히 연동.
*   **ORM 도입 고려**: SQLAlchemy 등.
*   **오류 처리 강화**: 재시도 로직, 서킷 브레이커 등.
*   **파라미터 최적화**.
*   **상태 관리 개선**.
*   **웹 대시보드 구축**.
*   **CI/CD 파이프라인 구축**.

---
*README 최종 업데이트: 2025-04-10*