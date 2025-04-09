# 자동매매 시스템 프로젝트

## 🎯 프로젝트 목표

안정적인 장기 투자(Core Portfolio)와 단기 고수익 추구(Challenge Trading) 전략을 결합한 자동매매 시스템 구축 및 운영.

## ✨ 주요 기능

*   **Core Portfolio (Alpaca)**: 미국 주요 ETF 기반의 자산 배분 및 리밸런싱 자동화 (`core_portfolio.py`)
*   **Challenge Trading (Binance/MEXC)**: 플라이트 챌린지 매매법 기반 고배율 선물 트레이딩 자동화 (`challenge_trading.py`)
*   **백테스팅**: `backtrader`를 이용한 챌린지 전략 성과 검증 (`backtest_runner.py`)
*   **뉴스 감성 분석**: News API 및 FinBERT 모델을 활용한 시장 심리 분석 (`sentiment_analysis.py`)
*   **변동성 이상 감지**: `prophet` 라이브러리를 이용한 가격 예측 및 이상 변동성 알림 (`volatility_alert.py`)
*   **유틸리티**: RSI, SMA, VIX 등 기술적 지표 계산 (`strategy_utils.py`)
*   **알림**: Slack을 통한 매매 신호, 오류, 이상 감지 등 실시간 알림 (`notifier.py`)
*   **설정 관리**: API 키, 전략 파라미터 등 환경 변수 관리 (`config.py`, `.env`)
*   **실행 제어**: `schedule` 라이브러리를 이용한 주기적 작업 실행 관리 (`main.py`)

## 🔧 기술 스택

*   Python 3.10+
*   **주요 라이브러리**:
    *   `pandas`, `numpy`
    *   `alpaca-trade-api`, `python-binance` (또는 MEXC API 라이브러리)
    *   `yfinance`
    *   `schedule`
    *   `python-dotenv`
    *   `matplotlib`, `plotly` (시각화)
    *   `slack_sdk`
    *   `backtrader`
    *   `newsapi-python`, `transformers`, `torch`
    *   `prophet`
    *   `logging`

## 📁 프로젝트 구조

```
/Trading/
├── main.py                  # 메인 실행 스크립트 (스케줄러)
├── core_portfolio.py        # Alpaca 기반 안정형 포트폴리오 전략
├── challenge_trading.py     # Binance/MEXC 기반 챌린지 전략
├── backtest_strategy.py     # Backtrader 전략 정의
├── backtest_runner.py       # Backtrader 실행 스크립트
├── sentiment_analysis.py    # 뉴스 감성 분석 모듈
├── volatility_alert.py      # Prophet 기반 변동성 감지 모듈
├── strategy_utils.py        # 기술적 지표 및 공통 유틸리티
├── notifier.py              # Slack 알림 모듈
├── config.py                # 설정 로드 및 관리
├── visualizer.py            # 데이터 시각화 (구현 예정)
├── requirements.txt         # (또는 pyproject.toml for Poetry) 의존성 목록
├── .env                     # API 키 등 환경 변수 (Git 무시 처리)
├── log_core.csv             # 안정형 전략 로그
└── log_challenge.csv        # 챌린지 전략 로그
```

## ⚙️ 설정 방법

1.  **.env 파일 생성**: 프로젝트 루트 디렉토리에 `.env` 파일을 생성하고 필요한 API 키 및 설정을 입력합니다. (`.env.example` 파일을 참조하여 필요한 변수 확인)

    ```dotenv
    # Alpaca API Keys
    ALPACA_API_KEY="YOUR_ALPACA_API_KEY"
    ALPACA_SECRET_KEY="YOUR_ALPACA_SECRET_KEY"
    ALPACA_PAPER="True" # True for paper trading, False for live trading

    # Binance API Keys (or MEXC)
    BINANCE_API_KEY="YOUR_BINANCE_API_KEY"
    BINANCE_SECRET_KEY="YOUR_BINANCE_SECRET_KEY"

    # NewsAPI Key
    NEWS_API_KEY="YOUR_NEWSAPI_KEY"

    # Slack Webhook URL
    SLACK_WEBHOOK_URL="YOUR_SLACK_WEBHOOK_URL"

    # Other configurations (e.g., strategy parameters)
    CORE_ASSETS="SPY,QQQ,GLD,SGOV,XLP"
    CORE_REBALANCE_THRESHOLD="0.15" # 15%
    CHALLENGE_SYMBOL="BTCUSDT"
    CHALLENGE_LEVERAGE="10"
    CHALLENGE_TP_RATIO="0.10" # 10% Take Profit
    CHALLENGE_SL_RATIO="0.05" # 5% Stop Loss
    VOLATILITY_THRESHOLD="5.0" # 5% Anomaly Threshold
    ```

2.  **의존성 설치**: (Poetry 사용 시)

    ```bash
    poetry install
    ```

    (pip 사용 시)

    ```bash
    pip install -r requirements.txt
    ```

## ▶️ 실행 방법

1.  **가상 환경 활성화**: (Poetry 사용 시)

    ```bash
    poetry shell
    ```

2.  **메인 시스템 실행**: `main.py`는 설정된 스케줄에 따라 각 전략 및 모니터링 작업을 실행합니다.

    ```bash
    python main.py
    ```

3.  **개별 모듈 테스트 실행**: 각 모듈(`sentiment_analysis.py`, `volatility_alert.py`, `backtest_runner.py` 등)은 자체 테스트 코드를 포함하고 있어 개별 실행하여 기능 확인이 가능합니다.

    ```bash
    python sentiment_analysis.py
    python volatility_alert.py
    python backtest_runner.py
    ```

## ⚠️ 주의 사항

*   **투자 위험**: 모든 투자는 원금 손실의 위험이 있습니다. 본 시스템은 교육 및 연구 목적으로 개발되었으며, 실제 투자 결정은 본인의 책임 하에 신중하게 진행해야 합니다.
*   **API 키 보안**: `.env` 파일에 저장된 API 키는 절대로 외부에 노출되지 않도록 주의해야 합니다. `.gitignore` 파일에 `.env`가 포함되어 있는지 확인하세요.
*   **테스트**: 실제 자금 투입 전 반드시 페이퍼 트레이딩 또는 소액 테스트를 통해 시스템의 안정성과 전략의 유효성을 충분히 검증해야 합니다.

## 🚀 향후 개선 방향 (TODO)

*   데이터베이스 연동 (SQLite, PostgreSQL 등)으로 로그 및 거래 내역 관리 효율화
*   웹 기반 대시보드 구축 (Flask/Django 또는 Streamlit/Dash)
*   머신러닝 기반 예측 모델 추가 통합
*   다양한 거래소 및 자산 지원 확대
*   Docker 기반 배포 자동화
*   보다 정교한 위험 관리 로직 추가 (예: 자금 관리 규칙)

---
*README 최종 업데이트: YYYY-MM-DD* 