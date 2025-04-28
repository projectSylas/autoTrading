# 자동매매 시스템 프로젝트 문서

## 1. 프로젝트 개요

본 프로젝트는 Python 기반의 자동매매 시스템 개발을 목표로 합니다. 다양한 금융 상품(주식, 암호화폐 등)에 대한 데이터 분석, AI 기반 예측, 전략 백테스팅 및 실시간 거래 실행 기능을 포함합니다.

주요 모듈은 다음과 같습니다:
- **Core Portfolio (`src/core_portfolio.py`)**: 안정적인 ETF 리밸런싱 전략 (현재 구현 예정)
- **Challenge Trading (`src/challenge_trading.py`)**: 고위험/고수익 암호화폐 선물 거래 전략 (기본 구조 구현 완료)
- **Utilities (`src/utils/`)**: 공통 함수 모음 (데이터 처리, API 연동, 알림 등)
- **AI Models (`src/ai_models/`)**: 가격 예측, 감성 분석 등 AI 모델 관련 로직
- **Backtesting (`src/backtesting/`)**: 전략 검증을 위한 백테스팅 프레임워크

## 2. 주요 기능 요구사항 (진행 중)

- **챌린지 전략 (`challenge_trading.py`)**:
    - 추세선 이탈, RSI 다이버전스, 거래량 급증 등 복합 조건 기반 진입/청산 로직
    - 실시간 Slack 알림 (진입, 익절, 손절)
- **Slack 알림 시스템 (`src/utils/notifier.py`)**:
    - 구조화된 메시지 (Block Kit) 기반 상세 알림 기능 구현 완료
    - 진입, 청산, 오류 등 상황별 맞춤 알림 템플릿
- **뉴스 심리 분석 (`sentiment_analysis.py` - 예정)**:
    - 뉴스 API 또는 FinBERT 기반 시장 심리 분석
    - 매수/관망 결정에 심리 지표 반영
- **백테스팅 (`backtrader` 기반 - 예정)**:
    - `challenge_trading.py` 전략의 성과 검증
    - 수익률, MDD 등 지표 시각화
- **시계열 이상 탐지 (`volatility_alert.py` - 예정)**:
    - Prophet 모델 기반 가격 예측 및 이상치 탐지
    - 큰 괴리 발생 시 Slack 알림

## 3. 기술 스택

- **언어**: Python 3.11+
- **패키지 관리**: Poetry
- **주요 라이브러리**: pandas, numpy, matplotlib, seaborn, scikit-learn, TA-Lib, backtrader, prophet, slack_sdk, yfinance, ccxt (예정), alpaca-trade-api (예정)
- **개발 환경**: Cursor IDE
- **버전 관리**: Git

## 4. 디렉토리 구조 (주요)

```
.
├── .venv/
├── data/
├── docs/              <-- 여기!
├── logs/
├── models/
├── model_weights/
├── notebooks/
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── core_portfolio.py  (예정)
│   ├── challenge_trading.py
│   ├── analysis/
│   ├── ai_models/
│   ├── backtesting/
│   ├── config/
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── common.py
│   │   ├── database.py (예정)
│   │   ├── notifier.py
│   ├── crypto_trader/
│   │   └── ... (API, 주문 관리 등 예정)
│   └── ...
├── tests/
├── config.json
├── poetry.lock
├── pyproject.toml
└── README.md (프로젝트 루트 README)
``` 