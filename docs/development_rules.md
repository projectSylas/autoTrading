# 개발 규칙

## 기본 원칙

- 개발시 무슨일이있어도 개발전에 해당 개발 코드 관련 코드들을 분석해서 비슷한형식으로 개발해라
- 개발시 항상 디렉토리구조와 프로젝트 구조를 파악하고 일관성있게개발한다.

## 📘 자동매매 시스템 개발 규칙서 (v1.0)

본 문서는 자동매매 시스템의 안정적 개발 및 운영을 위한 표준 개발 규칙을 정의합니다. 이 프로젝트는 AI 기반 시세 예측과 강화학습 전략을 포함한 복합 시스템입니다.

### 📁 디렉토리/파일 구조 규칙

- 모든 신규 모듈은 분야별 디렉토리에 배치 (예: `src/core/`, `src/challenge/`, `src/ai_models/`, `src/utils/`)
- AI 모델 파일은 `src/ai_models/` 내부에 작성 (예: `price_predictor.py`, `rl_trainer.py`)
- 학습된 모델은 `model_weights/` 폴더에 저장 (Git에 포함되지 않도록 `.gitignore` 설정)
- 실험용 노트북은 `notebooks/` 내에서만 사용

### 🔧 코드 작성 규칙

- 모든 파라미터는 `.env` 또는 `src/config/settings.py`를 통해 관리 (하드코딩 금지)
- 함수에는 타입 힌트를 포함하고 예외 처리를 명시할 것
- AI 모델 클래스는 `train`, `predict`, `save`, `load` 메서드를 분리하여 작성
- 데이터프레임 가공 시 원본 변경 방지를 위해 `.copy()` 사용 필수

### 💾 데이터/로그 관리

- 모든 로그는 `logs/` 폴더 또는 PostgreSQL DB (`src/utils/database.py`)로 저장
- AI 예측 결과는 날짜별 파일로 저장하고 실제값과의 오차도 기록할 것
- Slack 알림은 `src/utils/notifier.py`를 통해 일원화하여 관리
- `auto_report.py`는 하루 전체 전략 요약 리포트를 통합하여 Slack으로 전송 (예정)

### 🧠 AI/ML 관련 규칙

- AI 모델 학습/추론 관련 파라미터는 `.env` 또는 `settings.py`에서 관리
- 예측 결과는 반드시 시각화 가능하도록 구성 (`visualizer.py`와 연동 예정)
- 모델 성능 평가 지표(RMSE, MAE 등)를 계산하고 저장
- 모델은 학습 후 `.pt` 또는 유사 형식 파일로 저장, 추론 전 존재 여부 확인
- 실시간 추론 실패 시 조건 기반 전략으로 fallback 되도록 설계

### 🧪 테스트/안정성 관리

- `tests/` 디렉토리에 주요 모듈에 대한 유닛 테스트 작성 (pytest 또는 unittest)
- 실매매 전에는 반드시 Paper 계좌 또는 Testnet으로 테스트
- 예외 발생 시 Slack 알림과 함께 로그에 전체 기록
- 전략 실행 전 `is_market_ready()` 함수로 진입 조건 사전 검증 (필요시 구현)

### 🚀 실행 및 운영 규칙

- 실행 진입점은 `main.py` 단일화 (`schedule` 기반 실행 포함)
- 배포는 `docker-compose.yml`을 기준으로 진행 (개발/운영 분리 권장)
- 전략 결과는 Slack + DB 이중 저장
- 운영 중단 시 Slack 알림 전송 + 마지막 상태 저장 후 종료

### ✅ 예시 `.env` 파라미터 구조 (참고)

```dotenv
# AI 모델 관련
LSTM_WINDOW_SIZE=60
LSTM_EPOCHS=30
LSTM_BATCH_SIZE=32
RL_EPISODES=10000
RL_LEARNING_RATE=0.0003

# 전략 조건
CORE_RSI_THRESHOLD=35
CHALLENGE_SMA_PERIOD=7
CHALLENGE_SL_RATIO=0.05
CHALLENGE_TP_RATIO=0.1

# 경로
MODEL_SAVE_DIR=./model_weights/
LOG_DIR=./logs/
```

### 📌 한 줄 요약

하드코딩 없이, 구조는 명확히, 예측은 저장하고, 실패는 슬랙으로 보고, AI와 조건 기반 전략이 유기적으로 fallback할 수 있게 만든다.

*문서 버전: v1.0*
*최종 업데이트: 2025-04-10* 