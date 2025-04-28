# 파이썬 코드 상세 분석 요약

이 문서는 프로젝트 내 각 파이썬 파일(`.py`)을 분석한 상세 내용을 기록합니다. 각 파일의 목적, 주요 구현 방식, 현재 상태, 그리고 분석 과정에서 발견된 문제점 및 제안 사항을 포함합니다.

## src/config/settings.py

*   **목적**: 프로젝트의 모든 설정을 중앙에서 관리합니다. `.env` 파일과 환경 변수를 사용하여 API 키, 경로, 전략 파라미터, 데이터베이스 정보, 기능 플래그 등을 로드합니다.
*   **구현**:
    *   `pydantic-settings` 라이브러리의 `BaseSettings`를 사용하여 설정을 클래스 형태로 정의하고 타입 힌트 및 유효성 검사를 수행합니다.
    *   `dotenv`를 사용하여 프로젝트 루트의 `.env` 파일에서 설정을 로드합니다.
    *   API 키 (NewsAPI, Alpaca, Binance, Slack), 경로 (모델, 데이터, 로그), 전략 파라미터 (Core, Challenge), DB 접속 정보, 로깅 레벨, Slack 채널, 기능 활성화 플래그 등 다양한 설정 항목을 포함합니다.
    *   설정 로드 시 유효성 검사를 수행하고, 필수 키가 누락된 경우 경고를 로깅합니다 (`IS_CONFIG_VALID` 플래그).
    *   설정 로드 후 필요한 디렉토리(로그, 데이터, 모델 저장 등)를 자동으로 생성합니다.
    *   다른 모듈에서 `from src.config.settings import settings`로 쉽게 가져와 사용할 수 있도록 `settings` 인스턴스를 생성합니다.
*   **상태**: 잘 구조화되어 있으며, 다양한 설정 값을 체계적으로 관리하고 있습니다. `challenge_trading.py` 등에서 이 `settings` 객체를 참조하여 하드코딩된 값을 대체한 것은 좋은 방향입니다.

## src/utils/common.py

*   **목적**: 프로젝트 전반에서 사용될 수 있는 **범용 유틸리티 함수**들을 제공합니다. 데이터 로딩, VIX 지수 조회, 로그 저장 등을 담당합니다.
*   **구현**:
    *   `get_historical_data(symbol, period, interval)`: `yfinance`를 사용하여 특정 심볼의 시계열 데이터(OHLCV)를 가져옵니다. 필요한 컬럼만 선택하고 `.copy()`를 사용하여 반환합니다.
    *   `get_current_vix()`: VIX 지수(`^VIX`)의 최신 종가를 가져옵니다 (Yahoo Finance 사용).
    *   `save_log_to_csv(log_data, filename, log_dir)`: 딕셔너리 형태의 로그 데이터를 지정된 CSV 파일에 추가합니다.
    *   **제거된 함수**: `calculate_rsi`, `calculate_sma`, `detect_rsi_divergence`, `detect_trendline_breakout`, `detect_volume_spike`, `detect_support_resistance_by_volume` 등 핵심 기술 분석 함수들은 **`src/utils/strategy_utils.py`로 통합되었습니다.**
*   **상태**: 데이터 로딩, VIX 조회, 로깅 등 범용적인 유틸리티 함수만 포함하도록 **정리되었습니다.** 역할이 명확해졌습니다.
*   **결론 및 수정 필요 사항:**
    *   기술적 분석 함수들이 `src/utils/strategy_utils.py`로 성공적으로 통합되었으므로, `challenge_trading.py` 등 다른 모듈에서 기술적 분석 함수를 사용할 때는 **`src/utils/strategy_utils.py`를 import 해야 합니다.** (기존 `common.py` import 부분 확인 필요)
    *   `save_log_to_csv` 함수는 기본적인 로깅 기능만 제공하므로, DB 로깅(`src/utils/database.py`)과의 역할 분담 또는 통합을 고려할 수 있습니다.

## src/utils/strategy_utils.py

*   **목적**: `common.py`와 유사하게 전략 구현에 필요한 기술적 분석 및 유틸리티 함수들을 제공하는 것으로 보입니다. 하지만 내용상 중복되거나 더 발전된 버전의 함수들이 포함되어 있을 수 있습니다.
*   **구현**:
    *   **중복 구현 확인**:
        *   `calculate_sma(df, window, feature_col)`: `common.py`의 `calculate_sma`와 기능적으로 동일해 보입니다. 구현 방식(pandas rolling 사용)도 유사합니다.
        *   `calculate_rsi(df, window, feature_col)`: `common.py`의 `calculate_rsi`와 기능적으로 동일해 보입니다. 다만, 구현 방식이 약간 다를 수 있습니다 (EMA 사용 명시).
        *   `detect_rsi_divergence(df, price_col, rsi_col, lookback, threshold)`: `common.py`의 `detect_rsi_divergence`와 기능적으로 유사하지만, 구현 방식(low/high 찾는 방식, threshold 사용 등)이 다를 수 있습니다. 버전 2로 보이는 `detect_rsi_divergence_v2` 함수도 존재합니다.
        *   `detect_trendline_breakout_pullback(...)`: `common.py`의 `detect_trendline_breakout`과 유사한 기능을 수행할 것으로 보이며, 이름에 'pullback'이 명시되어 있습니다. Numba로 최적화된 버전(`find_breakout_pullback_nb`)도 있습니다.
        *   `calculate_poc(df, lookback, price_col, volume_col)`: POC(Point of Control) 계산 함수가 이 파일에 존재합니다. 여러 번 정의된 것으로 보이며, `common.py`의 `detect_support_resistance_by_volume`과는 다른 방식으로 구현되었을 수 있습니다.
    *   **추가 함수**:
        *   `calculate_bollinger_bands(df, window, num_std_dev, feature_col)`: 볼린저 밴드를 계산합니다.
        *   `check_sma_crossover(df, short_window, long_window, lookback)`: 단기/장기 SMA 골든크로스/데드크로스를 확인합니다.
        *   `numba` 데코레이터(`@numba.njit`)가 사용된 함수들 (`find_breakout_pullback_nb`, `check_last_crossing_nb`, `find_volume_breakout_pullback_nb` 등): 성능 향상을 위해 Numba로 최적화된 버전의 함수들이 다수 존재합니다. 이는 주로 루프 연산이 많은 계산(추세선, 크로스오버 확인 등)에 사용된 것으로 보입니다.
        *   `vectorbt` 라이브러리 import: 백테스팅 라이브러리인 `vectorbt`를 사용하는 함수가 있을 수 있습니다.
        *   `pandas_ta` 라이브러리 import: 기술적 분석 라이브러리인 `pandas_ta`를 사용하는 함수가 있을 수 있습니다 (ta-lib 래퍼 또는 자체 기능).
*   **상태**: 이 파일은 `common.py`와 상당 부분 **기능적으로 중복되는 함수들**(SMA, RSI, 다이버전스, 추세선, POC 등)을 포함하고 있습니다. 또한, Numba 최적화 버전, `vectorbt` 연관 기능 등 더 발전되거나 다른 방식으로 구현된 함수들도 존재합니다. 코드 라인 수가 많은 만큼 다양한 유틸리티가 포함되어 있을 가능성이 높습니다.
*   **결론 및 권장 조치:**
    1.  **심각한 중복**: `common.py`와 `strategy_utils.py` 사이에 핵심 기술 분석 함수들이 중복으로 구현되어 있습니다. 이는 혼란을 야기하고 유지보수를 어렵게 만듭니다.
    2.  **기능 통합 필요**: 두 파일에 흩어져 있는 유틸리티 함수들을 **하나의 파일로 통합**하거나, 명확한 기준(예: 기본 지표 vs. 복합 전략 신호, Numba 최적화 여부 등)에 따라 **역할을 분담**해야 합니다. 개발 규칙에 따라 일관성 있는 구조로 정리하는 것이 시급합니다.
    3.  **POC 함수 위치 확인**: `calculate_poc` 함수가 `strategy_utils.py`에 있는 것으로 확인되었으므로, `challenge_trading.py`는 이 파일을 참조해야 합니다.
    4.  **`challenge_trading.py` 수정 재확인**: `challenge_trading.py`의 import 문을 다시 검토하여, **`common.py` 또는 `strategy_utils.py` 중 어떤 파일의 함수를 사용할지 결정**하고 그에 맞게 수정해야 합니다. 가급적이면 더 완성도 높거나 Numba로 최적화된 버전이 있는 `strategy_utils.py`의 함수를 사용하는 것이 좋을 수 있으나, 코드 스타일이나 일관성을 고려하여 결정해야 합니다.
*   **제안:** 가장 먼저 `common.py`와 `strategy_utils.py`의 **역할을 명확히 정의하고 중복을 제거**하는 리팩토링을 진행하는 것이 좋겠습니다. 예를 들어,
    *   `common.py`: 데이터 로딩, 기본 로깅, 파일/DB 저장 등 범용 I/O 및 설정 관련 유틸리티만 남깁니다.
    *   `strategy_utils.py`: 모든 기술적 분석 지표 계산, 신호 감지(다이버전스, 추세선, 크로스오버 등), POC 계산 등 전략 구현에 직접적으로 사용되는 함수들을 모읍니다. (필요시 Numba 최적화 버전 사용)

## src/utils/notifier.py

*   **목적**: Slack 알림 발송 기능을 중앙에서 관리합니다.
*   **구현**: `requests` 라이브러리를 사용하여 Slack Webhook URL로 메시지를 보냅니다. Block Kit을 활용하여 구조화된 메시지(헤더, 필드, 마크다운 등)를 생성하는 헬퍼 함수들과, 특정 상황(진입, 종료, 오류)에 맞는 전용 알림 함수(`send_entry_alert`, `send_exit_alert`, `send_error_alert`)가 구현되어 있습니다. `settings.py`에서 Webhook URL 및 채널 정보를 가져옵니다.
*   **상태**: 최근 개선 작업을 통해 기능이 잘 구현되어 있으며, `challenge_trading.py`에서 사용하도록 업데이트되었습니다.

## src/utils/logging_config.py

*   **목적**: 애플리케이션 전체의 로깅 설정을 중앙에서 관리하고 초기화합니다.
*   **구현**:
    *   `setup_logging()` 함수를 제공하여 로깅 설정을 구성합니다.
    *   애플리케이션 로거 이름(`trading_app`)을 정의합니다.
    *   **로그 레벨을 `DEBUG`로 고정**하여 모든 수준의 로그가 기록되도록 설정했습니다. (기존에는 `settings.LOG_LEVEL`을 참조했을 수 있으나 현재 코드에서는 고정됨)
    *   **핸들러**:
        *   콘솔 핸들러 (`StreamHandler`): 모든 DEBUG 레벨 이상의 로그를 콘솔(stdout)에 출력합니다.
        *   파일 핸들러 (`RotatingFileHandler`): 모든 DEBUG 레벨 이상의 로그를 `logs/app.log` 파일에 기록합니다. 로그 파일 크기가 10MB에 도달하면 5개의 백업 파일을 유지하며 로테이션됩니다.
        *   오류 파일 핸들러 (`RotatingFileHandler`): ERROR 레벨 이상의 로그만 `logs/error.log` 파일에 기록합니다. 5MB 크기, 3개 백업.
    *   로그 포맷은 `%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s`로 설정되어 시간, 로거 이름, 라인 번호, 로그 레벨, 메시지를 포함합니다.
    *   핸들러 중복 추가를 방지하는 로직이 포함되어 있어, 여러 모듈에서 `setup_logging()`을 호출해도 안전합니다 (`force_reset=False` 기본값).
*   **상태**: 표준 로깅 설정을 제공하며, 콘솔과 파일(일반 로그, 오류 로그 분리)에 로그를 남기도록 잘 구성되어 있습니다. 로그 레벨이 DEBUG로 고정된 점은 개발 중에는 유용하지만, 운영 시에는 조정이 필요할 수 있습니다.

## src/utils/database.py

*   **목적**: PostgreSQL 데이터베이스와의 연결, 데이터 삽입 및 조회를 위한 유틸리티 함수를 제공합니다. 거래 내역, 예측 결과 등 다양한 로그를 DB에 저장하는 역할을 합니다.
*   **구현**:
    *   **Connection Pool**: `psycopg2.pool.SimpleConnectionPool`을 사용하여 DB 연결 풀을 생성하고 관리합니다. DB 접속 정보는 `settings.py` 또는 환경 변수에서 가져옵니다. (`get_db_connection`, `get_db_cursor` context manager 제공)
    *   **Table Creation**: `initialize_database()` 함수가 정의되어 있지만, 현재는 placeholder 상태입니다. 실제 테이블 생성 로직(예: `trades`, `prediction_logs`, `sentiment_logs` 등)은 다른 곳(아마도 DB 마이그레이션 스크립트나 초기 실행 시)에서 수행될 것으로 예상됩니다.
    *   **Data Insertion**:
        *   `log_trade_to_db()`: 거래 내역을 `trades` 테이블에 기록합니다. 전략 이름, 심볼, 사이드, 가격, 상태 등 다양한 정보를 인자로 받습니다.
        *   `log_prediction_to_db()`: AI 모델의 예측 결과를 `prediction_logs` 테이블에 기록합니다. 예측 시간, 심볼, 예측 가격, 실제 가격, 모델 정보 등을 인자로 받습니다.
        *   `log_sentiment_to_db()`, `log_volatility_to_db()`, `log_anomaly_to_db()`: 각 분석 결과를 해당 테이블에 기록하는 함수들이 placeholder 형태로 정의되어 있습니다.
    *   **Data Retrieval**:
        *   `get_log_data(table_name, days, limit, query_filter)`: 지정된 테이블에서 로그 데이터를 조회하여 Pandas DataFrame으로 반환합니다. 기간, 개수 제한, 추가 필터링 조건 적용이 가능합니다.
*   **상태**: PostgreSQL DB 연동을 위한 기본적인 유틸리티가 잘 갖춰져 있습니다. 거래 내역 및 예측 결과 로깅 함수는 구현되어 있으나, 다른 분석 결과 로깅은 아직 placeholder 상태입니다. 테이블 스키마 정의 및 생성 로직은 이 파일 외부에 있을 것으로 보입니다.

## src/utils/ticker_scanner.py

*   **목적**: 지정된 티커 목록(universe)에서 특정 조건(예: 거래량 급증)을 만족하는 티커를 스캔하여 찾아내는 유틸리티입니다.
*   **구현**:
    *   `fetch_data_for_universe(tickers, period, interval)`: `yfinance.download`를 사용하여 여러 티커의 데이터를 한 번에 효율적으로 가져옵니다. 결과를 티커별 DataFrame 딕셔너리로 반환합니다.
    *   `scan_for_volume_spikes(ticker_universe, avg_volume_period, volume_spike_ratio, min_volume_threshold)`:
        *   `fetch_data_for_universe`를 호출하여 티커 목록의 데이터를 가져옵니다.
        *   각 티커의 DataFrame에 대해 이동평균 거래량을 계산합니다.
        *   최신 거래량이 이동평균 거래량의 지정된 배수(ratio)를 초과하고, 동시에 최소 절대 거래량(threshold) 기준도 만족하는지 확인합니다.
        *   조건을 만족하는 티커 목록을 반환합니다.
*   **상태**: 거래량 급증이라는 특정 조건에 맞춰 티커를 스캔하는 기능이 잘 구현되어 있습니다. 다른 스캔 조건(예: 변동성 급증, 특정 패턴 발생 등)을 추가하여 확장할 수 있는 구조입니다.

## src/utils/mcp_runner.py

*   **목적**: MCP(Model/Meta-experiment Control Plane? 추정) 컨텍스트 파일을 기반으로 자동화된 백테스트 실험을 실행하고 결과를 저장하는 에이전트입니다. LLM 등에 의해 호출되어 실험을 수행하는 용도로 설계된 것으로 보입니다.
*   **구현**:
    *   **MCP 파일 로드 및 검증**:
        *   `load_mcp(filepath)`: JSON 형식의 MCP 파일을 로드하고, 필수 키(프로젝트, 전략 코어, 실행 조건, SL/TP, 전략 파라미터, 메트릭)가 있는지, `strategy_core` 값이 `STRATEGY_MAP`에 정의되어 있는지, 실행 조건 및 SL/TP 구조가 유효한지 등을 검증합니다.
    *   **전략 함수 매핑**:
        *   `STRATEGY_MAP`: MCP 파일의 `strategy_core` 문자열(예: "breakout-pullback")을 실제 실행할 백테스트 함수(예: `run_vectorbt_backtest`)에 매핑합니다.
    *   **백테스트 인자 준비**:
        *   `prepare_backtest_args(mcp, target_function)`: 로드된 MCP 데이터와 목표 백테스트 함수의 시그니처(`inspect.signature`)를 비교하여, 함수가 실제로 받는 인자들만 동적으로 구성합니다. MCP의 실행 조건, SL/TP, 전략 파라미터 등을 함수의 인자에 맞게 매핑합니다. 알 수 없는 파라미터는 무시하고 경고를 로깅합니다.
    *   **실험 실행**:
        *   `run_experiment_from_mcp(mcp)`: `STRATEGY_MAP`을 통해 적절한 백테스트 함수를 선택하고, `prepare_backtest_args`로 준비된 인자를 사용하여 함수를 호출합니다. 백테스트 결과(통계 딕셔너리)를 받습니다.
        *   반환된 통계에서 MCP 파일에 명시된 메트릭(`metrics`)만 추출하고, JSON 호환 타입으로 변환하여 결과를 구성합니다.
    *   **결과 저장**:
        *   `save_result(result, context_path, output_dir, output_filename)`: 실행 결과(추출된 메트릭 또는 오류 메시지), 원본 MCP 파일 경로, 실행 시간 등을 포함하는 JSON 파일을 지정된 경로(`mcp/results/`)에 저장합니다.
*   **상태**: JSON 파일을 이용한 백테스트 자동화 실행 프레임워크가 잘 구축되어 있습니다. 다양한 백테스트 함수와 파라미터를 유연하게 호출하고 결과를 정형화하여 저장할 수 있습니다. 주로 `vectorbt` 기반의 백테스트 함수(`run_vectorbt_backtest`)와 연동되는 것으로 보입니다.

## src/analysis/sentiment.py

*   **목적**: 뉴스 기사를 수집하고 분석하여 시장 또는 특정 키워드에 대한 감성(sentiment)을 파악합니다.
*   **구현**:
    *   **뉴스 수집**: `fetch_recent_news(keyword, days_ago, language)` 함수는 `NewsApiClient`를 사용하여 `settings.py`에 설정된 API 키로 특정 키워드에 대한 뉴스를 수집합니다.
    *   **Hugging Face 모델 초기화**: `initialize_sentiment_model()` 함수는 `settings.HF_SENTIMENT_MODEL_NAME`에 지정된 Hugging Face 감성 분석 모델과 토크나이저를 로드합니다. `transformers` 라이브러리의 `pipeline`을 사용하며, GPU 사용 여부를 설정에 따라 결정합니다 (`settings.HF_DEVICE`). 모델 로딩은 한 번만 수행됩니다 (`_model_loaded` 플래그).
    *   **감성 분석**: `analyze_sentiment_hf(texts)` 함수는 로드된 모델(`sentiment_pipeline`)을 사용하여 입력된 텍스트 목록의 감성을 분석합니다. 결과를 'positive', 'neutral', 'negative' 라벨과 0.0(부정)~1.0(긍정) 사이의 정규화된 점수로 변환합니다. 중립 임계값(`settings.SENTIMENT_NEUTRAL_THRESHOLD_LOW/HIGH`)을 사용하여 라벨을 결정합니다.
    *   **통합 함수**: `get_market_sentiment(keyword, days_ago, language)` 함수는 뉴스 수집과 감성 분석을 통합하여 수행합니다.
        *   키워드로 뉴스를 수집합니다.
        *   뉴스 제목과 설명을 합쳐 분석할 텍스트를 준비합니다.
        *   `analyze_sentiment_hf`를 호출하여 감성을 분석합니다.
        *   개별 분석 결과의 평균 점수를 계산하고, 평균 점수를 기준으로 전체 감성 라벨('positive', 'neutral', 'negative')을 결정합니다.
        *   결과를 DB에 로깅합니다 (`log_sentiment_to_db` 호출).
        *   최종적으로 (전체 라벨, 평균 점수, 원본 뉴스 DataFrame) 튜플을 반환합니다.
    *   **DB 로깅**: `log_sentiment_to_db` 함수를 사용하여 분석 결과를 DB에 저장하는 로직이 포함되어 있습니다 (함수 import 실패 시 경고 로깅).
*   **상태**: NewsAPI를 이용한 뉴스 수집과 Hugging Face 트랜스포머 모델 기반의 감성 분석 기능이 잘 구현되어 있습니다. 점수 정규화 및 전체 감성 판단 로직도 포함되어 있습니다. DB 로깅과의 연동도 고려되어 있습니다. `settings.py`를 통해 모델 이름, API 키, 기능 활성화 여부 등을 제어할 수 있습니다.

## src/analysis/volatility_alert.py

*   **목적**: 시계열 가격 데이터를 분석하여 변동성 이상 또는 예측 모델 기반의 이상 가격 움직임을 탐지하고 알림을 보냅니다. `Prophet` 라이브러리를 사용하여 가격 예측 및 이상 탐지를 수행합니다.
*   **구현**:
    *   **Prophet 의존성**: `prophet` 라이브러리 설치 여부를 확인하고, 설치되지 않은 경우 경고를 출력하고 기능을 비활성화합니다 (dummy 클래스 사용).
    *   **데이터 포맷팅**: `format_data_for_prophet(df)` 함수는 시계열 DataFrame(DatetimeIndex, 'close' 컬럼)을 Prophet 모델 입력 형식('ds', 'y' 컬럼)으로 변환합니다.
    *   **가격 예측**: `forecast_price(df_prophet, periods, freq)` 함수는 포맷팅된 데이터를 사용하여 Prophet 모델을 학습시키고, 지정된 기간(`periods`) 및 주기(`freq`)만큼 미래 가격(`yhat`)과 신뢰 구간(`yhat_lower`, `yhat_upper`)을 예측합니다.
    *   **이상 탐지 (구간 기반)**: `detect_anomaly(forecast_df, actual_price)` 함수는 최신 실제 가격(`actual_price`)이 가장 최근 예측의 신뢰 구간(`yhat_lower` ~ `yhat_upper`)을 벗어나는지 확인하여 이상 여부를 판단합니다.
    *   **통합 실행 및 알림**: `check_price_anomaly_and_notify(symbol, data_period, interval, forecast_periods, threshold_pct)` 함수는 전체 과정을 통합합니다.
        *   `get_historical_data`를 호출하여 데이터를 가져옵니다.
        *   데이터를 Prophet 형식으로 포맷합니다.
        *   `forecast_price`를 호출하여 예측을 생성합니다.
        *   (`threshold_pct`가 주어지지 않은 경우) `detect_anomaly`를 사용하여 예측 구간 기반 이상 탐지를 수행합니다.
        *   (`threshold_pct`가 주어진 경우) 최신 예측값(`yhat`)과 실제 가격(`latest_actual_price`)의 백분율 차이가 `threshold_pct`를 초과하는지 확인하여 이상 여부를 판단하는 로직이 필요합니다 (현재 코드 스니펫에는 명확히 보이지 않으나, 함수 인자로 추가된 것으로 보아 구현 의도가 있음).
        *   이상이 감지되면 `send_slack_notification`을 호출하여 Slack 알림을 보냅니다.
        *   최종적으로 이상 감지 여부(True/False)를 반환합니다.
*   **상태**: Prophet을 이용한 시계열 예측 및 이상 탐지 기능의 핵심 로직이 구현되어 있습니다. 데이터 가져오기, 포맷팅, 예측, 이상 탐지, 알림까지의 흐름이 잘 구성되어 있습니다. `threshold_pct`를 이용한 백분율 기반 이상 탐지 로직은 추가 확인이 필요합니다.

## src/analysis/visualizer.py

*   **목적**: 데이터베이스에 저장된 로그 데이터(거래 내역 등)를 기반으로 전략 성과를 요약하고 시각화하는 기능을 제공합니다.
*   **구현**:
    *   **데이터 로딩**: `src.utils.database.get_log_data` 함수를 사용하여 'trades' 테이블에서 특정 전략('core', 'challenge')의 데이터를 가져옵니다. (과거에는 CSV 파일 로딩이었을 수 있으나 현재는 DB 조회로 변경됨)
    *   **Core 전략 요약 및 시각화**: `plot_core_summary(df)` 함수는 안정형 포트폴리오(`core`)의 거래 로그를 받아 총 거래 횟수, 매수/매도 횟수, 주문 상태별 분포 등을 로깅합니다. Matplotlib을 사용하여 주문 상태 분포를 파이 차트로 시각화하고 이미지 파일(`core_status_summary.png`)로 저장합니다.
    *   **Challenge 전략 요약 및 시각화**: `plot_challenge_summary(df)` 함수는 챌린지 전략(`challenge`)의 거래 로그를 받아 총 진입/청산 횟수, 승률, 평균 손익비 등을 계산하여 로깅합니다. Plotly를 사용하여 누적 수익률 곡선을 그리고 HTML 파일(`challenge_cumulative_pnl.html`)로 저장합니다. PnL 계산 시 백분율 문자열 처리 로직이 포함되어 있습니다.
    *   **메인 실행**: `run_visualizer(days)` 함수는 지정된 기간(`days`) 동안의 로그 데이터를 DB에서 가져와 각 전략별 요약 및 시각화 함수를 호출합니다.
*   **상태**: Matplotlib과 Plotly를 사용하여 각 전략의 성과를 분석하고 시각화하는 기능이 구현되어 있습니다. 데이터 소스는 CSV 파일에서 DB 조회(`get_log_data`)로 변경되었습니다. 로그 데이터 형식(특히 PnL 컬럼)에 따라 일부 로직 수정이 필요할 수 있습니다.

## src/main.py

*   **목적**: 애플리케이션의 메인 진입점(entry point) 역할을 합니다. `schedule` 라이브러리를 사용하여 정의된 작업(분석 파이프라인, 이상 탐지 등)을 주기적으로 실행합니다.
*   **구현**:
    *   **모듈 임포트**: 백테스트, 감성 분석, 변동성 분석, 이상 탐지, 알림 등 필요한 기능 모듈들을 `try-except` 블록으로 감싸 안전하게 임포트합니다. 임포트 실패 시 해당 기능은 비활성화되고 오류 로그가 기록됩니다.
    *   **설정 로드**: `src.config.settings.settings` 인스턴스를 직접 임포트하여 사용합니다.
    *   **분석 파이프라인**:
        *   `run_analysis_pipeline()`: 백테스트, 감성 분석, 변동성 감지 등을 순차적으로 실행하는 함수입니다. 각 기능은 `settings.py`의 플래그(`ENABLE_BACKTEST`, `ENABLE_SENTIMENT`, `ENABLE_VOLATILITY`)에 따라 활성화/비활성화됩니다.
        *   각 단계의 실행 결과나 오류는 로깅되고, Slack 알림이 전송됩니다.
        *   (주석 처리된 부분) 과거에는 결과를 로컬 CSV 파일에 저장했을 수 있으나, 현재는 각 모듈이 내부적으로 DB에 로그를 저장하고, `main.py`는 실행 완료 여부나 간단한 요약 정보만 처리하는 구조로 변경된 것으로 보입니다.
    *   **이상 탐지 잡**:
        *   `run_anomaly_detection_job()`: `src.analysis.volatility_alert.check_price_anomaly_and_notify` 함수를 호출하여 가격 이상 탐지를 수행합니다. `settings.ENABLE_ANOMALY_DETECTION` 플래그로 활성화 여부를 제어합니다.
    *   **스케줄링**:
        *   `schedule` 라이브러리를 사용하여 정의된 잡들을 예약합니다.
        *   `run_analysis_pipeline`은 매일 오전 9시에 실행되도록 예약되어 있습니다 (설정 플래그 확인).
        *   `run_anomaly_detection_job`은 매시간 정각에 실행되도록 예약되어 있습니다 (설정 플래그 확인).
    *   **메인 루프**: `if __name__ == "__main__":` 블록에서 스케줄러를 시작하고 무한 루프를 돌며 예약된 작업을 확인하고 실행합니다 (`schedule.run_pending()`).
*   **상태**: 애플리케이션의 실행 흐름을 제어하고, 스케줄링을 통해 자동화를 구현하는 핵심 파일입니다. 기능별 활성화 플래그와 안전한 모듈 임포트, 상세 로깅 및 알림 기능을 갖추고 있습니다. 데이터 로깅 방식은 CSV에서 DB 중심으로 변경된 것으로 추정됩니다.

## src/reporting/performance_analyzer.py

*   **목적**: 다양한 전략(FinRL, FreqAI 등)의 백테스트 결과 파일을 로드하여 주요 성과 지표를 비교하고 시각화합니다.
*   **구현**:
    *   **데이터 로딩**:
        *   `load_finrl_backtest_results(file_path)`: FinRL 백테스트 결과 파일(CSV 가정)을 로드합니다. 파일 형식(컬럼 직접 저장 vs Metric/Value)에 따라 처리하는 로직 주석이 있습니다.
        *   `load_freqai_backtest_results(file_path)`: FreqAI 백테스트 결과 파일(JSON이 일반적이나 여기선 CSV로 가정)을 로드합니다. 실제 JSON 파싱 로직은 placeholder 상태입니다.
    *   **성과 비교**:
        *   `compare_strategies(results_list)`: 여러 전략의 결과 DataFrame 리스트를 입력받아, 정의된 핵심 지표(`key_metrics`: 총 수익률, 샤프 비율, MDD 등)를 추출하고 비교하는 단일 DataFrame을 생성합니다. 결과 파일 형식 차이를 일부 처리하려는 시도가 보입니다.
    *   **시각화**:
        *   `plot_comparison(comparison_df, save_path)`: 비교 DataFrame을 받아 특정 지표(예: 샤프 비율, 총 수익률)를 막대그래프로 시각화하고, `settings.REPORTING_SAVE_DIR`에 지정된 경로 또는 제공된 경로에 이미지 파일(`strategy_comparison_plot.png`)로 저장합니다. Matplotlib, Seaborn을 사용합니다.
*   **상태**: 여러 백테스팅 프레임워크(FinRL, FreqAI)의 결과를 통합하여 비교하고 시각화하는 기본 구조를 갖추고 있습니다. 다만, 각 프레임워크의 실제 결과 파일 형식에 맞춰 데이터 로딩 및 지표 추출 로직을 구체화해야 할 수 있습니다.

## src/backtesting/strategies/challenge_strategy_backtest.py

*   **목적**: `challenge_trading.py`에서 구현된 플라이트 챌린지 전략 로직을 `backtrader` 프레임워크 내에서 실행하여 백테스팅하기 위한 클래스입니다.
*   **구현**:
    *   **클래스 정의**: `ChallengeStrategyBacktest(bt.Strategy)` 클래스를 정의하여 `backtrader` 전략으로 작동합니다.
    *   **파라미터**: `params` 튜플을 통해 백테스트 실행 시 전달받을 파라미터(SMA 기간, RSI 임계값, 거래량 조건, SL/TP 비율, 감성 분석 키워드 및 활성화 여부 등)를 정의합니다.
    *   **초기화 (`__init__`)**:
        *   백테스트 실행 시 전략별 로그 파일을 생성합니다 (`_write_log` 함수 사용).
        *   데이터 라인(close, high, low, volume)을 참조합니다.
        *   주문 추적 변수들을 초기화합니다.
        *   **Backtrader 내부 지표 사용**: SMA, RSI, 거래량 이동평균, 최고가/최저가 등을 `bt.indicators`를 사용하여 계산합니다. 외부 라이브러리(`strategy_utils` 등) 의존성을 제거하려는 시도가 보입니다.
        *   감성 분석 관련 상태 변수를 초기화합니다.
    *   **로깅**: `_write_log(txt)` 함수는 타임스탬프와 함께 메시지를 전략 전용 로그 파일에 기록합니다.
    *   **주문/거래 알림 (`notify_order`, `notify_trade`)**: 주문 상태 변경 및 거래 종료 시 로그를 기록하고, SL/TP 가격을 설정/해제합니다.
    *   **핵심 로직 (`next`)**:
        *   각 bar마다 실행됩니다.
        *   지표가 준비되었는지 확인합니다.
        *   주문이 진행 중이면 대기합니다.
        *   **포지션 관리**: 현재 포지션이 있는 경우, 현재 가격이 설정된 SL/TP 가격에 도달했는지 확인하고, 도달 시 `self.close()`를 호출하여 포지션을 종료합니다.
        *   **감성 분석**: `enable_sentiment_filter`가 True이고 일정 주기(예: 24 bar)마다 `get_market_sentiment` 함수를 호출하여 시장 감성을 업데이트하고, 이를 `self.current_market_sentiment`에 저장합니다. (API 호출 성능 고려)
        *   **진입 신호**: 포지션이 없는 경우 진입 조건을 확인합니다.
            *   Backtrader 내부 지표를 사용하여 SMA 크로스오버, RSI 과매도, 거래량 급증, 다이버전스(내부 구현), 추세선 돌파/되돌림(내부 구현) 등 다양한 조건을 계산합니다.
            *   이 조건들을 조합하여 최종 Long 진입 신호(`long_signal`)를 결정합니다.
            *   **감성 필터 적용**: `enable_sentiment_filter`가 True이고 현재 감성이 'bearish'이면 `long_signal`이 True라도 진입하지 않습니다.
            *   최종 진입 조건 충족 시 `self.buy()`를 호출합니다. (Short 진입 로직은 현재 코드 스니펫에 명확히 보이지 않음)
    *   **종료 (`stop`)**: 백테스트 종료 시 필요한 정리 작업을 수행합니다 (예: 로그 파일 닫기).
*   **상태**: `backtrader`를 사용하여 챌린지 전략을 백테스팅하기 위한 클래스가 잘 구현되어 있습니다. 특히, 외부 라이브러리 의존성을 줄이고 Backtrader 내장 지표와 로직을 활용하려는 노력이 보입니다. 감성 분석 결과를 필터로 사용하는 기능도 통합되어 있습니다. 전략별 상세 로그 파일 생성 기능은 디버깅에 유용합니다. **이 파일이 `src/backtest` 디렉토리의 유사 파일들보다 완성도가 높아 보입니다.**

## src/backtesting/backtest_runner_vbt.py

*   **목적**: `vectorbt` 라이브러리를 사용하여 벡터화된 백테스팅을 실행합니다. 특히, AI 모델(`src.signals.ai_signal_generator`)이 생성한 진입/종료 신호를 기반으로 백테스팅을 수행하도록 구성되어 있습니다. 파라미터 최적화 기능도 포함할 수 있습니다.
*   **구현**:
    *   **데이터 로딩 및 검증**:
        *   `yf.download`를 사용하여 데이터를 가져오고, MultiIndex 컬럼을 처리합니다.
        *   컬럼명을 소문자로 통일하고, 필요한 컬럼(ohlcv) 존재 여부 및 NaN 값, 0 이하 가격 데이터를 검사하고 처리합니다.
    *   **지표 계산**:
        *   `vectorbt`의 내장 함수(`vbt.MA.run`, `vbt.RSI.run`, `vbt.ADX.run`, `vbt.ATR.run`)를 사용하여 SMA, RSI, ADX, ATR 등의 지표를 계산합니다. NaN 값 발생 시 경고 및 처리 로직이 포함되어 있습니다.
    *   **AI 신호 생성**:
        *   `src.signals.ai_signal_generator`의 `generate_ai_entry_signals` 함수를 호출하여 AI 모델 기반의 진입 신호(True/False Series)를 생성합니다. 모델 경로, 임계값, 윈도우 크기 등을 인자로 받습니다.
        *   `generate_ai_exit_signals` 함수도 호출하지만, 현재는 placeholder 상태(모두 False 반환)로 보입니다.
    *   **신호 조합 및 필터링**:
        *   ADX 지표(`adx_value`)가 `adx_threshold`보다 낮은 경우 (추세 약함) AI 진입 신호를 무시하는 필터링 로직(`use_trend_filter`)이 있습니다.
        *   AI 진입 신호와 AI 종료 신호 (현재는 placeholder)를 조합하여 최종 진입/종료 신호(`entries`, `exits`)를 생성합니다.
    *   **백테스팅 실행**:
        *   `vbt.Portfolio.from_signals`를 사용하여 준비된 진입/종료 신호, 가격 데이터, 초기 자금, 수수료 등을 바탕으로 백테스팅을 실행합니다.
        *   SL/TP 로직:
            *   `use_atr_exit=True`인 경우, 계산된 ATR 값을 기반으로 SL/TP 가격을 동적으로 설정합니다 (`sl_stop=atr * atr_sl_multiplier`, `tp_stop=atr * atr_tp_multiplier`).
            *   `use_atr_exit=False`인 경우, 설정에서 받아온 고정 비율(`sl_stop`, `tp_stop`)을 사용합니다.
    *   **결과 추출 및 반환**:
        *   백테스트 결과(`portfolio`)에서 주요 통계 지표(총 수익률, 승률, 샤프 비율, MDD 등)를 추출합니다 (`extract_stats` 함수 사용 추정).
        *   추출된 통계를 딕셔너리 형태로 반환합니다. 오류 발생 시 에러 메시지를 포함한 딕셔너리를 반환합니다.
    *   **파라미터 최적화**:
        *   `optimize_atr_params(...)` 함수는 ATR 관련 파라미터(window, sl_multiplier, tp_multiplier) 조합을 변경해가며 `run_vectorbt_backtest`를 반복 실행하고, 최적의 파라미터 조합을 찾는 기능을 수행합니다 (`itertools.product` 사용).
    *   **Plotting**: `plot=True`인 경우 백테스트 결과를 시각화합니다 (`portfolio.plot()`).
*   **상태**: AI 신호 기반의 벡터화된 백테스팅 및 파라미터 최적화 기능이 `vectorbt`를 사용하여 잘 구현되어 있습니다. 데이터 검증, 지표 계산, 신호 생성, 백테스팅 실행, 결과 추출까지의 파이프라인이 명확합니다. `mcp_runner.py`에서 이 함수를 호출하여 자동화된 실험을 수행할 수 있습니다.

## src/backtesting/backtest_runner.py

*   **목적**: `backtrader` 라이브러리를 사용하여 이벤트 기반 백테스팅을 실행하는 메인 스크립트입니다. 데이터 로딩, 전략 추가, Cerebro 엔진 설정 및 실행, 결과 분석 및 시각화까지의 전체 과정을 관리합니다.
*   **구현**:
    *   **환경 설정**: 프로젝트 루트 경로를 `sys.path`에 추가하고, 로깅 설정을 초기화합니다 (`src.utils.logging_config`). `settings.py`에서 설정을 로드하며, 실패 시 폴백 설정을 사용합니다.
    *   **전략 임포트**: `src.backtesting.strategies.challenge_strategy_backtest.ChallengeStrategyBacktest` 전략 클래스를 임포트합니다. 임포트 실패 시 최소 기능의 폴백 전략을 사용합니다.
    *   **Cerebro 설정**: `backtrader.Cerebro` 엔진 인스턴스를 생성합니다.
    *   **전략 추가**: `cerebro.addstrategy()`를 사용하여 임포트된 전략 클래스와 해당 파라미터 (`settings.py`에서 가져옴)를 Cerebro 엔진에 추가합니다.
    *   **데이터 로딩 및 추가**:
        *   `settings.py`에 정의된 데이터 소스(`yahoo`, `csv`, `db`)에 따라 데이터를 로드합니다.
        *   `yahoo`: `yf.download`를 사용하여 데이터를 가져옵니다. 기간 또는 시작/종료 날짜를 사용하며, MultiIndex 컬럼 처리 로직이 포함되어 있습니다.
        *   `csv`: `pd.read_csv`로 데이터를 로드하고, 날짜/시간 컬럼을 찾아 인덱스로 설정합니다. 날짜 범위 필터링을 지원합니다.
        *   `db`: `src.utils.common.get_historical_data`를 호출하여 데이터베이스에서 데이터를 가져옵니다 (DB 함수 구현 필요 가정).
        *   **데이터 검증 및 전처리**: 로드된 데이터의 컬럼명을 표준화(소문자, 스페이스->언더스코어)하고, 'adj_close'를 'close'로 사용하며, 필수 OHLCV 컬럼 존재 여부를 확인합니다. NaN 값을 처리(ffill/bfill)하고, 0 이하 가격 데이터를 검증합니다.
        *   최종적으로 준비된 데이터를 `bt.feeds.PandasData` 형식으로 변환하여 `cerebro.adddata()`를 통해 엔진에 추가합니다.
    *   **브로커 설정**:
        *   초기 자금 (`settings.BACKTEST_INITIAL_CASH`)과 수수료 (`settings.BACKTEST_COMMISSION`)를 설정합니다.
        *   매수 크기(`stake`)를 설정합니다 (`cerebro.addsizer`).
    *   **분석기 추가**:
        *   `TradeAnalyzer`: 거래별 상세 정보를 분석합니다.
        *   `SharpeRatio`: 샤프 비율을 계산합니다.
        *   `Returns`: 수익률 관련 지표를 계산합니다.
        *   `DrawDown`: 최대 낙폭(MDD)을 계산합니다.
    *   **백테스트 실행**:
        *   `cerebro.run()`을 호출하여 백테스트를 실행합니다.
        *   실행 결과를 `results` 변수에 저장합니다 (전략 인스턴스 리스트).
    *   **결과 분석 및 출력**:
        *   실행 결과에서 분석기(`analyzer`) 데이터를 추출하여 주요 지표(총 수익, 순수익, MDD, 샤프 비율 등)를 계산하고 로깅합니다.
    *   **시각화**: `settings.BACKTEST_PLOT`이 True이면 `cerebro.plot()`을 호출하여 백테스트 결과를 시각화합니다 (Matplotlib 사용).
*   **상태**: `backtrader` 기반의 백테스팅 실행기가 완전하게 구현되어 있습니다. 다양한 데이터 소스를 지원하고, 데이터 검증, 전략 설정, 분석기 추가, 결과 분석 및 시각화까지 포괄적인 기능을 제공합니다. `settings.py`를 통해 대부분의 설정을 제어할 수 있습니다. 이 파일은 `ChallengeStrategyBacktest` 전략을 실행하는 데 사용됩니다. **`src/backtest/runner.py`와 기능 중복.**

## src/core/portfolio.py

*   **목적**: 안정형 포트폴리오 전략(`core`)의 실행 로직을 담당합니다. Alpaca API를 사용하여 계정 정보를 조회하고, ETF 등의 자산을 리밸런싱하며, 특정 조건에 따라 매수/매도를 결정합니다.
*   **구현**:
    *   **Alpaca API 연동**: `alpaca_trade_api` 라이브러리를 사용하여 Alpaca 계정(`settings.py`의 API 키, URL 사용)에 연결하고, 계정 정보 조회(`get_account_details`), 보유 포지션 조회(`get_positions`), 주문 제출(`submit_order`) 등의 기능을 수행합니다.
    *   **매수 조건 확인**: `check_buy_conditions(symbol)` 함수는 특정 종목에 대해 매수 조건(RSI 과매도, VIX 임계값 초과, 금리 조건(TODO))을 `src/utils/common.py`의 유틸리티 함수(`get_current_vix`, `calculate_rsi`)를 사용하여 확인합니다.
    *   **목표 비중 계산**: `calculate_target_allocations(symbols)` 함수는 포트폴리오에 포함될 자산 목록을 받아 목표 비중(현재는 동일 비중)을 계산합니다.
    *   **리밸런싱 필요 여부 판단**: `needs_rebalance(current_positions, target_allocations)` 함수는 현재 포트폴리오의 자산별 비중과 목표 비중을 비교하여, 차이가 설정된 임계값(`settings.CORE_REBALANCE_THRESHOLD`)을 초과하면 리밸런싱이 필요하다고 판단합니다.
    *   **매수 실행**: `execute_buy_order(symbol, amount_usd)` 함수는 지정된 금액만큼 특정 종목을 시장가로 매수하는 주문을 제출합니다. 최소 주문 금액($1) 확인 로직이 포함되어 있습니다.
    *   **리밸런싱 실행**: `execute_rebalance(target_allocations)` 함수는 목표 비중에 맞춰 포트폴리오를 조정합니다.
        *   현재 포지션과 목표 비중을 비교하여 매도 또는 매수해야 할 금액을 계산합니다.
        *   Notional(금액 기준) 주문을 사용하여 매도 주문을 먼저 생성하고, 그 다음 매수 주문을 생성합니다.
        *   생성된 주문 목록을 Alpaca API를 통해 제출합니다 (매도 먼저).
    *   **거래 로깅**: `log_transaction(...)` 함수는 (아직 전체 코드는 안 보이지만) 거래 정보를 로깅하는 역할을 할 것으로 예상됩니다 (DB 또는 CSV).
    *   **월별 실행 제어**: `is_first_run_of_month()` 함수와 `_last_rebalance_check_month` 전역 변수를 사용하여 리밸런싱 검사 또는 특정 작업이 월 1회만 수행되도록 제어합니다.
    *   **메인 전략 실행**: `run_core_portfolio_strategy()` 함수는 전체 코어 전략 로직을 실행합니다.
        *   월 1회 리밸런싱 필요 여부를 확인하고, 필요시 `execute_rebalance`를 호출합니다.
        *   리밸런싱이 필요 없거나 완료된 후, 보유하지 않은 핵심 자산(`settings.CORE_ASSETS`)에 대해 `check_buy_conditions`를 확인하고, 조건 충족 시 일정 금액(`config.CORE_BUY_AMOUNT`)만큼 매수 주문(`execute_buy_order`)을 실행합니다.
*   **상태**: Alpaca API를 이용한 안정형 포트폴리오 리밸런싱 및 조건부 매수 전략의 핵심 로직이 구현되어 있습니다. 매수 조건 확인, 목표 비중 계산, 리밸런싱 필요성 판단, 주문 실행 등 필요한 기능들이 모듈화되어 있습니다. `strategy_utils` (`common.py`) 및 `notifier.py`와 연동되어 작동합니다. 금리 조건 확인은 아직 구현되지 않았습니다 (TODO).

## src/analysis/volatility.py

*   **목적**: `volatility_alert.py`와 유사하게 `Prophet` 라이브러리를 사용하여 시계열 가격을 예측하고, 실제 가격과의 **괴리율**을 계산하여 임계값 기반으로 이상 변동성을 탐지합니다. 탐지 시 Slack 알림을 보내고 결과를 DB에 로깅합니다.
*   **구현**:
    *   **Prophet 연동**: `volatility_alert.py`와 거의 동일한 `prepare_data_for_prophet` 및 `forecast_price` 함수를 사용하여 Prophet 모델을 학습하고 예측을 수행합니다.
    *   **이상 탐지 (괴리율 기반)**:
        *   `detect_anomaly(forecast, actual_df, price_col, threshold_percent)` 함수는 최신 실제 가격과 해당 시점의 예측 가격(`yhat`)을 비교합니다.
        *   `(실제가 - 예측가) / 예측가 * 100` 공식을 사용하여 괴리율(%)을 계산합니다.
        *   계산된 괴리율의 절대값이 설정된 임계값(`threshold_percent`)을 초과하면 이상 현상으로 판단합니다.
        *   이상이 감지되면 Slack으로 알림을 보냅니다.
        *   (이상 여부, 괴리율, 시점, 실제가, 예측가) 튜플을 반환합니다.
    *   **통합 실행**: `run_volatility_check(symbol, history_days, forecast_hours, interval, threshold)` 함수는 데이터 로딩(`common.py`의 `get_historical_data` 사용), Prophet 데이터 준비, 예측, 이상 탐지(괴리율 기반) 과정을 통합하여 실행하고, 결과를 DB에 로깅(`log_volatility_to_db`)합니다.
*   **상태**: `volatility_alert.py`와 상당히 유사한 기능을 수행하지만, 이상 탐지 방식에서 차이가 있습니다.
    *   `volatility_alert.py`: 예측 **신뢰 구간(yhat_lower ~ yhat_upper)**을 벗어나는지 확인하거나, 명시적 **괴리율 임계값(threshold_pct)**을 사용할 수 있습니다 (구현에 따라).
    *   `volatility.py`: 예측 **중심값(yhat)**과의 **괴리율**이 임계값을 초과하는지 확인합니다.
    *   두 파일 모두 Prophet 기반 예측 및 알림/로깅 기능을 포함하고 있어, **기능 중복 및 역할 모호성**이 존재합니다. 하나로 통합하거나 역할을 명확히 분리할 필요가 있습니다.

## src/optimizer/optimize_strategies.py

*   **목적**: 특정 전략(`run_vectorbt_backtest` 함수 사용)의 파라미터를 최적화하기 위해 파라미터 그리드 서치를 수행합니다. 여러 파라미터 조합에 대해 백테스트를 실행하고 결과를 비교하여 최상의 조합을 찾습니다.
*   **구현**:
    *   **설정**: 최적화할 대상 심볼, 기간, 타임프레임, 초기 자금 등 기본 실행 컨텍스트와 고정 파라미터(`FIXED_PARAMS`), 그리고 최적화할 파라미터 그리드(`param_grid`)를 정의합니다. 수집할 성능 지표(`METRICS_TO_COLLECT`)도 정의합니다.
    *   **파라미터 조합 생성**: `itertools.product`를 사용하여 `param_grid`에 정의된 모든 파라미터 조합을 생성합니다.
    *   **백테스트 반복 실행**:
        *   생성된 각 파라미터 조합에 대해 `tqdm`을 사용하여 진행 상황을 표시하며 루프를 돕니다.
        *   `src.backtesting.backtest_runner_vbt.run_vectorbt_backtest` 함수를 호출하기 위한 인자(`backtest_args`)를 준비합니다 (고정 파라미터 + 현재 조합 파라미터).
        *   백테스트를 실행하고 결과 통계(`stats`)를 받습니다.
        *   각 실행 결과(파라미터 조합 + 통계 지표)를 `results_list`에 저장합니다. 오류 발생 시 에러 메시지와 함께 NaN 값으로 기록합니다.
    *   **결과 처리 및 저장**:
        *   모든 백테스트 실행 후 `results_list`를 Pandas DataFrame으로 변환합니다.
        *   결과 DataFrame을 지정된 경로(`results/experiment_summary.csv`)에 CSV 파일로 저장합니다.
    *   **최적 결과 출력**:
        *   결과 DataFrame을 샤프 비율과 승률 기준으로 정렬하여 상위 5개 및 3개 조합을 콘솔에 출력합니다.
*   **상태**: `vectorbt` 백테스터와 연동하여 파라미터 그리드 서치 최적화를 수행하는 기능이 잘 구현되어 있습니다. 프로세스 자동화, 결과 저장, 상위 조합 출력까지 필요한 기능이 포함되어 있습니다.

## src/experiments/run_robustness_test.py

*   **목적**: 특정 트레이딩 전략(여기서는 "breakout-volume-surge")의 최적화된 파라미터를 사용하여, 다양한 자산(암호화폐, ETF, 개별 주식 등)과 기간에 걸쳐 백테스트를 실행하여 전략의 강건성(robustness)을 테스트합니다.
*   **구현**:
    *   **설정**:
        *   테스트할 전략 이름(`STRATEGY_NAME`), 고정할 최적화된 전략 파라미터(`FIXED_STRATEGY_PARAMS`), 그리고 테스트할 자산/기간 조합(`EXPERIMENT_COMBINATIONS`)을 정의합니다.
        *   `mcp_runner.py`와 연동하기 위한 MCP 컨텍스트 파일 경로(`CONTEXT_FILE_PATH`), 결과 디렉토리(`RESULTS_DIR`), MCP 실행 스크립트 경로(`MCP_RUNNER_SCRIPT`)를 설정합니다.
    *   **MCP 컨텍스트 생성**:
        *   `generate_mcp_context(combination)`: 각 실험 조합(자산, 기간 등)과 고정된 전략 파라미터를 사용하여 `mcp/experiment_context.json` 파일을 동적으로 생성합니다.
    *   **MCP 스크립트 실행**:
        *   `run_mcp_script()`: `subprocess.run`을 사용하여 `poetry run python mcp/run_mcp_experiment.py` 명령을 실행합니다. 이는 `mcp_runner.py`를 호출하여 생성된 컨텍스트 파일 기반으로 백테스트를 수행하게 합니다.
    *   **결과 수집**:
        *   `collect_result(combination)`: `mcp_runner.py`가 생성한 개별 결과 JSON 파일(`mcp/results/{symbol}_{start}_{end}_result.json`)을 읽어와, 필요한 지표(`METRICS_TO_EXTRACT`)를 추출하고 정리합니다.
    *   **메인 실행 루프**:
        *   `run_robustness_test()`: 정의된 `EXPERIMENT_COMBINATIONS`를 순회하며 각 조합에 대해 MCP 컨텍스트 생성 -> MCP 스크립트 실행 -> 결과 수집 과정을 반복합니다.
        *   모든 결과를 취합하여 Pandas DataFrame으로 만듭니다.
        *   최종 요약 결과를 CSV 파일(`results/robustness_summary_breakout_volume_surge.csv`)로 저장합니다.
        *   전체 테스트 결과를 콘솔에 출력합니다.
*   **상태**: `mcp_runner.py`와 연동하여 자동화된 강건성 테스트 파이프라인을 구축했습니다. 다양한 시장 환경에서 전략의 성능 일관성을 평가하는 데 유용합니다.

## src/signals/ai_signal_generator.py

*   **목적**: 학습된 AI 모델(여기서는 PyTorch GRU 기반 분류 모델 가정)을 사용하여 가격 데이터(OHLCV)를 입력받아 거래 진입 또는 종료 신호를 생성합니다.
*   **구현**:
    *   **모델 정의**: `GRUEntryClassifier` 클래스는 GRU 레이어와 최종 분류를 위한 Linear 레이어, Softmax 활성화 함수로 구성된 간단한 분류 모델 아키텍처를 정의합니다. (이 파일 내에 정의되어 있지만, 실제로는 `src/ai_models` 등 별도 파일에서 관리될 수 있음)
    *   **AI 진입 신호 생성**:
        *   `generate_ai_entry_signals(price_df, model_path, threshold, window_size, device)`:
            *   입력 데이터(OHLCV DataFrame)를 검증하고, 슬라이딩 윈도우(`window_size`) 방식으로 데이터를 준비합니다.
            *   지정된 경로(`model_path`)에서 학습된 PyTorch 모델(`GRUEntryClassifier` 가정)을 로드합니다.
            *   준비된 각 윈도우 데이터를 모델에 입력하여 추론(`model.eval()`, `torch.no_grad()`)을 수행합니다.
            *   모델 출력(Softmax 확률)에서 클래스 1(진입 신호로 가정)의 확률을 추출합니다.
            *   추출된 확률이 설정된 임계값(`threshold`)보다 높으면 1(진입), 아니면 0(보류)으로 시그널을 결정합니다.
            *   계산된 시그널들을 Pandas Series 형태로 반환합니다 (입력 DataFrame과 동일한 인덱스).
    *   **AI 종료 신호 생성**:
        *   `generate_ai_exit_signals(...)`: 현재는 **구현되지 않은 placeholder(Stub)** 함수입니다. 항상 0(퇴장 안 함)을 반환합니다. 실제 구현 시 진입 신호 생성과 유사한 방식으로 작동할 것으로 예상됩니다.
*   **상태**: AI 모델을 로드하여 진입 신호를 생성하는 핵심 기능이 구현되어 있습니다. 데이터 전처리, 모델 로딩, 추론, 임계값 기반 시그널 결정 과정이 포함되어 있습니다. 종료 신호 생성 기능은 아직 구현되지 않았습니다. `backtest_runner_vbt.py`에서 이 함수들을 호출하여 AI 기반 백테스팅을 수행합니다.

## src/insight/sentiment_analyzer.py

*   **목적**: `src/analysis/sentiment.py`와 유사하게 뉴스 기사를 기반으로 특정 티커 및 날짜에 대한 감성 점수를 계산합니다. 단, 이 파일은 집계된 점수를 반환하고 캐싱 기능을 포함하는 데 더 중점을 둔 것으로 보입니다.
*   **구현**:
    *   **감성 분석**: `analyze_text_sentiment(text, model_name, device)` 함수는 `analysis/sentiment.py`와 유사하게 Hugging Face 트랜스포머 파이프라인을 사용하여 텍스트 감성을 분석합니다. ('positive', 'negative', 'neutral' 라벨과 점수 반환)
    *   **뉴스 수집**: `fetch_news_for_ticker(ticker, date, api_key)` 함수는 특정 티커와 날짜에 대한 뉴스 제목 목록을 NewsAPI를 통해 가져옵니다. API 키가 없으면 더미 데이터를 반환합니다.
    *   **캐싱**:
        *   `load_cached_sentiment(ticker, date_str)`: 이전에 계산된 감성 점수를 로컬 파일 캐시(`data/sentiment_cache/`)에서 로드합니다.
        *   `save_sentiment_to_cache(ticker, date_str, sentiment_score)`: 계산된 감성 점수를 캐시에 저장합니다.
    *   **점수 계산**: `get_sentiment_score(ticker, date, api_key, model_name, device)` 함수는 특정 티커와 날짜에 대해 감성 점수를 반환합니다.
        *   캐시를 먼저 확인합니다.
        *   캐시가 없으면 `fetch_news_for_ticker`로 뉴스를 가져옵니다.
        *   뉴스 제목들에 대해 `analyze_text_sentiment`를 호출하고, 평균 감성 점수를 계산합니다.
        *   결과를 캐시에 저장하고 반환합니다.
*   **상태**: `src/analysis/sentiment.py`와 **기능적으로 중복**됩니다. 뉴스 수집 방식(NewsAPI 직접 호출 vs. `NewsApiClient` 사용), 반환값(평균 점수 vs. 전체 라벨 + 점수 + 원본 뉴스), 그리고 **캐싱 기능 포함 여부**에서 차이가 있습니다. **어떤 감성 분석 모듈을 주력으로 사용할지 결정하고 통합/정리**해야 합니다. 캐싱 기능이 필요하다면 이 파일을 기반으로 통합하는 것을 고려할 수 있습니다.

## src/challenge_trading.py

*   **목적**: 플라이트 챌린지 전략의 자동 실행 로직을 구현합니다. 기술적 지표, 로그, 알림 기능을 연동합니다. (초기 버전 또는 API 연동 전 버전일 가능성 있음)
*   **구현**:
    *   **설정 로드**: `src.config.settings`에서 전략 관련 파라미터(SMA 기간, SL/TP 비율 등)를 가져옵니다.
    *   **유틸리티 임포트**: `src.utils.common`, `src.utils.strategy_utils`, `src.utils.notifier` 등에서 필요한 함수를 임포트합니다. (**중복 유틸리티 파일 참조 문제**)
    *   **진입 조건 탐지**: `detect_entry_opportunity(df, settings)` 함수는 입력된 DataFrame과 설정을 기반으로 진입 조건을 확인합니다.
        *   RSI, SMA, POC, RSI 다이버전스, 추세선 이탈, 거래량 급증 등 다양한 기술적 지표를 계산합니다 (`common` 또는 `strategy_utils` 함수 호출 - **여기서 중복 문제가 발생**).
        *   계산된 지표들이 설정된 임계값을 만족하는지 확인하고, 조건이 모두 충족되면 True를 반환합니다.
    *   **거래 로깅**: `log_trade(...)` 함수는 거래 발생 시 상세 정보(시간, 심볼, 타입, 가격, PnL 등)를 CSV 파일(`settings.CHALLENGE_LOG_FILE`)에 기록합니다 (`save_log_to_csv` 사용).
    *   **메인 실행 루프**: `run_challenge_trading()` 함수는 전략 실행의 메인 루프입니다.
        *   현재 포지션 상태(`position_open`, `entry_price`)를 관리합니다.
        *   주기적으로 최신 시세 데이터(`common.get_historical_data`)를 가져옵니다.
        *   **포지션 관리**: 포지션이 열려 있는 경우, 현재 가격이 설정된 TP/SL 가격에 도달했는지 확인하고, 도달 시 청산 로직을 실행하고 로그 및 알림(`notifier.send_exit_alert`)을 보냅니다.
        *   **진입 탐색**: 포지션이 없는 경우, `detect_entry_opportunity`를 호출하여 진입 기회를 탐색합니다. 기회가 발생하면 진입 로직을 실행하고 로그 및 알림(`notifier.send_entry_alert`)을 보냅니다.
*   **상태**: 플라이트 챌린지 전략의 핵심 로직(진입 조건 탐지, 포지션 관리, 로깅, 알림)이 구현되어 있습니다. **하지만 다음과 같은 문제가 있습니다**:
    *   **기술적 분석 함수 중복 참조**: `common.py`와 `strategy_utils.py` 중 어느 파일의 함수를 사용해야 하는지 불명확합니다.
    *   **실제 API 연동 부재**: Binance Futures API 연동 로직(주문 실행 등)이 없습니다.
    *   **`src/challenge/strategy.py` 와 기능 중복**: `src/challenge/strategy.py` 가 이 파일의 기능을 포함하면서 실제 API 연동, AI/감성 분석 통합 등이 더 상세하게 구현되어 있습니다.
*   **권장 조치**: 이 파일은 **`src/challenge/strategy.py`로 통합되거나 제거**되어야 할 가능성이 높습니다. `src/challenge/strategy.py`가 더 완성된 버전으로 보입니다.

## src/backtest/runner.py

*   **상태**: `src/backtesting/backtest_runner.py`와 기능이 중복되고 덜 유연하며, 하드코딩된 설정과 미흡한 전략을 사용하므로 **제거되었습니다.**
*   **권장 조치**: `src/backtesting/backtest_runner.py`를 주력 `backtrader` 실행기로 사용합니다.

## src/backtest/strategies/flight_challenge_strategy.py

*   **상태**: `src/backtesting/strategies/challenge_strategy_backtest.py`와 기능 및 역할이 중복되며, 기능 구현 수준이 훨씬 낮으므로 **제거되었습니다.**
*   **권장 조치**: `src/backtesting/strategies/challenge_strategy_backtest.py`를 주력 `backtrader` 전략으로 사용합니다.

## src/backtest/backtest_strategy.py

*   **상태**: 실제 완성된 전략이라기보다는 전략 클래스의 기본 뼈대 또는 초기 개발 단계의 예시로 보이며, 다른 전략 파일들과 중복되므로 **제거되었습니다.**

## src/challenge/rl_strategy.py

*   **목적**: 강화학습(RL) 기반의 트레이딩 전략을 실행하는 클래스입니다. 사전 학습된 RL 모델(Stable Baselines3 알고리즘 가정)을 로드하고, 실시간 시장 데이터를 입력받아 행동(매수/매도/보류)을 예측하고 실행합니다.
*   **구현**:
    *   **모델 로딩**: `load_model()` 함수는 `settings.RL_ALGORITHM`과 `settings.RL_MODEL_PATH` 설정에 따라 지정된 RL 알고리즘 클래스(PPO, A2C, DDPG)를 사용하여 학습된 모델 파일을 로드합니다.
    *   **상태 관찰**: `_get_live_observation()` 함수는 실시간 시장 데이터(`get_historical_data` 사용)를 가져오고, 필요한 지표(RSI, SMA)를 계산합니다. 현재 포지션 상태와 PnL을 포함하여 RL 모델이 학습할 때 사용한 **정규화된 상태 벡터**를 생성합니다. 가격 정규화를 위해 **`settings.RL_NORMALIZATION_REFERENCE_PRICE`** 설정값을 사용합니다. 이 값이 유효하지 않으면 에러를 발생시킵니다.
    *   **행동 예측**: `run_step()` 함수 내에서 로드된 RL 모델(`self.model.predict`)을 사용하여 현재 관찰된 상태(`observation`)에 대한 최적의 행동(0: Hold, 1: Buy, 2: Sell)을 예측합니다. 실시간 거래에서는 `deterministic=True` 옵션을 사용하여 최적 행동을 선택합니다.
    *   **행동 실행**:
        *   예측된 행동에 따라 실제 주문을 실행하는 로직이 포함되어 있습니다. (주석 처리된 TODO 부분)
        *   현재는 실제 주문 대신 placeholder 로직과 로깅만 수행합니다.
        *   매수/매도 실행 시 포지션 상태(`self.current_position`, `self.entry_price`)를 업데이트합니다.
        *   실행된 거래 정보는 `log_trade_to_db`를 사용하여 DB에 로깅하고, `send_slack_notification`으로 알림을 보냅니다.
    *   **실행 루프**: `run_rl_trading_loop()` 함수 (아직 전체 구현은 안 보임)는 `run_step()` 함수를 주기적으로 호출하여 RL 전략을 계속 실행하는 역할을 할 것으로 예상됩니다.
*   **상태**: 강화학습 모델을 로드하고 실시간 데이터 기반으로 예측 및 (placeholder) 행동 실행을 수행하는 기본 프레임워크가 구현되어 있습니다. 핵심적인 부분은 **상태 정규화 방식이 학습 환경과 일치**해야 하며, 특히 `RL_NORMALIZATION_REFERENCE_PRICE` 설정값이 정확해야 합니다. 실제 주문 실행 로직은 아직 구현되지 않았습니다.

## src/challenge/strategy.py

*   **목적**: `src/challenge_trading.py`와 **상당히 유사한 기능**을 수행하는 것으로 보입니다. 플라이트 챌린지 전략을 구현하며, 여기서는 Binance Futures API 연동, AI/감성 분석 모듈 통합, 포지션 관리 로직 등이 더 구체적으로 포함되어 있습니다.
*   **구현**:
    *   **Binance API 연동**: `python-binance` 라이브러리를 사용하여 Binance Futures 클라이언트를 초기화하고, 현재 포지션 조회(`get_current_position`), 선물 주문 생성(`create_futures_order`), 시장 데이터 조회(`get_market_data`) 등의 API 호출 함수를 구현합니다. API 키는 `settings.py`에서 가져옵니다.
    *   **AI/감성 분석 통합**:
        *   `src.analysis.sentiment.get_market_sentiment` 및 `src.ai_models.price_predictor.run_price_prediction_flow` 함수를 임포트하여 사용합니다 (임포트 실패 시 경고).
        *   `analyze_prediction_trend(predictions_df)`: 가격 예측 결과 DataFrame을 받아 추세("predict_up", "predict_down", "predict_flat")를 분석합니다.
        *   `check_strategy_conditions(symbol, interval, lookback_period)`: 핵심 의사결정 함수.
            *   감성 분석 결과를 가져옵니다 (`get_market_sentiment`).
            *   가격 예측을 수행하고 추세를 분석합니다 (`run_price_prediction_flow`, `analyze_prediction_trend`).
            *   기술적 지표(RSI, SMA, 다이버전스, 추세선 등)를 계산합니다 (`strategy_utils` 사용).
            *   **조건 조합**: 기술적 신호, 감성 분석 결과, 예측 추세를 종합하여 최종 진입/보류 결정을 내립니다. 감성이나 예측이 부정적일 경우 기술적 매수 신호를 무시하는 로직이 포함될 수 있습니다 (`override_reason`).
            *   결과를 상세 정보(결정, 이유, 지표값, 감성/예측 정보 등)와 함께 딕셔너리로 반환합니다.
    *   **포지션 크기 계산**: `calculate_position_size(entry_price, stop_loss_price, risk_per_trade)` 함수는 진입 가격, 손절 가격, 거래당 감수할 위험 비율(`settings.CHALLENGE_RISK_PER_TRADE`)을 기반으로 적절한 주문 수량을 계산합니다.
    *   **포지션 관리**: `manage_open_position(symbol, current_price, position_info)` 함수는 현재 보유 중인 포지션에 대해 TP/SL 도달 여부를 확인하고, 도달 시 청산 주문(`create_futures_order` 호출, `reduce_only=True`)을 실행합니다. 고정 비율 TP/SL 로직이 구현되어 있습니다.
    *   **메인 실행 루프**: `run_challenge_strategy()` 함수는 전체 전략 실행 흐름을 관리합니다.
        *   `settings.CHALLENGE_SYMBOLS` 목록을 순회합니다.
        *   각 심볼에 대해 현재 포지션 유무를 확인합니다.
        *   포지션이 있으면 `manage_open_position`을 호출하여 관리합니다.
        *   포지션이 없으면 `check_strategy_conditions`를 호출하여 진입 기회를 탐색합니다.
        *   진입 신호('buy' 또는 'sell')가 발생하면, `calculate_position_size`로 주문 수량을 계산하고 `create_futures_order`를 호출하여 진입 주문을 실행합니다.
        *   거래 발생 시 `log_trade_to_db`로 DB에 로깅하고, `notifier`로 알림을 보냅니다.
*   **상태**: `src/challenge_trading.py`의 기능을 포함하면서 **실제 API 연동, AI/감성 분석 통합, 포지션 관리 로직이 더 상세하게 구현된 버전**으로 보입니다. `challenge_trading.py` 와 **역할이 중복**되므로 하나로 통합하거나 역할을 명확히 분리해야 합니다. 이 파일이 더 완성도가 높아 보이므로, `challenge_trading.py`의 내용을 이 파일로 통합하거나, `challenge_trading.py`를 제거하는 방향을 고려할 수 있습니다.

## src/stock_trader/live_trader_finrl.py

*   **목적**: 사전 학습된 FinRL 기반의 강화학습 에이전트를 사용하여 Alpaca API를 통해 주식 실시간 거래를 수행합니다.
*   **구현**:
    *   **라이브러리 의존성**: `alpaca-trade-api`, `finrl`, `stable-baselines3`, `pandas-ta` 등의 라이브러리 설치 여부를 확인하고, 미설치 시 경고 및 기능 제한/비활성화를 처리합니다.
    *   **Alpaca API 연동**: `get_alpaca_api()` 함수는 `settings.py`의 설정값을 사용하여 Alpaca API 클라이언트를 초기화하고 연결을 확인합니다.
    *   **기술적 지표 계산**: `calculate_technical_indicators(df, indicator_list)` 함수는 `pandas_ta` 라이브러리를 사용하여 FinRL 에이전트 학습 시 사용된 것과 **동일한 기술적 지표 목록(`tech_indicator_list`)**을 계산합니다. 이 목록은 에이전트의 상태(state) 구성에 필수적입니다.
    *   **FinRL 에이전트 로드**: `run_live_trading_finrl` 함수 내에서 `settings.FINRL_MODEL_DIR` 및 전달된 `agent_path`를 사용하여 학습된 Stable Baselines3 모델(PPO, A2C, DDPG 등)을 로드합니다.
    *   **실시간 거래 로직**: `run_live_trading_finrl` 함수는 실시간 거래의 핵심 로직을 포함합니다 (현재는 단일 실행 예시 형태).
        *   **데이터 가져오기**: Alpaca API를 통해 최신 분봉 데이터(`api.get_bars`)를 가져옵니다.
        *   **상태 구성**: 현재 계좌 잔고(`api.get_account`), 보유 주식 수(`api.get_position`), 그리고 `calculate_technical_indicators`로 계산된 기술적 지표 값들을 조합하여 **FinRL 에이전트가 학습할 때 사용한 것과 동일한 형식의 상태 벡터(state)**를 구성합니다. (이 부분 코드는 스니펫에 완전히 포함되지 않았으나, 필수 로직입니다.)
        *   **행동 예측**: 로드된 FinRL 에이전트(`trained_model.predict`)를 사용하여 구성된 상태 벡터에 대한 행동(매수/매도/보류)을 예측합니다.
        *   **주문 실행**: 예측된 행동에 따라 Alpaca API(`api.submit_order`)를 통해 실제 주문(시장가, 지정된 수량 `trade_quantity`)을 제출합니다.
        *   **로깅 및 알림**: 주문 결과 및 상태 변경을 로깅하고, 필요시 알림을 보냅니다.
*   **상태**: FinRL 에이전트를 이용한 주식 실시간 거래 프레임워크의 기본 구조를 갖추고 있습니다. Alpaca 연동, 지표 계산, 에이전트 로딩, 상태 구성, 행동 예측, 주문 실행 로직이 포함되어 있습니다. **가장 중요한 부분은 실시간 상태 구성이 학습 환경과 정확히 일치해야 한다는 점**이며, 특히 사용된 기술적 지표 목록과 계산 방식이 동일해야 합니다. 현재 코드는 루프가 아닌 단일 실행 예시로 구성되어 있으며, 실제 지속적인 거래를 위해서는 루프 및 스케줄링 로직이 필요합니다.

## src/stock_trader/agent_trainer.py

*   **목적**: FinRL 라이브러리를 사용하여 주식 거래를 위한 강화학습(DRL) 에이전트를 학습시킵니다.
*   **구현**:
    *   **라이브러리 의존성**: `finrl`, `stable-baselines3` 설치 여부를 확인하고, 미설치 시 경고 및 기능 비활성화를 처리합니다.
    *   **데이터 처리**:
        *   `finrl.meta.data_processor.DataProcessor`를 사용하여 지정된 티커와 기간에 대한 데이터를 다운로드(`yahoofinance` 사용)하고, 기술적 지표(`finrl.config.INDICATORS` 또는 사용자 정의 목록) 및 VIX 기반 변동성 지표를 추가하여 학습 데이터를 준비합니다.
    *   **학습 환경 설정**:
        *   `finrl.meta.env_stock_trading.env_stocktrading.StockTradingEnv`를 사용하여 학습 환경을 구성합니다. 처리된 데이터프레임, 초기 자금, 최대 거래 수량, 수수료, 상태 공간 차원, 보상 스케일링 등의 파라미터를 설정합니다.
    *   **에이전트 초기화 및 학습**:
        *   `finrl.agents.stablebaselines3.models.DRLAgent`를 사용하여 Stable Baselines3 기반의 RL 에이전트(PPO, A2C, DDPG 등)를 초기화합니다. 알고리즘별 하이퍼파라미터(`AGENT_KWARGS`)를 설정할 수 있습니다.
        *   `agent.train_model()`을 호출하여 지정된 총 타임스텝(`total_timesteps`) 동안 에이전트를 학습시킵니다. TensorBoard 로깅도 지원합니다.
    *   **모델 저장**: 학습이 완료된 모델을 지정된 경로(`model_save_path`, 기본값: `models/finrl/`)에 `.zip` 파일로 저장합니다.
    *   **메인 함수**: `train_finrl_agent(...)` 함수는 위의 데이터 처리, 환경 설정, 에이전트 초기화, 학습, 모델 저장 과정을 캡슐화하여 제공합니다.
*   **상태**: FinRL 라이브러리를 활용하여 주식 거래 DRL 에이전트를 학습시키는 파이프라인이 잘 구현되어 있습니다. 데이터 처리부터 모델 학습 및 저장까지의 과정이 포함되어 있습니다. 학습된 모델은 `live_trader_finrl.py`에서 사용될 수 있습니다.

## src/stock_trader/backtester.py

*   **목적**: `agent_trainer.py`에서 학습시킨 FinRL 기반 강화학습 에이전트를 사용하여 과거 데이터에 대한 백테스트를 수행하고, 성능 통계와 시각화 자료를 생성합니다.
*   **구현**:
    *   **라이브러리 의존성**: `agent_trainer.py`와 유사하게 `finrl`, `stable-baselines3` 등의 설치 여부를 확인합니다.
    *   **데이터 처리**: `finrl.meta.data_processor.DataProcessor`를 사용하여 백테스트 기간에 대한 데이터를 다운로드하고, 학습 시 사용된 것과 동일한 기술적 지표 및 변동성 지표를 추가하여 백테스트용 데이터를 준비합니다.
    *   **백테스트 환경 설정**: `finrl.meta.env_stock_trading.env_stocktrading.StockTradingEnv`를 사용하여 백테스트 환경을 구성합니다. 학습 환경과 동일한 파라미터(초기 자금, 수수료, 상태 공간 등)를 사용하되, `mode="trade"`로 설정합니다. Stable Baselines3와의 호환성을 위해 `DummyVecEnv`로 래핑합니다.
    *   **학습된 에이전트 로드**: 지정된 경로(`agent_path`)에서 학습된 Stable Baselines3 모델(PPO, A2C, DDPG 등)을 로드합니다. 로드 시 환경 정보(`env=backtest_env_instance`)를 전달하여 환경 호환성을 검증할 수 있습니다.
    *   **백테스트 시뮬레이션**:
        *   FinRL의 `DRLAgent.DRL_prediction` 함수를 사용하여 로드된 모델과 백테스트 환경을 기반으로 시뮬레이션을 수행하고, 계좌 가치 변화 DataFrame (`account_value_df`)을 얻습니다. (주석 처리된 부분에는 SB3의 `predict`를 직접 루프에서 사용하는 대체 구현 방식도 고려되어 있습니다.)
    *   **성능 통계 계산**: `finrl.plot.backtest_stats` 함수를 사용하여 시뮬레이션 결과(`account_value_df`)로부터 연간 수익률, 샤프 비율, 최대 낙폭(MDD) 등 주요 성능 지표를 계산합니다.
    *   **결과 저장 및 시각화**:
        *   계산된 성능 통계를 CSV 파일 (`results/finrl/stats_{...}.csv`)로 저장합니다.
        *   `plot=True`인 경우, `finrl.plot.backtest_plot` 함수를 사용하여 계좌 가치 변화와 기준 지수(예: S&P 500 `^GSPC`)를 비교하는 그래프를 이미지 파일 (`results/finrl/plot_{...}.png`)로 저장합니다.
    *   **메인 함수**: `run_backtest_finrl(...)` 함수는 위의 데이터 처리, 환경 설정, 에이전트 로드, 시뮬레이션, 통계 계산, 결과 저장 및 시각화 과정을 캡슐화하여 제공합니다.
*   **상태**: FinRL 에이전트를 이용한 백테스팅 파이프라인이 잘 구현되어 있습니다. 학습(`agent_trainer.py`)과 백테스트(`backtester.py`) 과정이 일관된 데이터 처리 및 환경 설정을 사용하도록 구성되어 있습니다. 결과 저장 및 시각화 기능도 포함되어 있습니다.

## src/crypto_trader/freqai_trainer.py

*   **목적**: FreqAI (Freqtrade의 AI 확장)와 호환되는 머신러닝 모델을 학습시키기 위한 헬퍼 스크립트 또는 placeholder입니다. **주요 내용**: Freqtrade의 `freqtrade freqai-train` 명령어를 통해 관리되는 실제 학습 프로세스 대신, **독립적인 모델 학습 또는 데이터 준비를 위한 개념적인 스크립트**임을 명시하고 있습니다.
*   **구현**:
    *   **Placeholder**: 실제 모델 학습 로직(LightGBM, GRU 등)은 주석 처리되거나 더미 데이터/객체로 대체되어 있습니다.
    *   **데이터 로딩**: FreqAI가 생성/사용하는 피처 데이터 파일(`data/features/{pair}_features.csv`)을 로드하는 부분을 가정합니다 (현재는 더미 DataFrame 사용).
    *   **모델 저장**: 학습된 모델을 저장하는 경로(`models/freqai/`) 및 로직을 가정합니다 (현재는 더미 파일 생성).
*   **상태**: **실제 FreqAI 모델 학습 코드가 아니라, 해당 프로세스를 보조하거나 개념을 설명하기 위한 placeholder 스크립트**입니다. FreqAI 모델 학습은 일반적으로 Freqtrade 프레임워크 내에서 구성 파일과 명령어를 통해 수행됩니다. 이 파일은 Freqtrade의 학습 파이프라인을 이해하거나 커스텀 모델을 외부에서 학습시켜 통합하려는 경우 참고 자료로 사용될 수 있습니다.

## src/crypto_trader/ExampleFreqAIStrategy.py

*   **목적**: Freqtrade 봇에서 FreqAI 확장 기능을 사용하는 **예시 전략** 클래스입니다. FreqAI를 통해 생성된 예측값을 기반으로 매수/매도 신호를 생성하는 방법을 보여줍니다. 또한, 커스텀 피처(기술적 지표 + 감성 분석 등)를 FreqAI 모델 학습 및 예측에 통합하는 방법을 예시합니다.
*   **구현**:
    *   **Freqtrade 전략 상속**: `freqtrade.strategy.IStrategy`를 상속받아 Freqtrade 봇에서 사용할 수 있는 전략 클래스를 정의합니다.
    *   **기본 설정**: `minimal_roi`, `stoploss`, `trailing_stop`, `timeframe`, `order_types` 등 Freqtrade 전략의 표준 파라미터들을 설정합니다.
    *   **FreqAI 정보**: `freqai_info` 딕셔너리는 Freqtrade 설정 파일(`config.json`)의 `freqai` 섹션과 연동될 정보를 정의합니다 (모델 식별자, 피처 엔지니어링/학습 설정 이름 등).
    *   **지표 및 피처 생성 (`populate_indicators`)**:
        *   기술적 지표와 커스텀 피처를 계산하여 DataFrame에 추가합니다.
        *   **`src.crypto_trader.custom_features.generate_custom_features` 함수를 호출**하여 커스텀 피처(이 예시에서는 TA + 감성 분석 포함 가정)를 생성하는 로직이 포함되어 있습니다. 이 함수가 FreqAI 모델의 입력 피처를 제공합니다.
    *   **매수 신호 생성 (`populate_entry_trend`)**:
        *   FreqAI가 예측하여 DataFrame에 추가한 **예측값 컬럼** (컬럼명은 FreqAI 설정에 따라 달라짐, 예: `&*_mean`)을 사용합니다.
        *   예측값이 특정 임계값(예: 1% 이상 상승 예측)을 초과하면 매수 신호(`enter_long = 1`)를 생성합니다.
    *   **매도 신호 생성 (`populate_exit_trend`)**:
        *   FreqAI 예측값 컬럼을 사용하여 매도 신호를 생성합니다 (예: 예측된 상승률이 0.1% 미만으로 떨어지면 매도).
        *   표준 기술적 지표 기반의 매도 조건을 추가할 수도 있습니다.
*   **상태**: FreqAI를 사용하는 Freqtrade 전략의 **훌륭한 예시**입니다. 커스텀 피처 생성, FreqAI 예측값 활용, 매수/매도 신호 생성까지의 흐름을 잘 보여줍니다. 실제 사용을 위해서는 Freqtrade `config.json` 파일에 해당 FreqAI 설정을 맞추고, `custom_features.py` 모듈을 구현해야 합니다.

## src/crypto_trader/custom_features.py

*   **목적**: Freqtrade/FreqAI 전략에서 사용할 커스텀 피처를 생성하는 함수들을 제공합니다. 표준 기술적 지표 외에 감성 분석 점수와 같은 외부 데이터를 피처로 통합하는 역할을 합니다.
*   **구현**:
    *   **기술적 지표 추가**: `add_technical_indicators(dataframe)` 함수는 입력 DataFrame에 SMA, RSI 등 표준 기술적 지표를 추가합니다. `pandas-ta` 라이브러리 사용을 시도하고, 실패 시 기본적인 rolling 계산으로 폴백합니다.
    *   **감성 점수 캐싱**: `load_cached_sentiment` 및 `save_sentiment_to_cache` 함수는 이전에 계산된 일별 감성 점수를 파일 시스템에 캐싱하여 중복 API 호출을 방지합니다 (`data/sentiment_cache/`).
    *   **감성 피처 추가**: `add_sentiment_feature(dataframe, ticker)` 함수는 각 날짜별로 감성 점수를 계산(캐시 확인 후 `src.insight.sentiment_analyzer.get_sentiment_score` 호출)하여 DataFrame에 'sentiment_score' 컬럼으로 추가합니다. NewsAPI 키는 환경 변수(`NEWS_API_KEY`)에서 가져옵니다.
    *   **메인 피처 생성 함수**: `generate_custom_features(dataframe, metadata)` 함수는 Freqtrade 전략 파일(`ExampleFreqAIStrategy.py`)에서 호출됩니다. 기술적 지표와 감성 점수를 순차적으로 계산하여 최종 피처 DataFrame을 반환합니다. 이 DataFrame은 FreqAI 모델의 학습 및 예측 입력으로 사용됩니다.
*   **상태**: 기술적 지표와 감성 점수를 결합하여 FreqAI 모델을 위한 풍부한 피처셋을 생성하는 로직이 구현되어 있습니다. 캐싱을 통해 효율성을 높였습니다. `src.insight.sentiment_analyzer`와 연동하여 감성 점수를 가져옵니다.

## src/ai_models/train_custom_env.py

*   **목적**: `src/ai_models/rl_environment.py`에 정의된 커스텀 트레이딩 환경(`TradingEnv`)을 사용하여 Stable Baselines3 기반의 강화학습 에이전트(여기서는 PPO 예시)를 학습시킵니다.
*   **구현**:
    *   **라이브러리 의존성**: `stable-baselines3` 및 커스텀 환경(`TradingEnv`) 임포트 가능 여부를 확인합니다.
    *   **설정 및 인수 파싱**: 학습 파라미터(타임스텝, 지표 기간, 초기 자금 등)의 기본값을 정의하고, 커맨드라인 인수(`argparse`)를 통해 학습 대상 티커와 총 타임스텝을 입력받습니다. 모델 및 로그 저장 경로는 `settings.py` 또는 기본값을 사용합니다.
    *   **데이터 로딩**: `src.utils.common.get_historical_data`를 사용하여 학습에 필요한 시계열 데이터를 로드합니다. (yfinance 1시간 데이터 제한 고려)
    *   **환경 생성**: `stable_baselines3.common.env_util.make_vec_env`를 사용하여 커스텀 `TradingEnv` 인스턴스를 생성합니다. 환경 생성 시 데이터프레임과 지표 파라미터 등을 전달합니다.
    *   **모델 설정**: Stable Baselines3의 PPO 모델(`stable_baselines3.PPO`)을 "MlpPolicy"와 함께 초기화합니다. TensorBoard 로깅 경로, 학습률, 엔트로피 계수 등의 하이퍼파라미터를 설정합니다.
    *   **학습 실행**: `model.learn()`을 호출하여 지정된 총 타임스텝 동안 에이전트를 학습시킵니다. (EvalCallback, StopTrainingOnRewardThreshold 등 콜백 사용 가능)
    *   **모델 저장**: 학습 완료 후 모델을 지정된 경로(`DEFAULT_MODEL_DIR` 아래)에 `.zip` 파일로 저장합니다. 파일명에는 티커, 환경 종류, 타임스텝 등이 포함됩니다.
    *   **메인 함수**: `train_custom_environment_agent(...)` 함수는 위의 데이터 로딩부터 모델 저장까지의 전체 학습 과정을 캡슐화합니다. `if __name__ == "__main__":` 블록에서 커맨드라인 인수를 받아 이 함수를 실행합니다.
*   **상태**: 커스텀 Gym(nasium) 환경(`TradingEnv`)을 사용하여 RL 에이전트를 학습시키는 표준적인 파이프라인을 구현했습니다. 데이터 로딩, 환경 생성, 모델 설정, 학습, 저장까지의 과정이 명확합니다. 커맨드라인 인수를 통해 학습 대상 티커와 타임스텝을 지정할 수 있어 유연성이 높습니다. 학습된 모델은 `src/challenge/rl_strategy.py` 등에서 실시간 거래에 사용될 수 있습니다. **`src/ai_models/rl_trainer.py`와 기능 중복.**

## src/ai_models/price_predictor.py

*   **목적**: AI 모델을 사용하여 시계열 가격 데이터를 기반으로 미래 가격을 예측합니다. Hugging Face 트랜스포머 모델 및 기존의 LSTM/Transformer 모델 구현을 모두 포함하고 있습니다.
*   **구현 (Hugging Face 부분)**:
    *   **모델 초기화**: `initialize_hf_timeseries_model()` 함수는 `settings.py`에 정의된 모델 이름(`HF_TIMESERIES_MODEL_NAME`), 장치(`HF_DEVICE`), 캐시 디렉토리 등을 사용하여 Hugging Face 라이브러리에서 사전 학습된 시계열 예측 모델 (`AutoModelForPrediction`)과 해당 설정(`AutoConfig`), 그리고 필요시 프로세서(`AutoProcessor`)를 로드합니다. 모델은 평가 모드(`eval()`)로 설정됩니다.
    *   **데이터 준비**: `prepare_data_for_hf_model(df, feature_col)` 함수는 입력 DataFrame에서 모델의 컨텍스트 길이(`context_length`)만큼의 마지막 시퀀스를 추출하여 모델 입력 형식(주로 `past_values` 텐서)으로 변환합니다. Processor가 있다면 이를 사용하여 데이터를 준비하고, 없다면 수동으로 텐서를 생성합니다. (스케일링 등 추가 전처리 필요 가능성 언급)
    *   **예측 수행**: `predict_with_hf_model(model_inputs, df_index)` 함수는 준비된 입력 텐서를 사용하여 로드된 HF 모델의 `.generate()` 메서드를 호출하여 미래 가격 시퀀스를 예측합니다. 예측된 값들에 대해 미래 타임스탬프를 생성하고 (입력 데이터의 빈도 추론), 결과를 'timestamp'와 'predicted_price' 컬럼을 가진 DataFrame으로 반환합니다. (역스케일링 필요 가능성 언급)
*   **구현 (기존 LSTM/Transformer 부분)**:
    *   **모델 정의**: `PricePredictorLSTM`과 `PricePredictorTransformer` 클래스는 각각 LSTM과 Transformer 기반의 가격 예측 모델 아키텍처를 PyTorch(`torch.nn.Module`)로 정의합니다.
    *   **데이터 전처리**: 데이터 스케일링(MinMaxScaler), 시퀀스 데이터 생성 등의 전처리 로직이 포함될 것으로 예상됩니다 (스니펫에는 명확히 안 보임).
    *   **학습**: `train_model(...)` 함수는 정의된 모델, 데이터 로더, 손실 함수, 옵티마이저를 사용하여 모델을 학습시키는 로직을 포함합니다.
    *   **예측**: `predict_next(...)` 함수는 학습된 모델과 마지막 시퀀스 데이터를 사용하여 다음 스텝(들)의 가격을 예측하고, 결과를 역스케일링합니다.
    *   **모델 저장/로드**: `save_model_weights` 및 `load_model_weights` 함수는 학습된 모델의 가중치와 스케일러 상태를 저장하고 로드하는 기능을 제공합니다.
*   **통합 실행 흐름**:
    *   `run_price_prediction_flow(train, predict, symbol, interval)`: 전체 워크플로우를 관리합니다.
        *   학습 모드(`train=True`)일 경우: 데이터 로드(`get_historical_data`), 전처리, 모델 학습(`train_model`), 모델 저장(`save_model_weights`)을 수행합니다.
        *   예측 모드(`predict=True`)일 경우:
            *   HF 모델 사용 시: `initialize_hf_timeseries_model`, `get_historical_data`, `prepare_data_for_hf_model`, `predict_with_hf_model` 순서로 호출하여 예측 결과를 생성합니다.
            *   (기존 모델 사용 시): 모델 로드(`load_model_weights`), 데이터 로드 및 전처리, 예측 수행(`predict_next`) 과정을 따릅니다.
        *   예측 결과를 DB에 로깅합니다 (`log_prediction_to_db`).
        *   최종적으로 (예측 결과 DataFrame, 실제 데이터 DataFrame) 튜플을 반환할 수 있습니다.
*   **상태**: Hugging Face 모델과 기존 LSTM/Transformer 모델을 이용한 가격 예측 기능이 모두 구현되어 있습니다. HF 모델 사용 시 초기화, 데이터 준비, 예측 과정이 명확하며, 기존 모델은 학습부터 예측, 저장/로드까지의 전체 파이프라인을 갖추고 있습니다. `settings.py`를 통해 HF 모델 사용 여부 및 관련 설정을 제어합니다. 예측 결과는 DB 로깅과 연동됩니다.

## src/ai_models/tune_hyperparameters.py

*   **목적**: `optuna` 라이브러리를 사용하여 `src/ai_models/rl_environment.py`의 커스텀 `TradingEnv` 환경에서 학습되는 Stable Baselines3 RL 에이전트(PPO 예시)의 하이퍼파라미터를 최적화합니다.
*   **구현**:
    *   **라이브러리 의존성**: `optuna`, `stable-baselines3`, `TradingEnv` 임포트 가능 여부를 확인합니다.
    *   **설정 및 인수 파싱**: 최적화 대상 티커, Optuna 시도 횟수(`n_trials`), 각 시도별 학습 타임스텝(`tune_timesteps`) 등을 커맨드라인 인수로 받습니다.
    *   **Objective 함수**: `objective(trial)` 함수는 Optuna가 각 시도마다 호출하는 핵심 함수입니다.
        *   **하이퍼파라미터 제안**: `trial.suggest_float`, `trial.suggest_categorical` 등을 사용하여 최적화할 하이퍼파라미터(학습률, 배치 크기, 엔트로피 계수 등)의 탐색 범위를 정의하고 값을 제안받습니다.
        *   **데이터 로드 및 환경 생성**: 제안된 파라미터로 학습을 수행하기 위해 데이터를 로드하고 `TradingEnv` 환경을 생성합니다.
        *   **모델 생성 및 학습**: 제안된 하이퍼파라미터를 사용하여 PPO 모델을 생성하고, `tune_timesteps` 동안 학습시킵니다.
        *   **모델 평가**: 학습된 모델을 사용하여 환경에서 일정 스텝 동안 실행하고, 성능 지표(여기서는 총 보상 `total_rewards`)를 계산합니다. (더 견고한 평가를 위해 전체 백테스트 실행 고려 가능 언급)
        *   계산된 성능 지표를 반환합니다 (Optuna는 이 값을 최대화/최소화).
    *   **Optuna Study 실행**:
        *   `optuna.create_study(direction="maximize")`를 사용하여 성능 지표(총 보상)를 최대화하는 연구(study)를 생성합니다.
        *   `study.optimize(objective, n_trials=...)`를 호출하여 지정된 횟수만큼 `objective` 함수를 실행하며 최적화를 수행합니다.
    *   **결과 출력**: 최적화 완료 후, 최상의 결과를 보인 시도(trial)의 성능 지표 값과 해당 하이퍼파라미터 조합을 로깅합니다.
*   **상태**: Optuna와 Stable Baselines3를 연동하여 커스텀 RL 환경에 대한 하이퍼파라미터 튜닝을 자동화하는 기능이 구현되어 있습니다. 하이퍼파라미터 탐색, 모델 학습, 평가, 결과 요약까지의 과정을 포함합니다.

## src/ai_models/rl_trainer.py

*   **목적**: `src/ai_models/train_custom_env.py`와 **동일한 목적**을 가집니다. 커스텀 RL 환경(`TradingEnv`)을 사용하여 Stable Baselines3 에이전트를 학습시킵니다.
*   **구현**:
    *   `train_rl_model(...)` 함수는 RL 모델 학습의 전체 과정을 캡슐화합니다.
    *   **설정 로드**: `settings.py`에서 RL 관련 설정(심볼, 알고리즘, 정책, 학습률, 배치 크기, 타임스텝, 모델 저장 경로 등)을 가져옵니다.
    *   **데이터 로딩**: `get_historical_data`를 사용하여 학습 데이터를 로드합니다. **중요**: 데이터 로드 시 첫 번째 종가를 **정규화 기준 가격(Normalization Reference Price)**으로 로깅하고, 사용자가 이 값을 `.env` 파일의 `RL_NORMALIZATION_REFERENCE_PRICE`에 설정하도록 안내합니다. 이는 실시간 거래(`src/challenge/rl_strategy.py`)에서 상태를 동일하게 정규화하기 위해 필수적입니다.
    *   **환경 생성**: 로드된 데이터를 사용하여 `TradingEnv` 환경을 생성합니다.
    *   **모델 초기화**: `settings.RL_ALGORITHM`에 지정된 알고리즘(PPO, A2C, DDPG) 클래스를 사용하여 모델을 초기화합니다. 모델 하이퍼파라미터는 `settings.py`에서 가져옵니다.
    *   **콜백 정의**: 평가 및 조기 종료를 위한 콜백(EvalCallback, StopTrainingOnRewardThreshold) 정의 부분이 주석 처리되어 있습니다 (현재는 사용 안 함).
    *   **학습 실행**: `model.learn()`을 호출하여 `settings.RL_TRAIN_TIMESTEPS` 동안 모델을 학습시킵니다.
    *   **모델 저장**: 학습 완료 후 모델을 `settings.RL_MODEL_PATH`에 저장합니다.
*   **상태**: `src/ai_models/train_custom_env.py`와 **기능 및 역할이 중복**됩니다. 주요 차이점은 이 파일이 `settings.py`에서 RL 관련 설정을 직접 가져와 사용하고, 정규화 기준 가격을 명시적으로 로깅하며, 콜백 예시가 포함되어 있다는 점입니다. **두 학습 스크립트 중 하나로 통합**해야 합니다. 이 파일이 설정 연동 및 정규화 가격 로깅 측면에서 조금 더 나아 보입니다.

## src/ai_models/backtest_custom_env.py

*   **목적**: `train_custom_env.py` 또는 `rl_trainer.py`에서 학습시킨 RL 에이전트를 사용하여, 커스텀 환경(`TradingEnv`) 내에서 백테스팅을 수행하고 결과를 분석 및 시각화합니다.
*   **구현**:
    *   **설정 및 인수 파싱**: 백테스트 대상 티커, 기간, 간격, 모델 경로 구성 요소(학습 스텝 수), 초기 자금 등의 파라미터를 정의하고, 커맨드라인 인수(`argparse`)를 통해 티커와 모델 스텝 수를 입력받아 모델 경로를 동적으로 구성합니다.
    *   **데이터 로딩**: `get_historical_data`를 사용하여 백테스트 기간에 대한 데이터를 로드합니다.
    *   **환경 생성**: 로드된 데이터를 사용하여 `TradingEnv` 백테스트 환경을 생성합니다. 학습 시 사용된 것과 동일한 초기 자금, 지표 파라미터 등을 사용해야 합니다.
    *   **모델 로드**: `stable_baselines3.PPO.load`를 사용하여 지정된 경로(`MODEL_PATH`)에서 학습된 PPO 에이전트를 로드합니다. 로드 시 환경(`env`)을 전달하여 관찰/행동 공간 호환성을 검증합니다.
    *   **백테스팅 시뮬레이션**:
        *   환경을 리셋하고 루프를 돌며 에이전트의 행동을 예측(`model.predict(deterministic=True)`)하고 환경 스텝을 진행(`env.step(action)`)합니다.
        *   매 스텝마다 포트폴리오 가치, 잔고, 보유 상태, PnL 등을 추적하고 상세 로그(DEBUG 레벨)를 기록합니다.
        *   환경 내부에서 발생하는 거래 정보(`info['trade']`)를 받아 `trade_log`에 저장하고 INFO 레벨로 로깅합니다.
    *   **결과 분석**: 시뮬레이션 종료 후, 최종 포트폴리오 가치, 총 수익률, 총 거래 횟수 등을 계산하여 요약 정보를 로깅합니다. (추가적인 상세 지표 계산 TODO 언급)
    *   **시각화**: Matplotlib을 사용하여 두 개의 서브플롯을 생성합니다.
        *   첫 번째 플롯: 시간에 따른 포트폴리오 가치 변화.
        *   두 번째 플롯: 가격 데이터와 함께 매수/매도 신호 발생 지점을 표시.
        *   생성된 플롯을 이미지 파일(`backtest_results_{...}.png`)로 저장합니다.
    *   **메인 함수**: `run_backtest(...)` 함수는 위의 과정을 캡슐화합니다. `if __name__ == "__main__":` 블록에서 커맨드라인 인수를 받아 이 함수를 실행합니다.
*   **상태**: 커스텀 RL 환경에서 학습된 에이전트의 성능을 백테스팅하고 평가하는 기능이 구현되어 있습니다. 데이터 로딩, 환경 생성, 모델 로드, 시뮬레이션, 결과 분석 및 시각화까지 완전한 파이프라인을 갖추고 있습니다. 커맨드라인 인수를 통해 대상 티커와 모델을 지정할 수 있습니다.

## src/ai_models/rl_environment.py

*   **목적**: Stable Baselines3 라이브러리와 호환되는 커스텀 강화학습 환경(`TradingEnv`)을 정의합니다. 이 환경은 주식/암호화폐 거래를 시뮬레이션하며, 에이전트는 이 환경과 상호작용하며 학습합니다.
*   **구현 (Gymnasium 기반)**:
    *   **클래스 정의**: `gymnasium.Env`를 상속받아 `TradingEnv` 클래스를 정의합니다.
    *   **초기화 (`__init__`)**:
        *   입력 데이터프레임(OHLCV), 초기 자금, 수수료, 지표 계산 파라미터(RSI, SMA, MACD, ATR, BB) 등을 받습니다.
        *   데이터프레임의 유효성(필수 컬럼, 최소 길이)을 검사합니다.
        *   **지표 계산**: `_add_indicators` 메서드를 호출하여 RSI, SMA, MACD, ATR, Bollinger Bands 등 필요한 모든 기술적 지표를 계산하고 DataFrame에 추가합니다 (`ta` 라이브러리 사용). 계산 후 NaN 값이 있는 행은 제거합니다.
        *   **상태 정규화**: `StandardScaler`를 사용하여 상태 벡터에 포함될 피처들(가격, 지표 값 등)을 정규화하기 위한 스케일러를 학습 데이터에 맞춰 `fit`합니다.
        *   **Action Space 정의**: 0 (Hold), 1 (Buy), 2 (Sell)의 3가지 이산적인 행동 공간을 정의합니다 (`spaces.Discrete(3)`).
        *   **Observation Space 정의**: 상태 벡터의 형태와 범위를 정의합니다 (`spaces.Box`). 상태 벡터는 **정규화된 피처 값들**, 현재 포지션 보유 여부(0 또는 1), 정규화된 PnL, 그리고 **정규화된 잔고(balance_norm)**를 포함합니다. (잔고 정규화는 초기 자금 대비 비율로 계산)
        *   환경 내부 상태 변수(현재 스텝, 잔고, 포지션 상태, 진입 가격, 총 PnL, 거래 내역 등)를 초기화합니다.
    *   **상태 관찰 (`_get_observation`)**: 현재 스텝의 데이터와 내부 상태를 기반으로 상태 벡터를 구성하고, **`fit`된 `StandardScaler`를 사용하여 피처들을 정규화**한 후 반환합니다. PnL과 잔고도 정규화됩니다.
    *   **보상 계산 (`_calculate_reward`)**: 에이전트의 행동 결과에 따라 보상을 계산합니다.
        *   **수익 실현 매도에 큰 양의 보상**을 제공합니다 (PnL * 높은 가중치).
        *   **손실 매도에 음의 보상(패널티)**을 제공합니다 (PnL * 높은 가중치).
        *   포지션을 보유하고 있을 때 **가격 변화에 따른 보상/패널티**를 제공합니다 (변동성 * 낮은 가중치).
        *   (수정됨) 거래 비용 패널티는 제거하여 거래를 장려하는 방향으로 변경되었습니다.
    *   **리셋 (`reset`)**: 환경을 초기 상태로 리셋하고 첫 번째 관찰 상태를 반환합니다.
    *   **스텝 진행 (`step`)**:
        *   에이전트로부터 행동(action)을 입력받습니다.
        *   행동에 따라 거래(매수/매도)를 시뮬레이션하고, 수수료를 적용하여 잔고를 업데이트합니다.
        *   포지션 상태(진입 가격, 보유 여부)를 업데이트합니다.
        *   다음 스텝으로 이동합니다.
        *   `_calculate_reward`를 호출하여 보상을 계산합니다.
        *   `_get_observation`을 호출하여 다음 상태를 얻습니다.
        *   종료 조건(마지막 스텝 도달)을 확인합니다.
        *   (관찰 상태, 보상, 종료 여부, 조기 종료 여부(truncated), 추가 정보 딕셔너리) 튜플을 반환합니다. 추가 정보(`info`) 딕셔너리에는 실제 거래 발생 시 관련 정보(타입, 가격, PnL 등)가 포함됩니다.
*   **상태**: Gym(nasium) 표준을 따르는 커스텀 트레이딩 환경이 잘 정의되어 있습니다. 상태 공간은 다양한 기술적 지표와 포트폴리오 상태를 포함하며 정규화됩니다. 보상 함수는 수익 실현 매도를 장려하고 손실 매도에 패널티를 부여하도록 설계되었습니다. 거래 시뮬레이션 및 상태 업데이트 로직이 포함되어 있으며, Stable Baselines3와 호환됩니다. 이 환경은 `train_custom_env.py` 및 `rl_trainer.py`에서 에이전트 학습에 사용되고, `backtest_custom_env.py`에서 백테스팅에 사용됩니다.

## src/analysis/sentiment_analysis.py

*   **목적**: NewsAPI를 통해 뉴스를 수집하고, 수집된 텍스트의 감성을 분석하여 특정 키워드에 대한 시장 감성을 판단합니다 (`src/analysis/sentiment.py`, `src/insight/sentiment_analyzer.py`와 **기능 중복**).
*   **구현**:
    *   **뉴스 수집**: `fetch_recent_news(keyword, ...)` 함수는 `requests` 라이브러리를 직접 사용하여 NewsAPI "everything" 엔드포인트에 HTTP GET 요청을 보내 뉴스를 가져옵니다. API 키는 `settings.py` 또는 환경 변수에서 가져옵니다.
    *   **감성 분석**: `analyze_sentiment(texts)` 함수는 **매우 기본적인 키워드 기반 방식**으로 감성을 분석합니다. 미리 정의된 긍정/부정 키워드 목록을 사용하여 텍스트 내 키워드 출현 빈도를 세고, 그 비율에 따라 "bullish", "bearish", "neutral"을 결정합니다. **FinBERT 같은 실제 모델 사용은 TODO 주석으로만 남아있습니다.**
    *   **통합 함수**: `get_market_sentiment(keyword, use_content)` 함수는 `fetch_recent_news`로 뉴스를 가져오고, 기사의 제목/설명(선택적으로 내용 포함)을 추출하여 `analyze_sentiment`로 전달한 후, 최종 감성("bullish", "bearish", "neutral")을 반환합니다.
*   **상태**: **뉴스 수집 기능은 `src/analysis/sentiment.py`와 중복**됩니다. **감성 분석 기능은 `src/analysis/sentiment.py` (Hugging Face 사용) 및 `src/insight/sentiment_analyzer.py` (Hugging Face 사용 + 캐싱)에 비해 매우 원시적인 키워드 기반 방식으로 구현되어 있어 정확도가 낮을 가능성이 높습니다.** 이 파일은 다른 감성 분석 파일들과 역할이 중복되고 기능 수준도 낮으므로, **정리/제거 대상**으로 보입니다.

## src/backtest/strategy.py

*   **목적**: `backtrader` 프레임워크를 사용하여 플라이트 챌린지 매매법 기반의 백테스팅 전략을 구현합니다. (`src/backtesting/strategies/challenge_strategy_backtest.py`, `src/backtest/strategies/flight_challenge_strategy.py`, `src/backtest/backtest_strategy.py`와 유사/중복)
*   **구현**:
    *   **임포트**: `backtrader`, `pandas`, `logging`을 임포트합니다.
    *   **유틸리티/설정 임포트**: `strategy_utils`와 `config` 모듈을 `try-except`로 안전하게 임포트합니다. 실패 시 경고 로깅 및 폴백 설정을 사용합니다.
    *   **클래스 정의**: `FlightBacktestStrategy(bt.Strategy)` 클래스를 정의합니다.
    *   **파라미터 (`params`)**:
        *   지표 계산 관련 파라미터: `rsi_period`, `sma_short/long_period`, `volume_sma_period`, `divergence_window`, `trendline_window`, `sr_window`, `volume_spike_factor` 등을 정의합니다.
        *   손익 관리: `tp_ratio`, `sl_ratio` (기본값 10%, 5%)
        *   주문: `stake` (고정 주문 수량)
    *   **로깅 (`log`)**: 표준 `backtrader` 로깅 함수를 정의합니다.
    *   **초기화 (`__init__`)**:
        *   데이터 라인(close, high, low, volume, open)을 참조합니다.
        *   주문 추적 변수(`order`, `buyprice`, `buycomm`, `bar_executed`)를 초기화합니다.
        *   **지표 변수 초기화**: `rsi`, `sma_short`, `sma_long`, `volume_sma`, `rsi_divergence`, `volume_spike`, `sr_levels`, `trend_event` 등의 변수를 선언합니다. 이 변수들은 `next` 메서드 내에서 **외부 `strategy_utils` 함수를 호출하여 계산**됩니다. (내부 `bt.indicators` 사용과 다름)
        *   주문 알림 (`notify_order`)**: 주문 상태 변경 시 로그를 기록합니다. 매수 완료 시 `buyprice`, `buycomm`을 저장하고, `bar_executed`를 업데이트합니다. 실패/거절 시 로그를 남깁니다. `self.order`를 초기화하여 다음 주문이 가능하도록 합니다.
        *   거래 알림 (`notify_trade`)**: 거래(매수/매도 쌍)가 종료되었을 때 순손익(pnlcomm)을 로깅합니다.
        *   데이터 변환 (`get_data_as_dataframe`)**: `next` 메서드 내에서 **`strategy_utils` 함수에 전달하기 위해** 현재까지의 `backtrader` 데이터를 Pandas DataFrame으로 변환하는 헬퍼 함수입니다. 매번 호출 시 DataFrame을 새로 생성하므로 성능에 영향을 줄 수 있습니다.
        *   핵심 로직 (`next`)**:
            *   진행 중인 주문이 있거나 지표 계산에 필요한 최소 데이터가 없으면 반환합니다.
            *   **지표 계산**:
                *   `get_data_as_dataframe`을 호출하여 최근 데이터를 DataFrame으로 가져옵니다.
                *   **`strategy_utils` 모듈의 함수들 (`calculate_rsi`, `calculate_sma`, `detect_rsi_divergence`, `detect_volume_spike`, `detect_support_resistance_by_volume`, `detect_trendline_breakout`)을 호출하여 지표 값을 계산하고 클래스 변수에 저장합니다.** (오류 발생 시 로그 기록)
                *   **진입/청산 로직**:
                    *   포지션이 없는 경우:
                        *   Long 조건 (`long_conditions_met` 리스트)과 Short 조건 (`short_conditions_met` 리스트)을 각각 확인합니다.
                        *   조건 확인 시 계산된 지표 변수(`self.rsi_divergence`, `support`, `resistance`, `self.volume_spike`, `self.sma_short`, `self.trend_event`)를 사용합니다. (SMA 비교 로직은 `self.data.sma_short[-1]` 참조 오류 가능성 있음)
                        *   조건 조합: 다이버전스가 있거나 다른 조건이 2개 이상(Long) 또는 1개 이상(Short) 만족하면 진입 결정 (`entry_side`, `entry_reason` 설정). (Short 조건 결정 부분에 `elif` 누락)
                        *   진입 결정 시 주문 실행 (`self.buy` 또는 `self.sell` 호출).
                        *   **TP/SL 동시 제출 시도**: 진입 주문(`self.order`)을 `parent`로 하여 Limit(TP) 주문과 Stop(SL) 주문을 함께 제출합니다. (`transmit` 파라미터 사용)
                    *   포지션이 있는 경우: 별도 로직 없음 (진입 시 TP/SL 설정됨 가정).
        *   종료 (`stop`)**: (스니펫에는 보이지 않음) 백테스트 종료 시 필요한 정리 작업을 수행합니다.
        *   상태**:
            *   기능 중복**: `src/backtesting/strategies/challenge_strategy_backtest.py`와 **매우 유사한 기능**을 가지며, **심각한 중복**입니다.
            *   외부 의존성**: 지표 계산을 `backtrader` 내부 지표(`bt.indicators`)가 아닌 외부 `strategy_utils` 모듈에 크게 의존합니다. 이는 `backtrader`의 최적화 기능을 활용하기 어렵게 만들 수 있습니다.
            *   성능 문제 가능성**: `next` 메서드 내에서 매번 DataFrame을 새로 생성하는 `get_data_as_dataframe` 호출은 성능 저하를 유발할 수 있습니다.
            *   구현 오류 가능성**: SMA 값 비교 시 `self.data.sma_short[-1]` 참조, Short 조건 `elif` 누락 등 잠재적 오류가 보입니다.
            *   와 `backtesting/strategies/challenge_strategy_backtest.py` 와의 비교**: `challenge_strategy_backtest.py`는 `bt.indicators`를 사용하여 내부적으로 지표를 계산하고 감성 분석 필터링까지 포함하는 등, 이 파일보다 더 `backtrader` 친화적이고 완성도가 높아 보입니다.
        *   권장 조치**: **이 파일(`src/backtest/strategy.py`)은 제거하는 것이 강력히 권장됩니다.** 기능이 중복되고, 외부 의존성이 높으며, 구현 방식이 비효율적입니다. `src/backtesting/strategies/challenge_strategy_backtest.py`를 주력 `backtrader` 전략으로 사용하고, `strategy_utils`와의 의존성 문제는 해당 파일 내에서 해결하거나 (`challenge_strategy_backtest`는 이미 내부 지표 사용), 혹은 `strategy_utils`를 정리하는 과정에서 해결해야 합니다.

## src/backtest/backtest_runner.py

*   **목적**: `backtrader` 프레임워크를 사용하여 백테스팅을 실행하는 스크립트입니다. (`src/backtesting/backtest_runner.py`와 유사/중복)
*   **구현**:
    *   **임포트**: `backtrader`, `pandas`, `matplotlib.pyplot`, `datetime`을 임포트합니다.
    *   **전략/유틸리티 임포트**:
        *   **`src.backtest.strategies.flight_challenge_strategy.FlightBacktestStrategy`** 를 임포트합니다. (이전 분석에서 이 전략은 기능 구현 수준이 낮고 중복됨)
        *   `src.utils.common.get_historical_data` 를 데이터 로딩 함수로 임포트합니다.
        *   `src.config.settings` 임포트 주석 처리됨.
    *   **Matplotlib 설정**: 플롯 크기와 배경색을 설정합니다. Jupyter 환경에서의 `%matplotlib inline` 사용 주석이 있습니다.
    *   **메인 실행 블록 (`if __name__ == '__main__':`)**:
        *   **Cerebro 엔진 생성**: `bt.Cerebro()` 호출.
        *   **전략 추가**: `cerebro.addstrategy(FlightBacktestStrategy)` 호출. 전략 파라미터 오버라이드 예시 주석 포함.
        *   **데이터 로딩**:
            *   심볼(`BTCUSDT`), 시작/종료 날짜, 인터벌(`1d`)을 하드코딩하여 정의합니다.
            *   `get_historical_data` 함수를 호출하여 데이터를 로드합니다.
            *   데이터 로딩 실패 시 오류 메시지 출력 후 종료.
            *   DataFrame 컬럼 이름 변경 필요성 주석 언급.
            *   `bt.feeds.PandasData`를 사용하여 데이터를 Cerebro에 추가합니다.
        *   **초기 자금 설정**: `start_cash = 10000.0` 하드코딩. `cerebro.broker.setcash()` 호출.
        *   **수수료 설정**: 0.1% (`0.001`) 하드코딩. `cerebro.broker.setcommission()` 호출.
        *   **분석기 추가**: `SharpeRatio`, `Returns`, `DrawDown`, `TradeAnalyzer` 분석기를 추가합니다.
        *   **백테스트 실행**: `cerebro.run()` 호출.
        *   **결과 분석 및 출력**:
            *   최종 포트폴리오 가치와 총 수익률을 계산하고 출력합니다.
            *   분석기 결과를 추출하여 샤프 비율, 연간 수익률, MDD, 총 거래 횟수, 승률 등을 계산하고 출력합니다. 결과 접근 시 발생할 수 있는 오류(`KeyError` 등) 처리 포함.
        *   **결과 시각화**:
            *   `cerebro.plot()`을 호출하여 결과를 플로팅합니다.
            *   캔들스틱 스타일, 색상, 볼륨 표시, 서브플롯 사용 등 다양한 플롯 설정을 `plot_config` 딕셔너리를 통해 지정합니다.
            *   DPI 설정 주석 포함. `iplot=False`로 정적 플롯 생성.
            *   플로팅 중 오류 발생 시 메시지 출력.
*   **상태**:
    *   기능 중복**: `src/backtesting/backtest_runner.py`와 **기능 및 역할이 거의 완전히 중복**됩니다. 기본적인 `backtrader` 실행 흐름은 동일합니다.
    *   차이점**:
        *   **전략 참조**: 이 파일은 `src/backtest/strategies/flight_challenge_strategy.py`를 사용하지만, `backtesting/backtest_runner.py`는 `src/backtesting/strategies/challenge_strategy_backtest.py`를 사용합니다. 후자가 더 완성도 높은 전략입니다.
        *   **설정 참조**: 이 파일은 대부분의 파라미터(심볼, 기간, 초기 자금, 수수료)를 **하드코딩**하는 반면, `backtesting/backtest_runner.py`는 `settings.py`에서 설정을 가져오는 구조입니다. 후자가 더 유연하고 관리하기 좋습니다.
        *   **데이터 로딩**: 데이터 로딩 방식은 유사하지만, `backtesting/backtest_runner.py`는 다양한 데이터 소스(yahoo, csv, db) 및 데이터 검증/전처리 로직을 포함하여 더 견고합니다.
        *   **구조**: 이 파일은 모든 로직이 `if __name__ == '__main__':` 블록 안에 있지만, `backtesting/backtest_runner.py`는 `run_backtest` 함수로 캡슐화하는 시도를 일부 보여줍니다 (비록 현재는 주석처리 되어 있을 수 있음).
    *   결론**: 이 파일은 `backtesting/backtest_runner.py`에 비해 **덜 유연하고, 덜 견고하며, 기능이 부족한 버전**입니다. 하드코딩된 설정과 기능이 덜 구현된 전략을 사용합니다.
*   권장 조치**: **이 파일(`src/backtest/backtest_runner.py`)은 제거해야 합니다.** `src/backtesting/backtest_runner.py`가 설정 기반으로 작동하고 더 완성된 전략 및 데이터 처리 로직을 사용하므로, 이를 주력 `backtrader` 실행기로 사용해야 합니다.

---

## 중간 요약 및 권장 사항 (분석 중 발견된 내용 기반)

지금까지 분석한 파일들을 종합해 보면 다음과 같은 주요 문제점과 권장 사항이 도출됩니다.

1.  **심각한 기능 중복** (일부 해결됨):
    *   **기술적 분석 유틸리티**: `src/utils/common.py` 와 `src/utils/strategy_utils.py` 에 RSI, SMA, 다이버전스, 추세선, POC 등 핵심 기능이 중복 구현되어 있었습니다. **-> `common.py` 정리 완료. `strategy_utils.py` 중심으로 통합됨.**
    *   **Backtrader 실행기**: `src/backtesting/backtest_runner.py` 와 `src/backtest/runner.py` 가 거의 동일한 기능을 수행했습니다. **-> `src/backtest/runner.py` 제거 완료. `src/backtesting/backtest_runner.py` 사용.**
    *   **Backtrader 전략**: `src/backtesting/strategies/challenge_strategy_backtest.py`, `src/backtest/strategies/flight_challenge_strategy.py`, `src/backtest/backtest_strategy.py` 가 모두 챌린지 전략 백테스트를 다루지만, 구현 수준이 다르고 중복되었습니다. **-> 중복/미흡 전략 파일들(`src/backtest/...`) 제거 완료. `src/backtesting/strategies/challenge_strategy_backtest.py` 사용.**
    *   **감성 분석**: `src/analysis/sentiment.py`, `src/insight/sentiment_analyzer.py`, `src/analysis/sentiment_analysis.py` 가 뉴스 수집 및 감성 분석 기능을 중복으로 구현하고 있습니다. 구현 방식(HF 모델 vs. 키워드)과 부가 기능(캐싱 등)에 차이가 있습니다. **-> 역할 정의 및 통합/정리 필요.** (HF 모델 기반 파일 중 하나 선택 추천)
    *   **Prophet 기반 분석**: `src/analysis/volatility_alert.py` 와 `src/analysis/volatility.py` 가 Prophet 예측 및 이상 탐지 기능을 유사하게 구현하고 있습니다 (탐지 방식에 차이). **-> 통합 또는 역할 분리 필요.**
    *   **RL 에이전트 학습**: `src/ai_models/train_custom_env.py` 와 `src/ai_models/rl_trainer.py` 가 커스텀 환경 기반 RL 학습 기능을 중복 구현하고 있습니다. **-> 하나로 통합 필요.** (`rl_trainer.py` 가 설정 연동 등에서 약간 우세)
    *   **챌린지 전략 실행**: `src/challenge_trading.py` 와 `src/challenge/strategy.py` 가 챌린지 전략 실행 로직을 중복 구현하고 있습니다. 후자가 API 연동, AI 통합 등 더 상세합니다. **-> 하나로 통합 필요.** (`src/challenge/strategy.py` 기반으로 통합 추천)

2.  **일관성 부족**: 디렉토리 구조(`backtest` vs `backtesting`), 파일명 규칙, 로깅 방식, 설정 참조 방식 등에서 일관성이 부족한 부분이 보입니다.
3.  **Placeholder/미완성 코드**: 일부 파일(`freqai_trainer.py`, `sentiment_analysis.py`의 모델 부분 등)은 실제 기능 대신 placeholder 로직만 포함하고 있습니다.
4.  **외부 라이브러리 관리**: 여러 파일에서 라이브러리 임포트 실패 시 폴백 로직을 포함하고 있어, `poetry`를 통한 의존성 관리가 중요함을 시사합니다.

---

(분석은 `src/crypto_trader/binance_api.py` 및 `src/crypto_trader/order_manager.py` 파일을 찾지 못해 중단되었습니다. 해당 파일이 존재하거나 다른 이름으로 있다면 추가 분석이 필요합니다.)
