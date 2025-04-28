# 챌린지 트레이딩 전략 (`src/challenge_trading.py`)

## 1. 개요

"플라이트 챌린지 매매법" 기반의 고위험/고수익 암호화폐 선물 트레이딩 전략입니다. Binance Futures API 연동을 가정하고 개발되었습니다.

## 2. 핵심 로직

- **데이터**: `yfinance` 또는 `ccxt`를 통해 시세 데이터(OHLCV)를 가져옵니다. (`src/utils/common.py`의 `get_historical_data` 활용)
- **지표 계산**: RSI, SMA, 거래량 등 기술적 지표를 계산합니다. (`src/utils/common.py` 활용)
- **진입 조건 (`detect_entry_opportunity`)**:
    - 추세선 이탈 후 되돌림 확인 (구현 예정)
    - RSI 다이버전스 발생 여부 (구현 예정)
    - 거래량 급증 확인
    - 이동평균선 (예: 7일 SMA) 이탈 방향 확인
    - VPVR 매물대 기반 지지/저항 확인 (거래량 기반으로 근사)
- **포지션 관리 (`manage_position`)**:
    - 진입 후 설정된 익절 (+10~15%) 및 손절 (-5%) 목표 도달 시 포지션 종료
- **알림**: 진입, 익절, 손절 발생 시 Slack으로 실시간 알림 전송 (`src/utils/notifier.py` 활용)

## 3. 주요 함수

- `detect_entry_opportunity(df: pd.DataFrame, symbol: str) -> Tuple[bool, str, float, float, str]`: 진입 시그널, 방향(long/short), 진입 가격, 수량, 진입 근거를 반환합니다.
- `manage_position(symbol: str, entry_price: float, position_side: str, quantity: float)`: 현재 포지션의 익절/손절 조건을 확인하고 청산 알림을 보냅니다.
- `run_challenge_strategy(symbols: List[str])`: 지정된 심볼 목록에 대해 전략을 실행하고 결과를 로깅합니다.

## 4. 향후 개선 사항

- 추세선 이탈 및 RSI 다이버전스 감지 로직 구체화
- Binance API 연동을 통한 실제 주문 실행 기능 추가
- 백테스팅 프레임워크(`backtrader`) 연동
- 로그 상세화 및 데이터베이스 저장 