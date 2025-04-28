# Slack 알림 모듈 (`src/utils/notifier.py`)

## 1. 개요

프로젝트 내 다양한 이벤트(진입, 청산, 오류 등) 발생 시 Slack으로 구조화된 알림을 전송하는 유틸리티 모듈입니다.

## 2. 주요 기능

- **Slack Block Kit 활용**: 메시지 가독성을 높이기 위해 Slack Block Kit 기반의 서식있는 메시지를 생성합니다.
- **상황별 알림 함수**: 진입, 청산(익절/손절), 오류 발생 등 특정 상황에 맞는 상세 정보를 포함하는 별도의 알림 함수를 제공합니다.
- **유연한 메시지 구성**: 헤더, 구분선, 마크다운 텍스트, 필드 섹션 등 재사용 가능한 블록 생성 함수를 제공하여 다양한 형태의 메시지를 쉽게 구성할 수 있습니다.
- **환경 변수 사용**: Slack Bot Token 및 채널 정보는 환경 변수(`SLACK_BOT_TOKEN`, `SLACK_CHANNEL`)에서 로드하여 보안을 강화합니다.

## 3. 주요 함수

- **블록 생성 함수**:
    - `create_slack_header_block(text: str) -> dict`: 헤더 블록 생성
    - `create_slack_divider_block() -> dict`: 구분선 블록 생성
    - `create_slack_markdown_section(text: str) -> dict`: 마크다운 텍스트 섹션 생성
    - `create_slack_fields_section(fields: List[Tuple[str, str]]) -> dict`: 키-값 형태의 필드 섹션 생성
- **특화된 알림 함수**:
    - `send_entry_alert(...)`: 진입 시그널 알림 (심볼, 방향, 가격, 수량, 근거 등 포함)
    - `send_exit_alert(...)`: 포지션 청산 알림 (심볼, 진입/청산 가격, 수량, PnL, 사유 등 포함, 수익/손실에 따라 이모지 표시)
    - `send_error_alert(subject: str, error_details: str)`: 오류 발생 알림 (오류 제목 및 상세 내용 포함)
- **기본 알림 함수**:
    - `send_slack_notification(message: str = None, blocks: list = None, level: str = "info")`: 일반 텍스트 또는 블록 형태의 메시지를 전송하는 핵심 함수. 로그 레벨(info, warning, error) 지정 가능.

## 4. 설정

- `.env` 파일에 `SLACK_BOT_TOKEN`과 `SLACK_CHANNEL` 환경 변수를 설정해야 합니다.

## 5. 사용 예시

```python
from src.utils.notifier import send_entry_alert, send_exit_alert, send_error_alert
import datetime

# 진입 알림 예시
send_entry_alert(
    timestamp=datetime.datetime.now(),
    symbol="BTC/USDT",
    side="Long",
    entry_price=70000.0,
    quantity=0.1,
    reason="RSI < 30 and Trendline Breakout"
)

# 익절 알림 예시
send_exit_alert(
    timestamp=datetime.datetime.now(),
    symbol="BTC/USDT",
    side="Long",
    entry_price=70000.0,
    exit_price=77000.0,
    quantity=0.1,
    pnl_ratio=0.10,
    reason="Take Profit Target Reached"
)

# 오류 알림 예시
try:
    result = 1 / 0
except Exception as e:
    send_error_alert(subject="Critical Calculation Error", error_details=str(e))

``` 