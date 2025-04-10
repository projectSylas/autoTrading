import requests
import json
import logging
import pandas as pd # 결과 DataFrame을 테이블 형태로 보내기 위해
from src.config.settings import SLACK_WEBHOOK_URL
from datetime import datetime

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Slack 메시지 섹션 구성 도우미 함수
def create_slack_block(title: str, text: str, block_type: str = "section") -> dict:
    """Slack Block Kit의 기본 섹션/헤더 블록을 생성합니다."""
    if block_type == "header":
        return {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": title,
                "emoji": True
            }
        }
    # 기본값: section
    return {
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": f"*{title}*\n{text}"
        }
    }

def format_dataframe_for_slack(df: pd.DataFrame, max_rows: int = 10) -> str:
    """DataFrame을 Slack 마크다운 테이블 형식으로 변환합니다 (간단 버전)."""
    if df is None or df.empty:
        return "(데이터 없음)"
    # 헤더 생성
    header = " | ".join([f"`{col}`" for col in df.columns])
    # 구분선 생성
    divider = " | ".join(["---" for _ in df.columns])
    # 데이터 행 생성 (최대 max_rows 까지만)
    rows = []
    for _, row in df.head(max_rows).iterrows():
        rows.append(" | ".join([f"{val}" for val in row.values]))

    table_str = f"{header}\n{divider}\n" + "\n".join(rows)
    if len(df) > max_rows:
        table_str += f"\n... ({len(df) - max_rows}개 행 생략)"
    # 코드 블록으로 감싸서 반환
    return f"```\n{table_str}\n```"


def send_slack_notification(
    message_title: str,
    message_body: str = "", # 일반 텍스트 메시지 (선택 사항)
    blocks: list | None = None, # 사용자 정의 블록 (선택 사항)
    level: str = "info" # 메시지 레벨 ('info', 'warning', 'error', 'success')
) -> bool:
    """통합된 형식으로 Slack 알림을 보냅니다.

    Args:
        message_title (str): 메시지의 주 제목 (항상 표시됨).
        message_body (str): 간단한 텍스트 본문 (blocks가 없으면 사용됨).
        blocks (list | None): Slack Block Kit 블록 리스트. None이면 기본 블록 생성.
        level (str): 메시지 중요도 ('info', 'warning', 'error', 'success'). 아이콘/색상에 영향.

    Returns:
        bool: 메시지 전송 성공 여부.
    """
    if not SLACK_WEBHOOK_URL:
        logging.warning("Slack Webhook URL이 설정되지 않아 알림을 보낼 수 없습니다.")
        return False

    # 레벨에 따른 아이콘/색상 설정 (간단 예시)
    icon_emoji = ":information_source:" # info (default)
    if level == "warning":
        icon_emoji = ":warning:"
    elif level == "error":
        icon_emoji = ":fire:"
    elif level == "success":
        icon_emoji = ":white_check_mark:"

    # 기본 헤더 블록 생성
    final_blocks = [{
        "type": "header",
        "text": {
            "type": "plain_text",
            "text": f"{icon_emoji} {message_title}",
            "emoji": True
        }
    }]

    # 사용자 정의 블록이 있으면 추가
    if blocks:
        final_blocks.extend(blocks)
    # 사용자 정의 블록 없고 메시지 본문만 있으면, 기본 섹션 블록 추가
    elif message_body:
        final_blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": message_body
            }
        })

    # 푸터 (시간 정보 추가)
    final_blocks.append({
        "type": "context",
        "elements": [
            {
                "type": "mrkdwn",
                "text": f"발송 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            }
        ]
    })

    # Slack Payload 구성
    slack_payload = {
        "blocks": final_blocks
    }

    try:
        response = requests.post(SLACK_WEBHOOK_URL, data=json.dumps(slack_payload),
                                 headers={'Content-Type': 'application/json'}, timeout=10)
        response.raise_for_status() # HTTP 오류 발생 시 예외 발생
        logging.info(f"Slack 알림 전송 성공: {message_title}")
        return True
    except requests.exceptions.RequestException as e:
        logging.error(f"Slack 알림 전송 실패: {e}")
        return False
    except Exception as e:
        logging.error(f"Slack 알림 처리 중 예외 발생: {e}", exc_info=True)
        return False

# --- 사용 예시 (테스트용) ---
if __name__ == "__main__":
    # 1. 간단한 정보 메시지
    send_slack_notification("시스템 정보", "자동매매 시스템이 시작되었습니다.", level="info")

    # 2. 경고 메시지
    send_slack_notification("API 연결 경고", "Binance API 응답 속도가 느립니다.", level="warning")

    # 3. 오류 메시지
    send_slack_notification("주문 실패", "Alpaca 주문 중 오류가 발생했습니다. 상세 로그를 확인하세요.", level="error")

    # 4. 성공 메시지 (결과 포함)
    backtest_result = {
        "전략": "Flight Challenge",
        "기간": "2023-01-01 ~ 2023-12-31",
        "최종 수익률": "+150.5%",
        "최대 낙폭 (MDD)": "-25.2%",
        "승률": "65%"
    }
    result_text = "\n".join([f"- {k}: {v}" for k, v in backtest_result.items()])
    send_slack_notification("백테스트 결과 요약", message_body=f"백테스트가 성공적으로 완료되었습니다.\n{result_text}", level="success")

    # 5. 복잡한 블록 구조 사용 (DataFrame 결과 포함)
    sentiment_df = pd.DataFrame({
        'Keyword': ['Bitcoin', 'Federal Reserve', 'Ethereum'],
        'Sentiment': ['neutral', 'negative', 'positive'],
        'Score': [0.55, -0.82, 0.91]
    })
    custom_blocks = [
        create_slack_block("분석 개요", "오늘의 주요 키워드 감성 분석 결과입니다."),
        create_slack_block("결과 테이블", format_dataframe_for_slack(sentiment_df)),
        {
            "type": "divider"
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*참고*: 부정적 감성(`negative`)이 발견되었습니다. 투자 결정에 참고하세요."
            }
        }
    ]
    send_slack_notification("시장 감성 분석 결과", blocks=custom_blocks, level="info") 