import requests
import json
import logging
import pandas as pd # 결과 DataFrame을 테이블 형태로 보내기 위해
import os

# Import settings instance directly
try:
    from src.config.settings import settings
    if settings is None:
        raise ImportError("Settings object is None after import.")
except ImportError:
    # Fallback if settings cannot be imported
    settings = type('obj', (object,), {
        'SLACK_WEBHOOK_URL': os.getenv("SLACK_WEBHOOK_URL") # Use getenv for fallback
    })()
    logging.getLogger(__name__).warning("Could not import settings instance. Using basic os.getenv fallbacks for Slack Webhook.")

# Setup logging
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

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

# --- Slack Notification Function ---
def send_slack_notification(message: str, channel: str | None = None) -> bool:
    """
    Sends a message to a specified Slack channel using a webhook URL from settings.

    Args:
        message (str): The message text to send.
        channel (str | None): Optional. The specific channel to send to (e.g., "#trading-alerts").
                               If None, sends to the webhook's default channel.

    Returns:
        bool: True if the message was sent successfully (HTTP 200), False otherwise.
    """
    webhook_url = settings.SLACK_WEBHOOK_URL

    if not webhook_url:
        logger.warning("SLACK_WEBHOOK_URL is not configured in settings/.env. Cannot send Slack notification.")
        return False

    payload = {
        "text": message
    }
    if channel:
        payload["channel"] = channel

    headers = {
        'Content-Type': 'application/json'
    }

    try:
        logger.debug(f"Sending Slack notification: {message[:50]}... to channel: {channel or 'default'}")
        response = requests.post(webhook_url, headers=headers, data=json.dumps(payload), timeout=10)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        if response.status_code == 200:
            logger.info(f"Successfully sent Slack notification to {channel or 'default channel'}.")
            return True
        else:
            # This case might not be reached due to raise_for_status, but kept for clarity
            logger.error(f"Failed to send Slack notification. Status Code: {response.status_code}, Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error sending Slack notification: {e}", exc_info=True)
        return False
    except Exception as e:
        logger.error(f"Unexpected error during Slack notification: {e}", exc_info=True)
        return False

# Example usage:
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger.info("Testing Slack Notifier...")
    
    if settings.SLACK_WEBHOOK_URL:
        print(f"SLACK_WEBHOOK_URL found: {settings.SLACK_WEBHOOK_URL[:30]}...")
        test_message = "✅ Test message from notifier.py!"
        # Send to default channel
        success_default = send_slack_notification(test_message)
        print(f"Sent to default channel: {'Success' if success_default else 'Failed'}")
        
        # Send to a specific channel (replace '#your-test-channel' if needed)
        # test_channel = "#your-test-channel"
        # success_specific = send_slack_notification(f"{test_message} (to specific channel)", channel=test_channel)
        # print(f"Sent to {test_channel}: {'Success' if success_specific else 'Failed'}")
    else:
        print("SLACK_WEBHOOK_URL not set in .env. Skipping test notification.")

# --- 사용 예시 (테스트용) ---
if __name__ == "__main__":
    # 1. 간단한 정보 메시지
    send_slack_notification("시스템 정보", level="info")

    # 2. 경고 메시지
    send_slack_notification("API 연결 경고", level="warning")

    # 3. 오류 메시지
    send_slack_notification("주문 실패", level="error")

    # 4. 성공 메시지 (결과 포함)
    backtest_result = {
        "전략": "Flight Challenge",
        "기간": "2023-01-01 ~ 2023-12-31",
        "최종 수익률": "+150.5%",
        "최대 낙폭 (MDD)": "-25.2%",
        "승률": "65%"
    }
    result_text = "\n".join([f"- {k}: {v}" for k, v in backtest_result.items()])
    send_slack_notification(f"백테스트 결과 요약\n{result_text}", level="success")

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