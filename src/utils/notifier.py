import requests
import json
import logging
import pandas as pd # 결과 DataFrame을 테이블 형태로 보내기 위해
import os
from typing import List, Tuple
from datetime import datetime

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

# --- Slack 메시지 포맷팅 및 블록 생성 함수 (개선) --- #
def format_dataframe_for_slack(df: pd.DataFrame, max_rows: int = 10) -> str:
    """DataFrame을 Slack 마크다운 테이블 형식으로 변환합니다 (간단 버전)."""
    if df is None or df.empty:
        return "(데이터 없음)"
    # Limit rows for Slack message length
    df_display = df.head(max_rows)
    # Simple markdown table format (may need escaping for special characters)
    header = " | ".join([f"`{col}`" for col in df_display.columns])
    divider = " | ".join(["---" for _ in df_display.columns])
    rows = []
    for _, row in df_display.iterrows():
        # Convert values to string, handle potential special characters simply for now
        row_values = [str(val).replace("|", "\|").replace("`", "\`") for val in row.values]
        rows.append(" | ".join(row_values))

    table_str = f"{header}\n{divider}\n" + "\n".join(rows)
    if len(df) > max_rows:
        table_str += f"\n... ({len(df) - max_rows} 행 생략)"
    # Return inside a code block for monospace font
    return f"```\n{table_str}\n```"

def create_slack_header_block(title: str) -> dict:
    """Slack Block Kit 헤더 블록을 생성합니다."""
    return {
        "type": "header",
        "text": {
            "type": "plain_text",
            "text": title,
            "emoji": True
        }
    }

def create_slack_divider_block() -> dict:
    """Slack Block Kit 구분선 블록을 생성합니다."""
    return {"type": "divider"}

def create_slack_markdown_section(text: str) -> dict:
    """Markdown 텍스트를 포함하는 Slack 섹션 블록을 생성합니다."""
    return {
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": text
        }
    }

def create_slack_fields_section(fields: List[Tuple[str, str]]) -> dict:
    """Key-Value 쌍 리스트를 사용하여 두 열 형식의 필드 섹션 블록을 생성합니다."""
    if not fields:
        return None
    
    # Slack 필드는 최대 10개, 각 필드 텍스트 길이 제한 등 고려 필요
    markdown_fields = []
    for key, value in fields:
        # key는 bold체로, value는 그대로 표시 (필요시 value 포맷팅 추가)
        markdown_fields.append(f"*{key}*\n{value}")
        
    return {
        "type": "section",
        "fields": [
            {"type": "mrkdwn", "text": field_text} for field_text in markdown_fields
        ]
    }

# --- Slack Notification Function (기존 구현 사용, level 처리 명시) --- #
def send_slack_notification(message: str, channel: str | None = None, blocks: list | None = None, level: str = 'info') -> bool:
    """
    Sends a message (text or blocks) to a specified Slack channel using a webhook URL.

    Args:
        message (str): The basic text message (used if blocks are not provided or as fallback).
        channel (str | None): Optional channel override (e.g., "#trading-alerts").
        blocks (list | None): Optional list of Slack Block Kit blocks for rich formatting.
        level (str): Message level ('info', 'warning', 'error', 'success', 'critical'). Used for logging.

    Returns:
        bool: True if sent successfully, False otherwise.
    """
    webhook_url = settings.SLACK_WEBHOOK_URL

    if not webhook_url:
        logger.warning("SLACK_WEBHOOK_URL is not configured. Cannot send Slack notification.")
        return False

    payload = {}
    if blocks:
        payload["blocks"] = blocks
        # Fallback text for notifications that don't show blocks
        payload["text"] = message 
    else:
        payload["text"] = message

    if channel:
        payload["channel"] = channel

    headers = {
        'Content-Type': 'application/json'
    }

    try:
        log_preview = f"Block count: {len(blocks)}" if blocks else f"Text: {message[:50]}..."
        logger.debug(f"Sending Slack notification ({log_preview}) to channel: {channel or 'default'}, Level: {level.upper()}")
        response = requests.post(webhook_url, headers=headers, data=json.dumps(payload), timeout=10)
        response.raise_for_status()

        if response.status_code == 200:
            logger.info(f"Successfully sent Slack notification ({level.upper()}) to {channel or 'default channel'}.")
            return True
        else:
            logger.error(f"Failed to send Slack notification. Status Code: {response.status_code}, Response: {response.text}")
            return False

    except requests.exceptions.RequestException as e:
        logger.error(f"Error sending Slack notification: {e}", exc_info=True)
        return False
    except Exception as e:
        logger.error(f"Unexpected error during Slack notification: {e}", exc_info=True)
        return False

# --- 특화된 알림 함수들 --- #
def send_entry_alert(symbol: str, side: str, entry_price: float, quantity: float, reason: str, timestamp: datetime | None = None):
    """진입 알림을 Slack으로 전송합니다."""
    ts = timestamp.strftime('%Y-%m-%d %H:%M:%S') if timestamp else "N/A"
    direction_emoji = "🚀" if side.lower() == 'long' else "↘️"
    title = f"{direction_emoji} 신규 진입 ({symbol} {side.upper()})"
    
    fields = [
        ("Timestamp", ts),
        ("Symbol", symbol),
        ("Side", side.upper()),
        ("Entry Price", f"{entry_price:.4f}"),
        ("Quantity", str(quantity)),
        ("Reason", reason[:500]) # 이유가 너무 길 경우 자르기
    ]
    
    blocks = [
        create_slack_header_block(title),
        create_slack_fields_section(fields)
    ]
    
    # Fallback text
    fallback_text = f"{title} - Price: {entry_price:.4f}, Qty: {quantity}, Reason: {reason[:50]}"
    
    send_slack_notification(fallback_text, blocks=blocks, level='info')

def send_exit_alert(symbol: str, side: str, entry_price: float, exit_price: float, quantity: float, pnl_ratio: float, reason: str, timestamp: datetime | None = None):
    """포지션 종료 알림을 Slack으로 전송합니다."""
    ts = timestamp.strftime('%Y-%m-%d %H:%M:%S') if timestamp else "N/A"
    result_emoji = "✅" if pnl_ratio >= 0 else "🔻"
    level = 'success' if pnl_ratio >= 0 else 'warning' # TP는 success, SL은 warning
    title = f"{result_emoji} 포지션 종료 ({symbol} {side.upper()})"
    
    fields = [
        ("Timestamp", ts),
        ("Symbol", symbol),
        ("Side", side.upper()),
        ("Entry Price", f"{entry_price:.4f}"),
        ("Exit Price", f"{exit_price:.4f}"),
        ("Quantity", str(quantity)),
        ("PnL Ratio", f"{pnl_ratio:.2%}"),
        ("Reason", reason[:500])
    ]
    
    blocks = [
        create_slack_header_block(title),
        create_slack_fields_section(fields)
    ]
    
    fallback_text = f"{title} - Exit: {exit_price:.4f}, PnL: {pnl_ratio:.2%}, Reason: {reason[:50]}"
    
    send_slack_notification(fallback_text, blocks=blocks, level=level)

def send_error_alert(subject: str, error_details: str, level: str = 'error'):
    """오류 알림을 Slack으로 전송합니다."""
    title = f"🚨 시스템 오류 발생: {subject}"
    blocks = [
        create_slack_header_block(title),
        create_slack_markdown_section(f"```\n{error_details[:1500]}\n```") # 코드 블록으로 표시
    ]
    fallback_text = f"{title} - {error_details[:100]}..."
    send_slack_notification(fallback_text, blocks=blocks, level=level)

# Example usage (개선된 테스트):
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger.info("Testing Enhanced Slack Notifier...")

    if settings.SLACK_WEBHOOK_URL:
        print(f"SLACK_WEBHOOK_URL found: {settings.SLACK_WEBHOOK_URL[:30]}...")

        # 1. Send Entry Alert
        print("\nTesting Entry Alert...")
        send_entry_alert(
            symbol="BTC/USDT",
            side="long",
            entry_price=65000.50,
            quantity=0.01,
            reason="RSI Bullish Divergence, Trendline Pullback",
            timestamp=datetime.now()
        )

        # 2. Send Exit Alert (Take Profit)
        print("\nTesting Exit Alert (TP)...")
        send_exit_alert(
            symbol="BTC/USDT",
            side="long",
            entry_price=65000.50,
            exit_price=71500.00,
            quantity=0.01,
            pnl_ratio=0.10, # 10%
            reason="Take Profit target reached",
            timestamp=datetime.now()
        )

        # 3. Send Exit Alert (Stop Loss)
        print("\nTesting Exit Alert (SL)...")
        send_exit_alert(
            symbol="ETH/USDT",
            side="short",
            entry_price=3500.00,
            exit_price=3675.00,
            quantity=0.1,
            pnl_ratio=-0.05, # -5%
            reason="Stop Loss triggered",
            timestamp=datetime.now()
        )

        # 4. Send Error Alert
        print("\nTesting Error Alert...")
        try:
            result = 1 / 0
        except Exception as e:
            import traceback
            error_details = f"Error: {e}\n{traceback.format_exc()}"
            send_error_alert(subject="Critical Calculation Error", error_details=error_details, level='critical')

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