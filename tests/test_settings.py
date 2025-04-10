import pytest
import os
from unittest.mock import patch, MagicMock
import logging

# Import the module to test
from src.config import settings

# Ensure logging is captured during tests
@pytest.fixture(autouse=True)
def capture_logs(caplog):
    caplog.set_level(logging.WARNING) # Capture WARNING level and above

def test_load_default_settings():
    """Test that settings load with default values when env vars are not set."""
    # Use patch.dict to temporarily clear relevant environment variables
    relevant_envs = ['ALPACA_API_KEY', 'CORE_ASSETS', 'CHALLENGE_LEVERAGE', 'SLACK_WEBHOOK_URL']
    with patch.dict(os.environ, {env: '' for env in relevant_envs}, clear=True):
        # Reload the settings module to apply the mocked environment
        import importlib
        importlib.reload(settings)

        # Assert default values (refer to settings.py for defaults)
        assert settings.ALPACA_API_KEY is None # Defaults to None if not set
        assert settings.CORE_ASSETS == ["SPY", "QQQ", "GLD", "SGOV", "XLP"] # Default list
        assert settings.CHALLENGE_LEVERAGE == 10 # Default int
        assert settings.SLACK_WEBHOOK_URL is None # Defaults to None

def test_load_settings_from_env():
    """Test that settings are loaded correctly from environment variables."""
    mock_env = {
        'ALPACA_API_KEY': 'test_key',
        'ALPACA_SECRET_KEY': 'test_secret',
        'CORE_ASSETS': 'BTC,ETH',
        'CHALLENGE_LEVERAGE': '25',
        'SLACK_WEBHOOK_URL': 'http://fake.slack.url',
        'ALPACA_PAPER': 'False' # Test boolean conversion
    }
    with patch.dict(os.environ, mock_env, clear=True):
        import importlib
        importlib.reload(settings)

        assert settings.ALPACA_API_KEY == 'test_key'
        assert settings.CORE_ASSETS == ['BTC', 'ETH'] # Check list conversion
        assert settings.CHALLENGE_LEVERAGE == 25 # Check int conversion
        assert settings.SLACK_WEBHOOK_URL == 'http://fake.slack.url'
        assert settings.ALPACA_PAPER is False # Check boolean conversion

def test_check_env_vars_missing(caplog):
    """Test check_env_vars logs warnings when essential variables are missing."""
    # Ensure critical env vars are missing
    required_core = ["ALPACA_API_KEY", "ALPACA_SECRET_KEY", "SLACK_WEBHOOK_URL"]
    with patch.dict(os.environ, {'SOME_OTHER_VAR': 'value'}, clear=True): # Clear all, set one irrelevant
        import importlib
        importlib.reload(settings) # Reload settings with cleared env

        # Call the function directly (it might be better to mock logging inside)
        settings.check_env_vars()

        assert len(caplog.records) > 0
        assert caplog.records[0].levelname == 'WARNING'
        # Check if the log message contains expected missing variables (adjust based on check_env_vars logic)
        assert "Core Portfolio" in caplog.text
        assert "ALPACA_API_KEY" in caplog.text
        assert "SLACK_WEBHOOK_URL" in caplog.text
        # Add checks for other categories (Challenge, Sentiment) if check_env_vars covers them

def test_check_env_vars_present(caplog):
    """Test check_env_vars logs info when essential variables are present."""
    mock_env = {
        'ALPACA_API_KEY': 'key',
        'ALPACA_SECRET_KEY': 'secret',
        'SLACK_WEBHOOK_URL': 'url',
        'BINANCE_API_KEY': 'bkey', # Assuming check_env_vars checks these too
        'BINANCE_SECRET_KEY': 'bsecret',
        'NEWS_API_KEY': 'nkey'
        # Add other required keys based on check_env_vars logic
    }
    # Set ENABLE flags if check_env_vars uses them
    os.environ['ENABLE_SENTIMENT'] = 'True'

    with patch.dict(os.environ, mock_env, clear=True):
        import importlib
        # It might be necessary to reload dependencies of settings too if they cache values
        importlib.reload(settings)

        # Reset caplog before calling the function
        caplog.clear()
        caplog.set_level(logging.INFO) # Capture INFO level for this test

        settings.check_env_vars()

        assert len(caplog.records) > 0
        assert caplog.records[-1].levelname == 'INFO' # Last message should be success info
        assert "필요한 환경 변수들이 잘 설정되었습니다." in caplog.text # Check for success message

    # Clean up env var set outside patch.dict
    del os.environ['ENABLE_SENTIMENT']

# Add more tests for specific type conversions (float, boolean variations) if needed 