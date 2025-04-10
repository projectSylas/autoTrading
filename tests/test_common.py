import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import logging

# Import functions to test from src.utils.common
from src.utils import common

# --- Fixtures for Test Data ---

@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Create a sample DataFrame for testing indicators."""
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05',
                          '2023-01-06', '2023-01-07', '2023-01-08', '2023-01-09', '2023-01-10',
                          '2023-01-11', '2023-01-12', '2023-01-13', '2023-01-14', '2023-01-15'])
    data = {
        'Open': [10, 11, 12, 11, 13, 14, 15, 14, 13, 12, 11, 10, 9, 8, 7],
        'High': [11, 12, 13, 12, 14, 15, 16, 15, 14, 13, 12, 11, 10, 9, 8],
        'Low':  [9, 10, 11, 10, 12, 13, 14, 13, 12, 11, 10, 9, 8, 7, 6],
        'Close':[11, 12, 11, 13, 14, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6],
        'Volume':[100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 500] # Add volume spike
    }
    df = pd.DataFrame(data, index=dates)
    return df

# --- Tests for Indicator Calculations ---

def test_calculate_sma(sample_dataframe):
    """Test SMA calculation."""
    sma_series = common.calculate_sma(dataframe=sample_dataframe, window=5, column='Close')
    assert isinstance(sma_series, pd.Series)
    assert not sma_series.empty
    # Check the last calculated SMA value (manually calculate for verification)
    # Close values: [11, 12, 11, 13, 14], [12, 11, 13, 14, 15], ... [9, 8, 7, 6]
    # Last 5 Close values: [10, 9, 8, 7, 6]
    expected_last_sma = (10 + 9 + 8 + 7 + 6) / 5
    assert np.isclose(sma_series.iloc[-1], expected_last_sma)
    # Test with insufficient data
    sma_short = common.calculate_sma(dataframe=sample_dataframe.head(3), window=5)
    assert sma_short is None
    # Test with column containing NaN
    df_with_nan = sample_dataframe.copy()
    df_with_nan.loc[df_with_nan.index[5], 'Close'] = np.nan
    sma_nan = common.calculate_sma(dataframe=df_with_nan, window=5)
    assert sma_nan is not None # ta library might handle internal NaNs
    assert isinstance(sma_nan, pd.Series)
    assert sma_nan.isnull().sum() > 0 # Expect some NaNs in the result

def test_calculate_rsi(sample_dataframe):
    """Test RSI calculation."""
    rsi_series = common.calculate_rsi(dataframe=sample_dataframe, window=6, column='Close') # Use shorter window for testing
    assert isinstance(rsi_series, pd.Series)
    assert not rsi_series.empty
    assert rsi_series.iloc[-1] < 50 # Last few prices are decreasing, RSI should be low
    # Test with insufficient data
    rsi_short = common.calculate_rsi(dataframe=sample_dataframe.head(5), window=6)
    assert rsi_short is None

# --- Tests for Detection Functions ---

def test_detect_volume_spike(sample_dataframe):
    """Test volume spike detection."""
    # Last volume (500) is much higher than previous (avg around 160 for last 10 excluding last)
    is_spike = common.detect_volume_spike(sample_dataframe, window=10, factor=2.5)
    assert is_spike is True
    # Test without spike
    no_spike_df = sample_dataframe.copy()
    no_spike_df.loc[no_spike_df.index[-1], 'Volume'] = 200 # No spike
    is_spike_false = common.detect_volume_spike(no_spike_df, window=10, factor=2.5)
    assert is_spike_false is False

def test_detect_rsi_divergence():
    """Test RSI divergence detection with synthetic data."""
    # Bullish Divergence: Price LL, RSI HL
    price_bull = pd.Series([10, 8, 12, 7, 9]) # LL: 8 -> 7
    rsi_bull = pd.Series([30, 25, 60, 35, 50]) # HL: 25 -> 35
    assert common.detect_rsi_divergence(price_bull, rsi_bull, window=5) == 'bullish'

    # Bearish Divergence: Price HH, RSI LH
    price_bear = pd.Series([10, 12, 11, 13, 12]) # HH: 12 -> 13
    rsi_bear = pd.Series([70, 80, 65, 75, 60]) # LH: 80 -> 75
    assert common.detect_rsi_divergence(price_bear, rsi_bear, window=5) == 'bearish'

    # No Divergence
    price_none = pd.Series([10, 12, 11, 13, 15]) # HH
    rsi_none = pd.Series([40, 60, 55, 70, 75])   # HH
    assert common.detect_rsi_divergence(price_none, rsi_none, window=5) is None

    # Insufficient data
    assert common.detect_rsi_divergence(price_bull.head(3), rsi_bull.head(3), window=5) is None

# --- Tests for Data Fetching (using mocking) ---

@patch('yfinance.Ticker') # Mock the Ticker class
def test_get_historical_data_success(mock_ticker, sample_dataframe):
    """Test get_historical_data success path with mocked yfinance."""
    # Configure the mock Ticker instance and its history method
    mock_instance = MagicMock()
    mock_instance.history.return_value = sample_dataframe # Return our fixture
    mock_ticker.return_value = mock_instance # When Ticker(symbol) is called, return our mock

    symbol = "TEST.SYM"
    df = common.get_historical_data(symbol, period="1mo", interval="1d")

    mock_ticker.assert_called_once_with(symbol)
    mock_instance.history.assert_called_once_with(period="1mo", interval="1d", auto_adjust=False)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    pd.testing.assert_frame_equal(df, sample_dataframe) # Check if returned df is the one we provided

@patch('yfinance.Ticker')
def test_get_historical_data_empty(mock_ticker):
    """Test get_historical_data returns empty DataFrame when yfinance gives empty."""
    mock_instance = MagicMock()
    mock_instance.history.return_value = pd.DataFrame() # Simulate empty return
    mock_ticker.return_value = mock_instance

    df = common.get_historical_data("EMPTY.SYM")
    assert isinstance(df, pd.DataFrame)
    assert df.empty

@patch('yfinance.Ticker')
def test_get_historical_data_error(mock_ticker, caplog):
    """Test get_historical_data handles yfinance exception."""
    mock_instance = MagicMock()
    mock_instance.history.side_effect = Exception("Test yfinance error") # Simulate exception
    mock_ticker.return_value = mock_instance

    caplog.set_level(logging.ERROR) # Capture ERROR logs for this test
    df = common.get_historical_data("ERROR.SYM")

    assert isinstance(df, pd.DataFrame)
    assert df.empty
    assert len(caplog.records) == 1
    assert "ERROR.SYM 과거 데이터 조회 중 오류 발생" in caplog.text
    assert "Test yfinance error" in caplog.text 