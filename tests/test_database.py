import pytest
import pandas as pd
from unittest.mock import patch, MagicMock, ANY, call  # Add call for checking multiple calls
import logging
from datetime import datetime, timedelta
import psycopg2  # For exception types
import pandas.testing as pd_testing # For comparing DataFrames

# Import functions/classes to test
from src.utils import database

# Mock the connection pool early
@pytest.fixture(scope='module', autouse=True)
def mock_db_pool():
    with patch('src.utils.database.connection_pool', MagicMock()) as mock_pool:
        mock_pool.__bool__.return_value = True
        yield mock_pool

@pytest.fixture
def mock_cursor_context():
    """Fixture to provide a mocked DB cursor within a context manager."""
    mock_cursor = MagicMock()
    mock_conn = MagicMock()
    mock_context = MagicMock()
    mock_context.__enter__.return_value = mock_cursor
    mock_context.__exit__.return_value = None
    with patch('src.utils.database.get_db_cursor', return_value=mock_context) as patched_get_cursor:
        yield mock_cursor, patched_get_cursor, mock_context

# --- Tests for log_prediction_to_db ---

def test_log_prediction_to_db_success(mock_cursor_context):
    """Test successful logging of prediction data."""
    mock_cursor, patched_get_cursor, _ = mock_cursor_context
    test_data = {
        'prediction_time': datetime(2024, 1, 1, 12, 0, 0),
        'symbol': 'TEST/USD',
        'predicted_price': 50000.50,
        'model_type': 'LSTM',
        'input_seq_len': 60
    }
    database.log_prediction_to_db(**test_data)
    patched_get_cursor.assert_called_once_with(commit=True)
    mock_cursor.execute.assert_called_once()
    args, kwargs = mock_cursor.execute.call_args
    sql_query = args[0]
    sql_params = args[1]
    assert "INSERT INTO prediction_logs" in sql_query
    # Check columns dynamically based on test_data keys for robustness
    expected_cols_str = ", ".join(sorted(test_data.keys()))
    assert expected_cols_str in sql_query.replace(" ", "") # Ignore spaces in check
    assert f"VALUES ({', '.join(['%s'] * len(test_data))})" in sql_query
    assert len(sql_params) == len(test_data)
    # Check parameters match data (order might vary based on dict iteration)
    for i, key in enumerate(sorted(test_data.keys())):
        assert sql_params[i] == test_data[key]

def test_log_prediction_to_db_missing_required(mock_cursor_context, caplog):
    """Test skipping log when required fields are missing."""
    mock_cursor, patched_get_cursor, _ = mock_cursor_context
    caplog.set_level(logging.WARNING)
    test_data_missing = {
        'prediction_time': datetime(2024, 1, 1, 13, 0, 0),
        # 'symbol': 'TEST/USD', # Missing symbol
        'predicted_price': 51000.00,
        'model_type': 'Transformer'
    }
    database.log_prediction_to_db(**test_data_missing)
    patched_get_cursor.assert_not_called()
    mock_cursor.execute.assert_not_called()
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == 'WARNING'
    assert "Required prediction data missing" in caplog.text

def test_log_prediction_to_db_db_error(mock_cursor_context, caplog):
    """Test error logging when cursor.execute fails."""
    mock_cursor, patched_get_cursor, mock_context = mock_cursor_context
    caplog.set_level(logging.ERROR)
    db_error = psycopg2.Error("Simulated DB Error")
    mock_cursor.execute.side_effect = db_error
    test_data = {
        'prediction_time': datetime(2024, 1, 1, 14, 0, 0),
        'symbol': 'ERROR/USD',
        'predicted_price': 49000.00
    }
    database.log_prediction_to_db(**test_data)
    mock_cursor.execute.assert_called_once()
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == 'ERROR'
    assert "Failed to log prediction to DB" in caplog.text
    assert "Simulated DB Error" in caplog.text

# --- Tests for initialize_database ---

def test_initialize_database_success(mock_cursor_context):
    """Test successful execution of all CREATE TABLE commands."""
    mock_cursor, patched_get_cursor, _ = mock_cursor_context
    database.initialize_database()
    patched_get_cursor.assert_called_once_with(commit=True)
    assert mock_cursor.execute.call_count >= 4 # Check at least 4 tables
    execute_calls = mock_cursor.execute.call_args_list
    sql_commands_executed = [c.args[0] for c in execute_calls]
    assert any("CREATE TABLE IF NOT EXISTS trades" in cmd for cmd in sql_commands_executed)
    assert any("CREATE TABLE IF NOT EXISTS sentiment_logs" in cmd for cmd in sql_commands_executed)
    assert any("CREATE TABLE IF NOT EXISTS volatility_logs" in cmd for cmd in sql_commands_executed)
    assert any("CREATE TABLE IF NOT EXISTS prediction_logs" in cmd for cmd in sql_commands_executed)

def test_initialize_database_db_error(mock_cursor_context, caplog):
    """Test error handling during table creation."""
    mock_cursor, patched_get_cursor, _ = mock_cursor_context
    caplog.set_level(logging.ERROR)
    db_error = psycopg2.Error("Simulated CREATE TABLE Error")
    mock_cursor.execute.side_effect = db_error
    with pytest.raises(psycopg2.Error, match="Simulated CREATE TABLE Error"):
        database.initialize_database()
    assert len(caplog.records) >= 1
    assert "Failed to initialize database" in caplog.text
    assert "Simulated CREATE TABLE Error" in caplog.text

# --- Tests for get_log_data ---

@pytest.fixture
def mock_dict_cursor_data():
    """Provides sample data simulating DictCursor fetchall."""
    # Simulate DictRow - accessing by key
    row1 = MagicMock()
    row1.__getitem__.side_effect = lambda key: {'id': 1, 'symbol': 'BTCUSDT', 'predicted_price': 60000, 'timestamp': datetime(2024, 1, 1, 10, 0, 0)}[key]
    row2 = MagicMock()
    row2.__getitem__.side_effect = lambda key: {'id': 2, 'symbol': 'ETHUSDT', 'predicted_price': 3000, 'timestamp': datetime(2024, 1, 1, 11, 0, 0)}[key]
    # Also make them behave like dicts for dict(row) conversion
    row1.keys.return_value = {'id', 'symbol', 'predicted_price', 'timestamp'}
    row2.keys.return_value = {'id', 'symbol', 'predicted_price', 'timestamp'}
    return [row1, row2]

def test_get_log_data_success(mock_cursor_context, mock_dict_cursor_data):
    """Test successful data retrieval and DataFrame conversion."""
    mock_cursor, patched_get_cursor, _ = mock_cursor_context
    mock_cursor.fetchall.return_value = mock_dict_cursor_data
    table = 'prediction_logs'
    df = database.get_log_data(table_name=table)
    patched_get_cursor.assert_called_once_with() # commit=False
    mock_cursor.execute.assert_called_once()
    sql_query, sql_params = mock_cursor.execute.call_args.args
    assert f"SELECT * FROM {table}" in sql_query
    assert "ORDER BY timestamp DESC" in sql_query
    assert not sql_params
    mock_cursor.fetchall.assert_called_once()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    # Use pandas testing to compare expected DataFrame
    expected_df = pd.DataFrame([
        {'id': 1, 'symbol': 'BTCUSDT', 'predicted_price': 60000, 'timestamp': datetime(2024, 1, 1, 10, 0, 0)},
        {'id': 2, 'symbol': 'ETHUSDT', 'predicted_price': 3000, 'timestamp': datetime(2024, 1, 1, 11, 0, 0)}
    ])
    # Adjust expected_df columns/dtypes if necessary based on actual DB schema/DictRow conversion
    pd_testing.assert_frame_equal(df, expected_df, check_dtype=False) # Ignore dtype for simplicity here

def test_get_log_data_with_filters(mock_cursor_context, mock_dict_cursor_data):
    """Test data retrieval with days, limit, and query_filter."""
    mock_cursor, patched_get_cursor, _ = mock_cursor_context
    mock_cursor.fetchall.return_value = mock_dict_cursor_data
    table = 'trades'
    days = 7
    limit = 10
    query_filter = "strategy = 'core'"
    df = database.get_log_data(table_name=table, days=days, limit=limit, query_filter=query_filter)
    mock_cursor.execute.assert_called_once()
    sql_query, sql_params = mock_cursor.execute.call_args.args
    assert f"SELECT * FROM {table}" in sql_query
    assert "WHERE timestamp >= NOW() - INTERVAL %s days" in sql_query
    assert f"AND ({query_filter})" in sql_query # Ensure filter is wrapped
    assert "ORDER BY timestamp DESC" in sql_query
    assert "LIMIT %s" in sql_query
    assert len(sql_params) == 2 # Should be days and limit only, filter is part of string
    assert sql_params[0] == days
    assert sql_params[1] == limit
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2

def test_get_log_data_db_error(mock_cursor_context, caplog):
    """Test error handling during data retrieval."""
    mock_cursor, patched_get_cursor, _ = mock_cursor_context
    caplog.set_level(logging.ERROR)
    db_error = psycopg2.Error("Simulated SELECT Error")
    mock_cursor.execute.side_effect = db_error
    table = 'some_table'
    df = database.get_log_data(table_name=table)
    mock_cursor.execute.assert_called_once()
    assert len(caplog.records) >= 1
    assert f"Failed to retrieve data from '{table}'" in caplog.text
    assert "Simulated SELECT Error" in caplog.text
    assert isinstance(df, pd.DataFrame)
    assert df.empty

# TODO: Add tests for other database functions like get_log_data, initialize_database
# Testing initialize_database would involve checking cursor.execute calls for CREATE TABLE
# Testing get_log_data would involve mocking cursor.fetchall() to return mock data 