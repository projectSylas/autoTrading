import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import logging
from sklearn.preprocessing import MinMaxScaler
import torch
import os

# Import functions/classes to test (adjust path as necessary)
# Assuming functions are directly in price_predictor.py or imported there
try:
    from src.ai_models.price_predictor import (
        prepare_sequence_data,
        save_model_weights,
        load_model_weights,
        PricePredictorLSTM # Import model for load/save tests
        # Add other necessary imports like settings if run_price_prediction_flow is tested
    )
    # Assuming models are defined in a submodule or locally
    # from src.ai_models.models import PricePredictorLSTM
except ImportError as e:
    pytest.skip(f"Skipping price predictor tests, failed to import: {e}", allow_module_level=True)

# --- Fixture for sample data ---
@pytest.fixture
def timeseries_df() -> pd.DataFrame:
    """Creates a sample time series DataFrame."""
    dates = pd.date_range(start='2023-01-01', periods=20, freq='D')
    data = {'Close': np.linspace(100, 150, 20) + np.random.normal(0, 5, 20)}
    df = pd.DataFrame(data, index=dates)
    return df

# --- Tests for prepare_sequence_data ---

def test_prepare_sequence_data_output_shape(timeseries_df):
    """Test the output shapes of X and y sequences."""
    input_seq_len = 5
    output_seq_len = 2
    X, y, scaler = prepare_sequence_data(timeseries_df, input_seq_len, output_seq_len, feature_col='Close')

    expected_samples = len(timeseries_df) - input_seq_len - output_seq_len + 1
    assert X.shape == (expected_samples, input_seq_len, 1)
    assert y.shape == (expected_samples, output_seq_len)
    assert isinstance(scaler, MinMaxScaler)

def test_prepare_sequence_data_insufficient_data(timeseries_df):
    """Test behavior with insufficient data length."""
    input_seq_len = 15
    output_seq_len = 10 # Total 25, but df only has 20 rows
    X, y, scaler = prepare_sequence_data(timeseries_df, input_seq_len, output_seq_len)
    assert X.shape == (0,) # Expect empty array
    assert y.shape == (0,) # Expect empty array
    assert scaler is None

def test_prepare_sequence_data_empty_or_missing_col():
    """Test with empty DataFrame or missing column."""
    empty_df = pd.DataFrame()
    X, y, scaler = prepare_sequence_data(empty_df, 5, 2)
    assert X.shape == (0,)
    assert y.shape == (0,)
    assert scaler is None

    df_wrong_col = pd.DataFrame({'Value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    X, y, scaler = prepare_sequence_data(df_wrong_col, 3, 1, feature_col='Close') # 'Close' is missing
    assert X.shape == (0,)
    assert y.shape == (0,)
    assert scaler is None

# --- Tests for save_model_weights and load_model_weights ---

@pytest.fixture
def dummy_model():
    """Creates a dummy PyTorch model instance for testing save/load."""
    # Use the actual model if simple, or a minimal mock
    # Assuming PricePredictorLSTM is importable
    model = PricePredictorLSTM(input_dim=1, hidden_dim=10, num_layers=1, output_dim=1, dropout_prob=0.1)
    return model

@patch('torch.save')
@patch('os.makedirs') # Mock makedirs to avoid actual directory creation
def test_save_model_weights_success(mock_makedirs, mock_torch_save, dummy_model):
    """Test successful model saving."""
    test_path = "mock_dir/model.pth"
    save_model_weights(dummy_model, test_path)

    # Assert os.makedirs was called to ensure directory exists
    mock_makedirs.assert_called_once_with(os.path.dirname(test_path), exist_ok=True)
    # Assert torch.save was called correctly
    mock_torch_save.assert_called_once_with(dummy_model.state_dict(), test_path)

@patch('torch.save')
@patch('os.makedirs')
def test_save_model_weights_error(mock_makedirs, mock_torch_save, dummy_model, caplog):
    """Test error handling during model saving."""
    caplog.set_level(logging.ERROR)
    test_path = "error_dir/model.pth"
    save_error = IOError("Disk full")
    mock_torch_save.side_effect = save_error # Simulate error on save

    save_model_weights(dummy_model, test_path)

    # Assert save was attempted
    mock_torch_save.assert_called_once()
    # Assert error was logged
    assert len(caplog.records) == 1
    assert "Failed to save model weights" in caplog.text
    assert "Disk full" in caplog.text

@patch('torch.load')
@patch('os.path.exists')
def test_load_model_weights_success(mock_os_exists, mock_torch_load, dummy_model):
    """Test successful model loading."""
    test_path = "mock_dir/model.pth"
    mock_os_exists.return_value = True # Simulate file exists
    mock_state_dict = dummy_model.state_dict() # Get a valid state dict
    mock_torch_load.return_value = mock_state_dict

    # Mock the model's load_state_dict method directly
    with patch.object(dummy_model, 'load_state_dict') as mock_load_state:
        loaded = load_model_weights(dummy_model, test_path, device='cpu')

        mock_os_exists.assert_called_once_with(test_path)
        mock_torch_load.assert_called_once_with(test_path, map_location='cpu')
        mock_load_state.assert_called_once_with(mock_state_dict)
        assert loaded is True
        assert dummy_model.training is False # Should be set to eval mode

@patch('os.path.exists')
def test_load_model_weights_file_not_found(mock_os_exists, dummy_model, caplog):
    """Test model loading when file does not exist."""
    caplog.set_level(logging.WARNING)
    test_path = "non_existent_dir/model.pth"
    mock_os_exists.return_value = False # Simulate file does not exist

    loaded = load_model_weights(dummy_model, test_path, device='cpu')

    mock_os_exists.assert_called_once_with(test_path)
    assert loaded is False
    assert len(caplog.records) == 1
    assert "Model weights file not found" in caplog.text

@patch('torch.load')
@patch('os.path.exists')
def test_load_model_weights_error(mock_os_exists, mock_torch_load, dummy_model, caplog):
    """Test error handling during model loading."""
    caplog.set_level(logging.ERROR)
    test_path = "mock_dir/model.pth"
    mock_os_exists.return_value = True # Simulate file exists
    load_error = RuntimeError("Invalid file format")
    mock_torch_load.side_effect = load_error # Simulate error on load

    loaded = load_model_weights(dummy_model, test_path, device='cpu')

    mock_os_exists.assert_called_once_with(test_path)
    mock_torch_load.assert_called_once()
    assert loaded is False
    assert len(caplog.records) == 1
    assert "Failed to load model weights" in caplog.text
    assert "Invalid file format" in caplog.text

# TODO: Consider adding tests for run_price_prediction_flow by mocking its internal calls 