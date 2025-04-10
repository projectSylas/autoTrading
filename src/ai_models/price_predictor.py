import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging
import os
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any

# HF Imports
# 필요한 모델 클래스로 변경될 수 있음 (e.g., TimesformerForPrediction, TimeSeriesTransformerForPrediction)
from transformers import AutoModelForPrediction, AutoConfig, AutoProcessor
# 예시: Timeseries 모델용 프로세서가 필요할 수 있음
# from transformers import AutoProcessor

# Project specific imports
from src.config import settings
from src.utils.database import get_log_data, log_prediction_to_db # Assuming DB logging function exists
from src.utils.common import get_historical_data # For yfinance data
from src.utils.notifier import send_slack_notification

logging.basicConfig(level=settings.LOG_LEVEL.upper(), format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

# --- Global Variables for HF Model --- #
hf_timeseries_model = None
hf_timeseries_config = None
hf_processor = None # 프로세서 추가 (선택적, 모델에 따라 필요)
hf_model_loaded = False

# --- HF Model Utility Functions --- #
def _get_hf_device() -> torch.device:
    """Determine the torch device for HF models based on settings."""
    if settings.HF_DEVICE == 'auto':
        # MPS (Apple Silicon GPU) 지원 확인 추가
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    return torch.device(settings.HF_DEVICE)

def initialize_hf_timeseries_model(force_reload: bool = False):
    """Initializes the Hugging Face time series model and processor based on settings."""
    global hf_timeseries_model, hf_timeseries_config, hf_processor, hf_model_loaded

    if not settings.ENABLE_HF_TIMESERIES:
        logging.info("Hugging Face time series prediction is disabled in settings.")
        hf_model_loaded = False
        return

    if hf_model_loaded and not force_reload:
        logging.info("HF Model already loaded.")
        return

    try:
        device = _get_hf_device()
        model_name = settings.HF_TIMESERIES_MODEL_NAME
        cache_dir = settings.MODEL_WEIGHTS_DIR
        logging.info(f"Initializing Hugging Face time series model: {model_name} on device: {device}")

        # 일부 모델은 processor가 필요할 수 있음 (데이터 전처리 담당)
        try:
            hf_processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir)
            logging.info(f"Loaded HF Processor for {model_name}")
        except Exception:
            hf_processor = None
            logging.info(f"No specific HF Processor found for {model_name}, proceeding without it.")

        # Load config and model
        hf_timeseries_config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)

        # 설정에서 prediction_length 가져오기 시도, 없으면 기본값 사용
        prediction_length = getattr(settings, 'HF_PREDICTION_LENGTH', getattr(hf_timeseries_config, 'prediction_length', 12)) # 예: 12 스텝 예측
        # 설정 파일에 따라 config 업데이트 (모델에 따라 필요한 설정이 다름)
        hf_timeseries_config.prediction_length = prediction_length
        logging.info(f"Set prediction_length to: {prediction_length}")

        # 모델 로드 (필요 시 AutoModelForPrediction을 구체적인 클래스로 변경)
        hf_timeseries_model = AutoModelForPrediction.from_pretrained(
            model_name,
            config=hf_timeseries_config,
            cache_dir=cache_dir
        )
        hf_timeseries_model.to(device)
        hf_timeseries_model.eval() # 평가 모드로 설정
        hf_model_loaded = True
        logging.info("✅ Hugging Face time series model loaded successfully.")

    except Exception as e:
        logging.error(f"❌ Failed to load Hugging Face time series model '{model_name}': {e}. Check model name, paths, dependencies, and internet connection.")
        hf_timeseries_model = None
        hf_timeseries_config = None
        hf_processor = None
        hf_model_loaded = False
        # Optionally notify via Slack
        # send_slack_notification("HF Model Load Failed", f"Model: {model_name}\nError: {e}")


def prepare_data_for_hf_model(df: pd.DataFrame, feature_col: str = 'Close') -> Optional[Dict[str, torch.Tensor]]:
    """Prepares the LAST sequence from the DataFrame for HF model prediction.

    Args:
        df (pd.DataFrame): Input DataFrame with a DatetimeIndex and feature columns.
        feature_col (str): The target column to use for prediction.

    Returns:
        Optional[Dict[str, torch.Tensor]]: Dictionary containing input tensors for the model,
                                            e.g., {"past_values": tensor}. Returns None on failure.
    """
    if not hf_model_loaded or hf_timeseries_config is None:
        logging.warning("HF Model not loaded, cannot prepare data.")
        return None
    if df is None or df.empty or feature_col not in df.columns:
        logging.warning(f"Input DataFrame is invalid or missing '{feature_col}' column.")
        return None

    # 모델 설정에서 context_length 가져오기
    # context_length = hf_timeseries_config.context_length # 대부분의 모델이 이 속성을 가짐
    context_length = getattr(settings, 'HF_CONTEXT_LENGTH', getattr(hf_timeseries_config, 'context_length', 128)) # 기본값 128

    if len(df) < context_length:
        logging.warning(f"Insufficient data length ({len(df)}) for HF model context length ({context_length}).")
        return None

    # 마지막 context_length 만큼의 데이터를 사용
    past_data = df[feature_col].values[-context_length:].astype(np.float32)

    # TODO: 모델에 따라 추가적인 전처리(스케일링, 시간 특징 추가 등)가 필요할 수 있음
    # 예: scaler = MinMaxScaler(); scaled_past_data = scaler.fit_transform(past_data.reshape(-1, 1)).flatten()

    # Processor가 있다면 Processor를 사용하여 인풋 생성
    if hf_processor:
        try:
            # Processor는 모델에 맞는 다양한 입력을 생성해줄 수 있음 (예: past_values, time_features 등)
            # 입력 형식이 모델마다 다르므로 확인 필요. 여기서는 간단히 리스트로 전달.
             model_inputs = hf_processor(
                 [list(past_data)], # 단일 시퀀스를 리스트 안에 넣어야 할 수 있음
                 return_tensors="pt"
            )
             logging.info(f"Data prepared using HF Processor. Input keys: {list(model_inputs.keys())}")
        except Exception as e:
             logging.error(f"Error using HF Processor for data preparation: {e}", exc_info=True)
             # Processor 실패 시 수동 방식 시도 (Fallback)
             logging.info("Falling back to manual tensor creation for 'past_values'.")
             past_values_tensor = torch.tensor(past_data).unsqueeze(0) # Add batch dimension
             model_inputs = {"past_values": past_values_tensor}
    else:
        # Processor가 없다면 수동으로 'past_values' 텐서 생성 (가장 일반적인 입력)
        past_values_tensor = torch.tensor(past_data).unsqueeze(0) # Add batch dimension [1, context_length]
        model_inputs = {"past_values": past_values_tensor}
        logging.info("Data prepared manually (created 'past_values' tensor).")

    return model_inputs


def predict_with_hf_model(model_inputs: Dict[str, torch.Tensor], df_index: pd.DatetimeIndex) -> Optional[pd.DataFrame]:
    """Performs prediction using the loaded HF time series model.

    Args:
        model_inputs (Dict[str, torch.Tensor]): Dictionary of input tensors from prepare_data_for_hf_model.
        df_index (pd.DatetimeIndex): The DatetimeIndex of the original input DataFrame (used for timestamp generation).

    Returns:
        Optional[pd.DataFrame]: DataFrame with 'timestamp' and 'predicted_price' columns,
                                 or None if prediction fails.
    """
    if not hf_model_loaded or hf_timeseries_model is None or hf_timeseries_config is None:
        logging.warning("HF Model not loaded, cannot predict.")
        return None
    if not model_inputs:
        logging.warning("Received empty model inputs for prediction.")
        return None
    if df_index is None or len(df_index) == 0:
         logging.warning("Received invalid DataFrame index for timestamp generation.")
         return None

    device = _get_hf_device()
    prediction_df = None

    try:
        with torch.no_grad():
            # 입력 텐서를 지정된 디바이스로 이동
            inputs_on_device = {k: v.to(device) for k, v in model_inputs.items()}
            logging.info(f"Performing HF prediction with input keys: {list(inputs_on_device.keys())} on device: {device}")

            # 모델 예측 수행 (.generate() 사용이 일반적)
            # generate 함수의 파라미터는 모델에 따라 다를 수 있음 (예: num_return_sequences)
            outputs = hf_timeseries_model.generate(
                **inputs_on_device
                # , max_length=hf_timeseries_config.context_length + hf_timeseries_config.prediction_length # 필요시 명시
            )

            # 출력 형태 확인 및 예측값 추출 (모델마다 다름)
            # 예: outputs 텐서의 shape 확인 후 슬라이싱
            # logging.debug(f"HF model output shape: {outputs.sequences.shape}") # 예시
            # 일반적으로 batch 차원과 sequence 차원을 가짐 [batch_size, sequence_length]
            # prediction_length 만큼의 예측값은 보통 시퀀스의 마지막 부분에 위치
            predicted_values_tensor = outputs.sequences[:, -hf_timeseries_config.prediction_length:]
            predicted_values = predicted_values_tensor[0].cpu().numpy() # 첫 번째 배치 결과, numpy 배열로 변환

            logging.info(f"Raw predicted values shape: {predicted_values.shape}")
            # TODO: 역변환(Inverse Transform) 필요 시 여기에 로직 추가
            # 예: if scaler is available: predicted_values = scaler.inverse_transform(predicted_values.reshape(-1, 1)).flatten()

            # --- 미래 타임스탬프 생성 --- 
            last_timestamp = df_index[-1]
            # 데이터프레임 인덱스에서 빈도(frequency) 추론 시도
            freq = pd.infer_freq(df_index)
            if freq is None:
                 # 빈도 추론 실패 시, 마지막 두 타임스탬프 간의 차이 사용
                 if len(df_index) >= 2:
                     time_delta = df_index[-1] - df_index[-2]
                     logging.warning(f"Could not infer frequency from index. Using time delta: {time_delta}")
                 else:
                     logging.error("Cannot determine time frequency for future timestamps.")
                     return None # 빈도 결정 불가 시 예측 중단
                 # freq를 Timedelta로 설정 (DateOffset 객체 사용 위함)
                 freq = time_delta

            # DateOffset 객체 생성
            try:
                 # 문자열 빈도(예: '1H', 'D')인 경우
                 date_offset = pd.tseries.frequencies.to_offset(freq)
            except ValueError:
                 # Timedelta 객체인 경우 (추론 실패 시)
                 date_offset = freq

            future_timestamps = [last_timestamp + date_offset * (i + 1) for i in range(hf_timeseries_config.prediction_length)]

            # 예측 결과 데이터프레임 생성
            prediction_df = pd.DataFrame({
                'timestamp': future_timestamps,
                'predicted_price': predicted_values.flatten() # 예측값 flatten
            })
            prediction_df.set_index('timestamp', inplace=True)

            logging.info("✅ HF model prediction successful.")
            logging.debug(f"Prediction DF:\n{prediction_df.head()}")

    except Exception as e:
        logging.error(f"❌ Error during HF model prediction: {e}", exc_info=True)
        return None

    return prediction_df

# --- Data Preprocessing (기존 LSTM/Transformer용) --- #
def prepare_sequence_data(df: pd.DataFrame, input_seq_len: int, output_seq_len: int, feature_col: str = 'Close') -> tuple[np.ndarray, np.ndarray, Optional[MinMaxScaler]]:
    """Prepares time series data into sequences for LSTM/Transformer."""
    if df is None or df.empty or feature_col not in df.columns:
        logging.warning("Cannot prepare sequence data: DataFrame is empty or feature column missing.")
        return np.array([]), np.array([]), None
    if len(df) < input_seq_len + output_seq_len:
        logging.warning(f"Cannot prepare sequence data: Insufficient data length ({len(df)}) for sequence lengths ({input_seq_len}+{output_seq_len}).")
        return np.array([]), np.array([]), None

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Use try-except for fit_transform as it might fail with all NaNs etc.
    try:
         # Ensure we only scale the target feature column
         scaled_data = scaler.fit_transform(df[[feature_col]].dropna().values) # Handle NaNs before scaling
         if scaled_data.size == 0:
             logging.warning("No valid data left after dropping NaNs for scaling.")
             return np.array([]), np.array([]), None
    except ValueError as e:
         logging.error(f"Error scaling data for {feature_col}: {e}. Check data for NaNs or constant values.")
         return np.array([]), np.array([]), None

    # Re-index the scaled data to match the original non-NaN index if needed, or work with the contiguous scaled array
    # For simplicity, assuming scaled_data aligns with a contiguous block usable for sequence creation
    X, y = [], []
    for i in range(len(scaled_data) - input_seq_len - output_seq_len + 1):
        X.append(scaled_data[i:(i + input_seq_len), 0])
        y.append(scaled_data[(i + input_seq_len):(i + input_seq_len + output_seq_len), 0])

    if not X or not y:
         logging.warning("No sequences generated after iterating through scaled data.")
         return np.array([]), np.array([]), scaler # Return scaler even if no sequences

    X, y = np.array(X), np.array(y)
    # Reshape X for LSTM/Transformer [samples, time_steps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    logging.info(f"Sequence data prepared: X shape={X.shape}, y shape={y.shape}")
    return X, y, scaler

# --- Model Definitions (기존) --- #
class PricePredictorLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_prob):
        super(PricePredictorLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Ensure input is 3D [batch, seq_len, features]
        if x.dim() == 2:
            x = x.unsqueeze(0) # Add batch dimension if missing
        out, _ = self.lstm(x)
        # Use output of the last time step
        out = self.fc(out[:, -1, :])
        return out

class PricePredictorTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim, dropout_prob, seq_len):
        super(PricePredictorTransformer, self).__init__()
        self.model_dim = model_dim
        # Input embedding layer
        self.embedding = nn.Linear(input_dim, model_dim)
        # Positional Encoding
        self.pos_encoder = nn.Parameter(torch.zeros(1, seq_len, model_dim))
        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout_prob, batch_first=True)
        # Transformer Encoder
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Final fully connected layer
        self.fc = nn.Linear(model_dim, output_dim)

        self._init_weights()

    def _init_weights(self):
        # Initialize positional encoder (optional but common)
        nn.init.uniform_(self.pos_encoder, -0.1, 0.1)

    def forward(self, src):
        # src shape: [batch_size, seq_len, input_dim]
        if src.size(1) != self.pos_encoder.size(1):
             # Dynamic adjustment or padding might be needed for variable lengths
             # For fixed length: raise error
             raise ValueError(f"Input sequence length {src.size(1)} does not match positional encoder length {self.pos_encoder.size(1)}")

        # Apply input embedding
        src = self.embedding(src) * np.sqrt(self.model_dim)
        # Add positional encoding
        src = src + self.pos_encoder
        # Pass through transformer encoder
        output = self.transformer_encoder(src)
        # Use the output corresponding to the last input token for prediction
        output = self.fc(output[:, -1, :])
        return output

# --- Training Function (기존) --- #
def train_model(model, dataloader, criterion, optimizer, device, epochs):
    """Trains the prediction model."""
    model.train()
    logging.info(f"Starting training for {epochs} epochs on device: {device}")
    for epoch in range(epochs):
        epoch_loss = 0.0
        processed_batches = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            # Ensure target shape matches output shape [batch_size, output_dim]
            if targets.dim() == 3 and targets.shape[-1] == 1: # e.g., [batch, seq, 1]
                targets = targets.squeeze(-1) # -> [batch, seq]
                if outputs.shape[1] == 1 and targets.shape[1] > 1: # output [batch, 1], target [batch, seq] -> use last target? Needs clarification
                     targets = targets[:, -1].unsqueeze(1) # Adjust target shape or model output logic
            elif targets.dim() == 2 and outputs.dim() == 2 and targets.shape[1] != outputs.shape[1]:
                 # Mismatch in output dimension, e.g., target [batch, out_seq], output [batch, 1]
                 # Adjust based on model's purpose. If predicting only next step from sequence:
                 if outputs.shape[1] == 1:
                      targets = targets[:, 0].unsqueeze(1) # Compare with the first step of target sequence
                 # Or if model should predict the whole sequence, ensure output_dim matches target seq len

            try:
                 loss = criterion(outputs, targets)
                 loss.backward()
                 optimizer.step()
                 epoch_loss += loss.item()
                 processed_batches += 1
            except RuntimeError as e:
                 logging.error(f"RuntimeError during loss calculation or backward pass: {e}")
                 logging.error(f"Output shape: {outputs.shape}, Target shape: {targets.shape}")
                 # Skip this batch or handle differently
                 continue

        if processed_batches > 0:
             avg_loss = epoch_loss / processed_batches
             logging.info(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}')
        else:
             logging.warning(f"Epoch [{epoch+1}/{epochs}] completed without processing any batches.")

    logging.info("Training finished.")

# --- Prediction Function (기존 LSTM/Transformer용) --- #
def predict_next(model, last_sequence, scaler, device, output_seq_len):
    """Predicts the next sequence using the trained LSTM/Transformer model."""
    if scaler is None:
        logging.error("Scaler is None, cannot perform prediction.")
        return None
    if last_sequence is None or last_sequence.size == 0:
        logging.error("Invalid last_sequence provided for prediction.")
        return None

    model.eval()
    with torch.no_grad():
        # Ensure last_sequence is a tensor and on the correct device
        if not isinstance(last_sequence, torch.Tensor):
             last_sequence_tensor = torch.tensor(last_sequence, dtype=torch.float32).to(device)
        else:
             last_sequence_tensor = last_sequence.to(device)

        # Ensure input has the correct dimensions [batch, seq_len, features]
        if last_sequence_tensor.dim() == 2: # e.g., [seq_len, features]
             last_sequence_tensor = last_sequence_tensor.unsqueeze(0) # -> [1, seq_len, features]
        elif last_sequence_tensor.dim() == 1: # e.g., [seq_len] -> needs feature dim
            last_sequence_tensor = last_sequence_tensor.unsqueeze(0).unsqueeze(-1) # -> [1, seq_len, 1]
        elif last_sequence_tensor.dim() != 3:
            logging.error(f"Unexpected input tensor dimension: {last_sequence_tensor.dim()}. Expected 3D [batch, seq, feature].")
            return None

        try:
             predicted_scaled = model(last_sequence_tensor)
        except Exception as e:
             logging.error(f"Error during model inference: {e}", exc_info=True)
             logging.error(f"Model type: {type(model)}, Input shape: {last_sequence_tensor.shape}")
             return None

        # Ensure predicted_scaled is on CPU and numpy for scaler
        predicted_scaled_np = predicted_scaled.cpu().numpy().flatten()

        # Inverse transform requires shape (n_samples, n_features)
        try:
            predicted = scaler.inverse_transform(predicted_scaled_np.reshape(-1, 1)).flatten()
        except ValueError as e:
             logging.error(f"Error during inverse transform: {e}. Scaler likely expected different shape or data range.")
             logging.error(f"Predicted scaled numpy shape: {predicted_scaled_np.shape}")
             return None # Return None if inverse transform fails

    logging.info(f"Prediction (LSTM/TF) for next {output_seq_len} steps (shape: {predicted.shape}): {predicted[:5]}...") # Log first 5
    return predicted

# --- Load/Save Functions (기존) --- #
def save_model_weights(model, scaler, path):
    """Saves the model's state dictionary and the scaler."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_dict = {
            'model_state_dict': model.state_dict(),
            'scaler': scaler
        }
        torch.save(save_dict, path)
        logging.info(f"Model weights and scaler saved to {path}")
    except Exception as e:
        logging.error(f"Error saving model weights and scaler to {path}: {e}")

def load_model_weights(model, path, device) -> Optional[MinMaxScaler]:
    """Loads model weights and the scaler from a file.
    
    Returns:
        Optional[MinMaxScaler]: The loaded scaler object, or None if loading fails.
    """
    try:
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            scaler = checkpoint['scaler']
            model.eval() # Set to evaluation mode after loading
            logging.info(f"Model weights and scaler loaded from {path}")
            return scaler # Return the loaded scaler
        else:
            logging.warning(f"Model weights file not found at {path}. Model not loaded.")
            return None # Return None if file not found
    except KeyError as ke:
         logging.error(f"Error loading model/scaler from {path}: Missing key {ke}. File might be corrupted or saved in old format.")
         return None
    except Exception as e:
        logging.error(f"Error loading model weights and scaler from {path}: {e}")
        return None

# --- Main Prediction Flow (UPDATED) --- #
def run_price_prediction_flow(
    train: bool = False,
    predict: bool = True,
    symbol: str = settings.CHALLENGE_SYMBOL, # strategy.py에서 symbol 전달받음
    interval: str = settings.CHALLENGE_INTERVAL # strategy.py에서 interval 전달받음
    ) -> Optional[Tuple[pd.DataFrame, Optional[pd.DataFrame]]]:
    """Runs the full price prediction flow: data fetch, train (optional), predict.

    Args:
        train (bool): Whether to train the primary (LSTM/Transformer) model.
        predict (bool): Whether to perform prediction using both models.
        symbol (str): The trading symbol (e.g., 'BTCUSDT').
        interval (str): The time interval for data (e.g., '1h', '4h').

    Returns:
        Optional[Tuple[pd.DataFrame, Optional[pd.DataFrame]]]:
            A tuple containing:
            - predictions_primary (pd.DataFrame): Predictions from the primary model (LSTM/Transformer). Empty if predict=False or failed.
            - predictions_hf (Optional[pd.DataFrame]): Predictions from the Hugging Face model. None if disabled, predict=False, or failed.
            Returns None if essential data loading fails.
    """
    logging.info(f"--- Starting Price Prediction Flow for {symbol} ({interval}) ---")
    logging.info(f"Train Primary: {train}, Predict: {predict}, Use HF: {settings.ENABLE_HF_TIMESERIES}")

    # Initialize HF model if enabled (safe to call multiple times)
    if settings.ENABLE_HF_TIMESERIES:
        initialize_hf_timeseries_model()

    # --- 1. 데이터 로드 --- 
    # DB 또는 yfinance 등에서 데이터 로드
    # 필요한 데이터 양은 모델의 input_seq_len/context_length + 학습 기간 고려
    lookback_days = getattr(settings, 'PREDICTOR_DATA_LOOKBACK_DAYS', 90) # 예: 90일치 데이터 사용
    try:
        # get_historical_data should return df with DatetimeIndex
        df = get_historical_data(symbol, interval, days_back=lookback_days)
        if df is None or df.empty:
            logging.error(f"Failed to load historical data for {symbol}. Prediction flow cannot continue.")
            return None
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
             logging.warning("Converting DataFrame index to DatetimeIndex.")
             df.index = pd.to_datetime(df.index)
        # Sort by date just in case
        df.sort_index(inplace=True)
        logging.info(f"Loaded {len(df)} data points for {symbol} from {df.index.min()} to {df.index.max()}")
    except Exception as e:
        logging.error(f"Error getting historical data for {symbol}: {e}", exc_info=True)
        return None

    # --- 2. 기본 모델 (LSTM/Transformer) 학습 (선택 사항) --- 
    primary_model_type = getattr(settings, 'PREDICTOR_MODEL_TYPE', 'LSTM').upper()
    input_seq_len = getattr(settings, 'PREDICTOR_INPUT_SEQ_LEN', 60)
    output_seq_len = getattr(settings, 'PREDICTOR_OUTPUT_SEQ_LEN', 12)
    feature_col = getattr(settings, 'PREDICTOR_FEATURE_COL', 'Close')
    model_path = os.path.join(settings.MODEL_WEIGHTS_DIR, f"{primary_model_type.lower()}_predictor_{symbol}_{interval}.pt")
    device = _get_hf_device() # Use the same device logic

    # 모델 정의 (기존 로직 사용)
    model = None
    if primary_model_type == 'LSTM':
        model = PricePredictorLSTM(
            input_dim=1, # 단일 feature 가정
            hidden_dim=settings.LSTM_HIDDEN_DIM,
            num_layers=settings.LSTM_NUM_LAYERS,
            output_dim=output_seq_len,
            dropout_prob=settings.LSTM_DROPOUT
        )
    elif primary_model_type == 'TRANSFORMER':
        model = PricePredictorTransformer(
            input_dim=1, model_dim=settings.TRANSFORMER_MODEL_DIM, num_heads=settings.TRANSFORMER_NUM_HEADS,
            num_layers=settings.TRANSFORMER_NUM_LAYERS, output_dim=output_seq_len,
            dropout_prob=settings.TRANSFORMER_DROPOUT, seq_len=input_seq_len
        )
    else:
        logging.warning(f"Unsupported primary model type: {primary_model_type}. Skipping primary model.")

    scaler_primary = None # Scaler should be accessible for prediction
    primary_model_available = False
    if model:
        if train:
            logging.info(f"Preparing data for {primary_model_type} training...")
            X_train, y_train, scaler_train = prepare_sequence_data(df, input_seq_len, output_seq_len, feature_col)
            scaler_primary = scaler_train # Store the scaler used for training

            if X_train.size > 0 and y_train.size > 0 and scaler_primary is not None:
                train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
                train_dataloader = DataLoader(train_dataset, batch_size=settings.PREDICTOR_BATCH_SIZE, shuffle=True)

                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=settings.PREDICTOR_LEARNING_RATE)
                model.to(device)
                train_model(model, train_dataloader, criterion, optimizer, device, settings.PREDICTOR_EPOCHS)

                # 학습된 모델과 scaler 저장
                save_model_weights(model, scaler_primary, model_path)
                primary_model_available = True
            else:
                logging.warning("Training skipped due to insufficient data or sequence preparation failure.")
        else:
            # 학습 안 할 경우, 기존 가중치와 scaler 로드 시도
            loaded_scaler = load_model_weights(model, model_path, device)
            if loaded_scaler:
                 model.to(device) # 로드 후에도 디바이스 지정 필요
                 scaler_primary = loaded_scaler # 로드된 scaler 사용
                 primary_model_available = True
                 if scaler_primary is None:
                      logging.warning("Scaler loaded as None. Prediction might fail.")
                      primary_model_available = False # Scaler 없으면 예측 불가
            else:
                 # model = None # Failed to load weights, disable primary model -> model is already defined
                 logging.warning(f"Primary model {primary_model_type} cannot be used as weights/scaler could not be loaded from {model_path}")
                 primary_model_available = False


    # --- 3. 예측 수행 --- 
    predictions_primary = pd.DataFrame() # 기본 모델 예측 결과
    predictions_hf = None # HF 모델 예측 결과 (Optional)

    if predict:
        # --- 3.a 기본 모델 예측 --- 
        if primary_model_available and scaler_primary:
            logging.info(f"Predicting with {primary_model_type} model...")
            # 예측을 위한 마지막 시퀀스 준비
            if len(df) >= input_seq_len:
                 try:
                     # Use the scaler obtained during training/loading
                     last_sequence_raw = df[[feature_col]].values[-input_seq_len:]
                     # Handle potential NaNs before scaling
                     if np.isnan(last_sequence_raw).any():
                          logging.warning(f"NaNs found in last sequence for {primary_model_type} prediction. Attempting prediction anyway, but results may be unreliable.")
                          # Option: Fill NaNs (e.g., ffill) or return empty prediction
                          # last_sequence_raw = pd.DataFrame(last_sequence_raw).ffill().values

                     # Check if scaler is fitted (should be if loaded correctly)
                     if not hasattr(scaler_primary, 'mean_') and not hasattr(scaler_primary, 'scale_'):
                          logging.error("Scaler for primary model is not fitted, even after loading/training. Cannot predict.")
                          # primary_model_available = False # Mark as unavailable? Or try fitting?
                     else:
                         last_sequence_scaled = scaler_primary.transform(last_sequence_raw)
                         # Ensure input is 3D: [1, seq_len, 1]
                         last_sequence_reshaped = np.reshape(last_sequence_scaled, (1, input_seq_len, 1))

                         predicted_values = predict_next(model, last_sequence_reshaped, scaler_primary, device, output_seq_len)

                         if predicted_values is not None and predicted_values.size > 0:
                             # 예측 결과 타임스탬프 생성
                             last_timestamp = df.index[-1]
                             freq = pd.infer_freq(df.index)
                             if freq is None and len(df.index) >= 2:
                                 freq = df.index[-1] - df.index[-2]

                             if freq:
                                 try: date_offset = pd.tseries.frequencies.to_offset(freq)
                                 except ValueError: date_offset = freq
                                 future_timestamps = [last_timestamp + date_offset * (i + 1) for i in range(len(predicted_values))] # Use actual length of predicted_values
                                 predictions_primary = pd.DataFrame({'timestamp': future_timestamps, 'predicted_price': predicted_values})
                                 predictions_primary.set_index('timestamp', inplace=True)
                                 logging.info(f"✅ {primary_model_type} prediction successful.")
                                 logging.debug(f"Primary Prediction DF ({len(predictions_primary)} steps):\n{predictions_primary.head()}")
                                 # DB 로깅
                                 if log_prediction_to_db:
                                     try: log_prediction_to_db(predictions_primary, model_name=f"{primary_model_type}_{symbol}_{interval}", type="primary")
                                     except Exception as db_err: logging.error(f"DB logging failed for primary prediction: {db_err}")
                             else:
                                  logging.warning("Could not determine frequency for primary prediction timestamps.")
                         else:
                              logging.warning(f"{primary_model_type} prediction function returned None or empty array.")
                 except ValueError as ve:
                      # Catch scaler transform errors specifically
                      logging.error(f"ValueError during primary model prediction (likely scaler issue): {ve}", exc_info=True)
                 except Exception as e:
                      logging.error(f"Error during {primary_model_type} prediction: {e}", exc_info=True)
            else:
                 logging.warning(f"{primary_model_type} prediction skipped: Insufficient data length ({len(df)} < {input_seq_len}).")
        elif predict and primary_model_type != 'NONE': # Only warn if primary model was intended but unavailable
             logging.warning(f"Primary model ({primary_model_type}) prediction skipped: Model or scaler is not available.")
        # elif not model:
        #      logging.info("Primary model prediction skipped: Model is not available (not selected or failed to load).")


        # --- 3.b Hugging Face 모델 예측 --- 
        if settings.ENABLE_HF_TIMESERIES and hf_model_loaded:
            logging.info("Preparing data and predicting with Hugging Face model...")
            # 데이터 준비 (전체 df 전달)
            hf_inputs = prepare_data_for_hf_model(df, feature_col=feature_col)

            if hf_inputs:
                # 예측 수행 (준비된 입력과 원본 df의 index 전달)
                predictions_hf = predict_with_hf_model(hf_inputs, df.index)

                if predictions_hf is not None and not predictions_hf.empty:
                     logging.info(f"✅ Hugging Face prediction successful ({len(predictions_hf)} steps)." )
                     logging.debug(f"HF Prediction DF:
{predictions_hf.head()}")
                     # DB 로깅
                     if log_prediction_to_db:
                         try: log_prediction_to_db(predictions_hf, model_name=f"{settings.HF_TIMESERIES_MODEL_NAME}_{symbol}_{interval}", type="hf")
                         except Exception as db_err: logging.error(f"DB logging failed for HF prediction: {db_err}")
                else:
                     logging.warning("Hugging Face prediction returned None or empty DataFrame.")
            else:
                logging.warning("Hugging Face prediction skipped: Data preparation failed.")
        elif settings.ENABLE_HF_TIMESERIES:
             logging.warning("Hugging Face prediction skipped: Model not loaded.")
        else:
            logging.info("Hugging Face prediction skipped: Disabled in settings.")


    logging.info(f"--- Finished Price Prediction Flow for {symbol} ({interval}) ---")
    if predict:
        return predictions_primary, predictions_hf
    else:
        return None

# --- Main Execution Guard --- #
if __name__ == "__main__":
    # Setup logging based on settings
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

    logging.info("Running price_predictor.py directly for testing.")

    # --- Test HF Model Initialization ---
    if settings.ENABLE_HF_TIMESERIES:
        print("\n--- Testing HF Model Initialization ---")
        initialize_hf_timeseries_model(force_reload=True)
        print(f"HF Model Loaded: {hf_model_loaded}")
        if hf_model_loaded:
             print(f"Model Class: {type(hf_timeseries_model)}")
             print(f"Config Keys: {list(hf_timeseries_config.to_dict().keys())[:5]}...") # Print first 5 keys
             # print(f"Processor available: {hf_processor is not None}")

    # --- Test Prediction Flow --- 
    print("\n--- Testing Prediction Flow (Predict Only) ---")
    # Use settings for symbol and interval for testing
    test_symbol = settings.CHALLENGE_SYMBOL
    test_interval = settings.CHALLENGE_INTERVAL
    try:
        # Set predict=True, train=False for testing prediction part
        primary_preds, hf_preds = run_price_prediction_flow(train=False, predict=True, symbol=test_symbol, interval=test_interval)

        print("\n--- Prediction Results ---")
        if primary_preds is not None and not primary_preds.empty:
            print(f"Primary Model ({getattr(settings, 'PREDICTOR_MODEL_TYPE', 'LSTM')}) Predictions:")
            print(primary_preds.head())
        else:
            print("Primary Model Prediction: Failed or Empty")

        if hf_preds is not None and not hf_preds.empty:
            print(f"\nHugging Face Model ({settings.HF_TIMESERIES_MODEL_NAME}) Predictions:")
            print(hf_preds.head())
        elif settings.ENABLE_HF_TIMESERIES:
            print("\nHugging Face Model Prediction: Failed or Empty")
        else:
            print("\nHugging Face Model Prediction: Disabled")

    except Exception as e:
        print(f"\n--- ERROR during test execution ---")
        logging.error(f"Error in main execution block: {e}", exc_info=True) 