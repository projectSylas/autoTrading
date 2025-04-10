import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging
import os
from datetime import datetime

# Project specific imports
from src.config import settings
from src.utils.database import get_log_data # Assuming a function to get data from DB
from src.utils.common import get_historical_data # For yfinance data
from src.utils.notifier import send_slack_notification

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Data Preprocessing --- #
def prepare_sequence_data(df: pd.DataFrame, input_seq_len: int, output_seq_len: int, feature_col: str = 'Close') -> tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """Prepares time series data into sequences for LSTM/Transformer.

    Args:
        df (pd.DataFrame): DataFrame with a DatetimeIndex and feature column.
        input_seq_len (int): Length of the input sequence.
        output_seq_len (int): Length of the output sequence (prediction horizon).
        feature_col (str): The column to use as the feature/target.

    Returns:
        tuple[np.ndarray, np.ndarray, MinMaxScaler]: (X_sequences, y_sequences, scaler)
                                                      Returns empty arrays if data is insufficient.
    """
    if df is None or df.empty or feature_col not in df.columns:
        logging.warning("Cannot prepare sequence data: DataFrame is empty or feature column missing.")
        return np.array([]), np.array([]), None
    if len(df) < input_seq_len + output_seq_len:
        logging.warning(f"Cannot prepare sequence data: Insufficient data length ({len(df)}) for sequence lengths ({input_seq_len}+{output_seq_len}).")
        return np.array([]), np.array([]), None

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[feature_col].values.reshape(-1, 1))

    X, y = [], []
    for i in range(len(scaled_data) - input_seq_len - output_seq_len + 1):
        X.append(scaled_data[i:(i + input_seq_len), 0])
        y.append(scaled_data[(i + input_seq_len):(i + input_seq_len + output_seq_len), 0])

    X, y = np.array(X), np.array(y)
    # Reshape X for LSTM/Transformer [samples, time_steps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    logging.info(f"Sequence data prepared: X shape={X.shape}, y shape={y.shape}")
    return X, y, scaler

# --- Model Definitions --- #
class PricePredictorLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_prob):
        super(PricePredictorLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        # Use the output of the last time step
        out = self.fc(out[:, -1, :])
        return out

class PricePredictorTransformer(nn.Module):
    # Basic Transformer Encoder structure - can be made more sophisticated
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim, dropout_prob):
        super(PricePredictorTransformer, self).__init__()
        self.model_dim = model_dim
        self.embedding = nn.Linear(input_dim, model_dim)
        self.pos_encoder = nn.Parameter(torch.zeros(1, settings.PREDICTOR_INPUT_SEQ_LEN, model_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout_prob, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(model_dim, output_dim)

    def forward(self, src):
        src = self.embedding(src) * np.sqrt(self.model_dim)
        src = src + self.pos_encoder # Add positional encoding
        output = self.transformer_encoder(src)
        # Use the output corresponding to the last input token
        output = self.fc(output[:, -1, :])
        return output

# --- Training Function --- #
def train_model(model, dataloader, criterion, optimizer, device, epochs):
    """Trains the prediction model."""
    model.train()
    logging.info(f"Starting training for {epochs} epochs on device: {device}")
    for epoch in range(epochs):
        epoch_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        logging.info(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}')
        # Add validation logic here if needed

    logging.info("Training finished.")

# --- Prediction Function --- #
def predict_next(model, last_sequence, scaler, device, output_seq_len):
    """Predicts the next sequence using the trained model."""
    model.eval()
    with torch.no_grad():
        last_sequence_tensor = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0).to(device)
        # Add batch dimension and feature dimension if needed (depends on model input)
        if len(last_sequence_tensor.shape) == 2: # If [seq_len, features]
             last_sequence_tensor = last_sequence_tensor.unsqueeze(0) # Add batch dim -> [1, seq_len, features]

        predicted_scaled = model(last_sequence_tensor)
        predicted_scaled_np = predicted_scaled.cpu().numpy().flatten()

        # Inverse transform to get actual predicted prices
        # Reshape for scaler (needs 2D array)
        predicted = scaler.inverse_transform(predicted_scaled_np.reshape(-1, 1)).flatten()

    logging.info(f"Prediction for next {output_seq_len} steps: {predicted}")
    return predicted

# --- Load/Save Functions --- #
def save_model_weights(model, path):
    """Saves the model's state dictionary."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(model.state_dict(), path)
        logging.info(f"Model weights saved to {path}")
    except Exception as e:
        logging.error(f"Failed to save model weights to {path}: {e}")

def load_model_weights(model, path, device):
    """Loads the model's state dictionary."""
    try:
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=device))
            model.to(device)
            model.eval() # Set to evaluation mode
            logging.info(f"Model weights loaded from {path}")
            return True
        else:
            logging.warning(f"Model weights file not found at {path}. Model not loaded.")
            return False
    except Exception as e:
        logging.error(f"Failed to load model weights from {path}: {e}")
        return False

# --- Main Orchestration --- #
def run_price_prediction_flow(train: bool = False, predict: bool = True):
    """Runs the full flow: data loading, preprocessing, training (optional), prediction.

    Args:
        train (bool): Whether to train the model.
        predict (bool): Whether to make a prediction using the latest data.
    """
    logging.info("===== 📈 Price Prediction Flow Start =====")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Load Data (from DB or yfinance)
    # Example: Use yfinance for simplicity here
    try:
        # Adjust period based on sequence length and desired history
        df = get_historical_data(settings.PREDICTOR_SYMBOL, period="1y", interval="1h") # Example: 1 year of hourly data
    except Exception as e:
        logging.error(f"Failed to load data for {settings.PREDICTOR_SYMBOL}: {e}")
        return

    if df is None or df.empty:
        logging.error("Data loading failed. Exiting prediction flow.")
        return

    # 2. Prepare Data
    X, y, scaler = prepare_sequence_data(df, settings.PREDICTOR_INPUT_SEQ_LEN, settings.PREDICTOR_OUTPUT_SEQ_LEN)
    if X.shape[0] == 0 or scaler is None:
        logging.error("Data preparation failed. Exiting prediction flow.")
        return

    # 3. Initialize Model
    input_dim = 1 # Since we use only 'Close' price
    output_dim = settings.PREDICTOR_OUTPUT_SEQ_LEN

    if settings.PREDICTOR_MODEL_TYPE == "LSTM":
        model = PricePredictorLSTM(
            input_dim=input_dim,
            hidden_dim=settings.PREDICTOR_HIDDEN_DIM,
            num_layers=settings.PREDICTOR_NUM_LAYERS,
            output_dim=output_dim,
            dropout_prob=settings.PREDICTOR_DROPOUT
        ).to(device)
    elif settings.PREDICTOR_MODEL_TYPE == "TRANSFORMER":
         model = PricePredictorTransformer(
             input_dim=input_dim,
             model_dim=settings.PREDICTOR_HIDDEN_DIM, # model_dim often same as hidden_dim
             num_heads=4, # Example, make configurable
             num_layers=settings.PREDICTOR_NUM_LAYERS,
             output_dim=output_dim,
             dropout_prob=settings.PREDICTOR_DROPOUT
         ).to(device)
    else:
        logging.error(f"Unsupported model type: {settings.PREDICTOR_MODEL_TYPE}")
        return

    logging.info(f"Initialized {settings.PREDICTOR_MODEL_TYPE} model.")

    # 4. Train Model (if requested)
    if train:
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
        dataloader = DataLoader(dataset, batch_size=settings.PREDICTOR_BATCH_SIZE, shuffle=True)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=settings.PREDICTOR_LR)

        train_model(model, dataloader, criterion, optimizer, device, settings.PREDICTOR_EPOCHS)
        save_model_weights(model, settings.PREDICTOR_MODEL_PATH)
    else:
        # If not training, try to load existing weights
        loaded = load_model_weights(model, settings.PREDICTOR_MODEL_PATH, device)
        if not loaded:
            logging.warning("Exiting prediction flow as model weights could not be loaded and training was not requested.")
            return

    # 5. Predict Next Steps (if requested)
    if predict:
        # Get the last sequence from the original scaled data
        if len(scaler.transform(df[['Close']].values)) >= settings.PREDICTOR_INPUT_SEQ_LEN:
            last_sequence_scaled = scaler.transform(df[['Close']].values)[-settings.PREDICTOR_INPUT_SEQ_LEN:]
            last_sequence_reshaped = last_sequence_scaled.reshape(settings.PREDICTOR_INPUT_SEQ_LEN, 1)

            predicted_prices = predict_next(model, last_sequence_reshaped, scaler, device, settings.PREDICTOR_OUTPUT_SEQ_LEN)

            # Get the timestamp for the first prediction
            last_timestamp = df.index[-1]
            # Assuming hourly data, calculate next timestamps
            # This needs adjustment based on actual data frequency (interval)
            time_delta = pd.Timedelta(hours=1) # Adjust based on interval ('1h', '1d', etc.)
            prediction_timestamps = [last_timestamp + time_delta * (i+1) for i in range(settings.PREDICTOR_OUTPUT_SEQ_LEN)]

            # Log, Notify, Save to DB (Example)
            prediction_output = "\n".join([f"  - {ts}: ${price:.2f}" for ts, price in zip(prediction_timestamps, predicted_prices)])
            logging.info(f"Predicted prices for {settings.PREDICTOR_SYMBOL}:\n{prediction_output}")

            if send_slack_notification:
                send_slack_notification(
                    f"Price Prediction ({settings.PREDICTOR_SYMBOL})",
                    f"Predicted prices for the next {settings.PREDICTOR_OUTPUT_SEQ_LEN} periods:\n{prediction_output}"
                )
            # TODO: Add logic to save prediction results to the database if needed
            # try:
            #     from src.utils.database import log_prediction_to_db
            #     log_prediction_to_db(symbol=settings.PREDICTOR_SYMBOL, predictions=dict(zip(prediction_timestamps, predicted_prices)))
            # except ImportError: pass
            # except Exception as db_err: logging.error(f"Failed to log prediction to DB: {db_err}")

        else:
            logging.warning("Not enough data to form the last sequence for prediction.")

    logging.info("===== 📈 Price Prediction Flow End =====")

# --- Example Execution --- #
if __name__ == "__main__":
    # Example: Train the model and then predict
    # run_price_prediction_flow(train=True, predict=True)

    # Example: Only predict using existing trained model
    run_price_prediction_flow(train=False, predict=True) 