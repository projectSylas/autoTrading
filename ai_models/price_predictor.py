from src.utils.database import get_log_data, log_prediction_to_db # Assume log_prediction_to_db exists
import pandas as pd
import logging
from datetime import datetime
import torch
import numpy as np
import os

# Project specific imports (assuming they exist and are correct)
from src.config import settings
from src.utils.common import get_historical_data # Assuming other necessary common utils are imported
from .data_preprocessing import prepare_sequence_data # Assuming this is defined locally or imported
from .models import PricePredictorLSTM, PricePredictorTransformer # Assuming models are defined locally or imported

# --- Database Import for Logging ---
try:
    from src.utils.database import log_prediction_to_db
    prediction_db_available = True
except ImportError:
    logging.warning("`log_prediction_to_db` not found in `utils.database`. Prediction DB logging disabled.")
    log_prediction_to_db = None
    prediction_db_available = False

# --- Slack Import ---
try:
    from src.utils.notifier import send_slack_notification
    slack_available = True
except ImportError:
     logging.warning("`send_slack_notification` not found. Slack notifications disabled.")
     send_slack_notification = None
     slack_available = False

# --- Utility functions for train/predict/save/load (assuming defined) ---
# def train_model(...):
# def predict_next(...):
# def save_model_weights(...):
# def load_model_weights(...):

# --- Main Orchestration --- #
def run_price_prediction_flow(train: bool = False, predict: bool = True) -> None:
    """Run the price prediction flow."""
    logging.info("===== Price Prediction Flow Start =====")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Load Data
    try:
        # Make sure settings has DATA_INTERVAL defined
        data_interval = getattr(settings, 'DATA_INTERVAL', '1h') # Default to '1h' if not set
        df = get_historical_data(settings.PREDICTOR_SYMBOL, period="1y", interval=data_interval)
        if df is None or df.empty:
            logging.error(f"Failed to load data for {settings.PREDICTOR_SYMBOL}. Exiting.")
            return
    except Exception as e:
        logging.error(f"Failed to load data for {settings.PREDICTOR_SYMBOL}: {e}")
        return

    # 2. Prepare Data
    X, y, scaler = prepare_sequence_data(df, settings.PREDICTOR_INPUT_SEQ_LEN, settings.PREDICTOR_OUTPUT_SEQ_LEN)
    if scaler is None or X.shape[0] == 0:
        logging.error("Data preparation failed. Exiting prediction flow.")
        return

    # 3. Initialize Model
    input_dim = 1
    output_dim = settings.PREDICTOR_OUTPUT_SEQ_LEN
    model = None
    try:
        if settings.PREDICTOR_MODEL_TYPE == "LSTM":
            model = PricePredictorLSTM(input_dim, settings.PREDICTOR_HIDDEN_DIM, settings.PREDICTOR_NUM_LAYERS, output_dim, settings.PREDICTOR_DROPOUT).to(device)
        elif settings.PREDICTOR_MODEL_TYPE == "TRANSFORMER":
            num_heads = getattr(settings, 'PREDICTOR_NUM_HEADS', 4) # Example default
            model = PricePredictorTransformer(input_dim, settings.PREDICTOR_HIDDEN_DIM, num_heads, settings.PREDICTOR_NUM_LAYERS, output_dim, settings.PREDICTOR_DROPOUT).to(device)
        else:
            logging.error(f"Unsupported model type: {settings.PREDICTOR_MODEL_TYPE}")
            return
        logging.info(f"Initialized {settings.PREDICTOR_MODEL_TYPE} model.")
    except Exception as e:
        logging.error(f"Failed to initialize model: {e}")
        return

    # 4. Train Model or Load Weights
    if train:
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
        dataloader = DataLoader(dataset, batch_size=settings.PREDICTOR_BATCH_SIZE, shuffle=True)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=settings.PREDICTOR_LR)
        train_model(model, dataloader, criterion, optimizer, device, settings.PREDICTOR_EPOCHS)
        save_model_weights(model, settings.PREDICTOR_MODEL_PATH)
    else:
        loaded = load_model_weights(model, settings.PREDICTOR_MODEL_PATH, device)
        if not loaded:
            logging.warning("Exiting: Model weights not loaded and training not requested.")
            return

    # 5. Predict Next Steps
    if predict:
        scaled_full_data = scaler.transform(df[['Close']].values)
        if len(scaled_full_data) >= settings.PREDICTOR_INPUT_SEQ_LEN:
            last_sequence_scaled = scaled_full_data[-settings.PREDICTOR_INPUT_SEQ_LEN:]
            last_sequence_reshaped = last_sequence_scaled.reshape(settings.PREDICTOR_INPUT_SEQ_LEN, 1)

            predicted_prices = predict_next(model, last_sequence_reshaped, scaler, device, settings.PREDICTOR_OUTPUT_SEQ_LEN)

            # Determine prediction timestamps based on data interval
            last_timestamp = df.index[-1]
            data_interval = getattr(settings, 'DATA_INTERVAL', '1h')
            try:
                time_delta = pd.Timedelta(data_interval) # Use pandas to parse interval string
            except ValueError:
                 logging.warning(f"Could not parse data interval '{data_interval}' using pandas. Defaulting to 1 hour.")
                 time_delta = pd.Timedelta(hours=1)

            prediction_timestamps = [last_timestamp + time_delta * (i + 1) for i in range(settings.PREDICTOR_OUTPUT_SEQ_LEN)]

            # Log predictions locally (Corrected Formatting - Separate message parts)
            prediction_details = "\n".join([f"  - {ts.strftime('%Y-%m-%d %H:%M')}: ${price:.2f}" for ts, price in zip(prediction_timestamps, predicted_prices)])
            log_message = f"Predicted prices for {settings.PREDICTOR_SYMBOL}:\n" + prediction_details # Build message separately
            logging.info(log_message) # Log the combined message

            # --- Save predictions to DB ---
            if prediction_db_available and log_prediction_to_db:
                try:
                    for i in range(len(predicted_prices)):
                        log_prediction_to_db(
                            prediction_time=prediction_timestamps[i],
                            symbol=settings.PREDICTOR_SYMBOL,
                            predicted_price=predicted_prices[i],
                            model_type=settings.PREDICTOR_MODEL_TYPE,
                            input_seq_len=settings.PREDICTOR_INPUT_SEQ_LEN
                        )
                    logging.info(f"Logged {len(predicted_prices)} predictions to database.")
                except Exception as db_err:
                    logging.error(f"Failed to log predictions to database: {db_err}")

            # --- Send Slack Notification ---
            if slack_available and send_slack_notification:
                first_pred_ts = prediction_timestamps[0].strftime('%Y-%m-%d %H:%M')
                first_pred_price = predicted_prices[0]
                try:
                    send_slack_notification(
                        title=f"📈 Price Prediction ({settings.PREDICTOR_SYMBOL})",
                        message_body=f"Predicted price at {first_pred_ts}: ${first_pred_price:.2f}",
                        level="info"
                    )
                except Exception as slack_err:
                     logging.error(f"Failed to send Slack notification: {slack_err}")
        else:
            logging.warning("Not enough data to form the last sequence for prediction.")

    logging.info("===== Price Prediction Flow End =====")


# Example usage
if __name__ == '__main__':
    logging.info("Running price prediction flow directly...")
    # Ensure necessary setup if running standalone
    # Example: Load .env if needed
    # from dotenv import load_dotenv
    # load_dotenv()
    # from src.config import settings # Reload settings after dotenv

    # Check if required functions/classes are defined or imported
    # For testing, you might need to define dummy versions or ensure imports work
    run_price_prediction_flow(train=False, predict=True) 