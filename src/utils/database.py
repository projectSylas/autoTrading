import psycopg2
import logging
import pandas as pd
from psycopg2 import pool
from psycopg2.extras import DictCursor
import os
from contextlib import contextmanager
from src.config import settings # Use settings for DB config
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Connection Pool --- #
# Use environment variables provided by docker-compose or host
DB_HOST = os.getenv("DB_HOST", "localhost") # Default to localhost if not in Docker
DB_PORT = os.getenv("DB_PORT", "5433")
DB_NAME = os.getenv("DB_NAME", "trading_db")
DB_USER = os.getenv("DB_USER", "trading_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "trading_password")

# Create a connection pool
try:
    connection_pool = psycopg2.pool.SimpleConnectionPool(
        minconn=1,
        maxconn=5, # Adjust max connections as needed
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    logging.info(f"Database connection pool created for {DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
except psycopg2.OperationalError as e:
    logging.error(f"Failed to create database connection pool: {e}")
    connection_pool = None

@contextmanager
def get_db_connection():
    """Context manager to get a connection from the pool."""
    if connection_pool is None:
        raise ConnectionError("Database connection pool is not available.")
    conn = None
    try:
        conn = connection_pool.getconn()
        yield conn
    except Exception as e:
        logging.error(f"Error getting DB connection: {e}")
        raise # Re-raise the exception
    finally:
        if conn:
            connection_pool.putconn(conn)

@contextmanager
def get_db_cursor(commit=False):
    """Context manager to get a cursor from a connection."""
    with get_db_connection() as conn:
        cursor = None
        try:
            cursor = conn.cursor(cursor_factory=DictCursor) # Return results as dicts
            yield cursor
            if commit:
                conn.commit()
                logging.debug("DB transaction committed.")
        except Exception as e:
            logging.error(f"Database cursor error: {e}", exc_info=True)
            if conn:
                 conn.rollback() # Rollback on error
                 logging.warning("DB transaction rolled back due to error.")
            raise
        finally:
            if cursor:
                cursor.close()

# --- Table Creation (Idempotent) --- #
def initialize_database():
    """Creates necessary tables if they don't exist."""
    commands = (
        """
        CREATE TABLE IF NOT EXISTS trades (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            strategy VARCHAR(50) NOT NULL, -- 'core' or 'challenge'
            symbol VARCHAR(50) NOT NULL,
            side VARCHAR(10) NOT NULL, -- 'buy' or 'sell'
            quantity NUMERIC,
            amount_usd NUMERIC,
            entry_price NUMERIC,
            exit_price NUMERIC,
            pnl_percent NUMERIC,
            status VARCHAR(50), -- 'open', 'closed_tp', 'closed_sl', 'rebalance', 'APIError', etc.
            order_id VARCHAR(100),
            reason TEXT
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS sentiment_logs (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            keyword VARCHAR(100) NOT NULL,
            sentiment VARCHAR(20), -- 'positive', 'negative', 'neutral'
            score NUMERIC,
            article_count INTEGER
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS volatility_logs (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            symbol VARCHAR(50) NOT NULL,
            check_time TIMESTAMP WITH TIME ZONE,
            is_anomaly BOOLEAN,
            actual_price NUMERIC,
            predicted_price NUMERIC,
            deviation_percent NUMERIC
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS prediction_logs (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP, -- Log time
            prediction_time TIMESTAMP WITH TIME ZONE NOT NULL, -- Timestamp the prediction is for
            symbol VARCHAR(50) NOT NULL,
            predicted_price NUMERIC NOT NULL,
            model_type VARCHAR(50),
            input_seq_len INTEGER,
            actual_price NUMERIC, -- To be filled later for error calculation
            error_percentage NUMERIC -- To be filled later
        );
        """
    )
    try:
        with get_db_cursor(commit=True) as cursor:
            for command in commands:
                cursor.execute(command)
        logging.info("Database initialized successfully (tables checked/created).")
    except Exception as e:
        logging.error(f"Failed to initialize database: {e}")
        # Decide if the application should stop if DB init fails
        raise

# --- Data Insertion Functions --- #
def log_trade_to_db(strategy: str, **kwargs):
    """Logs a trade record to the 'trades' table."""
    # Define columns based on kwargs, match with table definition
    allowed_cols = ['symbol', 'side', 'quantity', 'amount_usd', 'entry_price',
                    'exit_price', 'pnl_percent', 'status', 'order_id', 'reason']
    cols = ['strategy'] + [k for k in allowed_cols if k in kwargs]
    values_placeholder = ', '.join(['%s'] * (len(cols)))
    sql = f"INSERT INTO trades ({', '.join(cols)}) VALUES ({values_placeholder})"

    values = [strategy] + [kwargs.get(k) for k in allowed_cols if k in kwargs]

    try:
        with get_db_cursor(commit=True) as cursor:
            cursor.execute(sql, tuple(values))
        logging.debug(f"Trade logged to DB for strategy '{strategy}': {kwargs}")
    except Exception as e:
        logging.error(f"Failed to log trade to DB: {e}")

def log_sentiment_to_db(**kwargs):
    """Logs sentiment analysis results to 'sentiment_logs'."""
    allowed_cols = ['keyword', 'sentiment', 'score', 'article_count']
    cols = [k for k in allowed_cols if k in kwargs]
    if not cols:
        logging.warning("No valid columns found to log sentiment.")
        return
    values_placeholder = ', '.join(['%s'] * len(cols))
    sql = f"INSERT INTO sentiment_logs ({', '.join(cols)}) VALUES ({values_placeholder})"
    values = [kwargs.get(k) for k in cols]
    try:
        with get_db_cursor(commit=True) as cursor:
            cursor.execute(sql, tuple(values))
        logging.debug(f"Sentiment logged to DB: {kwargs}")
    except Exception as e:
        logging.error(f"Failed to log sentiment to DB: {e}")

def log_volatility_to_db(**kwargs):
    """Logs volatility check results to 'volatility_logs'."""
    allowed_cols = ['symbol', 'check_time', 'is_anomaly', 'actual_price',
                    'predicted_price', 'deviation_percent']
    cols = [k for k in allowed_cols if k in kwargs]
    if not cols:
        logging.warning("No valid columns found to log volatility.")
        return
    values_placeholder = ', '.join(['%s'] * len(cols))
    sql = f"INSERT INTO volatility_logs ({', '.join(cols)}) VALUES ({values_placeholder})"
    values = [kwargs.get(k) for k in cols]
    try:
        with get_db_cursor(commit=True) as cursor:
            cursor.execute(sql, tuple(values))
        logging.debug(f"Volatility logged to DB: {kwargs}")
    except Exception as e:
        logging.error(f"Failed to log volatility to DB: {e}")

def log_prediction_to_db(**kwargs):
    """Logs AI prediction results to the 'prediction_logs' table."""
    # Define allowed columns based on table schema (excluding id and timestamp)
    allowed_cols = ['prediction_time', 'symbol', 'predicted_price',
                    'model_type', 'input_seq_len', 'actual_price', 'error_percentage']

    # Filter kwargs to include only allowed columns
    cols_to_insert = {k: v for k, v in kwargs.items() if k in allowed_cols}

    if not cols_to_insert or 'prediction_time' not in cols_to_insert or \
       'symbol' not in cols_to_insert or 'predicted_price' not in cols_to_insert:
        logging.warning(f"Required prediction data missing (prediction_time, symbol, predicted_price). Log skipped. Data: {kwargs}")
        return

    cols = list(cols_to_insert.keys())
    values_placeholder = ', '.join(['%s'] * len(cols))
    sql = f"INSERT INTO prediction_logs ({ ', '.join(cols) }) VALUES ({ values_placeholder })"
    values = [cols_to_insert[k] for k in cols]

    try:
        with get_db_cursor(commit=True) as cursor:
            cursor.execute(sql, tuple(values))
        logging.debug(f"Prediction logged to DB: {cols_to_insert}")
    except Exception as e:
        logging.error(f"Failed to log prediction to DB: {e}")

# --- Data Retrieval Functions --- #
def get_log_data(table_name: str, days: int | None = None, limit: int | None = None, query_filter: str | None = None) -> pd.DataFrame:
    """Retrieves log data from a specified table.

    Args:
        table_name (str): The name of the table to query.
        days (int | None): Retrieve records from the last N days. If None, retrieve all.
        limit (int | None): Limit the number of records returned.
        query_filter (str | None): Additional SQL WHERE clause condition (e.g., "strategy = 'core'").

    Returns:
        pd.DataFrame: The retrieved data as a DataFrame.
    """
    base_sql = f"SELECT * FROM {table_name}"
    conditions = []
    params = []

    if days is not None:
        conditions.append("timestamp >= NOW() - INTERVAL %s days")
        params.append(days)

    if query_filter:
        conditions.append(f"({query_filter})") # Add filter condition

    if conditions:
        base_sql += " WHERE " + " AND ".join(conditions)

    base_sql += " ORDER BY timestamp DESC"

    if limit:
        base_sql += " LIMIT %s"
        params.append(limit)

    try:
        with get_db_cursor() as cursor:
            cursor.execute(base_sql, tuple(params))
            results = cursor.fetchall()
            df = pd.DataFrame([dict(row) for row in results]) # Convert DictRow to dict
            logging.info(f"Retrieved {len(df)} records from '{table_name}' (Filter: '{query_filter}', Days: {days}, Limit: {limit}).")
            return df
    except Exception as e:
        logging.error(f"Failed to retrieve data from '{table_name}': {e}")
        return pd.DataFrame() # Return empty DataFrame on error

# --- Example Usage & Initialization --- #
if __name__ == "__main__":
    print("Initializing Database (if needed)...")
    initialize_database()

    print("\nTesting log functions...")
    # Test trade log
    log_trade_to_db(
        strategy='challenge',
        symbol='BTCUSDT',
        side='buy',
        quantity=0.01,
        entry_price=60000,
        status='open',
        reason='Test entry'
    )
    log_trade_to_db(
        strategy='core',
        symbol='SPY',
        side='buy',
        amount_usd=500,
        status='filled',
        order_id='test_order_123'
    )

    # Test sentiment log
    log_sentiment_to_db(keyword='Bitcoin', sentiment='positive', score=0.8, article_count=10)

    # Test volatility log
    log_volatility_to_db(symbol='ETHUSDT', check_time=pd.Timestamp.now(tz='UTC'), is_anomaly=True, actual_price=3000, predicted_price=2800, deviation_percent=7.14)

    # Test prediction log
    print("\nTesting prediction log...")
    try:
        log_prediction_to_db(
            prediction_time=datetime.now() + timedelta(hours=1),
            symbol='BTCUSDT',
            predicted_price=65000.50,
            model_type='LSTM',
            input_seq_len=60
        )
        log_prediction_to_db(
            prediction_time=datetime.now() + timedelta(hours=2),
            symbol='BTCUSDT',
            predicted_price=65500.00,
            model_type='LSTM'
            # actual_price=65400.00 # Example for later
        )
        print("Prediction logs test successful (check DB).")
    except Exception as e:
         print(f"Error during prediction log test: {e}")

    print("\nTesting data retrieval...")
    df_trades = get_log_data('trades', days=7)
    print("Recent Trades:")
    print(df_trades.head())

    df_trades_core = get_log_data('trades', days=7, query_filter="strategy = 'core'")
    print("\nRecent Core Trades:")
    print(df_trades_core.head())

    df_sentiment = get_log_data('sentiment_logs', days=1)
    print("\nRecent Sentiment:")
    print(df_sentiment.head())

    print("\nDatabase utility tests completed.")

# Ensure DB is initialized when the module is imported if pool is available
if connection_pool:
     initialize_database() 