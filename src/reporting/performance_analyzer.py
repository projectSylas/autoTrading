import pandas as pd
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # Import numpy for np.isfinite
# Import settings
try:
    from src.config import settings
except ImportError:
    settings = type('obj', (object,), {
        'FINRL_RESULTS_DIR': 'results/finrl/',
        'FREQAI_RESULTS_DIR': 'results/freqai/',
        'REPORTING_SAVE_DIR': 'results/reports/'
    })()
    logging.getLogger(__name__).warning("Could not import settings module. Using default paths.")

# Setup logging
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Constants (Use settings) ---
FINRL_RESULTS_DIR = settings.FINRL_RESULTS_DIR
FREQAI_RESULTS_DIR = settings.FREQAI_RESULTS_DIR
REPORTING_SAVE_DIR = settings.REPORTING_SAVE_DIR

# --- Data Loading Functions ---
def load_finrl_backtest_results(file_path: str) -> pd.DataFrame | None:
    """
    Loads FinRL backtest statistics from a file (e.g., CSV).
    Assumes the file contains standard performance metrics.
    """
    logger.info(f"Loading FinRL backtest results from: {file_path}")
    if not os.path.exists(file_path):
        logger.error(f"FinRL results file not found: {file_path}")
        return None
    try:
        # Adjust reading based on how run_backtest_finrl saves stats
        # If it saves the DataFrame directly:
        stats_df = pd.read_csv(file_path)
        # If it saves the Metric/Value format from placeholder:
        # stats_df = pd.read_csv(file_path).set_index('Metric')['Value'].to_frame().T
        logger.info(f"Successfully loaded FinRL results. Shape: {stats_df.shape}")
        return stats_df
    except Exception as e:
        logger.error(f"Error loading FinRL results from {file_path}: {e}", exc_info=True)
        return None

def load_freqai_backtest_results(file_path: str) -> pd.DataFrame | None:
    """
    Loads FreqAI backtest statistics from a file (e.g., JSON or CSV).
    Freqtrade typically saves results in JSON format.
    """
    logger.info(f"Loading FreqAI backtest results from: {file_path}")
    if not os.path.exists(file_path):
        logger.error(f"FreqAI results file not found: {file_path}")
        return None
    try:
        # Freqtrade backtest results are often JSON per pair
        # This function might need logic to parse the JSON and extract key stats
        # Example: Load JSON and extract summary
        # import json
        # with open(file_path, 'r') as f:
        #     data = json.load(f)
        # summary = data.get('strategy_comparison', [{}])[0]
        # stats_df = pd.DataFrame([summary]) # Convert summary dict to DataFrame

        # Placeholder: Assume CSV for simplicity like FinRL
        stats_df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded FreqAI results (assuming CSV). Shape: {stats_df.shape}")
        return stats_df
    except Exception as e:
        logger.error(f"Error loading FreqAI results from {file_path}: {e}", exc_info=True)
        return None

# --- Performance Comparison Function ---
def compare_strategies(results_list: list[dict]) -> pd.DataFrame:
    """
    Combines results from multiple strategies into a single comparison DataFrame.

    Args:
        results_list (list[dict]): A list where each dict contains strategy info
                                     and its results DataFrame. Example:
                                     [{'name': 'FinRL_PPO_AAPL', 'type': 'Stock_DRL',
                                       'results_df': df1},
                                      {'name': 'FreqAI_LGBM_BTC', 'type': 'Crypto_ML',
                                       'results_df': df2}]

    Returns:
        pd.DataFrame: A DataFrame comparing key metrics across strategies.
    """
    comparison_data = []
    key_metrics = [ # Define metrics to compare
        'Total Return [%]', 'Annualized Return', 'Sharpe Ratio', 'Max Drawdown [%]', 'Max Drawdown',
        'Total Trades', 'Win Rate [%]'
    ]

    logger.info(f"Comparing {len(results_list)} strategies...")
    for item in results_list:
        name = item.get('name', 'Unknown Strategy')
        strategy_type = item.get('type', 'Unknown')
        df = item.get('results_df')

        if df is None or df.empty:
            logger.warning(f"Skipping strategy '{name}' due to missing or empty results.")
            continue

        # Extract metrics - handle potential differences in column names/structure
        strategy_stats = {'Strategy Name': name, 'Type': strategy_type}
        for metric in key_metrics:
            value = None
            if metric in df.columns:
                # Assuming one row of results per df
                value = df[metric].iloc[0] if not df[metric].empty else None
            # Handle placeholder Metric/Value format if necessary
            elif 'Metric' in df.columns and 'Value' in df.columns:
                 metric_row = df[df['Metric'] == metric]
                 if not metric_row.empty:
                     value = metric_row['Value'].iloc[0]
            # Handle Max Drawdown variations
            elif metric == 'Max Drawdown [%]' and 'Max Drawdown' in df.columns:
                 value = df['Max Drawdown'].iloc[0] * 100 # Assuming raw drawdown needs conversion
            elif metric == 'Max Drawdown' and 'Max Drawdown [%]' in df.columns:
                 value = df['Max Drawdown [%]'].iloc[0] / 100 # Assuming percentage needs conversion

            # Clean NaN/Inf for comparison
            if pd.isna(value) or (isinstance(value, (float, int)) and not np.isfinite(value)):
                value = None # Use None for consistency

            strategy_stats[metric.replace(' [%]', '')] = value # Clean metric name

        comparison_data.append(strategy_stats)

    if not comparison_data:
        logger.error("No valid strategy results found for comparison.")
        return pd.DataFrame()

    comparison_df = pd.DataFrame(comparison_data)
    # Reorder columns for better readability
    ordered_cols = ['Strategy Name', 'Type'] + [m.replace(' [%]', '') for m in key_metrics if m.replace(' [%]', '') in comparison_df.columns]
    comparison_df = comparison_df[ordered_cols]
    logger.info("Strategy comparison complete.")
    return comparison_df

# --- Visualization Function (Using Settings for Save Path) ---
def plot_comparison(comparison_df: pd.DataFrame, save_path: str | None = None):
    """
    Visualizes the strategy comparison results.
    Saves plot to REPORTING_SAVE_DIR from settings if save_path is None.
    """
    if comparison_df.empty:
        logger.warning("Comparison DataFrame is empty, skipping plotting.")
        return

    logger.info("Generating strategy comparison plot...")
    try:
        # Example: Bar plot of Sharpe Ratio and Total Return
        metrics_to_plot = ['Sharpe Ratio', 'Total Return']
        # Ensure columns exist before trying to plot
        metrics_to_plot = [m for m in metrics_to_plot if m in comparison_df.columns]
        if not metrics_to_plot:
             logger.warning("None of the specified metrics to plot are available.")
             return
             
        plot_df = comparison_df.set_index('Strategy Name')[metrics_to_plot].astype(float).dropna() # Convert to float for plotting

        if plot_df.empty:
            logger.warning("No valid data to plot after dropping NaNs or conversion.")
            return

        fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(8 * len(metrics_to_plot), 6)) # Increased height slightly
        if len(metrics_to_plot) == 1: axes = [axes] # Ensure axes is iterable

        for i, metric in enumerate(metrics_to_plot):
            sns.barplot(x=plot_df.index, y=plot_df[metric], ax=axes[i], palette="viridis") # Added color palette
            axes[i].set_title(f'{metric} Comparison')
            axes[i].set_ylabel(metric)
            axes[i].tick_params(axis='x', rotation=45, labelsize=10)
            # Add value labels on bars
            for container in axes[i].containers:
                 axes[i].bar_label(container, fmt='%.2f', label_type='edge', fontsize=9, padding=2)

        plt.tight_layout()

        # Use path from settings if not provided
        if save_path is None:
            # Ensure the reporting directory exists (settings.py should handle this on import)
            # os.makedirs(settings.REPORTING_SAVE_DIR, exist_ok=True) 
            save_path = os.path.join(settings.REPORTING_SAVE_DIR, "strategy_comparison_plot.png")

        plt.savefig(save_path)
        logger.info(f"Comparison plot saved to: {save_path}")
        plt.close(fig) # Close the figure to free memory

    except Exception as e:
        logger.error(f"Error generating comparison plot: {e}", exc_info=True)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger.info("Running Performance Analyzer...")
    print(f"FinRL Results Dir (from settings): {settings.FINRL_RESULTS_DIR}")
    print(f"FreqAI Results Dir (from settings): {settings.FREQAI_RESULTS_DIR}")
    print(f"Reporting Save Dir (from settings): {settings.REPORTING_SAVE_DIR}")

    # --- Dummy Data Loading & Comparison Example ---
    # Create dummy CSV result files for testing
    os.makedirs(settings.FINRL_RESULTS_DIR, exist_ok=True)
    os.makedirs(settings.FREQAI_RESULTS_DIR, exist_ok=True)
    finrl_dummy_path = os.path.join(settings.FINRL_RESULTS_DIR, "stats_AAPL_ppo_dummy.csv")
    freqai_dummy_path = os.path.join(settings.FREQAI_RESULTS_DIR, "stats_BTC_LGBM_dummy.csv")

    # Dummy FinRL results (Metric/Value format)
    pd.DataFrame({
        'Metric': ['Annualized Return', 'Sharpe Ratio', 'Max Drawdown', 'Total Trades'],
        'Value': [0.15, 0.95, -0.18, 60]
    }).to_csv(finrl_dummy_path, index=False)
    # Dummy FreqAI results (Direct column format)
    pd.DataFrame({
        'Total Return [%]': [180.0],
        'Max Drawdown [%]': [22.5],
        'Sharpe Ratio': [2.1],
        'Total Trades': [150],
        'Win Rate [%]': [68.0]
    }).to_csv(freqai_dummy_path, index=False)

    # Load the dummy results
    finrl_data = load_finrl_backtest_results(finrl_dummy_path)
    freqai_data = load_freqai_backtest_results(freqai_dummy_path)

    results_to_compare = []
    if finrl_data is not None:
         # Need to convert Metric/Value format to row for comparison
         finrl_stats_row = finrl_data.set_index('Metric')[['Value']].T
         results_to_compare.append({'name': 'FinRL_PPO_AAPL_Dummy', 'type': 'Stock_DRL', 'results_df': finrl_stats_row})
    if freqai_data is not None:
         results_to_compare.append({'name': 'FreqAI_LGBM_BTC_Dummy', 'type': 'Crypto_ML', 'results_df': freqai_data})

    if results_to_compare:
        comparison = compare_strategies(results_to_compare)
        print("\n--- Strategy Comparison Table ---")
        # Format float columns for printing
        float_cols = comparison.select_dtypes(include='number').columns
        print(comparison.to_string(float_format="{:.4f}".format))
        plot_comparison(comparison)
    else:
        print("\nCould not load any dummy results for comparison.")
 