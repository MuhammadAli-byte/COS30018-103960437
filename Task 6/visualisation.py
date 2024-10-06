import matplotlib.pyplot as plt
import mplfinance as mpf
from scipy.ndimage import gaussian_filter1d
import numpy as np


def plot_prediction(actual_prices, predicted_prices, ticker='Stock', smooth=False, sigma=2):
    """
    Plots the actual and predicted stock prices for multistep predictions.

    Parameters:
    - actual_prices (np.ndarray): Actual stock prices (samples, future_steps).
    - predicted_prices (np.ndarray): Predicted stock prices (samples, future_steps).
    - ticker (str): The stock ticker symbol used in the plot title.
    - smooth (bool): Whether to apply smoothing to the predicted prices for better visualization.
    - sigma (int): The sigma value for Gaussian smoothing.

    Returns:
    - None
    """
    # Flatten the arrays for plotting
    actual_prices_flat = actual_prices.flatten()
    predicted_prices_flat = predicted_prices.flatten()

    if smooth:
        predicted_prices_flat = gaussian_filter1d(predicted_prices_flat, sigma=sigma)

    plt.figure(figsize=(10, 6))
    plt.plot(actual_prices_flat, color='black', linewidth=1.5, label=f'Actual {ticker} Price')
    plt.plot(predicted_prices_flat, color='green', linestyle='--', linewidth=2, label=f'Predicted {ticker} Price')
    plt.title(f'{ticker} Share Price Prediction')
    plt.xlabel('Time Steps')
    plt.ylabel(f'{ticker} Share Price')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_candlestick(data, n_days=1, ticker='Stock', save_plot=False):
    """
    Plots a candlestick chart using mplfinance with the option to aggregate data over multiple trading days.

    Parameters:
    - data (pd.DataFrame): DataFrame containing stock data with columns ['Open', 'High', 'Low', 'Close', 'Volume'].
    - n_days (int): Number of trading days each candlestick should represent (e.g., 1 for daily, 5 for weekly).
    - ticker (str): The stock ticker symbol, used for labeling the plot.
    - save_plot (bool): If True, saves the plot as an image file; otherwise, it displays the plot.

    Returns:
    - None
    """
    plt.ioff()  # Disable interactive mode

    # Ensure data has the required columns
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(column in data.columns for column in required_columns):
        raise ValueError(f"Data must contain columns: {required_columns}")

    # Resample data if n_days > 1 to aggregate data over the specified number of days
    if n_days > 1:
        data = data.resample(f'{n_days}D').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()

    # Define custom style for improved readability
    style = mpf.make_mpf_style(
        base_mpf_style='charles',
        marketcolors=mpf.make_marketcolors(
            up='green', down='red',
            wick='inherit',
            edge='inherit',
            volume='in',
        ),
        gridcolor='lightgray',
        gridstyle='--',
        facecolor='white',
    )

    fig, axlist = mpf.plot(
        data,
        type='candle',
        style=style,
        title=f'{ticker} Candlestick Chart',
        ylabel='Price',
        volume=True,
        ylabel_lower='Volume',
        datetime_format='%Y-%b',
        xrotation=45,
        tight_layout=False,
        figsize=(14, 8),
        show_nontrading=False,
        returnfig=True
    )

    # Adjust the layout to avoid clipping
    fig.subplots_adjust(right=0.95, left=0.1, top=0.9, bottom=0.15)

    if save_plot:
        fig.savefig(f'{ticker}_candlestick.png', bbox_inches='tight')
    else:
        plt.show()


def plot_boxplot(data, window_size=20, step=5, ticker='Stock'):
    """
    Plots a boxplot chart for the closing prices of a stock using a moving window of n consecutive trading days.

    Parameters:
    - data (pd.DataFrame): DataFrame containing stock data with at least a 'Close' column.
    - window_size (int): The size of the moving window in trading days.
    - step (int): Step size to reduce the number of boxplots shown.
    - ticker (str): The stock ticker symbol, used for labeling the plot.

    Returns:
    - None
    """
    if 'Close' not in data.columns:
        raise ValueError("Data must contain 'Close' column for boxplot.")

    windowed_data = [
        data['Close'].iloc[i:i + window_size].values
        for i in range(0, len(data) - window_size + 1, step)
    ]

    plt.figure(figsize=(14, 7))
    plt.boxplot(
        windowed_data,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor='lightblue', color='blue'),
        whiskerprops=dict(color='blue'),
        capprops=dict(color='blue'),
        medianprops=dict(color='black')
    )

    plt.title(f'{ticker} Closing Prices Boxplot (Window Size = {window_size} Days)')
    plt.xlabel('Time Windows (Sampled)')
    plt.ylabel('Closing Price')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.xticks(ticks=range(1, len(windowed_data) + 1, max(1, len(windowed_data) // 10)), rotation=45)
    plt.tight_layout()
    plt.show()
