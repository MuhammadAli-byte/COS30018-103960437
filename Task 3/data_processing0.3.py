import os
import pandas as pd
import yfinance as yf
import mplfinance as mpf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def load_and_process_data_with_gap(ticker, start_date, end_date, handle_nan='drop',
                                   split_method='date', test_ratio=0.2, scale_data=True,
                                   save_local=False, load_local=False, local_dir='stock_data',
                                   feature_columns=None, gap_period='30D'):
    """
    Load and process stock market data with a gap between training and testing periods.

    This function fetches the stock data, processes it (e.g., handling NaNs, adding technical indicators),
    splits it into training and testing sets, and optionally scales the data.

    Parameters:
    - ticker (str): Stock ticker symbol.
    - start_date (str): Start date for data in 'YYYY-MM-DD' format.
    - end_date (str): End date for data in 'YYYY-MM-DD' format.
    - handle_nan (str): Method to handle NaN values ('drop', 'fill', 'none').
    - split_method (str): How to split the data ('date', 'random').
    - test_ratio (float): Ratio of the test set, if split_method is 'random'.
    - scale_data (bool): Whether to scale the data.
    - save_local (bool): Whether to save the data locally.
    - load_local (bool): Whether to load data from a local file if available.
    - local_dir (str): Directory to save/load the data.
    - feature_columns (list): List of columns to use as features.
    - gap_period (str): Gap period (e.g., '30D' for 30 days) between the training and testing periods.

    Returns:
    - X_train, X_test (np.ndarray): Training and testing feature matrices.
    - y_train, y_test (np.ndarray): Training and testing target vectors.
    - scaler (MinMaxScaler or None): The scaler object used for data normalization, if scaling is enabled.
    - df (pd.DataFrame): The original data frame used for visualization.
    """
    # Create a unique file name based on the ticker, start date, and end date
    file_name = f"{ticker}_{start_date.replace('-', '')}_to_{end_date.replace('-', '')}.csv"
    file_path = os.path.join(local_dir, file_name)

    # Load data from local file if available and matches the current date range
    if load_local and os.path.exists(file_path):
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)

        # Check if the loaded data matches the specified date range
        if df.index.min() <= pd.to_datetime(start_date) and df.index.max() >= pd.to_datetime(end_date):
            print(f"Loading data from local file: {file_path}")
        else:
            print("Local data does not match the specified date range. Fetching fresh data...")
            df = yf.download(ticker, start=start_date, end=end_date)
            df.to_csv(file_path)
    else:
        print("Fetching fresh data from Yahoo Finance...")
        df = yf.download(ticker, start=start_date, end=end_date)
        if save_local:
            if not os.path.exists(local_dir):
                os.makedirs(local_dir)
            df.to_csv(file_path)

    print(f"Data loaded with date range: {df.index.min()} to {df.index.max()}")

    # Handle NaN values
    if handle_nan == 'drop':
        df.dropna(inplace=True)
    elif handle_nan == 'fill':
        df.fillna(method='ffill', inplace=True)

    # Verify the date range in the DataFrame
    print(f"Data loaded with date range: {df.index.min()} to {df.index.max()}")

    # Determine features and target
    if feature_columns is None:
        feature_columns = df.columns.tolist()
        feature_columns.remove('Close')  # Assuming 'Close' is the target by default

    x = df[feature_columns].values
    y = df['Close'].values

    # Data splitting logic
    if split_method == 'date':
        gap = pd.to_timedelta(gap_period).days
        test_start_idx = int(len(df) * (1 - test_ratio)) + gap
        x_train, x_test = x[:test_start_idx], x[test_start_idx:]
        y_train, y_test = y[:test_start_idx], y[test_start_idx:]
    else:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_ratio, shuffle=True, random_state=42)

    # Data scaling
    scaler = None
    if scale_data:
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        y_scaler = MinMaxScaler()
        y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

    return x_train, x_test, y_train, y_test, scaler, df

def plot_candlestick(data, n_days=1, ticker='Stock', save_plot=False):
    """
    Plots a candlestick chart using mplfinance with the option to aggregate data over multiple trading days.

    The function enhances chart readability by adjusting styles, date formatting, and gridlines.
    Interactive zoom is enabled by default when displayed.

    Parameters:
    - data (pd.DataFrame): DataFrame containing stock data with columns ['Open', 'High', 'Low', 'Close', 'Volume'].
    - n_days (int): Number of trading days each candlestick should represent (e.g., 1 for daily, 5 for weekly).
    - ticker (str): The stock ticker symbol, used for labeling the plot.
    - save_plot (bool): If True, saves the plot as an image file; otherwise, it displays the plot.

    Returns:
    - None
    """
    # Enable interactive mode
    plt.ioff() # TODO: Currently not working

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
        base_mpf_style='charles',  # Clean and modern style
        marketcolors=mpf.make_marketcolors(
            up='green', down='red',  # Use green for up days and red for down days
            wick='inherit',  # Use the same colors for wicks
            edge='inherit',  # Use the same colors for edges
            volume='in',  # Match volume color to price movement
        ),
        gridcolor='lightgray',  # Light gray gridlines to minimize distraction
        gridstyle='--',  # Dashed gridlines
        facecolor='white',  # White background for simplicity
    )

    # Plotting the candlestick chart using mplfinance
    fig, axlist = mpf.plot(
        data,
        type='candle',  # Candlestick type plot
        style=style,  # Apply the custom style
        title=f'{ticker} Candlestick Chart',  # Title of the plot
        ylabel='Price',  # Label for the y-axis
        volume=True,  # Include volume in the plot
        ylabel_lower='Volume',  # Label for the volume axis
        datetime_format='%Y-%b',  # Format date as Year-Month (e.g., 2020-Aug)
        xrotation=45,  # Rotate x-axis labels for better readability
        tight_layout=False,  # Turn off tight layout to prevent clipping
        figsize=(14, 8),  # Increase figure size for more space
        show_nontrading=False,  # Exclude non-trading days
        returnfig=True  # Return the figure and axes to adjust further if needed
    )

    # Adjust the layout to avoid clipping
    fig.subplots_adjust(right=0.95, left=0.1, top=0.9, bottom=0.15)  # Adjust margins to prevent clipping

    # Save the plot as an image file if save_plot is True
    if save_plot:
        fig.savefig(f'{ticker}_candlestick.png', bbox_inches='tight')  # Ensure no clipping when saving
    else:
        plt.show()  # Display the plot with interactive mode enabled


def plot_boxplot(data, window_size=20, step=5, ticker='Stock'):
    """
    Plots a boxplot chart for the closing prices of a stock using a moving window of n consecutive trading days.
    Improves readability by adjusting the window size, step, and boxplot appearance.

    Parameters:
    - data (pd.DataFrame): DataFrame containing stock data with at least a 'Close' column.
    - window_size (int): The size of the moving window in trading days.
    - step (int): Step size to reduce the number of boxplots shown.
    - ticker (str): The stock ticker symbol, used for labeling the plot.

    Returns:
    - None
    """
    # Ensure data has the required 'Close' column
    if 'Close' not in data.columns:
        raise ValueError("Data must contain 'Close' column for boxplot.")

    # Generate moving windows of closing prices
    windowed_data = [
        data['Close'].iloc[i:i + window_size].values
        for i in range(0, len(data) - window_size + 1, step)
    ]

    # Plotting the boxplot
    plt.figure(figsize=(14, 7))
    plt.boxplot(
        windowed_data,
        patch_artist=True,  # Fill the boxes with color
        showfliers=False,  # Hide outliers to reduce clutter
        boxprops=dict(facecolor='lightblue', color='blue'),  # Box color
        whiskerprops=dict(color='blue'),  # Whisker color
        capprops=dict(color='blue'),  # Cap color
        medianprops=dict(color='black')  # Median line color
    )

    # Set plot title and labels
    plt.title(f'{ticker} Closing Prices Boxplot (Window Size = {window_size} Days)')
    plt.xlabel('Time Windows (Sampled)')
    plt.ylabel('Closing Price')
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # Light grid lines on y-axis for reference

    # Reduce the number of x-ticks and rotate them for better readability
    plt.xticks(ticks=range(0, len(windowed_data), max(1, len(windowed_data) // 10)), rotation=45)

    # Show the plot
    plt.tight_layout()
    plt.show()
