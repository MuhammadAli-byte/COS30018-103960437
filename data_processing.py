import os
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, GRU, SimpleRNN, Dropout, Flatten


def load_and_process_data_with_gap(ticker, start_date, end_date, handle_nan='drop',
                                   split_method='date', test_ratio=0.2, scale_data=True,
                                   save_local=False, load_local=False, local_dir='stock_data',
                                   feature_columns=None, gap_period='30D', target_column='Close'):
    """
    Load and process stock market data with a gap between training and testing periods.

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
    - target_column (str): Column name to be used as the target variable.

    Returns:
    - X_train, X_test (np.ndarray): Training and testing feature matrices.
    - y_train, y_test (np.ndarray): Training and testing target vectors.
    - scaler (MinMaxScaler or None): The scaler object used for data normalization, if scaling is enabled.
    - df (pd.DataFrame): The original data frame used for visualization.
    """
    # Create a unique file name based on the ticker, start date, and end date
    file_name = f"{ticker}_{start_date.replace('-', '')}_to_{end_date.replace('-', '')}.csv"
    file_path = os.path.join(local_dir, file_name)

    # Convert start and end dates to datetime for accurate comparison
    start_date_dt = pd.to_datetime(start_date)
    end_date_dt = pd.to_datetime(end_date)

    # Load data from local file if available and matches the current date range
    if load_local and os.path.exists(file_path):
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)

        # Adjust the loaded date range to account for potential missing or extra days
        local_start_date = df.index.min().normalize()  # Normalize to remove time component
        local_end_date = df.index.max().normalize()

        # Debug output to verify date comparison
        print(f"Local data start date: {local_start_date}, Expected start date: {start_date_dt}")
        print(f"Local data end date: {local_end_date}, Expected end date: {end_date_dt}")

        # Allow for minor discrepancies (e.g., one day off due to data fetching limits)
        if local_start_date <= start_date_dt and local_end_date >= (end_date_dt - pd.Timedelta(days=1)):
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

    # Determine features and target
    if feature_columns is None:
        feature_columns = df.columns.tolist()
        # Include 'Close' as a feature if desired
        # If you want to exclude the target column from features, uncomment the next line
        # feature_columns.remove(target_column)

    x = df[feature_columns].values
    y = df[target_column].values

    # Data splitting logic
    if split_method == 'date':
        gap = pd.to_timedelta(gap_period).days
        test_start_idx = int(len(df) * (1 - test_ratio)) + gap
        x_train, x_test = x[:test_start_idx], x[test_start_idx:]
        y_train, y_test = y[:test_start_idx], y[test_start_idx:]
    else:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_ratio, shuffle=True, random_state=42)

    # Data scaling (optional, but we will scale after reshaping for multistep predictions)
    scaler = None
    if scale_data:
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        y_scaler = MinMaxScaler()
        y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

    return x_train, x_test, y_train, y_test, scaler, df


def prepare_data_for_model(X, y, timesteps):
    """
    Prepares data for time series models by reshaping the input arrays
    into the required format (samples, timesteps, features).

    Parameters:
    - X (np.ndarray): Input feature matrix.
    - y (np.ndarray): Target vector.
    - timesteps (int): Number of timesteps for each sample (e.g., 60 for 60 days).

    Returns:
    - X_reshaped (np.ndarray): Reshaped input data for the model.
    - y_reshaped (np.ndarray): Corresponding reshaped target data.
    """
    n_samples = X.shape[0] - timesteps + 1

    # Reshape X and y for time series input
    X_reshaped = np.array([X[i:i + timesteps] for i in range(n_samples)])
    y_reshaped = y[timesteps - 1:]

    return X_reshaped, y_reshaped


def prepare_multistep_data(X, y, timesteps, future_steps):
    """
    Prepares data for time series models for multistep predictions.

    Parameters:
    - X (np.ndarray): Input feature matrix.
    - y (np.ndarray): Target vector.
    - timesteps (int): Number of timesteps for each sample.
    - future_steps (int): Number of future steps to predict.

    Returns:
    - X_reshaped (np.ndarray): Reshaped input data for the model.
    - y_reshaped (np.ndarray): Corresponding reshaped target data (multistep).
    """
    n_samples = X.shape[0] - timesteps - future_steps + 1

    X_reshaped = np.array([X[i:i + timesteps] for i in range(n_samples)])
    y_reshaped = np.array([y[i + timesteps:i + timesteps + future_steps] for i in range(n_samples)])

    return X_reshaped, y_reshaped


def create_backpropagation_model(input_shape, hidden_layers, hidden_units, activation_functions, output_size,
                             loss_function='mse', optimizer='adam'):
    """
    Creates a backpropagation neural network model.

    Parameters:
    - input_shape (tuple): Shape of the input data (timesteps, features).
    - hidden_layers (int): Number of hidden layers.
    - hidden_units (list of int): Number of units in each hidden layer.
    - activation_functions (list of str): Activation functions for each hidden layer.
    - output_size (int): Number of units in the output layer.
    - loss_function (str): Loss function to use for training.
    - optimizer (str): Optimizer to use for training.

    Returns:
    - model (Sequential): Compiled Keras feedforward model.
    """
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))

    for i in range(hidden_layers):
        model.add(Dense(hidden_units[i], activation=activation_functions[i]))

    model.add(Dense(output_size))  # Output layer
    model.compile(optimizer=optimizer, loss=loss_function)

    return model

def create_dl_model(input_shape, layer_types, layer_sizes, dropout_rates,
                    output_size, loss_function='mse', optimizer='adam',
                    return_sequences=None, activation_functions=None):
    """
    Creates a custom deep learning model based on the provided parameters.

    Parameters:
    - input_shape (tuple): Shape of the input data (timesteps, features).
    - layer_types (list of str): Types of layers to add (e.g., ['LSTM', 'GRU']).
    - layer_sizes (list of int): Number of units in each layer.
    - dropout_rates (list of float): Dropout rates for each layer.
    - output_size (int): Number of units in the output layer.
    - loss_function (str): Loss function to use for training (e.g., 'mse').
    - optimizer (str): Optimizer to use for training (e.g., 'adam').
    - return_sequences (list of bool): Whether each layer returns sequences.
    - activation_functions (list of str): Activation functions for each layer.

    Returns:
    - model (Model): Compiled Keras model ready for training.
    """
    inputs = Input(shape=input_shape)
    x = inputs

    n_layers = len(layer_types)
    if return_sequences is None:
        return_sequences = [False] * n_layers
    if activation_functions is None:
        activation_functions = ['relu'] * n_layers

    for i in range(n_layers):
        layer_type = layer_types[i].upper()
        units = layer_sizes[i]
        dropout_rate = dropout_rates[i] if i < len(dropout_rates) else 0.0
        return_seq = return_sequences[i]
        activation = activation_functions[i]

        if layer_type == 'LSTM':
            x = LSTM(units, return_sequences=return_seq)(x)
        elif layer_type == 'GRU':
            x = GRU(units, return_sequences=return_seq)(x)
        elif layer_type == 'RNN':
            x = SimpleRNN(units, return_sequences=return_seq)(x)
        elif layer_type == 'DENSE':
            x = Dense(units, activation=activation)(x)
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")

        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)

    outputs = Dense(output_size)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=optimizer, loss=loss_function)

    return model
