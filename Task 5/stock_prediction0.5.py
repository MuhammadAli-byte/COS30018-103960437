import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# Import functions from data_processing and visualization
from data_processing import (
    load_and_process_data_with_gap,
    create_dl_model,
    prepare_multistep_data
)
from visualisation import plot_candlestick, plot_boxplot, plot_prediction

# Load and Process Data
COMPANY = 'KO'  # Change the ticker symbol here
TRAIN_START = '2016-08-01'
TRAIN_END = '2024-08-31'
X_train, X_test, y_train, y_test, _, df = load_and_process_data_with_gap(
    ticker=COMPANY,
    start_date=TRAIN_START,
    end_date=TRAIN_END,
    handle_nan='drop',
    split_method='date',
    test_ratio=0.2,
    scale_data=False,  # We will scale after reshaping for multistep predictions
    save_local=True,
    load_local=True,
    local_dir='stock_data',
    feature_columns=None,  # Use all available features
    target_column='Close'  # Define the target column
)

# Plot charts of Data
plot_candlestick(df, n_days=5, ticker=COMPANY)
plot_boxplot(df, window_size=10, step=5, ticker=COMPANY)

# Prepare Data
PREDICTION_DAYS = 60
FUTURE_STEPS = 3  # Number of future days to predict

# Reshape training and test data for multistep prediction
X_train_reshaped, y_train_reshaped = prepare_multistep_data(X_train, y_train, PREDICTION_DAYS, FUTURE_STEPS)
X_test_reshaped, y_test_reshaped = prepare_multistep_data(X_test, y_test, PREDICTION_DAYS, FUTURE_STEPS)

# Scaling functions
def scale_X(X_train, X_test):
    n_samples_train, timesteps, n_features = X_train.shape
    n_samples_test = X_test.shape[0]
    X_train_flat = X_train.reshape(-1, n_features)
    X_test_flat = X_test.reshape(-1, n_features)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat).reshape(n_samples_train, timesteps, n_features)
    X_test_scaled = scaler.transform(X_test_flat).reshape(n_samples_test, timesteps, n_features)
    return X_train_scaled, X_test_scaled, scaler

def scale_y(y_train, y_test):
    n_samples_train, future_steps = y_train.shape
    n_samples_test = y_test.shape[0]
    y_train_flat = y_train.reshape(-1, 1)
    y_test_flat = y_test.reshape(-1, 1)
    scaler = MinMaxScaler()
    y_train_scaled = scaler.fit_transform(y_train_flat).reshape(n_samples_train, future_steps)
    y_test_scaled = scaler.transform(y_test_flat).reshape(n_samples_test, future_steps)
    return y_train_scaled, y_test_scaled, scaler

# Apply scaling
X_train_scaled, X_test_scaled, X_scaler = scale_X(X_train_reshaped, X_test_reshaped)
y_train_scaled, y_test_scaled, y_scaler = scale_y(y_train_reshaped, y_test_reshaped)

# Define the multivariate prediction function
def multivariate_prediction(k_days: int, company: str, train_start: str, train_end: str, prediction_days: int,
                            n_features: int = 6, test_ratio: float = 0.15, multivariate: bool = True):
    """
    Multivariate prediction using multiple features.
    """
    # Load and process data
    X_train, X_test, y_train, y_test, _, df = load_and_process_data_with_gap(
        ticker=company,
        start_date=train_start,
        end_date=train_end,
        handle_nan='drop',
        split_method='date',
        test_ratio=test_ratio,
        scale_data=False,
        save_local=True,
        load_local=True,
        local_dir='stock_data',
        feature_columns=None,
        target_column='Close'
    )

    # Prepare data for multistep prediction
    X_train_reshaped, y_train_reshaped = prepare_multistep_data(X_train, y_train, prediction_days, k_days)
    X_test_reshaped, y_test_reshaped = prepare_multistep_data(X_test, y_test, prediction_days, k_days)

    # Scale the data
    X_train_scaled, X_test_scaled, X_scaler = scale_X(X_train_reshaped, X_test_reshaped)
    y_train_scaled, y_test_scaled, y_scaler = scale_y(y_train_reshaped, y_test_reshaped)

    # Define the model
    model = Sequential()
    model.add(LSTM(150, input_shape=(prediction_days, n_features), return_sequences=False, activation='tanh'))
    model.add(Dropout(0.3))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(k_days, activation='linear'))

    model.compile(loss='huber', optimizer='adam')

    # Train the model
    early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
    model.fit(
        X_train_scaled, y_train_scaled,
        epochs=100, batch_size=32, verbose=1,
        validation_split=0.2, callbacks=[early_stopping]
    )

    # Predict on the test set
    predicted_prices = model.predict(X_test_scaled)
    predicted_prices_inv = y_scaler.inverse_transform(predicted_prices.reshape(-1, 1)).reshape(-1, k_days)
    actual_prices_inv = y_scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).reshape(-1, k_days)

    # Calculate performance metrics
    mae = mean_absolute_error(actual_prices_inv.flatten(), predicted_prices_inv.flatten())
    mse = mean_squared_error(actual_prices_inv.flatten(), predicted_prices_inv.flatten())
    rmse = np.sqrt(mse)

    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")

    return predicted_prices_inv, actual_prices_inv, model

# Example usage of multivariate prediction
k_days = 5  # Number of future days to predict
predictions, actual, model = multivariate_prediction(
    k_days=k_days,
    company=COMPANY,
    train_start=TRAIN_START,
    train_end=TRAIN_END,
    prediction_days=PREDICTION_DAYS
)

# Display predictions vs actual
print(f"Predictions for the next {k_days} days: {predictions}")
print(f"Actual closing prices: {actual}")

# Plot the Test Predictions
plot_prediction(actual, predictions, ticker=COMPANY, smooth=True, sigma=2)
