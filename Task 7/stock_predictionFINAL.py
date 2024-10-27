import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler

# Import functions from data_processing and visualization
from data_processing import (
    load_and_process_data_with_gap,
    create_dl_model,
    create_backpropagation_model,
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
    scale_data=False,
    save_local=True,
    load_local=True,
    local_dir='stock_data',
    feature_columns=None,
    target_column='Close'
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

# Build the DL model
input_shape_dl = (PREDICTION_DAYS, X_train_scaled.shape[2])
layer_types = ['LSTM', 'GRU', 'Dense']
layer_sizes = [150, 100, 50]
dropout_rates = [0.3, 0.3, 0.2]
return_sequences = [True, False, False]
activation_functions = ['tanh', 'tanh', 'relu']

dl_model = create_dl_model(
    input_shape=input_shape_dl,
    layer_types=layer_types,
    layer_sizes=layer_sizes,
    dropout_rates=dropout_rates,
    output_size=FUTURE_STEPS,
    loss_function='huber',
    optimizer='adam',
    return_sequences=return_sequences,
    activation_functions=activation_functions
)
dl_model.summary()

# Define early stopping and checkpoint for the DL model
checkpoint_dl = ModelCheckpoint('best_dl_model.h5', save_best_only=True, monitor='val_loss')
early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

# Train the DL model
history_dl = dl_model.fit(
    X_train_scaled, y_train_scaled,
    epochs=100, batch_size=32, verbose=1,
    validation_split=0.2, callbacks=[early_stopping, checkpoint_dl]
)

# Predict with the DL model
dl_model.load_weights('best_dl_model.h5')
predicted_dl = dl_model.predict(X_test_scaled)

# Reshape and inverse transform DL predictions
predicted_dl_flat = predicted_dl.reshape(-1, 1)
predicted_dl_inv = y_scaler.inverse_transform(predicted_dl_flat).reshape(-1, FUTURE_STEPS)

# Define input shape and parameters for the backpropagation model
input_shape_bp = (PREDICTION_DAYS, X_train_scaled.shape[2])
hidden_layers = 3
hidden_units = [100, 50, 25]
activation_functions_bp = ['relu', 'relu', 'relu']

# Create and compile the backpropagation model
backpropagation_model = create_backpropagation_model(
    input_shape=input_shape_bp,
    hidden_layers=hidden_layers,
    hidden_units=hidden_units,
    activation_functions=activation_functions_bp,
    output_size=FUTURE_STEPS,
    loss_function='huber',
    optimizer='adam'
)

# Define early stopping and checkpoint for the backpropagation model
checkpoint_bp = ModelCheckpoint('best_bp_model.h5', save_best_only=True, monitor='val_loss')

# Train the backpropagation model
history_bp = backpropagation_model.fit(
    X_train_scaled, y_train_scaled,
    epochs=100, batch_size=32, verbose=1,
    validation_split=0.2, callbacks=[early_stopping, checkpoint_bp]
)

# Predict with the backpropagation model
backpropagation_model.load_weights('best_bp_model.h5')
predicted_bp = backpropagation_model.predict(X_test_scaled)

# Reshape and inverse transform backpropagation predictions
predicted_bp_flat = predicted_bp.reshape(-1, 1)
predicted_bp_inv = y_scaler.inverse_transform(predicted_bp_flat).reshape(-1, FUTURE_STEPS)

# Prepare data for SARIMA
train_series = pd.Series(y_train.flatten(), index=pd.date_range(start=TRAIN_START, periods=len(y_train)))
test_series = pd.Series(y_test.flatten(), index=pd.date_range(start=TRAIN_END, periods=len(y_test)))

from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Use pmdarima's auto_arima to find the best SARIMA parameters
auto_sarima_model = auto_arima(train_series, seasonal=True, m=12, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
best_order = auto_sarima_model.order
best_seasonal_order = auto_sarima_model.seasonal_order

# Fit the SARIMA model with the best parameters
sarima_model = SARIMAX(train_series, order=best_order, seasonal_order=best_seasonal_order)
sarima_result = sarima_model.fit(disp=False)

# Predict using the SARIMA model
sarima_predictions = sarima_result.predict(start=len(train_series), end=len(train_series) + len(test_series) - 1)
sarima_predictions = sarima_predictions.to_numpy()

# Inverse transform the actual test data
y_test_flat = y_test_scaled.reshape(-1, 1)
actual_prices_inv = y_scaler.inverse_transform(y_test_flat).reshape(-1, FUTURE_STEPS)

# Flatten for evaluation
actual_prices_flat = actual_prices_inv.flatten()

# Flatten all predictions
predicted_dl_flat = predicted_dl_inv.flatten()
predicted_bp_flat = predicted_bp_inv.flatten()
sarima_predictions_flat = sarima_predictions.flatten()

# Determine the minimum length for truncation
min_length = min(len(predicted_dl_flat), len(predicted_bp_flat), len(sarima_predictions_flat), len(actual_prices_flat))

# Truncate all predictions and actual values to the same length
predicted_dl_flat = predicted_dl_flat[:min_length]
predicted_bp_flat = predicted_bp_flat[:min_length]
sarima_predictions_flat = sarima_predictions_flat[:min_length]
actual_prices_flat = actual_prices_flat[:min_length]

# Ensure the length is divisible by FUTURE_STEPS for reshaping
total_steps = (min_length // FUTURE_STEPS) * FUTURE_STEPS

# Truncate to make the length divisible by FUTURE_STEPS
predicted_dl_flat = predicted_dl_flat[:total_steps]
predicted_bp_flat = predicted_bp_flat[:total_steps]
sarima_predictions_flat = sarima_predictions_flat[:total_steps]
actual_prices_flat = actual_prices_flat[:total_steps]

# Reshape for plotting
actual_prices_plot = actual_prices_flat.reshape(-1, FUTURE_STEPS)
ensemble_predictions_flat = (
    0.5 * predicted_dl_flat +
    0.2 * sarima_predictions_flat +
    0.3 * predicted_bp_flat
)
ensemble_predictions_plot = ensemble_predictions_flat.reshape(-1, FUTURE_STEPS)

# Plot the updated ensemble predictions using the plot_prediction function
plot_prediction(
    actual_prices=actual_prices_plot,
    predicted_prices=ensemble_predictions_plot,
    ticker=COMPANY,
    smooth=True,
    sigma=2
)
