import numpy as np
import matplotlib.pyplot as plt
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

# Display reshaped data shapes
print("Reshaped X_train shape:", X_train_reshaped.shape)
print("Reshaped y_train shape:", y_train_reshaped.shape)
print("Reshaped X_test shape:", X_test_reshaped.shape)
print("Reshaped y_test shape:", y_test_reshaped.shape)

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

# Build the model using the function
input_shape = (PREDICTION_DAYS, X_train_scaled.shape[2])
layer_types = ['LSTM', 'GRU', 'Dense']
layer_sizes = [150, 100, 50]
dropout_rates = [0.3, 0.3, 0.2]
output_size = FUTURE_STEPS
return_sequences = [True, False, False]
activation_functions = ['tanh', 'tanh', 'relu']

model = create_dl_model(
    input_shape=input_shape,
    layer_types=layer_types,
    layer_sizes=layer_sizes,
    dropout_rates=dropout_rates,
    output_size=output_size,
    loss_function='huber',
    optimizer='adam',
    return_sequences=return_sequences,
    activation_functions=activation_functions
)
model.summary()

# Train the Model
early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss')

history = model.fit(
    X_train_scaled, y_train_scaled,
    epochs=100, batch_size=32, verbose=1,
    validation_split=0.2, callbacks=[early_stopping, checkpoint]
)

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Test the Model Accuracy on Existing Data
model.load_weights('best_model.h5')

# Predict prices using the test set
predicted_prices = model.predict(X_test_scaled)

# Reshape to (-1, 1) for inverse transform
predicted_prices_flat = predicted_prices.reshape(-1, 1)
actual_prices_flat = y_test_scaled.reshape(-1, 1)

# Inverse transform
predicted_prices_inv = y_scaler.inverse_transform(predicted_prices_flat).reshape(-1, FUTURE_STEPS)
actual_prices_inv = y_scaler.inverse_transform(actual_prices_flat).reshape(-1, FUTURE_STEPS)

# Flatten for evaluation
predicted_prices_flat = predicted_prices_inv.flatten()
actual_prices_flat = actual_prices_inv.flatten()

# Evaluate Model Performance
mae = mean_absolute_error(actual_prices_flat, predicted_prices_flat)
mse = mean_squared_error(actual_prices_flat, predicted_prices_flat)
rmse = np.sqrt(mse)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Plot the Test Predictions
plot_prediction(actual_prices_inv, predicted_prices_inv, ticker=COMPANY, smooth=True, sigma=2)

# Predict Next Sequence of Days
feature_columns = df.columns.tolist()
# Adjust if you excluded the target column from features in load_and_process_data_with_gap
# feature_columns.remove('Close')

last_sequence = df[feature_columns].values[-PREDICTION_DAYS:]
last_sequence_scaled = X_scaler.transform(last_sequence).reshape(1, PREDICTION_DAYS, -1)

next_days_prediction = model.predict(last_sequence_scaled)
next_days_prediction = y_scaler.inverse_transform(next_days_prediction).flatten()

print(f"Next {FUTURE_STEPS} Days Prediction: {next_days_prediction}")
