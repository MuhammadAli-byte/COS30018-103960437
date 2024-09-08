import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Import functions from data_processing and visualization
from data_processing import load_and_process_data_with_gap, create_dl_model, prepare_data_for_model
from visualisation import plot_candlestick, plot_boxplot, plot_prediction

# Load and Process Data
COMPANY = 'KO'  # Change the ticker symbol here
TRAIN_START = '2016-08-01'
TRAIN_END = '2024-08-31'
X_train, X_test, y_train, y_test, scalers, df = load_and_process_data_with_gap(
    ticker=COMPANY,
    start_date=TRAIN_START,
    end_date=TRAIN_END,
    handle_nan='drop',
    split_method='date',
    test_ratio=0.2,
    scale_data=True,
    save_local=True,
    load_local=True,
    local_dir='stock_data',
    feature_columns=None  # Use all available features
)

# Plot charts of Data
# Plot candlestick chart of closing prices
plot_candlestick(df, n_days=5, ticker=COMPANY)  # Adjust n_days as needed
# Plot boxplot of closing prices
plot_boxplot(df, window_size=10, step=5, ticker=COMPANY)


# Prepare Data
PREDICTION_DAYS = 60
# Reshape training and test data
X_train_reshaped, y_train_reshaped = prepare_data_for_model(X_train, y_train, PREDICTION_DAYS)
X_test_reshaped, y_test_reshaped = prepare_data_for_model(X_test, y_test, PREDICTION_DAYS)
# Display reshaped data shapes
print("Reshaped X_train shape:", X_train_reshaped.shape)
print("Reshaped X_test shape:", X_test_reshaped.shape)


# Build the model using the function
# Define the parameters for the custom model
input_shape = (PREDICTION_DAYS, X_train.shape[1])
layer_types = ['LSTM', 'GRU', 'Dense']
layer_sizes = [150, 100, 50]
dropout_rates = [0.3, 0.3, 0.2]
output_size = 1
return_sequences = [True, False, False]
activation_functions = ['tanh', 'tanh', 'relu']
# Create the model using the function
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
# Early stopping and checkpointing to save the best model
early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss')
# Train the model with validation and callbacks
history = model.fit(X_train_reshaped, y_train_reshaped, epochs=100, batch_size=32, verbose=1,
                    validation_split=0.2, callbacks=[early_stopping, checkpoint])
# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Test the Model Accuracy on Existing Data
# Load the best model weights
model.load_weights('best_model.h5')
# Predict prices using the test set
predicted_prices = model.predict(X_test_reshaped)
y_scaler = MinMaxScaler()
y_train = y_train.reshape(-1, 1)
y_scaler.fit(y_train)
# Inverse transform the predictions to get actual predicted prices
predicted_prices = y_scaler.inverse_transform(predicted_prices)
# Inverse transform actual test prices for comparison
actual_prices = y_test_reshaped.reshape(-1, 1)
actual_prices = y_scaler.inverse_transform(actual_prices)

# Evaluate Model Performance
# Calculate evaluation metrics
mae = mean_absolute_error(actual_prices, predicted_prices)
mse = mean_squared_error(actual_prices, predicted_prices)
rmse = np.sqrt(mse)
# Print evaluation results
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
# Plot the Test Predictions using the function from visualization.py
plot_prediction(actual_prices, predicted_prices, ticker=COMPANY, smooth=True, sigma=2)

# Predict Next Day
# Use the last PREDICTION_DAYS days from the test set to predict the next day's price
real_data = X_test_reshaped[-1].reshape((1, PREDICTION_DAYS, X_train.shape[1]))
# Predict the next day's price
next_day_prediction = model.predict(real_data)
next_day_prediction = y_scaler.inverse_transform(next_day_prediction)
print(f"Next Day Prediction: {next_day_prediction[0][0]}")
