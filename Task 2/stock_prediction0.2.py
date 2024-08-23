import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU
from tensorflow.keras.regularizers import l2, l1_l2
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import EarlyStopping
from load_and_process import load_and_process_data_with_gap

#------------------------------------------------------------------------------
# Load Data with Additional Features
#------------------------------------------------------------------------------
COMPANY = 'CBA.AX'
TRAIN_START = '2022-08-01'
TRAIN_END = '2024-08-23'

# Load and process data
X_train, X_test, y_train, y_test, scalers = load_and_process_data_with_gap(
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

#------------------------------------------------------------------------------
# Prepare Data
#------------------------------------------------------------------------------
PREDICTION_DAYS = 60

# Print the original shape of the training data
print("Original shape of X_train:", X_train.shape)

# Number of features (columns)
n_features = X_train.shape[1]

# Determine the number of samples and time steps
n_samples = X_train.shape[0]

# Calculate possible time steps based on the total number of elements
n_timesteps = X_train.size // (n_samples * n_features)

print(f"Calculated time steps: {n_timesteps}")

# Reshape data for LSTM model
X_train = np.reshape(X_train, (n_samples, n_timesteps, n_features))

print("Reshaped X_train shape:", X_train.shape)

# Repeat the process for X_test
print("Original shape of X_test:", X_test.shape)
n_samples_test = X_test.shape[0]
X_test = np.reshape(X_test, (n_samples_test, n_timesteps, n_features))
print("Reshaped X_test shape:", X_test.shape)

#------------------------------------------------------------------------------
# Build the Model
#------------------------------------------------------------------------------
# Initialize the Sequential model
model = Sequential()

# Add the first LSTM layer with dropout
model.add(LSTM(units=150, return_sequences=True, input_shape=(n_timesteps, n_features)))
model.add(Dropout(0.2))

# Add the second LSTM layer with dropout
model.add(LSTM(units=100, return_sequences=False))
model.add(Dropout(0.2))

# Add the output layer
model.add(Dense(units=1))

# Compile the model using Adam optimizer and Huber loss
model.compile(optimizer='adam', loss=Huber())

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with validation and early stopping
history = model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1, validation_split=0.2, callbacks=[early_stopping])

#------------------------------------------------------------------------------
# Plot the Training and Validation Loss
#------------------------------------------------------------------------------
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

#------------------------------------------------------------------------------
# Test the Model Accuracy on Existing Data
#------------------------------------------------------------------------------
# Predict prices using the test set
predicted_prices = model.predict(X_test)

# Print raw predictions before inverse_transform
print("Raw predictions before inverse_transform:", predicted_prices[:5])

# Assuming y_train/y_test were scaled separately with a specific scaler for y
y_scaler = MinMaxScaler()
y_train = y_train.reshape(-1, 1)
y_scaler.fit(y_train)

# Inverse transform the predictions to get actual predicted prices
predicted_prices = y_scaler.inverse_transform(predicted_prices)

# Print inverse-transformed predictions
print("Inverse-transformed predictions:", predicted_prices[:5])

# Display the actual and predicted prices for each day
for i in range(len(predicted_prices)):
    print(f"Day {i + 1}: Actual Price = {y_test[i]}, Predicted Price = {predicted_prices[i][0]}")

#------------------------------------------------------------------------------
# Plot the Test Predictions
#------------------------------------------------------------------------------
actual_prices = y_test.reshape(-1, 1)
actual_prices = y_scaler.inverse_transform(actual_prices)

plt.plot(actual_prices, color="black", label=f"Actual {COMPANY} Price")
plt.plot(predicted_prices, color="green", label=f"Predicted {COMPANY} Price")
plt.title(f"{COMPANY} Share Price")
plt.xlabel("Time")
plt.ylabel(f"{COMPANY} Share Price")
plt.legend()
plt.show()

#------------------------------------------------------------------------------
# Predict Next Day
#------------------------------------------------------------------------------
# Use the last PREDICTION_DAYS days from the test set to predict the next day's price
real_data = X_test[-PREDICTION_DAYS:]
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], n_features))

# Predict the next day's price
prediction = model.predict(real_data)
prediction = y_scaler.inverse_transform(prediction)
print(f"Next Day Prediction: {prediction[0][0]}")
