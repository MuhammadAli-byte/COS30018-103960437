import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, GRU, Attention, Input, Concatenate
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# Importing the functions you provided
from data_processing import load_and_process_data_with_gap, plot_candlestick, plot_boxplot

#------------------------------------------------------------------------------
# Load Data with Additional Features
#------------------------------------------------------------------------------
COMPANY = 'KO'  # Change the ticker symbol here
TRAIN_START = '2020-08-01'
TRAIN_END = '2024-08-31'

# Load and process data
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

# Plot candlestick chart
plot_candlestick(df, n_days=5, ticker=COMPANY)  # Adjust n_days as needed

# Plot boxplot of closing prices
plot_boxplot(df, window_size=10, step=5, ticker=COMPANY)

#------------------------------------------------------------------------------
# Prepare Data
#------------------------------------------------------------------------------
PREDICTION_DAYS = 60

# Print the original shape of the training data
print("Original shape of X_train:", X_train.shape)

# Number of features (columns)
n_features = X_train.shape[1]

# Define timesteps manually to ensure correct reshaping
n_timesteps = PREDICTION_DAYS
n_samples = X_train.shape[0] - n_timesteps + 1

# Reshape data for LSTM model
X_train_reshaped = np.array([X_train[i:i+n_timesteps] for i in range(n_samples)])
y_train_reshaped = y_train[n_timesteps-1:]

print("Reshaped X_train shape:", X_train_reshaped.shape)

# Repeat the process for X_test
n_samples_test = X_test.shape[0] - n_timesteps + 1
X_test_reshaped = np.array([X_test[i:i+n_timesteps] for i in range(n_samples_test)])
y_test_reshaped = y_test[n_timesteps-1:]

print("Reshaped X_test shape:", X_test_reshaped.shape)

#------------------------------------------------------------------------------
# Build the Enhanced Model with GRU and Attention
#------------------------------------------------------------------------------
# Define the input layer
inputs = Input(shape=(n_timesteps, n_features))

# Add a Bidirectional LSTM layer
x = Bidirectional(LSTM(units=150, return_sequences=True, kernel_regularizer=l2(0.001)))(inputs)
x = Dropout(0.3)(x)

# Add a GRU layer
x = GRU(units=100, return_sequences=True, kernel_regularizer=l2(0.001))(x)
x = Dropout(0.3)(x)

# Add Attention Layer
attention = Attention()([x, x])
x = Concatenate()([x, attention])

# Final GRU layer
x = GRU(units=50, return_sequences=False)(x)
x = Dropout(0.3)(x)

# Dense layers for final prediction
x = Dense(50, activation='relu')(x)
x = Dropout(0.2)(x)
outputs = Dense(1)(x)

# Create the model
model = Model(inputs, outputs)
model.compile(optimizer=Adam(learning_rate=0.0005), loss=Huber())

# Early stopping and checkpointing to save the best model
early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss')

# Train the model with validation and callbacks
history = model.fit(X_train_reshaped, y_train_reshaped, epochs=100, batch_size=32, verbose=1,
                    validation_split=0.2, callbacks=[early_stopping, checkpoint])

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
# Load the best model weights
model.load_weights('best_model.h5')

# Predict prices using the test set
predicted_prices = model.predict(X_test_reshaped)

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

#------------------------------------------------------------------------------
# Plot the Test Predictions
#------------------------------------------------------------------------------
def plot_prediction(actual_prices, predicted_prices, ticker='Stock', smooth=False, sigma=2):
    """
    Plots the actual and predicted stock prices with optional smoothing of the predicted line.

    Parameters:
    - actual_prices (np.ndarray): Actual stock prices.
    - predicted_prices (np.ndarray): Predicted stock prices by the model.
    - ticker (str): The stock ticker symbol used in the plot title.
    - smooth (bool): Whether to apply smoothing to the predicted prices for better visualization.
    - sigma (int): The sigma value for Gaussian smoothing.

    Returns:
    - None
    """
    # Optionally smooth the predicted prices for better visualization
    if smooth:
        from scipy.ndimage import gaussian_filter1d
        predicted_prices = gaussian_filter1d(predicted_prices.flatten(), sigma=sigma)

    plt.figure(figsize=(10, 6))
    plt.plot(actual_prices, color='black', linewidth=1.5, label=f'Actual {ticker} Price')
    plt.plot(predicted_prices, color='green', linestyle='--', linewidth=2, label=f'Predicted {ticker} Price')
    plt.title(f'{ticker} Share Price')
    plt.xlabel('Time')
    plt.ylabel(f'{ticker} Share Price')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Display the improved plot with optional smoothing
actual_prices = y_test_reshaped.reshape(-1, 1)
actual_prices = y_scaler.inverse_transform(actual_prices)
plot_prediction(actual_prices, predicted_prices, ticker=COMPANY, smooth=True, sigma=2)

#------------------------------------------------------------------------------
# Predict Next Day
#------------------------------------------------------------------------------
# Use the last PREDICTION_DAYS days from the test set to predict the next day's price
real_data = X_test_reshaped[-1].reshape((1, n_timesteps, n_features))

# Predict the next day's price
prediction = model.predict(real_data)
prediction = y_scaler.inverse_transform(prediction)
print(f"Next Day Prediction: {prediction[0][0]}")
