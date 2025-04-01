import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler

# Importing the dataset
df = pd.read_csv('Palantir.csv', parse_dates=['Date'], sep=',', index_col='Date')
df_close = df['Close']

# Plotting the Close Price History
plt.figure(figsize=(16, 8))
plt.title('Palantir Close Price History')
plt.plot(df_close)
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.grid()
plt.show()


scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(np.array(df_close).reshape(-1, 1))


def create_dataset(data, time_step=3):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), 0]) 
        y.append(data[i + time_step, 0])  
    return np.array(X), np.array(y)


training_size = int(len(scaled_data) * 0.8)
test_size = len(scaled_data) - training_size
train_data, test_data = scaled_data[0:training_size], scaled_data[training_size:len(scaled_data)]

time_step = 3
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

model.summary()

model.fit(X_train, y_train, batch_size=1, epochs=100)

test_predict = model.predict(X_test)

test_predict_inverse = scaler.inverse_transform(test_predict)
y_test_inverse = scaler.inverse_transform(y_test.reshape(-1, 1))


plt.figure(figsize=(16, 8))
plt.title('Model Predictions vs Actual Prices')
plt.plot(test_predict_inverse, label='Predicted Close Price')
plt.plot(y_test_inverse, label='Actual Close Price')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.legend(loc='upper left')
plt.grid()
plt.show()

last_3_days = df_close[-3:].values
last_3_days_scaled = scaler.transform(last_3_days.reshape(-1, 1))
last_3_days_scaled = last_3_days_scaled.reshape(1, 3, 1)
next_day_prediction = model.predict(last_3_days_scaled)
next_day_prediction_inverse = scaler.inverse_transform(next_day_prediction)
print(f'Next day prediction: {next_day_prediction_inverse[0][0]}')

