import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. DOWNLOAD THE DATA (UP TO TODAY)
ticker = "BTC-USD"
data = yf.download(ticker, start="2019-01-01", progress=False)

print(f"Data downloaded for {ticker}")
print(f"Total days: {len(data)}")
print(f"Last available day: {data.index[-1].date()}\n")

# 2. DISPLAY HISTORICAL PRICES
plt.figure(figsize=(12, 5))
plt.plot(data.index, data['Close'])
plt.title(f'Historical price of {ticker}')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.grid()
plt.show()

# 3. PREPARE THE DATA
prices = data['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler()
prices_scaled = scaler.fit_transform(prices)

X = [] 
y = []
window = 3

for i in range(len(prices_scaled) - window):
    X.append(prices_scaled[i:i+window].flatten())
    y.append(prices_scaled[i+window])

X = np.array(X)
y = np.array(y)

# 4. SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# 5. TRAIN MODEL
model = LinearRegression()
model.fit(X_train, y_train)

# 6. PREDICT
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# 7. EVALUATE
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"\nMODEL RESULTS:")
print(f"Train RMSE: {train_rmse:.6f} | Test RMSE: {test_rmse:.6f}")
print(f"Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f}")
# Test RMSE: 0.016874

# 8. DISPLAY PREDICTIONS
plt.figure(figsize=(14, 6))
train_indices = range(len(y_train))
test_indices = range(len(y_train), len(y_train) + len(y_test))

plt.plot(train_indices, y_train, label='Actual (train)', alpha=0.7)
plt.plot(train_indices, y_train_pred, label='Predicted (train)', alpha=0.7)
plt.plot(test_indices, y_test, label='Actual (test)', alpha=0.7)
plt.plot(test_indices, y_test_pred, label='Predicted (test)', alpha=0.7, linestyle='--')

plt.title(f'Price prediction for {ticker}')
plt.xlabel('Time')
plt.ylabel('Normalized price')
plt.legend()
plt.grid()
plt.show()

# 9. FORECAST FOR TOMORROW 🎯
print("\n" + "="*60)
print("🎯 FORECAST FOR TOMORROW 🎯")
print("="*60)

# Take the last 3 days
last_3_days = prices_scaled[-3:].flatten()

# Make the prediction
tomorrow_prediction_scaled = model.predict([last_3_days])

# Denormalize to get the actual price
tomorrow_prediction_real = scaler.inverse_transform(tomorrow_prediction_scaled)[0][0]

# Today's price
today_price = data['Close'].iloc[-1].item()  # Added .item()

# Change
percentage_change = ((tomorrow_prediction_real - today_price) / today_price) * 100

print(f"\nToday's price ({data.index[-1].date()}): ${today_price:.2f}")
print(f"Forecast for tomorrow: ${tomorrow_prediction_real:.2f}")
print(f"Expected change: ${tomorrow_prediction_real - today_price:+.2f} ({percentage_change:+.2f}%)")
print("="*60)