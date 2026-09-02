# AAPL price forecast using ARIMAX - ARIMA extended with exogenous variables (volume and moving averages) 
# to see whether external features improve on the plain ARIMA baseline.


import yfinance as yf
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# 1. DOWNLOAD DATA
ticker = "AAPL"
data = yf.download(ticker, start="2020-01-01", progress=False)

dates = data.index
data = data.reset_index(drop=True)

print(f"Data downloaded for {ticker}")
print(f"Total days: {len(data)}\n")

# 2. CREATE ADDITIONAL FEATURES
data['Volume_scaled'] = data['Volume'] / data['Volume'].max()  # Normalize volume
data['MA_5'] = data['Close'].rolling(window=5).mean()  # 5-day moving average
data['MA_10'] = data['Close'].rolling(window=10).mean()  # 10-day moving average
data['Volatility'] = data['Close'].rolling(window=5).std()  # Volatility

# Remove NaN (the first values do not have a moving average)
data = data.dropna()

print("Features added:")
print(f"  - Volume (normalized)")
print(f"  - 5-day Moving Average")
print(f"  - 10-day Moving Average")
print(f"  - Volatility (5 days)\n")

# 3. SPLIT DATA
train_size = int(len(data) * 0.9)
train_price = data['Close'][:train_size]
test_price = data['Close'][train_size:]

# Features for train and test
train_exog = data[['Volume_scaled', 'MA_5', 'MA_10', 'Volatility']][:train_size]
test_exog = data[['Volume_scaled', 'MA_5', 'MA_10', 'Volatility']][train_size:]

# 4. TRAIN ARIMAX
print("Training ARIMAX...")
model = ARIMA(train_price, order=(5, 1, 2), exog=train_exog)
results = model.fit()

print(results.summary())

# 5. PREDICT ON TEST
forecast_test = results.get_forecast(steps=len(test_price), exog=test_exog)
y_pred = forecast_test.predicted_mean.values

rmse_arimax = np.sqrt(mean_squared_error(test_price, y_pred))
print(f"\n Test RMSE (ARIMAX): {rmse_arimax:.6f}")

# 6. FORECAST FOR TOMORROW
# Create the feature values for tomorrow (using the latest values)
tomorrow_exog = pd.DataFrame({
    'Volume_scaled': [data['Volume_scaled'].iloc[-1]],
    'MA_5': [data['MA_5'].iloc[-1]],
    'MA_10': [data['MA_10'].iloc[-1]],
    'Volatility': [data['Volatility'].iloc[-1]]
})

# Reset the test_exog index
test_exog_reset = test_exog.reset_index(drop=True)

forecast_tomorrow = results.get_forecast(
    steps=len(test_price) + 1,
    exog=pd.concat([test_exog_reset, tomorrow_exog], ignore_index=True)
)
tomorrow_price = forecast_tomorrow.predicted_mean.values[-1]
today_price = data['Close'].iloc[-1]

percentage_change = ((tomorrow_price - today_price) / today_price) * 100

print(f"\n" + "="*60)
print("FORECAST FOR TOMORROW (ARIMAX)")
print("="*60)
print(f"Today's price: ${today_price:.2f}")
print(f"Forecast for tomorrow: ${tomorrow_price:.2f}")
print(f"Change: ${tomorrow_price - today_price:+.2f} ({percentage_change:+.2f}%)")
