# AAPL price forecast using ARIMA (AutoRegressive Integrated Moving Average).
# Unlike the tree/linear models, ARIMA models the time series purely from its own past values, without external features.
# Test size = 0.1

import yfinance as yf
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# 1 DOWNLOAD THE DATA
ticker = "AAPL"
data = yf.download(ticker, start="2020-01-01", progress=False)

dates = data.index
data = data.reset_index(drop=True)

print(f"Data downloaded for {ticker}")
print(f"Total days: {len(data)}")
print(f"Last day analyzed: {dates[-1].date()}\n")

# 2 SPLIT THE DATA
train_size = int(len(data)*0.9)
train_data = data["Close"][:train_size]
test_data = data["Close"][train_size:]

# 3. TRAIN ARIMA
print("Training ARIMA...\n")
model = ARIMA(train_data, order=(5, 1, 2))
results = model.fit()

# PREDICT ON TEST
forecast_test = results.get_forecast(steps=len(test_data))
y_pred = forecast_test.predicted_mean.values

rmse = np.sqrt(mean_squared_error(test_data, y_pred))
print(f"Test RMSE: {rmse:.6f}")

# 4 TOMORROW'S FORECAST
forecast_tomorrow = results.get_forecast(steps=len(test_data) + 1)
tomorrow_price = forecast_tomorrow.predicted_mean.values[-1]
today_price = data["Close"].iloc[-1].item()

percentage_change = ((tomorrow_price - today_price)/today_price) * 100

print("FORECAST FOR TOMORROW\n")
print(f"Today's price : ${today_price:.2f}")
print(f"Tomorrow's price : ${tomorrow_price:.2f}")
print(f"Price change : ${tomorrow_price - today_price:.2f}")
print(f"Percentage change : {percentage_change:.2f}%")
