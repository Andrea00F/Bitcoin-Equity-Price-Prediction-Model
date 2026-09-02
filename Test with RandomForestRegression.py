import yfinance as yf
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 1 DOWNLOAD DATA UP TO TODAY
ticker = "AAPL"
data = yf.download(ticker, start="2020-01-01", progress=False)

print(f"Data downloaded for {ticker}")
print(f"Total days: {len(data)}")
print(f"Last available day: {data.index[-1].date()}\n")

# 2 PREPARE THE DATA
prices = data["Close"].values.reshape(-1,1)
scaler = MinMaxScaler()
prices_scaled = scaler.fit_transform(prices)

X = []
y = []
window = 5

for i in range(len(prices_scaled) - window):
    X.append(prices_scaled[i:i+window].flatten())
    y.append(prices_scaled[i+window])

X = np.array(X)
y = np.array(y).ravel()

# 3 SPLIT THE DATA INTO TRAIN AND TEST
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)

# 4 TRAIN THE MODEL
model = RandomForestRegressor()
model.fit(X_train, y_train)

# 5 PREDICT
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# EVALUATE THE MODEL
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
print(f"Test RMSE: {train_rmse:.6f}")
# Test RMSE: 0.004765


# 6 FORECAST FOR TOMORROW
print("FORECAST FOR TOMORROW")
print("="*60)

# take the last 5 days
last_5_days = prices_scaled[-5:].flatten()

# make the prediction
tomorrow_prediction_scaled = model.predict([last_5_days]).reshape(-1,1)

# denormalize
tomorrow_prediction_real = scaler.inverse_transform(tomorrow_prediction_scaled)[0][0]

# today's price
today_price = data["Close"].iloc[-1].item()

# change
percentage_change = ((tomorrow_prediction_real - today_price) / today_price) *100

print(f"Today's price : ${today_price:.2f}")
print(f"Forecast for tomorrow : ${tomorrow_prediction_real:.2f}")
print(f"Expected change : ${tomorrow_prediction_real - today_price:.2f}")
print(f"Percentage change : {percentage_change:.2f} %")