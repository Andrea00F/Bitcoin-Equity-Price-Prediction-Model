# BTC stock price preditcion using RandomForestRegression with technical indicators (RSI, ATR, moving averages, momentum, ROC),
# plus Monte Carlo simulation to estimate a range of likely outcomes instead of a single point forecast.
# Test size = 0.1

import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 1. DOWNLOAD BITCOIN DATA
ticker = "BTC-USD"
data = yf.download(ticker, start="2020-01-01", progress=False)

dates = data.index
data = data.reset_index(drop=True)

print(f"Data downloaded for {ticker}")
print(f"Total days: {len(data)}\n")


# 2. CALCULATE RSI
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = np.where(loss != 0, gain / loss, 0)
    rsi = 100 - (100 / (1 + rs))

    # ✅ Flatten to convert from 2D to 1D
    return pd.Series(rsi.flatten(), index=prices.index)


# 3. CALCULATE ATR
def calculate_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


# 4. ADD FEATURES
print("Creating features...")
data['Volume_scaled'] = data['Volume'] / data['Volume'].max()
data['MA_7'] = data['Close'].rolling(window=7).mean()
data['MA_21'] = data['Close'].rolling(window=21).mean()
data['RSI'] = calculate_rsi(data['Close'], period=14)
data['Volatility'] = data['Close'].rolling(window=14).std()
data['ATR'] = calculate_atr(data['High'], data['Low'], data['Close'], period=14)
data['Momentum'] = data['Close'].pct_change() * 100
data['ROC'] = ((data['Close'] - data['Close'].shift(10)) / data['Close'].shift(10)) * 100

data = data.dropna()

print("Features added ✅\n")

# 5. PREPARE DATA FOR 7 DAYS
window = 5
prediction_days = 7  # ✅ Predict 7 days

prices = data['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler()
prices_scaled = scaler.fit_transform(prices)

X = []
y = []

# ✅ OPTION 1: Predict only the 7th day (easier)
print(f"Creating windows (prediction: {prediction_days} days)...")
for i in range(len(prices_scaled) - window - prediction_days + 1):
    price_window = prices_scaled[i:i + window].flatten()

    features = data[[
        'Volume_scaled', 'MA_7', 'MA_21', 'RSI',
        'Volatility', 'ATR', 'Momentum', 'ROC'
    ]].iloc[i + window - 1].values

    combined = np.concatenate([price_window, features])
    X.append(combined)

    # Price 7 days later
    y.append(prices_scaled[i + window + prediction_days - 1])

X = np.array(X)
y = np.array(y).ravel()

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}\n")

# 6. SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, shuffle=False
)

print(f"Train: {len(X_train)} samples")
print(f"Test: {len(X_test)} samples\n")

# 7. TRAIN
print("Training RandomForest (7 days)...")
model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42)
model.fit(X_train, y_train)

# 8. EVALUATE
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_r2 = model.score(X_train, y_train)
test_r2 = model.score(X_test, y_test)

print(f"\n" + "=" * 60)
print("MODEL RESULTS (7-day Forecast)")
print("=" * 60)
print(f"Train RMSE: {train_rmse:.6f}")
print(f"Test RMSE: {test_rmse:.6f}")
print(f"Train R²: {train_r2:.4f}")
print(f"Test R²: {test_r2:.4f}")

# 9. 7-DAY FORECAST FROM TODAY
last_5_prices = prices_scaled[-5:].flatten()

latest_features = data[[
    'Volume_scaled', 'MA_7', 'MA_21', 'RSI',
    'Volatility', 'ATR', 'Momentum', 'ROC'
]].iloc[-1].values

tomorrow_features = np.concatenate([last_5_prices, latest_features])
prediction_scaled = model.predict([tomorrow_features])[0]

# Denormalize
prediction_real = scaler.inverse_transform([[prediction_scaled]])[0][0]
today_price = data['Close'].iloc[-1].item()

percentage_change = ((prediction_real - today_price) / today_price) * 100

print(f"\n" + "=" * 60)
print("🎯 7-DAY FORECAST (BITCOIN)")
print("=" * 60)
print(f"Today's price: ${today_price:,.2f}")
print(f"Forecast in 7 days: ${prediction_real:,.2f}")
print(f"Expected change: ${prediction_real - today_price:+,.2f} ({percentage_change:+.2f}%)")
print("=" * 60)

# Comparison
actual_tomorrow_price = data['Close'].iloc[-1].item()
print(f"\nInterpretation:")
if percentage_change > 0:
    print(f"⬆️  Increase: the price should rise by {abs(percentage_change):.2f}%")
else:
    print(f"⬇️  Decrease: the price should fall by {abs(percentage_change):.2f}%")

import matplotlib.pyplot as plt

# 10. MONTE CARLO SIMULATION
print(f"\n" + "=" * 60)
print("🎲 MONTE CARLO SIMULATION (1000 scenarios)")
print("=" * 60)

# Number of simulations
n_simulations = 1000

# Calculate the model's historical error
residuals = y_test - y_test_pred  # Model errors
error_std = np.std(residuals)  # Standard deviation of errors

print(f"\nModel statistics:")
print(f"  • Mean error: {np.mean(residuals):.6f}")
print(f"  • Standard deviation of errors: {error_std:.6f}")

# Base prediction (what the model normally predicts)
base_prediction = model.predict([tomorrow_features])[0]

# Generate 1000 predictions by adding noise (random errors)
simulations = []

print(f"\nGenerating {n_simulations} scenarios...")
for i in range(n_simulations):
    # Add random noise based on the error distribution
    noise = np.random.normal(0, error_std)  # Mean=0, Std.Dev=error_std

    simulated_prediction = base_prediction + noise
    simulations.append(simulated_prediction)

simulations = np.array(simulations)

# Denormalize all simulations
simulations_real = scaler.inverse_transform(simulations.reshape(-1, 1)).flatten()

# Simulation statistics
sim_mean = np.mean(simulations_real)
sim_median = np.median(simulations_real)
sim_std = np.std(simulations_real)
sim_min = np.min(simulations_real)
sim_max = np.max(simulations_real)

# Percentiles (confidence intervals)
percentile_5 = np.percentile(simulations_real, 5)
percentile_25 = np.percentile(simulations_real, 25)
percentile_50 = np.percentile(simulations_real, 50)
percentile_75 = np.percentile(simulations_real, 75)
percentile_95 = np.percentile(simulations_real, 95)

print(f"\n" + "=" * 60)
print("📊 MONTE CARLO RESULTS")
print("=" * 60)
print(f"\nENTRY PRICE (today): ${today_price:,.2f}")
print(f"\nPESSIMISTIC SCENARIO (5th percentile): ${percentile_5:,.2f}")
print(f"  → Change: {((percentile_5 - today_price) / today_price) * 100:+.2f}%")

print(f"\nBEARISH SCENARIO (25th percentile): ${percentile_25:,.2f}")
print(f"  → Change: {((percentile_25 - today_price) / today_price) * 100:+.2f}%")

print(f"\nMEDIAN SCENARIO (50th percentile): ${percentile_50:,.2f}")
print(f"  → Change: {((percentile_50 - today_price) / today_price) * 100:+.2f}%")

print(f"\nBULLISH SCENARIO (75th percentile): ${percentile_75:,.2f}")
print(f"  → Change: {((percentile_75 - today_price) / today_price) * 100:+.2f}%")

print(f"\nOPTIMISTIC SCENARIO (95th percentile): ${percentile_95:,.2f}")
print(f"  → Change: {((percentile_95 - today_price) / today_price) * 100:+.2f}%")

print(f"\nMean of all simulations: ${sim_mean:,.2f}")
print(f"Standard deviation: ${sim_std:,.2f}")
print(f"Range: ${sim_min:,.2f} - ${sim_max:,.2f}")

# Profit probability
prob_profit = np.sum(simulations_real > today_price) / n_simulations * 100
prob_loss = np.sum(simulations_real < today_price) / n_simulations * 100

print(f"\n" + "=" * 60)
print("💰 PROFIT/LOSS PROBABILITY")
print("=" * 60)
print(f"Probability of PROFIT (price > ${today_price:,.2f}): {prob_profit:.1f}%")
print(f"Probability of LOSS (price < ${today_price:,.2f}): {prob_loss:.1f}%")

# Simulation graph
plt.figure(figsize=(14, 6))

# Histogram
plt.hist(simulations_real, bins=50, alpha=0.7, color='blue', edgecolor='black')
plt.axvline(today_price, color='red', linestyle='--', linewidth=2, label=f'Entry: ${today_price:,.0f}')
plt.axvline(sim_mean, color='green', linestyle='-', linewidth=2, label=f'Mean: ${sim_mean:,.0f}')
plt.axvline(percentile_5, color='orange', linestyle=':', linewidth=2, label=f'5th percentile: ${percentile_5:,.0f}')
plt.axvline(percentile_95, color='purple', linestyle=':', linewidth=2, label=f'95th percentile: ${percentile_95:,.0f}')

plt.title('Monte Carlo Simulation - Price distribution in 7 days', fontsize=14, fontweight='bold')
plt.xlabel('BTC Price ($)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Graph 2: Percentiles (confidence intervals)
percentiles = [5, 25, 50, 75, 95]
prices = [percentile_5, percentile_25, percentile_50, percentile_75, percentile_95]
colors = ['red', 'orange', 'green', 'lightgreen', 'purple']

plt.figure(figsize=(12, 6))
plt.barh(percentiles, prices, color=colors, edgecolor='black', linewidth=2)
plt.axvline(today_price, color='blue', linestyle='--', linewidth=2, label='Entry price')
plt.xlabel('BTC Price ($)', fontsize=12)
plt.ylabel('Percentile', fontsize=12)
plt.title('Monte Carlo - Price Percentiles', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3, axis='x')
plt.tight_layout()
plt.show()
