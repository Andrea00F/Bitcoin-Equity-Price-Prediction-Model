# Bitcoin & Equity Price Prediction Model

A Python pipeline for forecasting asset prices, combining engineered 
financial features with multiple predictive modeling approaches.

## Overview
This project builds and compares statistical and machine learning models 
to predict short-term price movements for Bitcoin and equities, evaluating 
performance across different market conditions.

## Feature Engineering
- RSI (Relative Strength Index)
- ATR (Average True Range)
- Moving averages
- Momentum indicators

## Models Implemented
- Linear Regression
- ARIMA / ARIMAX
- RandomForestRegressor

Performance was compared across models to evaluate predictive accuracy 
under varying market conditions.

The performance of each model was evaluated by calculating the RMSE. The results are:


## Technical Notes
Resolved compatibility issues across the Python 3.14 data science stack 
(pandas, scikit-learn, statsmodels), involving debugging of dependency 
conflicts and version mismatches.

## Setup
```bash
pip install -r requirements.txt
python main.py
```

## Status
Ongoing project (2025–present).
