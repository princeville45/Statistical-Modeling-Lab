import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_revenue_data(months=24):
    """Generates synthetic revenue data with trend and seasonality."""
    np.random.seed(42)
    dates = [datetime(2024, 1, 1) + timedelta(days=30*i) for i in range(months)]
    time_index = np.arange(months)
    
    # Components: Trend + Seasonality (sine wave) + Noise
    trend = 5000 + (200 * time_index)
    seasonality = 1000 * np.sin(2 * np.pi * time_index / 12)
    noise = np.random.normal(0, 300, months)
    
    revenue = trend + seasonality + noise
    return pd.Series(revenue, index=dates)

def simple_linear_regression_forecast(data, steps=3):
    """Fits a trend line and predicts future steps."""
    y = data.values
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1)
    
    future_x = np.arange(len(y), len(y) + steps)
    forecast = slope * future_x + intercept
    return forecast

def moving_average_forecast(data, window=3, steps=3):
    """Calculates moving average and extends it."""
    ma_value = data.rolling(window=window).mean().iloc[-1]
    return np.full(steps, ma_value)

def exponential_smoothing(data, alpha=0.3, steps=3):
    """Simple Exponential Smoothing implementation."""
    y = data.values
    result = [y[0]]
    for n in range(1, len(y)):
        result.append(alpha * y[n] + (1 - alpha) * result[n-1])
    
    forecast_val = result[-1]
    return np.full(steps, forecast_val)

def calculate_errors(actual, predicted):
    """Calculates MAE and RMSE."""
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted)**2))
    return mae, rmse

if __name__ == "__main__":
    rev_data = generate_revenue_data()
    print("--- Revenue Forecasting Report ---")
    print(f"Last 6 months revenue:\n{rev_data.tail(6).to_string()}\n")
    
    # Forecasts
    steps = 3
    lr_f = simple_linear_regression_forecast(rev_data, steps)
    ma_f = moving_average_forecast(rev_data, window=3, steps=steps)
    es_f = exponential_smoothing(rev_data, alpha=0.4, steps=steps)
    
    print(f"Next {steps} Months Prediction:")
    print(f"Linear Regression: {lr_f}")
    print(f"Moving Average:    {ma_f}")
    print(f"Exp. Smoothing:    {es_f}")
