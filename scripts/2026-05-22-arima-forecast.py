from statsmodels.tsa.arima.model import ARIMA
import pandas as pd

def forecast_demand_arima(series, order=(5,1,0), steps=12):
    """Fits an ARIMA model and forecasts future demand values."""
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast