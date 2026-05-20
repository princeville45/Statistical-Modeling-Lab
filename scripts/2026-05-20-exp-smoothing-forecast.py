from statsmodels.tsa.holtwinters import SimpleExpSmoothing

def forecast_sales_exp_smoothing(data, periods=3, smoothing_level=0.8):
    """Applies Simple Exponential Smoothing for short-term sales forecasting."""
    model = SimpleExpSmoothing(data).fit(smoothing_level=smoothing_level, optimized=False)
    forecast = model.forecast(periods)
    return forecast