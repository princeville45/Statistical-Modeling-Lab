from statsmodels.tsa.arima.model import ARIMA

def evaluate_arima_model(data, order):
    """Calculates AIC for a specific ARIMA order to assist in grid search."""
    try:
        model = ARIMA(data, order=order)
        results = model.fit()
        return results.aic
    except:
        return float('inf')