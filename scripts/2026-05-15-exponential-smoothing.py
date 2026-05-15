def exponential_smoothing(series, alpha):
    """Performs simple exponential smoothing for time series forecasting."""
    result = [series[0]]
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])
    return result