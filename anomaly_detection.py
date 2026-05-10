import numpy as np
import pandas as pd

def generate_anomalous_data(n=100):
    np.random.seed(42)
    data = np.random.normal(1000, 50, n)
    # Inject anomalies
    data[10] = 2000 # High spike
    data[50] = 200  # Low drop
    data[85] = 1800 # High spike
    return pd.Series(data)

def detect_zscore(data, threshold=3):
    mean = np.mean(data)
    std = np.std(data)
    z_scores = (data - mean) / std
    return np.abs(z_scores) > threshold

def detect_iqr(data):
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return (data < lower_bound) | (data > upper_bound)

def detect_rolling_sigma(data, window=7, n_sigma=3):
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    upper = rolling_mean + (n_sigma * rolling_std)
    lower = rolling_mean - (n_sigma * rolling_std)
    return (data > upper) | (data < lower)

if __name__ == "__main__":
    revenue = generate_anomalous_data()
    
    z_anomalies = detect_zscore(revenue)
    iqr_anomalies = detect_iqr(revenue)
    rolling_anomalies = detect_rolling_sigma(revenue)
    
    print("--- Anomaly Detection Report ---")
    print(f"Z-Score detected: {revenue[z_anomalies].index.tolist()}")
    print(f"IQR detected:     {revenue[iqr_anomalies].index.tolist()}")
    print(f"Rolling Sigma:    {revenue[rolling_anomalies].index.tolist()}")
