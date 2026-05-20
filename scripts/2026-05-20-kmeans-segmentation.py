from sklearn.cluster import KMeans
import pandas as pd

def segment_customers(df, n_clusters=4):
    """Performs K-Means clustering for customer segmentation based on behavior."""
    # Features: recency, frequency, monetary
    features = df[['recency', 'frequency', 'monetary']]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['segment'] = kmeans.fit_predict(features)
    return df