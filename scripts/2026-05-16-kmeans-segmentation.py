from sklearn.cluster import KMeans
import pandas as pd

def segment_customers(df, n_clusters=4):
    """Applies K-Means clustering for behavioral customer segmentation."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df[['frequency', 'monetary_value']])
    return df