import pandas as pd
import numpy as np

def generate_customer_data(n=100):
    """Generates synthetic RFM data."""
    np.random.seed(42)
    data = {
        'customer_id': range(1, n+1),
        'recency': np.random.randint(1, 365, n),
        'frequency': np.random.randint(1, 50, n),
        'monetary': np.random.randint(100, 10000, n)
    }
    return pd.DataFrame(data)

def rfm_segmentation(df):
    """Implements RFM scoring and segmentation."""
    # Scoring 1-5 (5 is best)
    # Recency: lower is better
    df['R'] = pd.qcut(df['recency'], 5, labels=[5, 4, 3, 2, 1])
    # Frequency & Monetary: higher is better
    df['F'] = pd.qcut(df['frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
    df['M'] = pd.qcut(df['monetary'], 5, labels=[1, 2, 3, 4, 5])
    
    df['RFM_Score'] = df['R'].astype(str) + df['F'].astype(str) + df['M'].astype(str)
    
    def segment_name(row):
        score = int(row['R']) + int(row['F']) + int(row['M'])
        if score >= 13: return 'Champions'
        if score >= 10: return 'Loyal'
        if score >= 7: return 'At Risk'
        if score >= 5: return 'Lost'
        return 'Inactive'
    
    df['Segment'] = df.apply(segment_name, axis=1)
    return df

if __name__ == "__main__":
    customers = generate_customer_data(200)
    rfm_df = rfm_segmentation(customers)
    
    print("--- Customer Segmentation Report ---")
    summary = rfm_df.groupby('Segment').agg({
        'customer_id': 'count',
        'monetary': 'mean'
    }).rename(columns={'customer_id': 'Count', 'monetary': 'Avg Revenue'})
    
    print(summary.sort_values('Avg Revenue', ascending=False))
    
    top_segment = summary['Count'].idxmax()
    print(f"\nStrategic Insight: Priority segment is '{top_segment}' with {summary.loc[top_segment, 'Count']} customers.")
