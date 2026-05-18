def estimate_avg_transaction_value(frequency, monetary_value):
    """Simplified Gamma-Gamma submodel to estimate expected average transaction value."""
    # Logic for expected value of the monetary value distribution
    # In practice, uses Bayesian estimation. This is a simplified proxy for RevOps.
    return round(monetary_value * (1 + (1 / frequency)), 2)