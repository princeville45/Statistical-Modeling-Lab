import numpy as np
from sklearn.linear_model import LinearRegression

def predict_conversion_rate(ad_spend, conversions):
    """Simple linear model to predict conversions based on ad spend."""
    model = LinearRegression()
    X = np.array(ad_spend).reshape(-1, 1)
    y = np.array(conversions)
    model.fit(X, y)
    return model.coef_[0], model.intercept_