from sklearn.ensemble import RandomForestRegressor
import pandas as pd

def train_lead_score_model(X, y):
    """Trains a Random Forest Regressor to predict lead conversion probability."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model