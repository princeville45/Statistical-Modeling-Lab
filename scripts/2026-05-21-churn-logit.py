import statsmodels.api as sm
import pandas as pd

def fit_churn_logit(df):
    """Fits a logistic regression model to predict customer churn."""
    # Features: tenure, monthly_charges, total_charges
    X = df[['tenure', 'monthly_charges', 'total_charges']]
    y = df['churned']
    X = sm.add_constant(X)
    model = sm.Logit(y, X).fit()
    return model.summary()