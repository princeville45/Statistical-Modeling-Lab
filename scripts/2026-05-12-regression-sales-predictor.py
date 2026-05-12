"""
Multiple Linear Regression: Sales Performance Predictor
Author: Irem Victor Chinonso | Statistical Business Architect
Date: 2026-05-12
Repo: Statistical-Modeling-Lab

Builds a multiple linear regression model to predict
monthly sales revenue from operational variables.
Outputs coefficients, R-squared, residual diagnostics.
"""

import numpy as np
import pandas as pd
from datetime import datetime


def generate_sales_data(n=120):
    """Generate synthetic operational + revenue data."""
    np.random.seed(21)
    data = {
        "visits_per_day": np.random.randint(20, 80, n),
        "avg_transaction_ngn": np.random.normal(1800, 400, n).clip(500),
        "stock_fill_rate_pct": np.random.uniform(70, 100, n),
        "marketing_spend_ngn": np.random.uniform(0, 5000, n),
        "num_reps": np.random.randint(1, 5, n)
    }
    df = pd.DataFrame(data)

    # True relationship + noise
    df["monthly_revenue_ngn"] = (
        df["visits_per_day"] * df["avg_transaction_ngn"] * 0.4
        + df["stock_fill_rate_pct"] * 200
        + df["marketing_spend_ngn"] * 3.5
        + df["num_reps"] * 8000
        + np.random.normal(0, 15000, n)
    ).round(2)

    return df


def add_intercept(X):
    """Add intercept column to design matrix."""
    ones = np.ones((X.shape[0], 1))
    return np.hstack([ones, X])


def ols_fit(X, y):
    """Ordinary Least Squares via normal equation: β = (XᵀX)⁻¹Xᵀy"""
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return beta


def compute_r_squared(y, y_hat):
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - ss_res / ss_tot


def compute_adj_r_squared(r2, n, k):
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)


def compute_standard_errors(X, residuals):
    """Compute standard errors of coefficients."""
    n, k = X.shape
    sigma2 = np.sum(residuals ** 2) / (n - k)
    cov_matrix = sigma2 * np.linalg.pinv(X.T @ X)
    return np.sqrt(np.diag(cov_matrix))


def t_statistics(beta, se):
    return beta / se


def run_regression():
    print("=" * 60)
    print("MULTIPLE LINEAR REGRESSION: SALES PREDICTOR")
    print("Statistical Modeling Lab | Irem Victor Chinonso")
    print("=" * 60)

    df = generate_sales_data(120)
    feature_cols = ["visits_per_day", "avg_transaction_ngn",
                    "stock_fill_rate_pct", "marketing_spend_ngn", "num_reps"]
    target_col = "monthly_revenue_ngn"

    X_raw = df[feature_cols].values.astype(float)
    y = df[target_col].values.astype(float)

    # Normalize features
    X_mean = X_raw.mean(axis=0)
    X_std = X_raw.std(axis=0)
    X_scaled = (X_raw - X_mean) / X_std
    X = add_intercept(X_scaled)

    beta = ols_fit(X, y)
    y_hat = X @ beta
    residuals = y - y_hat

    r2 = compute_r_squared(y, y_hat)
    adj_r2 = compute_adj_r_squared(r2, len(y), len(feature_cols))
    se = compute_standard_errors(X, residuals)
    t_stats = t_statistics(beta, se)

    print(f"\nSample Size: {len(df)} observations")
    print(f"R-Squared:          {r2:.4f}")
    print(f"Adjusted R-Squared: {adj_r2:.4f}")
    print(f"RMSE:               ₦{np.sqrt(np.mean(residuals**2)):,.0f}")

    print("\n--- COEFFICIENT TABLE ---")
    labels = ["Intercept"] + feature_cols
    print(f"{'Variable':<30} {'Coefficient':>15} {'Std Error':>12} {'t-stat':>10}")
    print("-" * 70)
    for label, b, s, t in zip(labels, beta, se, t_stats):
        sig = "**" if abs(t) > 2.0 else ""
        print(f"{label:<30} {b:>15,.2f} {s:>12,.2f} {t:>10.3f} {sig}")

    print("\n--- RESIDUAL DIAGNOSTICS ---")
    print(f"Mean Residual:   ₦{residuals.mean():,.2f}  (should be ~0)")
    print(f"Max Residual:    ₦{residuals.max():,.0f}")
    print(f"Min Residual:    ₦{residuals.min():,.0f}")
    print(f"Residual StdDev: ₦{residuals.std():,.0f}")

    print("\n--- INTERPRETATION ---")
    for label, b in zip(feature_cols, beta[1:]):
        direction = "increases" if b > 0 else "decreases"
        print(f"  1 unit increase in {label} {direction} monthly revenue by ₦{abs(b):,.0f} (normalized)")

    print("\nRegression complete.")


if __name__ == "__main__":
    run_regression()
