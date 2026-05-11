import numpy as np

def multiple_linear_regression(X, y):
    """
    Fits a multiple linear regression model using the normal equation.
    X: Independent variables matrix (n_samples, n_features)
    y: Dependent variable vector (n_samples,)
    """
    X_b = np.c_[np.ones((len(X), 1)), X]
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    y_pred = X_b.dot(theta_best)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    residuals = y - y_pred
    return theta_best, r_squared, residuals

if __name__ == "__main__":
    X = np.array([
        [100, 500, 1], [150, 700, 2], [130, 600, 3],
        [180, 800, 4], [200, 900, 5], [250, 1100, 6], [220, 1000, 7]
    ])
    y = np.array([1200, 1700, 1500, 2100, 2400, 3000, 2700])
    theta, r2, res = multiple_linear_regression(X, y)
    print("--- Multiple Linear Regression Model ---")
    print(f"Intercept: {theta[0]:.2f}")
    print(f"R-squared: {r2:.4f}")
