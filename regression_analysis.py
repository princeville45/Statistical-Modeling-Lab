import numpy as np

class MultiLinearRegression:
    """Multiple Linear Regression from scratch using Matrix Algebra."""
    def __init__(self):
        self.beta = None
        self.r_squared = None

    def fit(self, X, y):
        # Add intercept column
        X = np.column_stack([np.ones(X.shape[0]), X])
        
        # Matrix Algebra: beta = (X'X)^-1 X'y
        xtx_inv = np.linalg.inv(X.T @ X)
        self.beta = xtx_inv @ X.T @ y
        
        # R-squared calculation
        y_pred = X @ self.beta
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        self.r_squared = 1 - (ss_res / ss_tot)
        
    def predict(self, X):
        X = np.column_stack([np.ones(X.shape[0]), X])
        return X @ self.beta

if __name__ == "__main__":
    # Synthetic Sales Data: [Marketing Spend, Competitor Price, Seasonality Index]
    # Revenue = 5000 + 5*Spend - 20*CompPrice + 1000*Season
    np.random.seed(42)
    X = np.random.rand(100, 3) * [2000, 500, 10]
    y = 5000 + 5*X[:,0] - 20*X[:,1] + 1000*X[:,2] + np.random.normal(0, 500, 100)
    
    model = MultiLinearRegression()
    model.fit(X, y)
    
    print("--- Regression Analysis: Revenue Drivers ---")
    print(f"R-Squared: {model.r_squared:.4f}")
    print(f"Intercept: {model.beta[0]:.2f}")
    print(f"Marketing Impact: {model.beta[1]:.2f} (Revenue increase per unit spend)")
    print(f"Competitor Impact: {model.beta[2]:.2f} (Revenue change per unit competitor price)")
    print(f"Seasonality Impact: {model.beta[3]:.2f}")
