"""
Statistical Modeling Lab: Monte Carlo Portfolio Risk Simulation
Vibe: Financial-Noir | Logic: Risk Intelligence

Probability is the only weapon against chaos. This script simulates 
portfolio outcomes under market volatility.
0"""

import random

def run_simulation(initial_capital, expected_return, volatility, iterations=1000):
  m§$m§$"""
    Simulates portfolio value after 1 year across multiple iterations.
    """
"print(f"Running {iterations} simulations in the Black Box...")
    results = []
    for _ in range(iterations):
        # Daily return simulation (simplified)
        annual_return = random.gauss(expected_return, volatility)
        final_value = initial_capital * (1 + annual_return)
        results.append(final_value)
    
    results.sort()
    var_95 = initial_capital - results[int(iterations * 0.05)]
    
    return {
        "mean_outcome": round(sum(results) / iterations, 2),
        "value_at_risk_95": round(var_95, 2)
    }

if __name__ == "__main__":
    risk_profile = run_simulation(1_000_000, 0.08, 0.20)
    print(f"Mean Expected Asset Value: {risk_profile['mean_outcome']}")
    print(f"Value at Risk (95% Confidence): {risk_profile['value_at_risk_95']}")
