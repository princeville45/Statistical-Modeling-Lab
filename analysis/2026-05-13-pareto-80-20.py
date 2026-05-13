import pandas as pd
def pareto_analysis(df, col="revenue"):
    df = df.sort_values(by=col, ascending=False)
    df["cum_pct"] = 100 * df[col].cumsum() / df[col].sum()
    return df[df["cum_pct"] <= 80]