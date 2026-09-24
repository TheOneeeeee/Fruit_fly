import pandas as pd

conn = pd.read_parquet("data/2025_Connectivity_783.parquet")
print(conn.columns.tolist())
print(conn.head())
print(f"\n{len(conn)} total synapse rows")