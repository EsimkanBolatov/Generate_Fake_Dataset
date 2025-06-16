import pandas as pd

def load_parquet(path):
    return pd.read_parquet(path)

def load_csv(path):
    return pd.read_csv(path)

def save_parquet(df, path):
    df.to_parquet(path, index=False)

def save_csv(df, path):
    df.to_csv(path, index=False)
