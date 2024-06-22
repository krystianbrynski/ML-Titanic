import pandas as pd


def read_data(data: str) -> pd.DataFrame:
    return pd.read_csv(data)
