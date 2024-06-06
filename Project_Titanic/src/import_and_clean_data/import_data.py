import pandas as pd


def read_train_data():
    return pd.read_csv('../data/Titanic-Dataset.csv')

def read_test_data():
    return pd.read_csv('../data/tested.csv')
