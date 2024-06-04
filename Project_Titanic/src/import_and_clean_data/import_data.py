import pandas as pd

def write_data(titanic_df):
    print(titanic_df)

def read_data():
    titanic_df = pd.read_csv('../data/Titanic-Dataset.csv')
    write_data(titanic_df)
