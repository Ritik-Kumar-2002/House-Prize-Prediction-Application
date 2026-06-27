
import pandas as pd 

def load_dataset():
    path = 'dataset/raw/Housing.csv'
    return pd.read_csv(path)