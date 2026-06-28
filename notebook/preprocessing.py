import numpy as np

def preprocess_data(df):
    # Convert the right skewed prize column into a Normal Distribution curve
    df['price'] = np.log1p(df['price']) # Saw EDA.ipynb file for visual

    # drop Unnecessar Column
    # df.drop(columns=['hotwaterheating'], inplace=True)
    return df