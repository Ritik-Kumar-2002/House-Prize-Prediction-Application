
from sklearn.preprocessing import OrdinalEncoder

def feature_encoding(df):
    binary_col = df.select_dtypes("object").drop(columns=['furnishingstatus']).columns
    # ordinal_col = ['furnishingstatus']


    # Binary Encoding
    binaryEncoder = OrdinalEncoder(categories=[['no', 'yes']]*len(binary_col))
    df[binary_col] = binaryEncoder.fit_transform(df[binary_col])

    # One Hot Encoding
    furnishingEncoder = OrdinalEncoder(categories=[['unfurnished',
                           'semi-furnished',
                           'furnished']])


    df[['furnishingstatus']] = furnishingEncoder.fit_transform(df[['furnishingstatus']])
    
    # Show dataset
    # print('Feature Scaling call: \n',df)

    return df

