import pandas as pd

def profile_data(df):

    profile = {}
    profile["rows"] = df.shape[0]
    profile["columns"] = df.shape[1]

    profile["missing_values"] = df.isnull().sum().to_dict()

    profile["data_types"] = df.dtypes.astype(str).to_dict()

    profile["duplicate_rows"] = df.duplicated().sum()

    profile["unique_values"] = df.nunique().to_dict()

    return profile