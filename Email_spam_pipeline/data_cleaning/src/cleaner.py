import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

def standardize_missing(df):

    missing_values = [
        "", " ", "NaN", "nan",
        "NULL", "null",
        "N/A", "n/a",
        "-", "######", "########"
    ]

    df.replace(missing_values, np.nan, inplace=True)

    return df
def handle_missing(df):

    for column in df.columns:

        if pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column].fillna(df[column].median())

        else:
            df[column] = df[column].fillna("Unknown")

    return df
def standardize_text(df):

    for column in df.select_dtypes(include="object"):

        df[column] = df[column].str.strip()
        df[column] = df[column].str.lower()

    return df
def fix_data_types(df):

    for column in df.columns:

        try:
            df[column] = pd.to_numeric(df[column])
        except:
            pass

    return df
def handle_outliers(df):

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    for col in numeric_cols:

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)

        IQR = Q3 - Q1
        
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df[col] = df[col].clip(lower, upper)

    return df