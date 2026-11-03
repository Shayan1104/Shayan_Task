import pandas as pd
import chardet

def load_dataset(file_path):

    with open(file_path, "rb") as f:
        result = chardet.detect(f.read(100000))

    encoding = result["encoding"]

    print(f"Detected encoding: {encoding}")

    df = pd.read_csv(file_path, encoding=encoding, low_memory=False)

    return df