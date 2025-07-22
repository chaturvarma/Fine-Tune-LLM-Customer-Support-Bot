import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_dataset(dataset_path, dataset_size=None):
    df = pd.read_csv(dataset_path)

    if dataset_size is None:
        df_small = df
    else:
        df_small, _ = train_test_split(
            df, train_size=dataset_size, stratify=df['category'], random_state=42
        )

    df_train, df_temp = train_test_split(
        df_small, test_size=0.2, stratify=df_small['category'], random_state=42
    )

    df_val, df_test = train_test_split(
        df_temp, test_size=0.5, stratify=None, random_state=42
    )

    return df_train, df_val, df_test