import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_dataset(dataset_path, dataset_size=None):
    df = pd.read_csv(dataset_path)
    
    # If dataset_size is not specified, use the entire dataset
    if dataset_size is None:
        df_small = df
    else:
        # Ensure class distribution is preserved using stratified sampling
        df_small, _ = train_test_split(
            df, train_size=dataset_size, stratify=df['category'], random_state=42
        )
    
    # Split the sampled dataset into training and temporary sets (80% train, 20% temp)
    df_train, df_temp = train_test_split(
        df_small, test_size=0.2, stratify=df_small['category'], random_state=42
    )

    # Further split the temporary set evenly into validation and test sets (10% each)
    df_val, df_test = train_test_split(
        df_temp, test_size=0.5, stratify=None, random_state=42
    )

    return df_train, df_val, df_test