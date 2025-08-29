import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

def load_and_split_dataset(dataset_path, dataset_size=None, min_intent_count=5, upsample=True, rare_intent_threshold=2):
    df = pd.read_csv(dataset_path)

    # Filter out very rare intents based on rare_intent_threshold
    intent_counts = df['intent'].value_counts()
    df = df[df['intent'].isin(intent_counts[intent_counts >= rare_intent_threshold].index)]

    # If a smaller dataset size is specified, sample it while keeping stratification
    if dataset_size is not None and dataset_size < len(df):
        df, _ = train_test_split(
            df,
            train_size=dataset_size,
            stratify=df['category'] + '_' + df['intent'],
            random_state=42
        )

    # Split the dataset into training (80%) and temporary (20%) sets
    df_train, df_temp = train_test_split(
        df,
        test_size=0.2,
        stratify=df['category'] + '_' + df['intent'],
        random_state=42
    )

    # Split the temporary set into validation (10%) and test (10%) sets
    df_val, df_test = train_test_split(
        df_temp,
        test_size=0.5,
        stratify=df_temp['category'] + '_' + df_temp['intent'],
        random_state=42
    )

    # Balance intent distribution within each split
    def balance_intents(df_split, min_intent_count=5, upsample=False, drop=True):
        balanced = []
        for category in df_split['category'].unique():
            df_cat = df_split[df_split['category'] == category]
            intent_counts = df_cat['intent'].value_counts()

            for intent, count in intent_counts.items():
                df_group = df_cat[df_cat['intent'] == intent]
                
                if count < min_intent_count:
                    if upsample:
                        df_group = resample(
                            df_group,
                            replace=True,
                            n_samples=min_intent_count,
                            random_state=42
                        )
                        balanced.append(df_group)
                    elif not drop:
                        # Keep small intents without upsampling
                        balanced.append(df_group)
                else:
                    balanced.append(df_group)

        if not balanced:
            return pd.DataFrame(columns=df_split.columns)

        return pd.concat(balanced).sample(frac=1, random_state=42).reset_index(drop=True)

    df_train = balance_intents(df_train, min_intent_count=min_intent_count, upsample=upsample)
    df_val   = balance_intents(df_val, min_intent_count=min_intent_count, upsample=False, drop=False)
    df_test  = balance_intents(df_test, min_intent_count=min_intent_count, upsample=False, drop=False)

    return df_train, df_val, df_test