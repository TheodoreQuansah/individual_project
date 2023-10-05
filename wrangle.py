import pandas as pd
from sklearn.preprocessing import MinMaxScaler



def minmax_scale(train, validate, test):
    """
    Apply Min-Max scaling to train, validate, and test DataFrames.

    Args:
    - train, validate, test: DataFrames containing the datasets to be scaled.

    Returns:
    - DataFrame, DataFrame, DataFrame: The min-max scaled train, validate, and test DataFrames.
    """
    # Define columns to be scaled (excluding non-numeric columns)
    numeric_cols = train.select_dtypes(include=['number']).columns.tolist()

    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()

    # Fit the scaler on the training data and transform all sets
    train[numeric_cols] = scaler.fit_transform(train[numeric_cols])
    validate[numeric_cols] = scaler.transform(validate[numeric_cols])
    test[numeric_cols] = scaler.transform(test[numeric_cols])

    return train, validate, test


def one_hot_encode_and_rename(train, validate, test, convert_bool_to_int=True):
    """
    Apply one-hot encoding to specified columns in train, validate, and test DataFrames 
    and remove "_Yes" suffix. Optionally, convert boolean columns to integers.

    Args:
    - train, validate, test: DataFrames containing the datasets to be encoded and renamed.
    - convert_bool_to_int: Boolean flag to indicate whether to convert boolean columns to integers.

    Returns:
    - DataFrame, DataFrame, DataFrame: The one-hot encoded and renamed train, validate, and test DataFrames.
    """
    # Define columns to encode
    categorical_columns = ['sex', 'race', 'age_category', 'gen_health']
    additional_categorical_columns = ['heart_disease', 'smoking', 'alcohol_drinking', 'stroke', 'diff_walking', 'diabetic', 'physical_activity', 'asthma', 'kidney_disease', 'skin_cancer']
    
    # Apply one-hot encoding to specified columns
    train_encoded = pd.get_dummies(train, columns=categorical_columns + additional_categorical_columns, drop_first=True)
    validate_encoded = pd.get_dummies(validate, columns=categorical_columns + additional_categorical_columns, drop_first=True)
    test_encoded = pd.get_dummies(test, columns=categorical_columns + additional_categorical_columns, drop_first=True)
    
    # Remove "_Yes" suffix from column names
    cleaned_feature_names = [name[:-4] if name.endswith('_Yes') else name for name in train_encoded.columns]
    train_encoded.columns = cleaned_feature_names
    validate_encoded.columns = cleaned_feature_names
    test_encoded.columns = cleaned_feature_names
    
    # Convert boolean columns to integers if specified
    if convert_bool_to_int:
        bool_columns = train_encoded.select_dtypes(include=['bool']).columns
        train_encoded[bool_columns] = train_encoded[bool_columns].astype(int)
        validate_encoded[bool_columns] = validate_encoded[bool_columns].astype(int)
        test_encoded[bool_columns] = test_encoded[bool_columns].astype(int)
    
    return train_encoded, validate_encoded, test_encoded




def data_split(df, target_column):
    """
    Split a DataFrame into features (X) and the target variable (y).

    Args:
    - df: DataFrame containing the dataset.
    - target_column: Name of the target variable.

    Returns:
    - DataFrame, Series: The features (X) and the target variable (y).
    """
    X = df.drop(columns=[target_column])
    y = pd.DataFrame(df[target_column])
    return X, y



