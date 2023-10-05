import pandas as pd
from sklearn.model_selection import train_test_split

def format_column_names(df):
    df.columns = [col[0].lower() + ''.join(['_' + c.lower() if c.isupper() else c for c in col[1:]]) for col in df.columns]
    return df



def split_data(df, random_seed=42):
    # First, split the data into training (70%) and temp (30%)
    train, test = train_test_split(df, test_size=0.30, random_state=random_seed)

    # Then, split the temp data into validation (50%) and test (50%)
    val, test = train_test_split(test, test_size=0.50, random_state=random_seed)

    return train, val, test

def drop_outliers_iqr_all_columns(df, threshold=1.5):
    """
    Detect and drop outliers from all numeric columns in a DataFrame using the IQR method.

    Args:
    - df: The DataFrame to remove outliers from.
    - threshold: The threshold value to determine outliers (default is 1.5).

    Returns:
    - DataFrame: A new DataFrame with outliers removed from all numeric columns.
    """
    # Create a copy of the original DataFrame
    df_no_outliers = df.copy()
    
    # Loop through all numeric columns
    for column_name in df.select_dtypes(include=['number']).columns:
        # Calculate the IQR for the column
        Q1 = df[column_name].quantile(0.25)
        Q3 = df[column_name].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define the lower and upper bounds for outliers
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        # Filter the DataFrame to keep only non-outliers in the current column
        df_no_outliers = df_no_outliers[(df_no_outliers[column_name] >= lower_bound) & (df_no_outliers[column_name] <= upper_bound)]
    
    return df_no_outliers


