import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def filter_and_transform_data(df, target_col, date_col='date', 
                             start_date=None, end_date=None, 
                             filter_cols=None, transform_method='none'):
    """
    Filter and transform data for modeling.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    target_col : str
        Target column for modeling
    date_col : str, optional
        Date column name
    start_date : str, optional
        Start date for filtering data (inclusive)
    end_date : str, optional
        End date for filtering data (inclusive)
    filter_cols : list, optional
        List of columns to keep (if None, keep all columns)
    transform_method : str, optional
        Method for transforming the target variable.
        Options: 'none', 'log', 'sqrt', 'box-cox', 'standardize', 'normalize'
    
    Returns:
    --------
    pandas.DataFrame
        Filtered and transformed DataFrame
    dict
        Transformation parameters for inverse transform
    """
    # Create a copy to avoid modifying the original
    df_processed = df.copy()
    
    # Ensure date column is datetime type
    if date_col in df_processed.columns:
        df_processed[date_col] = pd.to_datetime(df_processed[date_col])
    
    # Filter by date range if specified
    if start_date is not None:
        start_date = pd.to_datetime(start_date)
        df_processed = df_processed[df_processed[date_col] >= start_date]
    
    if end_date is not None:
        end_date = pd.to_datetime(end_date)
        df_processed = df_processed[df_processed[date_col] <= end_date]
    
    # Filter by columns if specified
    if filter_cols is not None:
        # Always include date_col and target_col
        cols_to_keep = filter_cols.copy()
        if date_col not in cols_to_keep:
            cols_to_keep.append(date_col)
        if target_col not in cols_to_keep:
            cols_to_keep.append(target_col)
        
        # Keep only the specified columns
        df_processed = df_processed[cols_to_keep]
    
    # Transform target variable if specified
    transform_params = {'method': transform_method}
    
    if transform_method == 'none':
        # No transformation
        pass
    
    elif transform_method == 'log':
        # Log transform (add small constant to handle zeros)
        min_val = df_processed[target_col].min()
        constant = 0 if min_val > 0 else abs(min_val) + 1
        transform_params['constant'] = constant
        
        df_processed[target_col] = np.log1p(df_processed[target_col] + constant)
    
    elif transform_method == 'sqrt':
        # Square root transform (handle negative values)
        min_val = df_processed[target_col].min()
        constant = 0 if min_val >= 0 else abs(min_val) + 1
        transform_params['constant'] = constant
        
        df_processed[target_col] = np.sqrt(df_processed[target_col] + constant)
    
    elif transform_method == 'box-cox':
        # Box-Cox transform (requires positive values)
        from scipy import stats
        
        min_val = df_processed[target_col].min()
        constant = 0 if min_val > 0 else abs(min_val) + 1
        transform_params['constant'] = constant
        
        transformed_data, lam = stats.boxcox(df_processed[target_col] + constant)
        transform_params['lambda'] = lam
        df_processed[target_col] = transformed_data
    
    elif transform_method == 'standardize':
        # Standardize (z-score)
        scaler = StandardScaler()
        df_processed[target_col] = scaler.fit_transform(df_processed[[target_col]])
        transform_params['mean'] = scaler.mean_[0]
        transform_params['std'] = scaler.scale_[0]
    
    elif transform_method == 'normalize':
        # Min-max scaling to [0, 1]
        scaler = MinMaxScaler()
        df_processed[target_col] = scaler.fit_transform(df_processed[[target_col]])
        transform_params['min'] = scaler.data_min_[0]
        transform_params['max'] = scaler.data_max_[0]
    
    else:
        logger.warning(f"Unknown transform method: {transform_method}. No transformation applied.")
    
    return df_processed, transform_params

def inverse_transform(values, transform_params):
    """
    Apply inverse transformation to values.
    
    Parameters:
    -----------
    values : numpy.ndarray or pandas.Series
        Values to inverse transform
    transform_params : dict
        Transformation parameters returned by filter_and_transform_data
    
    Returns:
    --------
    numpy.ndarray or pandas.Series
        Inverse transformed values
    """
    method = transform_params.get('method', 'none')
    
    if method == 'none':
        # No transformation
        return values
    
    elif method == 'log':
        # Inverse of log transform
        constant = transform_params.get('constant', 0)
        return np.expm1(values) - constant
    
    elif method == 'sqrt':
        # Inverse of square root transform
        constant = transform_params.get('constant', 0)
        return np.square(values) - constant
    
    elif method == 'box-cox':
        # Inverse of Box-Cox transform
        from scipy import special
        constant = transform_params.get('constant', 0)
        lam = transform_params.get('lambda', 1)
        
        if np.abs(lam) < 1e-8:
            inverse_values = np.exp(values)
        else:
            inverse_values = np.power(lam * values + 1, 1 / lam)
        
        return inverse_values - constant
    
    elif method == 'standardize':
        # Inverse of standardization
        mean = transform_params.get('mean', 0)
        std = transform_params.get('std', 1)
        return values * std + mean
    
    elif method == 'normalize':
        # Inverse of min-max scaling
        min_val = transform_params.get('min', 0)
        max_val = transform_params.get('max', 1)
        return values * (max_val - min_val) + min_val
    
    else:
        logger.warning(f"Unknown transform method: {method}. Returning values as is.")
        return values

def create_train_val_test_split(df, date_col='date', train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split data into train, validation, and test sets based on time.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    date_col : str, optional
        Date column name
    train_ratio : float, optional
        Ratio of data to use for training
    val_ratio : float, optional
        Ratio of data to use for validation
    test_ratio : float, optional
        Ratio of data to use for testing
    
    Returns:
    --------
    tuple
        (train_df, val_df, test_df)
    """
    # Ensure date column is datetime type
    df = df.copy()
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
    
    # Sort by date
    df.sort_values(by=date_col, inplace=True)
    
    # Calculate split indices
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    # Split data
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    return train_df, val_df, test_df

def aggregate_data(df, date_col='date', groupby_cols=None, agg_dict=None):
    """
    Aggregate data at different levels.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    date_col : str, optional
        Date column name
    groupby_cols : list, optional
        Columns to group by (including date_col)
    agg_dict : dict, optional
        Dictionary mapping columns to aggregation functions
    
    Returns:
    --------
    pandas.DataFrame
        Aggregated DataFrame
    """
    # Create a copy to avoid modifying the original
    df_agg = df.copy()
    
    # If no groupby columns specified, just return the original
    if groupby_cols is None:
        return df_agg
    
    # Ensure date_col is in groupby_cols
    if date_col not in groupby_cols:
        groupby_cols = [date_col] + groupby_cols
    
    # Default aggregation: sum for numeric, first for others
    if agg_dict is None:
        agg_dict = {}
        numeric_cols = df_agg.select_dtypes(include=['number']).columns
        for col in df_agg.columns:
            if col not in groupby_cols:
                if col in numeric_cols:
                    agg_dict[col] = 'sum'
                else:
                    agg_dict[col] = 'first'
    
    # Perform aggregation
    df_agg = df_agg.groupby(groupby_cols).agg(agg_dict).reset_index()
    
    return df_agg

def prepare_hierarchical_forecasting(df, date_col='date', level_cols=None, target_col='sales'):
    """
    Prepare data for hierarchical forecasting.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    date_col : str, optional
        Date column name
    level_cols : list, optional
        List of columns defining hierarchy levels, from top to bottom
    target_col : str, optional
        Target column to forecast
    
    Returns:
    --------
    dict
        Dictionary with aggregated DataFrames at each level
    """
    results = {}
    
    # If no level columns, treat as single level
    if level_cols is None or len(level_cols) == 0:
        results['all'] = df.copy()
        return results
    
    # Create aggregations at each level
    for i in range(len(level_cols) + 1):
        if i == 0:
            # Top level: aggregate across all levels
            agg_df = df.groupby(date_col)[target_col].sum().reset_index()
            level_name = 'all'
        else:
            # Intermediate level: aggregate by specified levels
            level_subset = level_cols[:i]
            groupby_cols = [date_col] + level_subset
            agg_df = df.groupby(groupby_cols)[target_col].sum().reset_index()
            level_name = '_'.join(level_subset)
        
        results[level_name] = agg_df
    
    # Add the original data as the most granular level
    level_name = '_'.join(level_cols) + '_full'
    results[level_name] = df.copy()
    
    return results 