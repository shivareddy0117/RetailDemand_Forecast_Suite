#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data cleaning and preprocessing module for retail demand forecasting.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_datetime_columns(df, date_columns):
    """
    Convert columns to datetime format.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        date_columns (list): List of column names to convert
        
    Returns:
        pd.DataFrame: DataFrame with converted datetime columns
    """
    df_copy = df.copy()
    
    for col in date_columns:
        if col in df_copy.columns:
            df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
    
    return df_copy


def handle_missing_values(df, strategy='ffill'):
    """
    Handle missing values in the DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        strategy (str): Strategy to handle missing values
            - 'ffill': Forward fill
            - 'bfill': Backward fill
            - 'interpolate': Linear interpolation
            - 'mean': Replace with column mean
            - 'median': Replace with column median
            - 'zero': Replace with zeros
            - 'drop': Drop rows with missing values
        
    Returns:
        pd.DataFrame: DataFrame with handled missing values
    """
    df_copy = df.copy()
    
    # Count missing values before handling
    missing_before = df_copy.isnull().sum().sum()
    
    if strategy == 'ffill':
        df_copy = df_copy.fillna(method='ffill')
        # Fill any remaining NaNs at the beginning with backward fill
        df_copy = df_copy.fillna(method='bfill')
    
    elif strategy == 'bfill':
        df_copy = df_copy.fillna(method='bfill')
        # Fill any remaining NaNs at the end with forward fill
        df_copy = df_copy.fillna(method='ffill')
    
    elif strategy == 'interpolate':
        df_copy = df_copy.interpolate(method='linear', axis=0)
        # Fill any remaining NaNs at the edges
        df_copy = df_copy.fillna(method='ffill').fillna(method='bfill')
    
    elif strategy == 'mean':
        for col in df_copy.columns:
            if pd.api.types.is_numeric_dtype(df_copy[col]):
                df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
    
    elif strategy == 'median':
        for col in df_copy.columns:
            if pd.api.types.is_numeric_dtype(df_copy[col]):
                df_copy[col] = df_copy[col].fillna(df_copy[col].median())
    
    elif strategy == 'zero':
        df_copy = df_copy.fillna(0)
    
    elif strategy == 'drop':
        df_copy = df_copy.dropna()
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Count missing values after handling
    missing_after = df_copy.isnull().sum().sum()
    
    logger.info(f"Missing values: {missing_before} -> {missing_after}")
    
    return df_copy


def remove_outliers(df, columns, method='zscore', threshold=3.0):
    """
    Remove outliers from the DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of column names to check for outliers
        method (str): Method to detect outliers
            - 'zscore': Z-score method
            - 'iqr': Interquartile range method
        threshold (float): Threshold for outlier detection
            - For zscore: Number of standard deviations
            - For iqr: Multiplier for IQR
        
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    df_copy = df.copy()
    rows_before = len(df_copy)
    
    for col in columns:
        if col not in df_copy.columns:
            logger.warning(f"Column {col} not found in DataFrame")
            continue
        
        if not pd.api.types.is_numeric_dtype(df_copy[col]):
            logger.warning(f"Column {col} is not numeric, skipping")
            continue
        
        if method == 'zscore':
            # Z-score method
            z_scores = np.abs((df_copy[col] - df_copy[col].mean()) / df_copy[col].std())
            df_copy = df_copy[z_scores <= threshold]
        
        elif method == 'iqr':
            # IQR method
            Q1 = df_copy[col].quantile(0.25)
            Q3 = df_copy[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            df_copy = df_copy[(df_copy[col] >= lower_bound) & (df_copy[col] <= upper_bound)]
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    rows_after = len(df_copy)
    rows_removed = rows_before - rows_after
    
    logger.info(f"Outlier removal: {rows_removed} rows removed ({rows_removed/rows_before:.2%})")
    
    return df_copy


def resample_time_series(df, date_column, value_columns, freq='D', agg_func='sum'):
    """
    Resample time series data to a different frequency.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        date_column (str): Name of the date column
        value_columns (list): List of value column names
        freq (str): Frequency for resampling
            - 'D': Daily
            - 'W': Weekly
            - 'M': Monthly
            - 'Q': Quarterly
            - 'Y': Yearly
        agg_func (str or dict): Aggregation function(s)
            - 'sum': Sum values
            - 'mean': Mean of values
            - 'median': Median of values
            - 'min': Minimum value
            - 'max': Maximum value
            - dict: Dictionary mapping column names to aggregation functions
        
    Returns:
        pd.DataFrame: Resampled DataFrame
    """
    # Make a copy and ensure datetime format
    df_copy = df.copy()
    df_copy[date_column] = pd.to_datetime(df_copy[date_column])
    
    # Set date as index
    df_copy = df_copy.set_index(date_column)
    
    # Select only required columns
    if value_columns:
        df_copy = df_copy[value_columns]
    
    # Handle aggregation function
    if isinstance(agg_func, dict):
        # Check if all value columns are covered
        for col in value_columns:
            if col not in agg_func:
                logger.warning(f"No aggregation function specified for {col}, using 'sum'")
                agg_func[col] = 'sum'
        
        # Resample with different aggregation functions for each column
        resampled = df_copy.resample(freq).agg(agg_func)
    
    else:
        # Use the same aggregation function for all columns
        resampled = df_copy.resample(freq).agg(agg_func)
    
    logger.info(f"Resampled data from {len(df_copy)} rows to {len(resampled)} rows with frequency '{freq}'")
    
    return resampled


def fill_missing_dates(df, date_column, freq='D', value_columns=None, fill_method='ffill'):
    """
    Fill missing dates in time series data.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        date_column (str): Name of the date column
        freq (str): Frequency for resampling
        value_columns (list): List of value column names to fill
        fill_method (str): Method to fill missing values
            - 'ffill': Forward fill
            - 'bfill': Backward fill
            - 'interpolate': Linear interpolation
            - 'zero': Fill with zeros
        
    Returns:
        pd.DataFrame: DataFrame with filled missing dates
    """
    # Make a copy and ensure datetime format
    df_copy = df.copy()
    df_copy[date_column] = pd.to_datetime(df_copy[date_column])
    
    # Get min and max dates
    min_date = df_copy[date_column].min()
    max_date = df_copy[date_column].max()
    
    # Create a complete date range
    date_range = pd.date_range(start=min_date, end=max_date, freq=freq)
    
    # Create a template DataFrame with the complete date range
    template_df = pd.DataFrame({date_column: date_range})
    
    # Merge with the original data
    merged_df = pd.merge(template_df, df_copy, on=date_column, how='left')
    
    # Fill missing values
    if value_columns:
        for col in value_columns:
            if col in merged_df.columns:
                if fill_method == 'ffill':
                    merged_df[col] = merged_df[col].fillna(method='ffill')
                elif fill_method == 'bfill':
                    merged_df[col] = merged_df[col].fillna(method='bfill')
                elif fill_method == 'interpolate':
                    merged_df[col] = merged_df[col].interpolate(method='linear')
                elif fill_method == 'zero':
                    merged_df[col] = merged_df[col].fillna(0)
    else:
        if fill_method == 'ffill':
            merged_df = merged_df.fillna(method='ffill')
        elif fill_method == 'bfill':
            merged_df = merged_df.fillna(method='bfill')
        elif fill_method == 'interpolate':
            merged_df = merged_df.interpolate(method='linear')
        elif fill_method == 'zero':
            merged_df = merged_df.fillna(0)
    
    # Count added rows
    added_rows = len(merged_df) - len(df_copy)
    
    logger.info(f"Added {added_rows} missing dates to the data")
    
    return merged_df


def lag_features(df, columns, lags, drop_na=False):
    """
    Create lag features for time series columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame (with DatetimeIndex)
        columns (list): List of column names to create lags for
        lags (list): List of lag values
        drop_na (bool): Whether to drop rows with NaN values
        
    Returns:
        pd.DataFrame: DataFrame with lag features
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex")
    
    df_copy = df.copy()
    
    for col in columns:
        for lag in lags:
            lag_name = f"{col}_lag_{lag}"
            df_copy[lag_name] = df_copy[col].shift(lag)
    
    if drop_na:
        df_copy = df_copy.dropna()
    
    logger.info(f"Created {len(columns) * len(lags)} lag features")
    
    return df_copy


def rolling_features(df, columns, windows, functions=['mean'], drop_na=False):
    """
    Create rolling window features for time series columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame (with DatetimeIndex)
        columns (list): List of column names to create features for
        windows (list): List of window sizes
        functions (list): List of aggregation functions
            - 'mean': Rolling mean
            - 'std': Rolling standard deviation
            - 'min': Rolling minimum
            - 'max': Rolling maximum
            - 'median': Rolling median
            - 'sum': Rolling sum
        drop_na (bool): Whether to drop rows with NaN values
        
    Returns:
        pd.DataFrame: DataFrame with rolling features
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex")
    
    df_copy = df.copy()
    
    for col in columns:
        for window in windows:
            for func in functions:
                feature_name = f"{col}_rolling_{window}_{func}"
                
                if func == 'mean':
                    df_copy[feature_name] = df_copy[col].rolling(window=window).mean()
                elif func == 'std':
                    df_copy[feature_name] = df_copy[col].rolling(window=window).std()
                elif func == 'min':
                    df_copy[feature_name] = df_copy[col].rolling(window=window).min()
                elif func == 'max':
                    df_copy[feature_name] = df_copy[col].rolling(window=window).max()
                elif func == 'median':
                    df_copy[feature_name] = df_copy[col].rolling(window=window).median()
                elif func == 'sum':
                    df_copy[feature_name] = df_copy[col].rolling(window=window).sum()
                else:
                    logger.warning(f"Unknown function: {func}, skipping")
    
    if drop_na:
        df_copy = df_copy.dropna()
    
    logger.info(f"Created {len(columns) * len(windows) * len(functions)} rolling features")
    
    return df_copy


def add_date_features(df, date_column=None, drop_date=False):
    """
    Add date-based features to the DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        date_column (str): Name of the date column (not needed if index is DatetimeIndex)
        drop_date (bool): Whether to drop the original date column
        
    Returns:
        pd.DataFrame: DataFrame with date features
    """
    df_copy = df.copy()
    
    # If date_column is provided, use it; otherwise use the index
    if date_column:
        if date_column not in df_copy.columns:
            raise ValueError(f"Column {date_column} not found in DataFrame")
        
        dates = pd.to_datetime(df_copy[date_column])
    elif isinstance(df_copy.index, pd.DatetimeIndex):
        dates = df_copy.index
    else:
        raise ValueError("Either date_column must be provided or DataFrame index must be DatetimeIndex")
    
    # Extract date components
    df_copy['year'] = dates.year
    df_copy['month'] = dates.month
    df_copy['day'] = dates.day
    df_copy['day_of_week'] = dates.dayofweek
    df_copy['day_of_year'] = dates.dayofyear
    df_copy['week_of_year'] = dates.isocalendar().week
    df_copy['quarter'] = dates.quarter
    df_copy['is_weekend'] = dates.dayofweek.isin([5, 6]).astype(int)
    df_copy['is_month_start'] = dates.is_month_start.astype(int)
    df_copy['is_month_end'] = dates.is_month_end.astype(int)
    df_copy['is_quarter_start'] = dates.is_quarter_start.astype(int)
    df_copy['is_quarter_end'] = dates.is_quarter_end.astype(int)
    df_copy['is_year_start'] = dates.is_year_start.astype(int)
    df_copy['is_year_end'] = dates.is_year_end.astype(int)
    
    # Add cyclical features
    df_copy['month_sin'] = np.sin(2 * np.pi * dates.month / 12)
    df_copy['month_cos'] = np.cos(2 * np.pi * dates.month / 12)
    df_copy['day_sin'] = np.sin(2 * np.pi * dates.day / 31)
    df_copy['day_cos'] = np.cos(2 * np.pi * dates.day / 31)
    df_copy['day_of_week_sin'] = np.sin(2 * np.pi * dates.dayofweek / 7)
    df_copy['day_of_week_cos'] = np.cos(2 * np.pi * dates.dayofweek / 7)
    df_copy['day_of_year_sin'] = np.sin(2 * np.pi * dates.dayofyear / 366)
    df_copy['day_of_year_cos'] = np.cos(2 * np.pi * dates.dayofyear / 366)
    
    # Drop the original date column if requested
    if drop_date and date_column and date_column in df_copy.columns:
        df_copy = df_copy.drop(columns=[date_column])
    
    logger.info(f"Added 22 date-based features")
    
    return df_copy


def one_hot_encode_categories(df, columns):
    """
    One-hot encode categorical columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of categorical column names
        
    Returns:
        pd.DataFrame: DataFrame with one-hot encoded categories
    """
    df_copy = df.copy()
    
    for col in columns:
        if col not in df_copy.columns:
            logger.warning(f"Column {col} not found in DataFrame")
            continue
        
        # Create one-hot encoding
        dummies = pd.get_dummies(df_copy[col], prefix=col, dummy_na=False)
        
        # Add to DataFrame
        df_copy = pd.concat([df_copy, dummies], axis=1)
        
        # Drop original column
        df_copy = df_copy.drop(columns=[col])
    
    logger.info(f"One-hot encoded {len(columns)} categorical columns")
    
    return df_copy


def merge_external_data(main_df, external_df, date_column, how='left', fill_method='ffill'):
    """
    Merge external data with the main DataFrame based on dates.
    
    Args:
        main_df (pd.DataFrame): Main DataFrame
        external_df (pd.DataFrame): External data DataFrame
        date_column (str): Name of the date column in both DataFrames
        how (str): Merge method ('left', 'right', 'inner', 'outer')
        fill_method (str): Method to fill missing values after merge
        
    Returns:
        pd.DataFrame: Merged DataFrame
    """
    # Make copies and ensure datetime format
    main_copy = main_df.copy()
    external_copy = external_df.copy()
    
    main_copy[date_column] = pd.to_datetime(main_copy[date_column])
    external_copy[date_column] = pd.to_datetime(external_copy[date_column])
    
    # Perform merge
    merged_df = pd.merge(main_copy, external_copy, on=date_column, how=how)
    
    # Fill missing values
    if fill_method == 'ffill':
        merged_df = merged_df.fillna(method='ffill')
    elif fill_method == 'bfill':
        merged_df = merged_df.fillna(method='bfill')
    elif fill_method == 'interpolate':
        merged_df = merged_df.interpolate(method='linear')
    
    logger.info(f"Merged external data with {external_copy.shape[1] - 1} features")
    
    return merged_df


def clean_retail_data(df):
    """
    Clean retail data by removing outliers, handling missing values, and ensuring proper data types.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame with retail data
        
    Returns:
    --------
    pandas.DataFrame
        Cleaned DataFrame
    """
    # Create a copy to avoid modifying the original
    df_cleaned = df.copy()
    
    # Ensure date column is datetime type
    if 'date' in df_cleaned.columns:
        df_cleaned['date'] = pd.to_datetime(df_cleaned['date'])
    
    # Sort by date
    if 'date' in df_cleaned.columns:
        df_cleaned = df_cleaned.sort_values('date')
    
    # Handle missing values
    for column in df_cleaned.columns:
        # Skip date column
        if column == 'date':
            continue
        
        # Check if column is numeric
        if pd.api.types.is_numeric_dtype(df_cleaned[column]):
            # Calculate missing value percentage
            missing_pct = df_cleaned[column].isna().mean() * 100
            
            if missing_pct > 0 and missing_pct < 5:
                # Less than 5% missing: fill with mean
                df_cleaned[column] = df_cleaned[column].fillna(df_cleaned[column].mean())
            elif missing_pct >= 5 and missing_pct < 20:
                # Between 5% and 20% missing: forward fill then backward fill
                df_cleaned[column] = df_cleaned[column].fillna(method='ffill').fillna(method='bfill')
            elif missing_pct >= 20:
                # More than 20% missing: fill with median
                df_cleaned[column] = df_cleaned[column].fillna(df_cleaned[column].median())
        else:
            # For non-numeric columns, fill with mode
            if df_cleaned[column].isna().any():
                mode_value = df_cleaned[column].mode()[0]
                df_cleaned[column] = df_cleaned[column].fillna(mode_value)
    
    return df_cleaned


def handle_outliers(df, method='winsorize', column=None, limits=(0.01, 0.01), threshold=3, groupby=None):
    """
    Handle outliers in the data using various methods.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    method : str, optional
        Method to handle outliers:
        - 'winsorize': Cap outliers at specified percentiles
        - 'trim': Remove outliers
        - 'zscore': Remove outliers based on z-score
        - 'iqr': Remove outliers based on interquartile range
    column : str, optional
        Column to process. If None, process all numeric columns
    limits : tuple, optional
        Lower and upper limits for winsorization (as percentiles)
    threshold : float, optional
        Threshold for zscore method
    groupby : str or list, optional
        Column(s) to group by before handling outliers
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with handled outliers
    """
    # Create a copy to avoid modifying the original
    df_handled = df.copy()
    
    # If column not specified, process all numeric columns
    if column is None:
        numeric_columns = df_handled.select_dtypes(include=[np.number]).columns.tolist()
    else:
        numeric_columns = [column]
    
    for col in numeric_columns:
        if groupby is not None:
            # Handle outliers within groups
            for group_name, group_df in df_handled.groupby(groupby):
                group_indices = group_df.index
                
                if method == 'winsorize':
                    # Winsorize: Cap outliers at specified percentiles
                    lower_limit, upper_limit = limits
                    lower_val = np.percentile(group_df[col], lower_limit * 100)
                    upper_val = np.percentile(group_df[col], (1 - upper_limit) * 100)
                    df_handled.loc[group_indices, col] = df_handled.loc[group_indices, col].clip(lower_val, upper_val)
                
                elif method == 'trim':
                    # Trim: Remove outliers
                    lower_limit, upper_limit = limits
                    lower_val = np.percentile(group_df[col], lower_limit * 100)
                    upper_val = np.percentile(group_df[col], (1 - upper_limit) * 100)
                    outlier_indices = group_indices[~group_df[col].between(lower_val, upper_val)]
                    df_handled = df_handled.drop(outlier_indices)
                
                elif method == 'zscore':
                    # Z-score: Remove outliers based on z-score
                    z_scores = stats.zscore(group_df[col])
                    outlier_indices = group_indices[np.abs(z_scores) > threshold]
                    df_handled = df_handled.drop(outlier_indices)
                
                elif method == 'iqr':
                    # IQR: Remove outliers based on interquartile range
                    Q1 = group_df[col].quantile(0.25)
                    Q3 = group_df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_val = Q1 - 1.5 * IQR
                    upper_val = Q3 + 1.5 * IQR
                    outlier_indices = group_indices[~group_df[col].between(lower_val, upper_val)]
                    df_handled = df_handled.drop(outlier_indices)
        
        else:
            # Handle outliers globally
            if method == 'winsorize':
                # Winsorize: Cap outliers at specified percentiles
                lower_limit, upper_limit = limits
                lower_val = np.percentile(df_handled[col], lower_limit * 100)
                upper_val = np.percentile(df_handled[col], (1 - upper_limit) * 100)
                df_handled[col] = df_handled[col].clip(lower_val, upper_val)
            
            elif method == 'trim':
                # Trim: Remove outliers
                lower_limit, upper_limit = limits
                lower_val = np.percentile(df_handled[col], lower_limit * 100)
                upper_val = np.percentile(df_handled[col], (1 - upper_limit) * 100)
                df_handled = df_handled[df_handled[col].between(lower_val, upper_val)]
            
            elif method == 'zscore':
                # Z-score: Remove outliers based on z-score
                z_scores = stats.zscore(df_handled[col])
                df_handled = df_handled[np.abs(z_scores) <= threshold]
            
            elif method == 'iqr':
                # IQR: Remove outliers based on interquartile range
                Q1 = df_handled[col].quantile(0.25)
                Q3 = df_handled[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_val = Q1 - 1.5 * IQR
                upper_val = Q3 + 1.5 * IQR
                df_handled = df_handled[df_handled[col].between(lower_val, upper_val)]
    
    return df_handled


def detect_outliers(df, column, method='zscore', threshold=3, groupby=None):
    """
    Detect outliers in the data using various methods.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    column : str
        Column to check for outliers
    method : str, optional
        Method to detect outliers:
        - 'zscore': Detect outliers based on z-score
        - 'iqr': Detect outliers based on interquartile range
    threshold : float, optional
        Threshold for zscore method
    groupby : str or list, optional
        Column(s) to group by before detecting outliers
        
    Returns:
    --------
    pandas.Series
        Boolean Series indicating outliers
    """
    # Create an empty Series to store outlier indicators
    outliers = pd.Series(False, index=df.index)
    
    if groupby is not None:
        # Detect outliers within groups
        for group_name, group_df in df.groupby(groupby):
            group_indices = group_df.index
            group_values = group_df[column]
            
            if method == 'zscore':
                # Z-score: Detect outliers based on z-score
                z_scores = stats.zscore(group_values)
                group_outliers = np.abs(z_scores) > threshold
                outliers.loc[group_indices] = group_outliers
            
            elif method == 'iqr':
                # IQR: Detect outliers based on interquartile range
                Q1 = group_values.quantile(0.25)
                Q3 = group_values.quantile(0.75)
                IQR = Q3 - Q1
                lower_val = Q1 - 1.5 * IQR
                upper_val = Q3 + 1.5 * IQR
                group_outliers = ~group_values.between(lower_val, upper_val)
                outliers.loc[group_indices] = group_outliers
    
    else:
        # Detect outliers globally
        if method == 'zscore':
            # Z-score: Detect outliers based on z-score
            z_scores = stats.zscore(df[column])
            outliers = np.abs(z_scores) > threshold
        
        elif method == 'iqr':
            # IQR: Detect outliers based on interquartile range
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_val = Q1 - 1.5 * IQR
            upper_val = Q3 + 1.5 * IQR
            outliers = ~df[column].between(lower_val, upper_val)
    
    return outliers 