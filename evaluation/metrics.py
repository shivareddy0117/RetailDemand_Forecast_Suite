#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluation metrics for forecasting models.
This module provides functions to calculate various performance metrics for time series forecasting.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error (MAPE).
    
    Parameters:
    -----------
    y_true : array-like
        Actual values
    y_pred : array-like
        Predicted values
        
    Returns:
    --------
    float
        MAPE value
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Handle division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def weighted_mape(y_true, y_pred, weights=None):
    """
    Calculate Weighted Mean Absolute Percentage Error (WMAPE).
    
    Parameters:
    -----------
    y_true : array-like
        Actual values
    y_pred : array-like
        Predicted values
    weights : array-like, optional
        Weights for each observation
        
    Returns:
    --------
    float
        WMAPE value
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    if weights is None:
        # Default to weighting by true value (equivalent to MAD/MAV)
        weights = np.abs(y_true)
    
    # Handle division by zero
    sum_weights = np.sum(weights)
    if sum_weights == 0:
        return np.nan
    
    return np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights) * 100

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculate Symmetric Mean Absolute Percentage Error (SMAPE).
    
    Parameters:
    -----------
    y_true : array-like
        Actual values
    y_pred : array-like
        Predicted values
        
    Returns:
    --------
    float
        SMAPE value
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    
    # Handle division by zero
    mask = denominator != 0
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100

def root_mean_squared_error(y_true, y_pred):
    """
    Calculate Root Mean Squared Error (RMSE).
    
    Parameters:
    -----------
    y_true : array-like
        Actual values
    y_pred : array-like
        Predicted values
        
    Returns:
    --------
    float
        RMSE value
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mean_absolute_scaled_error(y_true, y_pred, y_train=None, seasonality=1):
    """
    Calculate Mean Absolute Scaled Error (MASE).
    
    Parameters:
    -----------
    y_true : array-like
        Actual values for the test period
    y_pred : array-like
        Predicted values for the test period
    y_train : array-like, optional
        Training data to compute the scale
    seasonality : int, optional
        Seasonality of the time series (default=1 for non-seasonal)
        
    Returns:
    --------
    float
        MASE value
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if y_train is None:
        warnings.warn("Training data not provided for MASE calculation. Using test data instead.")
        y_train = y_true
    
    # Calculate the scale (mean absolute error of the naive forecast)
    naive_errors = np.abs(np.array(y_train[seasonality:]) - np.array(y_train[:-seasonality]))
    scale = np.mean(naive_errors)
    
    # Handle division by zero
    if scale == 0:
        warnings.warn("Scale is zero in MASE calculation. Returning infinity.")
        return np.inf
    
    # Calculate MASE
    return np.mean(np.abs(y_true - y_pred)) / scale

def calculate_forecast_bias(y_true, y_pred):
    """
    Calculate forecast bias (average error).
    
    Parameters:
    -----------
    y_true : array-like
        Actual values
    y_pred : array-like
        Predicted values
        
    Returns:
    --------
    float
        Bias value
    """
    return np.mean(y_pred - y_true)

def calculate_all_metrics(y_true, y_pred, y_train=None, seasonality=1):
    """
    Calculate all forecast evaluation metrics.
    
    Parameters:
    -----------
    y_true : array-like
        Actual values
    y_pred : array-like
        Predicted values
    y_train : array-like, optional
        Training data for MASE calculation
    seasonality : int, optional
        Seasonality period for MASE calculation
        
    Returns:
    --------
    dict
        Dictionary containing all metrics
    """
    metrics = {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': root_mean_squared_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'Bias': calculate_forecast_bias(y_true, y_pred)
    }
    
    # Metrics that can have division by zero issues
    try:
        metrics['MAPE'] = mean_absolute_percentage_error(y_true, y_pred)
    except:
        metrics['MAPE'] = np.nan
        
    try:
        metrics['SMAPE'] = symmetric_mean_absolute_percentage_error(y_true, y_pred)
    except:
        metrics['SMAPE'] = np.nan
    
    try:
        metrics['WMAPE'] = weighted_mape(y_true, y_pred)
    except:
        metrics['WMAPE'] = np.nan
    
    try:
        metrics['MASE'] = mean_absolute_scaled_error(y_true, y_pred, y_train, seasonality)
    except:
        metrics['MASE'] = np.nan
    
    return metrics

def compare_models(actual_values, model_forecasts, y_train=None, seasonality=1):
    """
    Compare multiple forecasting models using various metrics.
    
    Parameters:
    -----------
    actual_values : array-like
        Actual values
    model_forecasts : dict
        Dictionary of model forecasts {model_name: forecast_values}
    y_train : array-like, optional
        Training data for MASE calculation
    seasonality : int, optional
        Seasonality period for MASE calculation
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with metrics for each model
    """
    results = {}
    
    for model_name, forecast in model_forecasts.items():
        metrics = calculate_all_metrics(actual_values, forecast, y_train, seasonality)
        results[model_name] = metrics
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results).T
    
    return results_df

def calculate_prediction_intervals(forecasts, confidence_level=0.95):
    """
    Calculate prediction intervals for ensemble forecasts.
    
    Parameters:
    -----------
    forecasts : dict or DataFrame
        Dictionary or DataFrame of forecasts from different models
    confidence_level : float, optional
        Confidence level for the prediction interval (default=0.95)
        
    Returns:
    --------
    tuple
        (lower_bound, upper_bound) arrays
    """
    if isinstance(forecasts, dict):
        forecasts_array = np.array(list(forecasts.values()))
    else:
        forecasts_array = np.array(forecasts)
    
    # Calculate mean forecast
    mean_forecast = np.mean(forecasts_array, axis=0)
    
    # Calculate standard deviation of forecasts
    std_forecast = np.std(forecasts_array, axis=0)
    
    # Calculate z-score for the given confidence level
    z_score = abs(np.percentile(np.random.normal(0, 1, 10000), 
                              [(1 - confidence_level) / 2 * 100, 
                               (1 + confidence_level) / 2 * 100]))
    
    # Calculate prediction intervals
    lower_bound = mean_forecast - z_score[1] * std_forecast
    upper_bound = mean_forecast + z_score[1] * std_forecast
    
    # Ensure lower bound is not negative for count data
    lower_bound = np.maximum(lower_bound, 0)
    
    return lower_bound, upper_bound 