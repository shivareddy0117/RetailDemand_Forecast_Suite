#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization utilities for forecast evaluation.
This module provides functions to visualize forecasts and their performance metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
import datetime

# Set default style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

def plot_forecast_vs_actual(dates, actual, forecast, model_name=None, prediction_intervals=None, 
                           figsize=(12, 6), title=None, ylabel='Value', xlabel='Date'):
    """
    Plot forecasted values against actual values.
    
    Parameters:
    -----------
    dates : array-like
        Dates or x-values for the plot
    actual : array-like
        Actual values
    forecast : array-like
        Forecasted values
    model_name : str, optional
        Name of the forecasting model
    prediction_intervals : tuple, optional
        Tuple of (lower_bound, upper_bound) for prediction intervals
    figsize : tuple, optional
        Figure size
    title : str, optional
        Plot title
    ylabel : str, optional
        Y-axis label
    xlabel : str, optional
        X-axis label
    
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Check if inputs are valid
    if len(dates) == 0 or len(actual) == 0:
        ax.text(0.5, 0.5, "No data available for plotting", 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=14)
        return fig
    
    # Convert inputs to arrays for consistent handling
    dates = np.array(dates)
    actual = np.array(actual)
    forecast = np.array(forecast)
    
    # Plot actual values
    ax.plot(dates, actual, 'b-', label='Actual', marker='o')
    
    # Plot forecasted values
    ax.plot(dates, forecast, 'r--', label=f'Forecast{" (" + model_name + ")" if model_name else ""}', 
            color='#ff7f0e', alpha=0.7)
    
    # Plot prediction intervals if provided
    if prediction_intervals is not None:
        lower_bound, upper_bound = prediction_intervals
        ax.fill_between(dates, lower_bound, upper_bound, color='#ff7f0e', alpha=0.2, 
                        label=f'95% Prediction Interval')
    
    # Set title and labels
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(f'Forecast vs Actual{" (" + model_name + ")" if model_name else ""}', fontsize=14)
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    # Format x-axis for dates
    if len(dates) > 0:
        try:
            first_date = dates[0]
            if isinstance(first_date, (pd.Timestamp, np.datetime64)) or pd.api.types.is_datetime64_any_dtype(first_date):
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                fig.autofmt_xdate()
        except (IndexError, TypeError, AttributeError):
            pass  # If dates can't be formatted, just continue
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_multiple_forecasts(dates, actual, forecasts, figsize=(12, 6), title=None, xlabel='Date', ylabel='Value'):
    """
    Plot actual values and multiple forecast models on the same graph.
    
    Parameters:
    -----------
    dates : array-like
        Dates for the x-axis
    actual : array-like
        Actual values
    forecasts : dict
        Dictionary of forecast values with model names as keys
    figsize : tuple, optional
        Figure size
    title : str, optional
        Plot title
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Convert dates to list if it's a pandas Series or other iterable
    dates_list = dates.tolist() if hasattr(dates, 'tolist') else list(dates)
    
    # Plot actual values
    ax.plot(dates, actual, 'o-', label='Actual', color='#1f77b4', linewidth=2, alpha=0.8)
    
    # Color palette for forecasts
    colors = sns.color_palette('viridis', len(forecasts))
    
    # Plot each forecast
    for i, (model_name, forecast) in enumerate(forecasts.items()):
        ax.plot(dates, forecast, 'o-', label=f'{model_name}', color=colors[i], alpha=0.7)
    
    # Set title and labels
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title('Comparison of Forecast Models', fontsize=14)
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    # Format x-axis for dates - check the first element of our list
    if len(dates_list) > 0 and isinstance(dates_list[0], (pd.Timestamp, np.datetime64, datetime.datetime)):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_forecast_errors(dates, errors, model_name=None, figsize=(12, 6), title=None):
    """
    Plot forecast errors over time.
    
    Parameters:
    -----------
    dates : array-like
        Dates for the x-axis
    errors : array-like
        Forecast errors (actual - forecast)
    model_name : str, optional
        Name of the forecasting model
    figsize : tuple, optional
        Figure size
    title : str, optional
        Plot title
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot errors
    ax.bar(dates, errors, alpha=0.7, color='#2ca02c')
    ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    
    # Set title and labels
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(f'Forecast Errors{" (" + model_name + ")" if model_name else ""}', fontsize=14)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Error (Actual - Forecast)', fontsize=12)
    
    # Format x-axis for dates
    if isinstance(dates[0], (pd.Timestamp, np.datetime64)):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_error_distribution(errors, model_name=None, figsize=(10, 6), title=None):
    """
    Plot the distribution of forecast errors.
    
    Parameters:
    -----------
    errors : array-like
        Forecast errors (actual - forecast)
    model_name : str, optional
        Name of the forecasting model
    figsize : tuple, optional
        Figure size
    title : str, optional
        Plot title
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot error distribution
    sns.histplot(errors, kde=True, ax=ax, color='#2ca02c', alpha=0.7)
    
    # Add vertical line at zero
    ax.axvline(x=0, color='r', linestyle='--', alpha=0.7)
    
    # Add mean error line
    mean_error = np.mean(errors)
    ax.axvline(x=mean_error, color='blue', linestyle='--', alpha=0.7, 
               label=f'Mean Error: {mean_error:.2f}')
    
    # Set title and labels
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(f'Error Distribution{" (" + model_name + ")" if model_name else ""}', fontsize=14)
    
    ax.set_xlabel('Error (Actual - Forecast)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_metrics_comparison(metrics_df, figsize=(12, 8), title=None):
    """
    Plot comparison of metrics across different models.
    
    Parameters:
    -----------
    metrics_df : pandas.DataFrame
        DataFrame with metrics for each model
    figsize : tuple, optional
        Figure size
    title : str, optional
        Plot title
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    # Number of metrics and models
    n_metrics = metrics_df.shape[1]
    n_models = metrics_df.shape[0]
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_metrics, 1, figsize=figsize, sharex=True)
    
    # If only one metric, axes is not a list
    if n_metrics == 1:
        axes = [axes]
    
    # Plot each metric
    for i, metric in enumerate(metrics_df.columns):
        ax = axes[i]
        
        # Sort by metric value (ascending or descending based on metric)
        ascending = metric in ['R2']  # Higher is better for R2
        sorted_df = metrics_df.sort_values(by=metric, ascending=not ascending)
        
        # Plot horizontal bar chart
        bars = ax.barh(sorted_df.index, sorted_df[metric], alpha=0.7)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width * 1.01
            ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.4f}',
                   va='center')
        
        # Set title and labels
        ax.set_title(f'{metric}', fontsize=12)
        ax.set_xlabel(metric, fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # Set overall title
    if title:
        fig.suptitle(title, fontsize=16, y=1.02)
    else:
        fig.suptitle('Model Performance Comparison', fontsize=16, y=1.02)
    
    plt.tight_layout()
    return fig

def plot_residual_analysis(dates, residuals, model_name=None, figsize=(12, 10), title=None):
    """
    Create a comprehensive residual analysis plot with time series, histogram, and Q-Q plot.
    
    Parameters:
    -----------
    dates : array-like
        Dates for the x-axis
    residuals : array-like
        Residual values (actual - forecast)
    model_name : str, optional
        Name of the forecasting model
    figsize : tuple, optional
        Figure size
    title : str, optional
        Plot title
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    """
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
    
    # Convert dates to list if it's a pandas Series or other iterable
    dates_list = dates.tolist() if hasattr(dates, 'tolist') else list(dates)
    
    # Residuals over time
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(dates, residuals, 'o-', alpha=0.7)
    ax1.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    ax1.set_title('Residuals Over Time', fontsize=12)
    ax1.set_xlabel('Date', fontsize=10)
    ax1.set_ylabel('Residual', fontsize=10)
    
    # Format x-axis for dates - check the first element of our list
    if len(dates_list) > 0 and isinstance(dates_list[0], (pd.Timestamp, np.datetime64, datetime.datetime)):
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()
    
    # Histogram of residuals
    ax2 = fig.add_subplot(gs[1, 0])
    sns.histplot(residuals, kde=True, ax=ax2, color='#2ca02c', alpha=0.7)
    ax2.axvline(x=0, color='r', linestyle='--', alpha=0.7)
    ax2.set_title('Residual Distribution', fontsize=12)
    ax2.set_xlabel('Residual', fontsize=10)
    ax2.set_ylabel('Frequency', fontsize=10)
    
    # QQ plot of residuals
    ax3 = fig.add_subplot(gs[1, 1])
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=ax3)
    ax3.set_title('Q-Q Plot', fontsize=12)
    
    # Set overall title
    if title:
        fig.suptitle(title, fontsize=14, y=0.98)
    else:
        fig.suptitle(f'Residual Analysis{" (" + model_name + ")" if model_name else ""}', 
                    fontsize=14, y=0.98)
    
    plt.tight_layout()
    return fig

def plot_feature_importance(feature_names, importances, model_name=None, figsize=(10, 6), title=None):
    """
    Plot feature importance for tree-based models.
    
    Parameters:
    -----------
    feature_names : array-like
        Names of features
    importances : array-like
        Importance scores for each feature
    model_name : str, optional
        Name of the model
    figsize : tuple, optional
        Figure size
    title : str, optional
        Plot title
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    # Create DataFrame for sorting
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot horizontal bar chart
    bars = ax.barh(importance_df['Feature'], importance_df['Importance'], alpha=0.7)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width * 1.01
        ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.4f}',
               va='center')
    
    # Set title and labels
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(f'Feature Importance{" (" + model_name + ")" if model_name else ""}', fontsize=14)
    
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_forecast_components(model, forecast, figsize=(12, 10), title=None):
    """
    Plot the components of a Prophet forecast.
    
    Parameters:
    -----------
    model : Prophet
        Fitted Prophet model
    forecast : pandas.DataFrame
        Prophet forecast DataFrame
    figsize : tuple, optional
        Figure size
    title : str, optional
        Plot title
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    try:
        # This only works with Prophet models
        fig = model.plot_components(forecast, figsize=figsize)
        
        # Set overall title
        if title:
            fig.suptitle(title, fontsize=16, y=0.98)
        else:
            fig.suptitle('Forecast Components', fontsize=16, y=0.98)
        
        plt.tight_layout()
        return fig
    except:
        print("This function is designed for Prophet models only.")
        return None

def plot_seasonal_decomposition(result, figsize=(12, 10), title=None):
    """
    Plot the results of seasonal decomposition.
    
    Parameters:
    -----------
    result : statsmodels.tsa.seasonal.DecomposeResult
        Result of seasonal decomposition
    figsize : tuple, optional
        Figure size
    title : str, optional
        Plot title
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    fig = plt.figure(figsize=figsize)
    
    # Plot observed, trend, seasonal, and residual components
    ax1 = fig.add_subplot(411)
    ax1.plot(result.observed, 'k', alpha=0.7)
    ax1.set_ylabel('Observed', fontsize=10)
    ax1.set_xticklabels([])
    
    ax2 = fig.add_subplot(412)
    ax2.plot(result.trend, 'g', alpha=0.7)
    ax2.set_ylabel('Trend', fontsize=10)
    ax2.set_xticklabels([])
    
    ax3 = fig.add_subplot(413)
    ax3.plot(result.seasonal, 'r', alpha=0.7)
    ax3.set_ylabel('Seasonal', fontsize=10)
    ax3.set_xticklabels([])
    
    ax4 = fig.add_subplot(414)
    ax4.plot(result.resid, 'b', alpha=0.7)
    ax4.set_ylabel('Residual', fontsize=10)
    
    # Set overall title
    if title:
        fig.suptitle(title, fontsize=14, y=0.98)
    else:
        fig.suptitle('Seasonal Decomposition', fontsize=14, y=0.98)
    
    plt.tight_layout()
    return fig 