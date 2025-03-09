#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Facebook Prophet implementation for time series forecasting.
This module provides optimized Prophet models for retail demand forecasting.
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
import os
from concurrent.futures import ProcessPoolExecutor
import itertools

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def prepare_prophet_data(df, date_column, target_column, regressors=None):
    """
    Prepare data for Prophet forecasting.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    date_column : str
        Name of the date column
    target_column : str
        Name of the target column
    regressors : list, optional
        List of regressor column names
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame formatted for Prophet
    """
    # Create a copy of the input DataFrame to avoid modifying the original
    prophet_df = df.copy()
    
    # Rename columns to Prophet's required format
    prophet_df = prophet_df.rename(columns={date_column: 'ds', target_column: 'y'})
    
    # Ensure date column is datetime type
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
    
    # Select only required columns
    cols = ['ds', 'y']
    
    # Handle missing regressors
    if regressors:
        available_regressors = []
        for regressor in regressors:
            if regressor in prophet_df.columns:
                available_regressors.append(regressor)
            else:
                print(f"Warning: Regressor '{regressor}' not found in the data. Skipping.")
                # If it's a holiday indicator, we'll create a dummy column
                if regressor == 'is_holiday':
                    prophet_df['is_holiday'] = 0
                    available_regressors.append('is_holiday')
                    print(f"Created dummy 'is_holiday' column with zeros.")
                elif regressor == 'temperature' and any('temperature' in col for col in prophet_df.columns):
                    # Try to find a column with temperature in the name
                    temp_col = next((col for col in prophet_df.columns if 'temperature' in col), None)
                    if temp_col:
                        prophet_df['temperature'] = prophet_df[temp_col]
                        available_regressors.append('temperature')
                        print(f"Using '{temp_col}' as 'temperature' regressor.")
        
        cols.extend(available_regressors)
    
    return prophet_df[cols]

class OptimizedProphet:
    """
    Optimized Prophet model for retail forecasting.
    Includes hyperparameter optimization and cross-validation.
    """
    
    def __init__(self, 
                 seasonality_mode='additive',
                 changepoint_prior_scale=0.05,
                 seasonality_prior_scale=10.0,
                 holidays_prior_scale=10.0,
                 daily_seasonality=False,
                 weekly_seasonality=True,
                 yearly_seasonality=True):
        """
        Initialize OptimizedProphet model.
        
        Parameters:
        -----------
        seasonality_mode : str, optional
            Seasonality mode ('additive' or 'multiplicative')
        changepoint_prior_scale : float, optional
            Flexibility of the trend
        seasonality_prior_scale : float, optional
            Strength of the seasonality
        holidays_prior_scale : float, optional
            Strength of the holiday effects
        daily_seasonality : bool, optional
            Whether to include daily seasonality
        weekly_seasonality : bool or int, optional
            Whether to include weekly seasonality or number of Fourier terms
        yearly_seasonality : bool or int, optional
            Whether to include yearly seasonality or number of Fourier terms
        """
        self.seasonality_mode = seasonality_mode
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.holidays_prior_scale = holidays_prior_scale
        self.daily_seasonality = daily_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.yearly_seasonality = yearly_seasonality
        self.model = None
        self.trained = False
    
    def _create_model(self):
        """Create a Prophet model with the specified parameters."""
        model = Prophet(
            seasonality_mode=self.seasonality_mode,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            holidays_prior_scale=self.holidays_prior_scale,
            daily_seasonality=self.daily_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            yearly_seasonality=self.yearly_seasonality
        )
        return model
    
    def fit(self, df, regressors=None):
        """
        Fit the Prophet model.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame in Prophet format (with 'ds' and 'y' columns)
        regressors : list, optional
            List of regressor column names
            
        Returns:
        --------
        self
        """
        # Create Prophet model
        self.model = self._create_model()
        
        # Add regressors
        if regressors:
            for regressor in regressors:
                if regressor in df.columns:
                    self.model.add_regressor(regressor)
        
        # Fit the model
        self.model.fit(df)
        self.trained = True
        
        return self
    
    def predict(self, periods=30, freq='D', future_df=None, include_history=False):
        """
        Generate forecast for future periods.
        
        Parameters:
        -----------
        periods : int, optional
            Number of periods to forecast
        freq : str, optional
            Frequency of the forecast
        future_df : pandas.DataFrame, optional
            DataFrame with future dates and regressors
        include_history : bool, optional
            Whether to include historical dates in the forecast
            
        Returns:
        --------
        pandas.DataFrame
            Forecast DataFrame
        """
        if not self.trained:
            raise ValueError("Model must be trained before making predictions")
        
        if future_df is None:
            # Create future DataFrame
            future = self.model.make_future_dataframe(periods=periods, freq=freq, 
                                                     include_history=include_history)
        else:
            # Use provided DataFrame
            future = future_df
        
        # Make predictions
        forecast = self.model.predict(future)
        
        return forecast
    
    def cross_validate(self, initial='730 days', period='180 days', horizon='30 days'):
        """
        Perform cross-validation on the trained model.
        
        Parameters:
        -----------
        initial : str, optional
            Initial training period
        period : str, optional
            Period between cutoffs
        horizon : str, optional
            Forecast horizon
            
        Returns:
        --------
        tuple
            (cv_results, cv_metrics)
        """
        if not self.trained:
            raise ValueError("Model must be trained before cross-validation")
        
        # Perform cross-validation
        cv_results = cross_validation(
            self.model, 
            initial=initial, 
            period=period, 
            horizon=horizon
        )
        
        # Calculate metrics
        cv_metrics = performance_metrics(cv_results)
        
        return cv_results, cv_metrics
    
    def optimize_hyperparameters(self, df, params_grid=None, 
                              initial='730 days', period='180 days', horizon='30 days'):
        """
        Optimize hyperparameters using grid search.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame in Prophet format (with 'ds' and 'y' columns)
        params_grid : dict, optional
            Dictionary of parameter grid
        initial, period, horizon : str, optional
            Cross-validation parameters
            
        Returns:
        --------
        dict
            Best parameters
        """
        # Default parameter grid if not provided
        if params_grid is None:
            params_grid = {
                'seasonality_mode': ['additive', 'multiplicative'],
                'changepoint_prior_scale': [0.001, 0.01, 0.05, 0.1, 0.5],
                'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
                'holidays_prior_scale': [0.01, 0.1, 1.0, 10.0]
            }
        
        # Generate all combinations of parameters
        param_combinations = list(itertools.product(*params_grid.values()))
        params_list = [dict(zip(params_grid.keys(), combo)) for combo in param_combinations]
        
        # Initialize best parameters and best RMSE
        best_params = None
        best_rmse = float('inf')
        
        # Evaluate each combination of parameters
        for params in params_list:
            # Create and fit model with current parameters
            model = Prophet(
                seasonality_mode=params['seasonality_mode'],
                changepoint_prior_scale=params['changepoint_prior_scale'],
                seasonality_prior_scale=params['seasonality_prior_scale'],
                holidays_prior_scale=params['holidays_prior_scale'],
                daily_seasonality=self.daily_seasonality,
                weekly_seasonality=self.weekly_seasonality,
                yearly_seasonality=self.yearly_seasonality
            )
            model.fit(df)
            
            # Perform cross-validation
            try:
                cv_results = cross_validation(
                    model, 
                    initial=initial, 
                    period=period, 
                    horizon=horizon
                )
                cv_metrics = performance_metrics(cv_results)
                
                # Check if current model is better
                rmse = cv_metrics['rmse'].mean()
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_params = params
            except Exception as e:
                # Skip parameter combination if cross-validation fails
                print(f"Skipping parameter combination due to error: {e}")
                continue
        
        # Update model parameters with best parameters
        if best_params:
            self.seasonality_mode = best_params['seasonality_mode']
            self.changepoint_prior_scale = best_params['changepoint_prior_scale']
            self.seasonality_prior_scale = best_params['seasonality_prior_scale']
            self.holidays_prior_scale = best_params['holidays_prior_scale']
            
            # Create and fit model with best parameters
            self.model = self._create_model()
            self.model.fit(df)
            self.trained = True
        
        return best_params
    
    def get_components(self):
        """
        Get trend and seasonality components of the forecast.
        
        Returns:
        --------
        dict
            Components including trend, seasonality, and holidays
        """
        if not self.trained:
            raise ValueError("Model must be trained before getting components")
        
        # Make forecast for the training period
        forecast = self.predict(periods=0, include_history=True)
        
        # Extract components
        components = {
            'trend': forecast[['ds', 'trend']],
            'weekly': forecast[['ds', 'weekly']] if 'weekly' in forecast.columns else None,
            'yearly': forecast[['ds', 'yearly']] if 'yearly' in forecast.columns else None,
            'holidays': forecast[['ds', 'holidays']] if 'holidays' in forecast.columns else None
        }
        
        # Add regressors if available
        regressors = [col for col in forecast.columns if col.startswith('extra_regressors')]
        if regressors:
            for regressor in regressors:
                components[regressor] = forecast[['ds', regressor]]
        
        return components


class ProphetHyperparameterTuner:
    """
    Hyperparameter tuning for Prophet models.
    """
    
    def __init__(self, 
                 param_grid=None, 
                 initial='365 days',
                 period='30 days',
                 horizon='30 days',
                 parallel=True,
                 n_jobs=-1):
        """
        Initialize the hyperparameter tuner.
        
        Args:
            param_grid (dict): Dictionary of parameter grids
            initial (str): Initial training period
            period (str): Period between cutoffs
            horizon (str): Forecast horizon
            parallel (bool): Whether to use parallel processing
            n_jobs (int): Number of jobs for parallel processing
        """
        self.param_grid = param_grid or {
            'seasonality_mode': ['additive', 'multiplicative'],
            'changepoint_prior_scale': [0.001, 0.01, 0.05, 0.1, 0.5],
            'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
            'holidays_prior_scale': [0.01, 0.1, 1.0, 10.0],
            'changepoint_range': [0.8, 0.9, 0.95]
        }
        self.initial = initial
        self.period = period
        self.horizon = horizon
        self.parallel = parallel
        self.n_jobs = n_jobs
        self.best_params = None
        self.best_model = None
        self.best_rmse = float('inf')
        self.results = None
    
    def _evaluate_model(self, params, df, holidays=None, regressors=None):
        """Evaluate a single model with given parameters."""
        model = Prophet(
            seasonality_mode=params['seasonality_mode'],
            changepoint_prior_scale=params['changepoint_prior_scale'],
            seasonality_prior_scale=params['seasonality_prior_scale'],
            holidays_prior_scale=params['holidays_prior_scale'],
            changepoint_range=params['changepoint_range'],
            holidays=holidays
        )
        
        # Add regressors if provided
        if regressors:
            for regressor in regressors:
                model.add_regressor(regressor)
        
        # Fit the model
        model.fit(df)
        
        # Cross-validation
        df_cv = cross_validation(
            model=model,
            initial=self.initial,
            period=self.period,
            horizon=self.horizon,
            parallel='processes' if self.parallel else None
        )
        
        # Calculate metrics
        metrics = performance_metrics(df_cv)
        rmse = metrics['rmse'].mean()
        
        return {
            'params': params,
            'rmse': rmse,
            'metrics': metrics,
            'model': model
        }
    
    def tune(self, df, holidays=None, regressors=None):
        """
        Tune hyperparameters to find the best model.
        
        Args:
            df (pd.DataFrame): DataFrame with 'ds' and 'y' columns
            holidays (pd.DataFrame): DataFrame with holiday definitions
            regressors (list): List of regressor column names
            
        Returns:
            dict: Best parameters
        """
        logger.info("Starting hyperparameter tuning...")
        
        # Generate all combinations of parameters
        all_params = [dict(zip(self.param_grid.keys(), values)) 
                     for values in itertools.product(*self.param_grid.values())]
        
        logger.info(f"Testing {len(all_params)} parameter combinations")
        
        results = []
        
        # Use parallel processing if enabled
        if self.parallel:
            with ProcessPoolExecutor(max_workers=self.n_jobs if self.n_jobs > 0 else None) as executor:
                futures = [executor.submit(self._evaluate_model, params, df, holidays, regressors) 
                          for params in all_params]
                for future in futures:
                    result = future.result()
                    results.append(result)
                    
                    # Update best model if better
                    if result['rmse'] < self.best_rmse:
                        self.best_rmse = result['rmse']
                        self.best_params = result['params']
                        self.best_model = result['model']
                        
                    logger.info(f"Tested parameters: {result['params']}, RMSE: {result['rmse']:.4f}")
        else:
            # Sequential processing
            for params in all_params:
                result = self._evaluate_model(params, df, holidays, regressors)
                results.append(result)
                
                # Update best model if better
                if result['rmse'] < self.best_rmse:
                    self.best_rmse = result['rmse']
                    self.best_params = result['params']
                    self.best_model = result['model']
                    
                logger.info(f"Tested parameters: {result['params']}, RMSE: {result['rmse']:.4f}")
        
        # Sort results by RMSE
        self.results = sorted(results, key=lambda x: x['rmse'])
        
        logger.info(f"Hyperparameter tuning complete. Best RMSE: {self.best_rmse:.4f}")
        logger.info(f"Best parameters: {self.best_params}")
        
        return self.best_params
    
    def get_best_model(self):
        """Get the best model from tuning."""
        if self.best_model is None:
            raise ValueError("Tuning must be performed before getting the best model")
        
        return self.best_model
    
    def get_optimized_prophet(self):
        """Get an OptimizedProphet instance with the best parameters."""
        if self.best_params is None:
            raise ValueError("Tuning must be performed before getting an optimized model")
        
        model = OptimizedProphet(
            seasonality_mode=self.best_params['seasonality_mode'],
            changepoint_prior_scale=self.best_params['changepoint_prior_scale'],
            seasonality_prior_scale=self.best_params['seasonality_prior_scale'],
            holidays_prior_scale=self.best_params['holidays_prior_scale'],
            changepoint_range=self.best_params['changepoint_range'],
            parallel=self.parallel,
            n_jobs=self.n_jobs
        )
        
        return model


def create_holiday_features(holidays_df, date_col='date', name_col='holiday_name', is_major_col='is_major_holiday'):
    """
    Create holiday features for Prophet.
    
    Args:
        holidays_df (pd.DataFrame): DataFrame with holiday information
        date_col (str): Name of the date column
        name_col (str): Name of the holiday name column
        is_major_col (str): Name of the column indicating if it's a major holiday
        
    Returns:
        pd.DataFrame: DataFrame formatted for Prophet holidays
    """
    # Create a copy to avoid modifying the original
    holidays = holidays_df.copy()
    
    # Rename columns to Prophet's required format
    holidays = holidays.rename(columns={date_col: 'ds', name_col: 'holiday'})
    
    # Ensure datetime format
    holidays['ds'] = pd.to_datetime(holidays['ds'])
    
    # Add lower and upper window for major holidays
    if is_major_col in holidays.columns:
        holidays['lower_window'] = holidays[is_major_col].apply(lambda x: -2 if x else 0)
        holidays['upper_window'] = holidays[is_major_col].apply(lambda x: 2 if x else 0)
    
    return holidays[['ds', 'holiday', 'lower_window', 'upper_window']]


def forecast_multiple_series(data_dict, forecast_periods=30, freq='D', parallel=True, n_jobs=-1):
    """
    Forecast multiple time series in parallel.
    
    Args:
        data_dict (dict): Dictionary of DataFrames with 'ds' and 'y' columns
        forecast_periods (int): Number of periods to forecast
        freq (str): Frequency of the forecast
        parallel (bool): Whether to use parallel processing
        n_jobs (int): Number of jobs for parallel processing
        
    Returns:
        dict: Dictionary of forecast results
    """
    def _forecast_single(key, df):
        """Forecast a single time series."""
        model = OptimizedProphet(parallel=False)  # Avoid nested parallelism
        model.fit(df)
        forecast = model.predict(periods=forecast_periods, freq=freq)
        return key, forecast
    
    logger.info(f"Forecasting {len(data_dict)} time series...")
    
    results = {}
    
    if parallel and len(data_dict) > 1:
        with ProcessPoolExecutor(max_workers=n_jobs if n_jobs > 0 else None) as executor:
            futures = [executor.submit(_forecast_single, key, df) for key, df in data_dict.items()]
            for future in futures:
                key, forecast = future.result()
                results[key] = forecast
    else:
        for key, df in data_dict.items():
            key, forecast = _forecast_single(key, df)
            results[key] = forecast
    
    logger.info("Forecasting complete")
    return results 