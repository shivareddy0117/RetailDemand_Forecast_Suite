#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LightGBM implementation for retail demand forecasting.
This module provides optimized LightGBM models for time series forecasting.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import joblib
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TimeSeriesLGBMModel:
    """
    LightGBM model optimized for time series forecasting.
    """
    
    def __init__(self, params=None):
        """
        Initialize TimeSeriesLGBMModel.
        
        Parameters:
        -----------
        params : dict, optional
            LightGBM parameters
        """
        # Default parameters for time series forecasting
        self.default_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
        
        # Update with user-provided parameters if any
        self.params = self.default_params.copy()
        if params:
            self.params.update(params)
        
        self.model = None
        self.trained = False
        self.feature_importance = None
        self.target_column = None
        self.feature_columns = None
    
    def fit(self, df, target_column='sales', feature_columns=None,
          categorical_features=None, n_estimators=100, early_stopping_rounds=50,
          validation_size=0.2):
        """
        Fit LightGBM model for time series forecasting.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with time series data
        target_column : str, optional
            Name of the target column
        feature_columns : list, optional
            List of feature column names
        categorical_features : list, optional
            List of categorical feature names
        n_estimators : int, optional
            Number of boosting iterations
        early_stopping_rounds : int, optional
            Number of early stopping rounds
        validation_size : float, optional
            Proportion of data to use for validation
            
        Returns:
        --------
        self
        """
        # Store target and feature column names
        self.target_column = target_column
        
        # If feature columns are not provided, use all columns except target
        if feature_columns is None:
            self.feature_columns = [col for col in df.columns if col != target_column and not pd.api.types.is_datetime64_any_dtype(df[col])]
        else:
            self.feature_columns = feature_columns
        
        # Split data into features and target
        X = df[self.feature_columns]
        y = df[target_column]
        
        # Split data into train and validation sets
        train_size = int(len(df) * (1 - validation_size))
        X_train, X_val = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_val = y.iloc[:train_size], y.iloc[train_size:]
        
        # Create dataset for LightGBM
        train_data = lgb.Dataset(
            X_train, 
            label=y_train, 
            categorical_feature=categorical_features
        )
        
        val_data = lgb.Dataset(
            X_val, 
            label=y_val, 
            categorical_feature=categorical_features,
            reference=train_data
        )
        
        # Train the model
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=n_estimators,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(early_stopping_rounds),
                lgb.log_evaluation(100)
            ]
        )
        
        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'Feature': self.feature_columns,
            'Importance': self.model.feature_importance('gain')
        }).sort_values(by='Importance', ascending=False)
        
        self.trained = True
        
        return self
    
    def predict(self, X):
        """
        Generate predictions for X.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            DataFrame with feature columns
            
        Returns:
        --------
        numpy.ndarray
            Predictions
        """
        if not self.trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Extract features
        if isinstance(X, pd.DataFrame):
            X_features = X[self.feature_columns]
        else:
            X_features = X
        
        # Make predictions
        predictions = self.model.predict(X_features)
        
        return predictions
    
    def recursive_forecast(self, df, forecast_horizon=30, include_history=False):
        """
        Generate multi-step recursive forecasts.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with time series data
        forecast_horizon : int, optional
            Number of steps to forecast
        include_history : bool, optional
            Whether to include historical values in the output
            
        Returns:
        --------
        numpy.ndarray
            Forecasted values
        """
        if not self.trained:
            raise ValueError("Model must be trained before making forecasts")
        
        # Create a copy of the data
        forecast_df = df.copy()
        
        # Get last date and create date index for forecast
        last_date = None
        if isinstance(forecast_df.index, pd.DatetimeIndex):
            last_date = forecast_df.index[-1]
            forecast_date_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon)
        
        # Initialize historical values
        historical_values = None
        if include_history:
            historical_values = forecast_df[self.target_column].values
        
        # Generate forecasts recursively
        forecasts = []
        
        # Create a df for forecasting
        forecast_window = forecast_df.iloc[-1:].copy()
        
        for i in range(forecast_horizon):
            # Make prediction for next time step
            next_pred = self.predict(forecast_window)
            forecasts.append(next_pred[0])
            
            # Update forecast window for next prediction
            forecast_window.loc[forecast_window.index[-1], self.target_column] = next_pred[0]
            
            # Shift forecast window forward
            new_row = forecast_window.iloc[-1:].copy()
            
            # If we have a date index, update the index of the new row
            if last_date is not None:
                new_row.index = [forecast_date_index[i]]
            else:
                new_row.index = [forecast_window.index[-1] + 1]
            
            forecast_window = new_row
        
        # Combine historical and forecast values if requested
        if include_history:
            return np.concatenate([historical_values, np.array(forecasts)])
        else:
            return np.array(forecasts)
    
    def optimize_hyperparameters(self, df, target_column='sales', feature_columns=None,
                              categorical_features=None, param_grid=None, n_splits=5,
                              n_estimators=100, early_stopping_rounds=20):
        """
        Optimize hyperparameters using time series cross-validation.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with time series data
        target_column : str, optional
            Name of the target column
        feature_columns : list, optional
            List of feature column names
        categorical_features : list, optional
            List of categorical feature names
        param_grid : dict, optional
            Dictionary of parameter grid
        n_splits : int, optional
            Number of cross-validation splits
        n_estimators : int, optional
            Number of boosting iterations
        early_stopping_rounds : int, optional
            Number of early stopping rounds
            
        Returns:
        --------
        dict
            Best hyperparameters
        """
        # Store target and feature column names
        self.target_column = target_column
        
        # If feature columns are not provided, use all columns except target
        if feature_columns is None:
            self.feature_columns = [col for col in df.columns if col != target_column and not pd.api.types.is_datetime64_any_dtype(df[col])]
        else:
            self.feature_columns = feature_columns
        
        # Default parameter grid if not provided
        if param_grid is None:
            param_grid = {
                'learning_rate': [0.01, 0.05, 0.1],
                'num_leaves': [15, 31, 63],
                'feature_fraction': [0.7, 0.8, 0.9],
                'bagging_fraction': [0.7, 0.8, 0.9],
                'max_depth': [-1, 5, 10]
            }
        
        # Split data into features and target
        X = df[self.feature_columns]
        y = df[target_column]
        
        # Set up time series cross-validation
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # Generate all combinations of parameters
        import itertools
        param_keys = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(itertools.product(*param_values))
        params_list = [dict(zip(param_keys, combo)) for combo in param_combinations]
        
        # Initialize best parameters and best RMSE
        best_params = None
        best_rmse = float('inf')
        
        # Evaluate each combination of parameters
        for params in params_list:
            # Update parameters
            current_params = self.params.copy()
            current_params.update(params)
            
            # Initialize scores for this parameter set
            cv_scores = []
            
            # Perform time series cross-validation
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Create datasets
                train_data = lgb.Dataset(
                    X_train,
                    label=y_train,
                    categorical_feature=categorical_features
                )
                
                val_data = lgb.Dataset(
                    X_val,
                    label=y_val,
                    categorical_feature=categorical_features,
                    reference=train_data
                )
                
                # Train model
                model = lgb.train(
                    current_params,
                    train_data,
                    num_boost_round=n_estimators,
                    valid_sets=[val_data],
                    callbacks=[lgb.early_stopping(early_stopping_rounds)],
                    verbose_eval=False
                )
                
                # Make predictions and calculate RMSE
                y_pred = model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                cv_scores.append(rmse)
            
            # Calculate average RMSE across all folds
            avg_rmse = np.mean(cv_scores)
            
            # Update best parameters if current is better
            if avg_rmse < best_rmse:
                best_rmse = avg_rmse
                best_params = params.copy()
        
        # Update model parameters with best parameters
        self.params.update(best_params)
        
        # Train the final model with the best parameters
        self.fit(
            df,
            target_column=target_column,
            feature_columns=feature_columns,
            categorical_features=categorical_features,
            n_estimators=n_estimators,
            early_stopping_rounds=early_stopping_rounds
        )
        
        return best_params
    
    def get_feature_importance(self, plot=False, figsize=(10, 6)):
        """
        Get feature importance.
        
        Parameters:
        -----------
        plot : bool, optional
            Whether to plot feature importance
        figsize : tuple, optional
            Figure size
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with feature importance
        """
        if not self.trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        if plot:
            plt.figure(figsize=figsize)
            plt.barh(self.feature_importance['Feature'], self.feature_importance['Importance'])
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.title('Feature Importance')
            plt.tight_layout()
            plt.show()
        
        return self.feature_importance
    
    def evaluate(self, df, target_column=None, feature_columns=None, metrics=None):
        """
        Evaluate model on test data.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with test data
        target_column : str, optional
            Name of the target column (default: same as used for training)
        feature_columns : list, optional
            List of feature column names (default: same as used for training)
        metrics : list, optional
            List of metrics to calculate
            
        Returns:
        --------
        dict
            Dictionary with evaluation metrics
        """
        if not self.trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Use stored target and feature columns if not provided
        if target_column is None:
            target_column = self.target_column
        
        if feature_columns is None:
            feature_columns = self.feature_columns
        
        # Default metrics
        if metrics is None:
            metrics = ['rmse', 'mae', 'r2']
        
        # Extract features and target
        X = df[feature_columns]
        y = df[target_column]
        
        # Make predictions
        y_pred = self.predict(X)
        
        # Calculate metrics
        results = {}
        for metric in metrics:
            if metric.lower() == 'rmse':
                results['rmse'] = np.sqrt(mean_squared_error(y, y_pred))
            elif metric.lower() == 'mae':
                results['mae'] = mean_absolute_error(y, y_pred)
            elif metric.lower() == 'r2':
                results['r2'] = r2_score(y, y_pred)
            elif metric.lower() == 'mape':
                # Handle zero values
                mask = y != 0
                results['mape'] = np.mean(np.abs((y[mask] - y_pred[mask]) / y[mask])) * 100
        
        return results
    
    def plot_forecast(self, df, forecast_horizon=30, figsize=(12, 6), title=None):
        """
        Plot historical data and forecasts.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with historical data
        forecast_horizon : int, optional
            Number of steps to forecast
        figsize : tuple, optional
            Figure size
        title : str, optional
            Plot title
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        if not self.trained:
            raise ValueError("Model must be trained before plotting forecast")
        
        # Generate forecast
        forecasts = self.recursive_forecast(df, forecast_horizon, include_history=False)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Determine x-axis (dates or integers)
        if isinstance(df.index, pd.DatetimeIndex):
            # Date index
            historical_x = df.index
            forecast_x = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_horizon)
        else:
            # Integer index
            historical_x = df.index
            forecast_x = range(df.index[-1] + 1, df.index[-1] + forecast_horizon + 1)
        
        # Plot historical data
        ax.plot(historical_x, df[self.target_column], label='Historical', color='blue')
        
        # Plot forecast
        ax.plot(forecast_x, forecasts, label='Forecast', color='red', linestyle='--')
        
        # Add shaded area to indicate forecast period
        ax.axvspan(historical_x[-1], forecast_x[-1], alpha=0.2, color='gray')
        
        # Add labels and legend
        ax.set_xlabel('Date' if isinstance(df.index, pd.DatetimeIndex) else 'Time')
        ax.set_ylabel(self.target_column)
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'LightGBM Forecast ({forecast_horizon} steps)')
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig


class LGBMHyperparameterTuner:
    """
    Hyperparameter tuning for LightGBM time series models.
    """
    
    def __init__(self, 
                 param_grid=None,
                 n_estimators=1000,
                 early_stopping_rounds=50,
                 cv=5,
                 parallel=True,
                 n_jobs=-1):
        """
        Initialize the hyperparameter tuner.
        
        Args:
            param_grid (dict): Dictionary of parameter grids
            n_estimators (int): Number of boosting iterations
            early_stopping_rounds (int): Early stopping patience
            cv (int): Number of cross-validation folds
            parallel (bool): Whether to use parallel processing
            n_jobs (int): Number of jobs for parallel processing
        """
        self.param_grid = param_grid or {
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [-1, 5, 7, 9],
            'num_leaves': [31, 63, 127],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
        }
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.cv = cv
        self.parallel = parallel
        self.n_jobs = n_jobs
        self.best_params = None
        self.best_score = float('inf')
        self.results = None
    
    def _create_param_combinations(self):
        """Create all parameter combinations from the parameter grid."""
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        param_combinations = list(product(*values))
        return [dict(zip(keys, combo)) for combo in param_combinations]
    
    def _evaluate_params(self, params, X, y, cv):
        """Evaluate a single parameter combination."""
        # Create TimeSeriesSplit for cross-validation
        tscv = TimeSeriesSplit(n_splits=cv)
        
        # Initialize scores
        scores = []
        
        # Run cross-validation
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Create LightGBM datasets
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            # Set fixed parameters
            fixed_params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'n_jobs': 1  # Avoid nested parallelism
            }
            
            # Combine fixed and tunable parameters
            lgb_params = {**fixed_params, **params}
            
            # Train model
            model = lgb.train(
                lgb_params,
                train_data,
                num_boost_round=self.n_estimators,
                valid_sets=[val_data],
                early_stopping_rounds=self.early_stopping_rounds,
                verbose_eval=False
            )
            
            # Make predictions and calculate RMSE
            y_pred = model.predict(X_val)
            score = np.sqrt(mean_squared_error(y_val, y_pred))
            scores.append(score)
        
        # Calculate average score
        mean_score = np.mean(scores)
        
        return {
            'params': params,
            'score': mean_score,
            'scores': scores
        }
    
    def tune(self, X, y):
        """
        Tune hyperparameters to find the best model.
        
        Args:
            X (pd.DataFrame): Feature DataFrame
            y (pd.Series): Target series
            
        Returns:
            dict: Best parameters
        """
        logger.info("Starting hyperparameter tuning for LightGBM...")
        
        # Create parameter combinations
        param_combinations = self._create_param_combinations()
        logger.info(f"Testing {len(param_combinations)} parameter combinations")
        
        # Run evaluation
        results = []
        
        if self.parallel:
            # Parallel evaluation
            with ProcessPoolExecutor(max_workers=self.n_jobs if self.n_jobs > 0 else None) as executor:
                futures = [executor.submit(self._evaluate_params, params, X, y, self.cv) 
                          for params in param_combinations]
                
                for future in futures:
                    result = future.result()
                    results.append(result)
                    
                    # Update best parameters if better
                    if result['score'] < self.best_score:
                        self.best_score = result['score']
                        self.best_params = result['params']
                    
                    logger.info(f"Tested parameters: {result['params']}, RMSE: {result['score']:.4f}")
        else:
            # Sequential evaluation
            for params in param_combinations:
                result = self._evaluate_params(params, X, y, self.cv)
                results.append(result)
                
                # Update best parameters if better
                if result['score'] < self.best_score:
                    self.best_score = result['score']
                    self.best_params = result['params']
                
                logger.info(f"Tested parameters: {result['params']}, RMSE: {result['score']:.4f}")
        
        # Sort results by score
        self.results = sorted(results, key=lambda x: x['score'])
        
        logger.info(f"Hyperparameter tuning complete. Best RMSE: {self.best_score:.4f}")
        logger.info(f"Best parameters: {self.best_params}")
        
        return self.best_params
    
    def get_best_params(self):
        """Get the best parameters from tuning."""
        if self.best_params is None:
            raise ValueError("Tuning must be performed before getting best parameters")
        
        return self.best_params


def forecast_multiple_store_product(df, store_col, product_col, date_col, target_col,
                                  forecast_horizon=30, parallel=True, n_jobs=-1):
    """
    Forecast sales for multiple store-product combinations in parallel.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        store_col (str): Store column name
        product_col (str): Product column name
        date_col (str): Date column name
        target_col (str): Target column name
        forecast_horizon (int): Number of periods to forecast
        parallel (bool): Whether to use parallel processing
        n_jobs (int): Number of jobs for parallel processing
        
    Returns:
        pd.DataFrame: Combined forecast results
    """
    def _forecast_single(store_id, product_id, data):
        """Forecast a single store-product combination."""
        # Set date as index
        data = data.set_index(date_col)
        
        # Create and fit model
        model = TimeSeriesLGBMModel()
        model.fit(data, target_col)
        
        # Generate forecast
        forecast = model.recursive_forecast(data, forecast_horizon)
        
        # Add store and product IDs
        forecast['store_id'] = store_id
        forecast['product_id'] = product_id
        
        return forecast
    
    logger.info(f"Forecasting {forecast_horizon} periods for multiple store-product combinations...")
    
    # Get unique store-product combinations
    combinations = df[[store_col, product_col]].drop_duplicates()
    
    forecasts = []
    
    if parallel and len(combinations) > 1:
        # Parallel forecasting
        with ProcessPoolExecutor(max_workers=n_jobs if n_jobs > 0 else None) as executor:
            futures = []
            
            for _, row in combinations.iterrows():
                store_id = row[store_col]
                product_id = row[product_col]
                
                # Filter data for this store-product
                store_product_data = df[(df[store_col] == store_id) & 
                                       (df[product_col] == product_id)].copy()
                
                futures.append(executor.submit(_forecast_single, store_id, product_id, store_product_data))
            
            for future in futures:
                forecast = future.result()
                forecasts.append(forecast)
    else:
        # Sequential forecasting
        for _, row in combinations.iterrows():
            store_id = row[store_col]
            product_id = row[product_col]
            
            # Filter data for this store-product
            store_product_data = df[(df[store_col] == store_id) & 
                                   (df[product_col] == product_id)].copy()
            
            forecast = _forecast_single(store_id, product_id, store_product_data)
            forecasts.append(forecast)
    
    # Combine all forecasts
    combined_forecast = pd.concat(forecasts, ignore_index=True)
    
    logger.info("Multiple store-product forecasting complete")
    return combined_forecast 