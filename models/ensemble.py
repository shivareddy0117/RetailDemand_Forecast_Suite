#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ensemble modeling for time series forecasting.
This module provides methods for combining forecasts from different models.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

class ForecastEnsemble:
    """
    Ensemble model for combining forecasts from multiple models.
    """
    
    def __init__(self, method='simple_average', weights=None, meta_model=None):
        """
        Initialize ForecastEnsemble.
        
        Parameters:
        -----------
        method : str, optional
            Ensemble method:
            - 'simple_average': Simple average of forecasts
            - 'weighted_average': Weighted average of forecasts
            - 'median': Median of forecasts
            - 'meta_learner': Train a meta-model on forecasts
        weights : dict, optional
            Dictionary of weights for each model (only for weighted_average)
        meta_model : object, optional
            Meta-model for stacking (sklearn-compatible regressor)
        """
        self.method = method
        self.weights = weights
        self.meta_model = meta_model
        self.trained = False
        self.model_weights = None
        self.model_scores = None
        self.models = None
    
    def fit(self, forecasts, actuals):
        """
        Fit the ensemble to the data.
        
        Parameters:
        -----------
        forecasts : dict or DataFrame
            Dictionary or DataFrame of forecasts from different models
        actuals : array-like
            Actual values
            
        Returns:
        --------
        self
        """
        # Store model names
        if isinstance(forecasts, dict):
            self.models = list(forecasts.keys())
            forecasts_array = np.column_stack([forecasts[model] for model in self.models])
        else:
            # Assume DataFrame where columns are model names
            self.models = forecasts.columns.tolist()
            forecasts_array = forecasts.values
        
        # Convert actuals to numpy array
        actuals = np.array(actuals)
        
        # Check dimensions
        if len(actuals) != forecasts_array.shape[0]:
            raise ValueError("Number of actual values must match number of forecasts")
        
        # Calculate model-specific performance metrics
        self.model_scores = {}
        for i, model in enumerate(self.models):
            forecast = forecasts_array[:, i]
            
            # Calculate error metrics
            rmse = np.sqrt(mean_squared_error(actuals, forecast))
            mae = mean_absolute_error(actuals, forecast)
            
            # Store scores
            self.model_scores[model] = {
                'rmse': rmse,
                'mae': mae
            }
        
        # Determine or train weights based on method
        if self.method == 'simple_average':
            # Equal weights for all models
            self.model_weights = {model: 1.0 / len(self.models) for model in self.models}
            
        elif self.method == 'weighted_average':
            if self.weights is not None:
                # Use user-provided weights
                self.model_weights = self.weights
            else:
                # Calculate weights based on inverse RMSE
                inverse_rmse = {model: 1.0 / self.model_scores[model]['rmse'] for model in self.models}
                total_inverse_rmse = sum(inverse_rmse.values())
                self.model_weights = {model: inverse_rmse[model] / total_inverse_rmse for model in self.models}
        
        elif self.method == 'median':
            # No weights needed for median
            self.model_weights = None
            
        elif self.method == 'meta_learner':
            # Train a meta-learner on the forecasts
            if self.meta_model is None:
                # Default to Ridge regression
                self.meta_model = Ridge(alpha=1.0)
            
            # Train meta-model
            self.meta_model.fit(forecasts_array, actuals)
            
        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")
        
        self.trained = True
        return self
    
    def predict(self, forecasts):
        """
        Generate ensemble forecast.
        
        Parameters:
        -----------
        forecasts : dict or DataFrame
            Dictionary or DataFrame of forecasts from different models
            
        Returns:
        --------
        numpy.ndarray
            Ensemble forecast
        """
        if not self.trained:
            raise ValueError("Ensemble must be trained before making predictions")
        
        # Convert forecasts to array
        if isinstance(forecasts, dict):
            # Dictionary of model forecasts
            forecast_models = list(forecasts.keys())
            # Check if all trained models are present
            missing_models = [model for model in self.models if model not in forecast_models]
            if missing_models:
                raise ValueError(f"Missing forecasts for models: {missing_models}")
            
            forecasts_array = np.column_stack([forecasts[model] for model in self.models])
        else:
            # Assume DataFrame where columns are model names
            forecast_models = forecasts.columns.tolist()
            # Check if all trained models are present
            missing_models = [model for model in self.models if model not in forecast_models]
            if missing_models:
                raise ValueError(f"Missing forecasts for models: {missing_models}")
            
            forecasts_array = forecasts[self.models].values
        
        # Generate ensemble forecast based on method
        if self.method == 'simple_average':
            # Simple average of all forecasts
            ensemble_forecast = np.mean(forecasts_array, axis=1)
            
        elif self.method == 'weighted_average':
            # Weighted average of forecasts
            weighted_forecasts = np.zeros(forecasts_array.shape[0])
            for i, model in enumerate(self.models):
                weighted_forecasts += self.model_weights[model] * forecasts_array[:, i]
            
            ensemble_forecast = weighted_forecasts
            
        elif self.method == 'median':
            # Median of forecasts
            ensemble_forecast = np.median(forecasts_array, axis=1)
            
        elif self.method == 'meta_learner':
            # Predictions from meta-learner
            ensemble_forecast = self.meta_model.predict(forecasts_array)
            
        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")
        
        return ensemble_forecast
    
    def get_weights(self):
        """
        Get model weights or importance scores.
        
        Returns:
        --------
        dict
            Weights or importance scores for each model
        """
        if not self.trained:
            raise ValueError("Ensemble must be trained before getting weights")
        
        if self.method == 'meta_learner':
            # For meta-learners, return coefficients or feature importances
            if hasattr(self.meta_model, 'coef_'):
                # Linear models
                coef = self.meta_model.coef_
                return {model: coef[i] for i, model in enumerate(self.models)}
            elif hasattr(self.meta_model, 'feature_importances_'):
                # Tree-based models
                importances = self.meta_model.feature_importances_
                return {model: importances[i] for i, model in enumerate(self.models)}
            else:
                raise ValueError("Meta-model does not expose coefficients or feature importances")
        else:
            # For other methods, return weights
            return self.model_weights if self.model_weights is not None else {model: 1.0 / len(self.models) for model in self.models}
    
    def get_model_scores(self):
        """
        Get performance scores for each component model.
        
        Returns:
        --------
        pandas.DataFrame
            Performance metrics for each model
        """
        if not self.trained:
            raise ValueError("Ensemble must be trained before getting model scores")
        
        # Convert scores to DataFrame
        scores_df = pd.DataFrame(self.model_scores).T
        
        return scores_df
    
    def evaluate(self, forecasts, actuals, metrics=None):
        """
        Evaluate ensemble performance.
        
        Parameters:
        -----------
        forecasts : dict or DataFrame
            Dictionary or DataFrame of forecasts from different models
        actuals : array-like
            Actual values
        metrics : list, optional
            List of metrics to calculate
            
        Returns:
        --------
        dict
            Dictionary of performance metrics
        """
        if not self.trained:
            raise ValueError("Ensemble must be trained before evaluation")
        
        # Default metrics
        if metrics is None:
            metrics = ['rmse', 'mae', 'r2']
        
        # Generate ensemble forecast
        ensemble_forecast = self.predict(forecasts)
        
        # Convert actuals to numpy array
        actuals = np.array(actuals)
        
        # Calculate metrics
        results = {}
        for metric in metrics:
            if metric.lower() == 'rmse':
                results['rmse'] = np.sqrt(mean_squared_error(actuals, ensemble_forecast))
            elif metric.lower() == 'mae':
                results['mae'] = mean_absolute_error(actuals, ensemble_forecast)
            elif metric.lower() == 'r2':
                results['r2'] = r2_score(actuals, ensemble_forecast)
            elif metric.lower() == 'mape':
                # Handle zero values
                mask = actuals != 0
                results['mape'] = np.mean(np.abs((actuals[mask] - ensemble_forecast[mask]) / actuals[mask])) * 100
        
        return results

def create_stacked_ensemble(forecasts, actuals, meta_models=None, cv=5):
    """
    Create a stacked ensemble using multiple meta-models.
    
    Parameters:
    -----------
    forecasts : dict or DataFrame
        Dictionary or DataFrame of forecasts from different models
    actuals : array-like
        Actual values
    meta_models : list, optional
        List of meta-models to use
    cv : int, optional
        Number of cross-validation folds
        
    Returns:
    --------
    ForecastEnsemble
        Best performing ensemble
    """
    # Default meta-models
    if meta_models is None:
        meta_models = [
            ('ridge', Ridge(alpha=1.0)),
            ('lasso', Lasso(alpha=0.1)),
            ('elastic_net', ElasticNet(alpha=0.1, l1_ratio=0.5)),
            ('random_forest', RandomForestRegressor(n_estimators=100, random_state=42)),
            ('gradient_boosting', GradientBoostingRegressor(n_estimators=100, random_state=42))
        ]
    
    # Convert forecasts to array
    if isinstance(forecasts, dict):
        models = list(forecasts.keys())
        forecasts_array = np.column_stack([forecasts[model] for model in models])
    else:
        models = forecasts.columns.tolist()
        forecasts_array = forecasts.values
    
    # Convert actuals to numpy array
    actuals = np.array(actuals)
    
    # Set up cross-validation
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=cv)
    
    # Evaluate each meta-model
    best_score = float('inf')
    best_model = None
    best_model_name = None
    
    for name, model in meta_models:
        scores = []
        
        # Perform cross-validation
        for train_idx, val_idx in tscv.split(forecasts_array):
            X_train, X_val = forecasts_array[train_idx], forecasts_array[val_idx]
            y_train, y_val = actuals[train_idx], actuals[val_idx]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_val)
            score = np.sqrt(mean_squared_error(y_val, y_pred))
            scores.append(score)
        
        # Calculate average score
        avg_score = np.mean(scores)
        
        # Update best model if current is better
        if avg_score < best_score:
            best_score = avg_score
            best_model = model
            best_model_name = name
    
    # Create ensemble with best meta-model
    ensemble = ForecastEnsemble(method='meta_learner', meta_model=best_model)
    ensemble.fit(forecasts, actuals)
    
    return ensemble

def create_optimal_ensemble(forecasts, actuals, methods=None):
    """
    Create the best performing ensemble by trying different methods.
    
    Parameters:
    -----------
    forecasts : dict or DataFrame
        Dictionary or DataFrame of forecasts from different models
    actuals : array-like
        Actual values
    methods : list, optional
        List of methods to try
        
    Returns:
    --------
    ForecastEnsemble
        Best performing ensemble
    """
    # Default methods
    if methods is None:
        methods = ['simple_average', 'weighted_average', 'median', 'meta_learner']
    
    # Try each method
    best_score = float('inf')
    best_ensemble = None
    best_method = None
    
    for method in methods:
        # Create ensemble
        if method == 'meta_learner':
            # Try different meta-models
            ensemble = create_stacked_ensemble(forecasts, actuals)
        else:
            ensemble = ForecastEnsemble(method=method)
            ensemble.fit(forecasts, actuals)
        
        # Evaluate
        ensemble_forecast = ensemble.predict(forecasts)
        score = np.sqrt(mean_squared_error(actuals, ensemble_forecast))
        
        # Update best ensemble if current is better
        if score < best_score:
            best_score = score
            best_ensemble = ensemble
            best_method = method
    
    return best_ensemble 