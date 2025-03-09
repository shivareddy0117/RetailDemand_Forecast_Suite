#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Distributed model training module using PySpark.
This module provides functions to train forecasting models at scale.
"""

import pandas as pd
import numpy as np
import os
import logging
import json
import pickle
from datetime import datetime
from functools import partial

# PySpark imports
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, pandas_udf, PandasUDFType, window, explode, array
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, ArrayType, DateType, IntegerType
import pyspark.sql.functions as F

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_spark_session(app_name="RetailForecasting", master="local[*]", config=None):
    """
    Create and configure a Spark session.
    
    Args:
        app_name (str): Name of the Spark application
        master (str): Master URL of the Spark cluster
        config (dict): Additional Spark configuration options
        
    Returns:
        SparkSession: Configured Spark session
    """
    try:
        # Start with a basic builder
        builder = SparkSession.builder.appName(app_name).master(master)
        
        # Add default configurations for better performance
        builder = builder.config("spark.executor.memory", "4g") \
                        .config("spark.driver.memory", "6g") \
                        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
                        .config("spark.sql.shuffle.partitions", "100")
        
        # Add additional configurations if provided
        if config:
            for key, value in config.items():
                builder = builder.config(key, value)
        
        # Create and return the session
        spark = builder.getOrCreate()
        
        logger.info(f"Created Spark session with app name: {app_name}")
        return spark
    
    except ImportError:
        logger.error("PySpark is not installed. Install it with: pip install pyspark")
        raise


def load_data_to_spark(spark, data_path, format_type="csv", header=True, inferSchema=True, options=None):
    """
    Load data into a Spark DataFrame.
    
    Args:
        spark (SparkSession): Active Spark session
        data_path (str): Path to the data file or directory
        format_type (str): File format type (csv, parquet, json, etc.)
        header (bool): Whether the file has a header (for CSV)
        inferSchema (bool): Whether to infer the schema
        options (dict): Additional options for data loading
        
    Returns:
        pyspark.sql.DataFrame: Spark DataFrame
    """
    # Set up basic options
    read_options = {"header": header, "inferSchema": inferSchema}
    
    # Add additional options if provided
    if options:
        read_options.update(options)
    
    # Load the data
    df = spark.read.format(format_type).options(**read_options).load(data_path)
    
    logger.info(f"Loaded data from {data_path} into Spark DataFrame with {df.count()} rows and {len(df.columns)} columns")
    return df


def partition_data_by_store_product(spark_df, date_col="date", store_col="store_id", product_col="product_id"):
    """
    Partition data by store and product for parallel processing.
    
    Args:
        spark_df (pyspark.sql.DataFrame): Input Spark DataFrame
        date_col (str): Name of the date column
        store_col (str): Name of the store ID column
        product_col (str): Name of the product ID column
        
    Returns:
        pyspark.sql.DataFrame: Partitioned Spark DataFrame
    """
    # Ensure date column is properly formatted
    spark_df = spark_df.withColumn(date_col, F.to_date(col(date_col)))
    
    # Repartition the data by store and product
    partitioned_df = spark_df.repartition(col(store_col), col(product_col))
    
    logger.info("Partitioned data by store and product")
    return partitioned_df


def preprocess_data_parallel(spark_df, date_col="date", group_cols=None):
    """
    Preprocess data in parallel using Spark.
    
    Args:
        spark_df (pyspark.sql.DataFrame): Input Spark DataFrame
        date_col (str): Name of the date column
        group_cols (list): Columns to group by for parallel processing
        
    Returns:
        pyspark.sql.DataFrame: Preprocessed Spark DataFrame
    """
    if group_cols is None:
        group_cols = ["store_id", "product_id"]
    
    # Register a UDF for preprocessing
    schema = spark_df.schema
    
    # Define the UDF for preprocessing
    @pandas_udf(schema, PandasUDFType.GROUPED_MAP)
    def preprocess_group(group_df):
        # Convert to pandas DataFrame
        pandas_df = group_df.copy()
        
        # Ensure date is sorted
        pandas_df = pandas_df.sort_values(by=date_col)
        
        # Fill missing values
        numeric_cols = pandas_df.select_dtypes(include=['number']).columns
        pandas_df[numeric_cols] = pandas_df[numeric_cols].fillna(method='ffill').fillna(method='bfill')
        
        # Handle outliers using IQR method
        for col in numeric_cols:
            if col != date_col and pandas_df[col].nunique() > 1:
                Q1 = pandas_df[col].quantile(0.25)
                Q3 = pandas_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                pandas_df[col] = pandas_df[col].clip(lower_bound, upper_bound)
        
        return pandas_df
    
    # Apply the UDF to process each group in parallel
    preprocessed_df = spark_df.groupby(*group_cols).apply(preprocess_group)
    
    logger.info("Applied parallel preprocessing to data")
    return preprocessed_df


def create_time_features_spark(spark_df, date_col="date"):
    """
    Create time-based features using Spark functions.
    
    Args:
        spark_df (pyspark.sql.DataFrame): Input Spark DataFrame
        date_col (str): Name of the date column
        
    Returns:
        pyspark.sql.DataFrame: Spark DataFrame with time features
    """
    # Convert to date type if needed
    spark_df = spark_df.withColumn(date_col, F.to_date(col(date_col)))
    
    # Extract date components
    spark_df = spark_df.withColumn("year", F.year(col(date_col)))
    spark_df = spark_df.withColumn("month", F.month(col(date_col)))
    spark_df = spark_df.withColumn("day", F.dayofmonth(col(date_col)))
    spark_df = spark_df.withColumn("day_of_week", F.dayofweek(col(date_col)))
    spark_df = spark_df.withColumn("day_of_year", F.dayofyear(col(date_col)))
    spark_df = spark_df.withColumn("week_of_year", F.weekofyear(col(date_col)))
    spark_df = spark_df.withColumn("quarter", F.quarter(col(date_col)))
    
    # Create weekend flag
    spark_df = spark_df.withColumn("is_weekend", 
                                  F.when((col("day_of_week") == 1) | (col("day_of_week") == 7), 1).otherwise(0))
    
    # Create month start/end flags
    spark_df = spark_df.withColumn("is_month_start", F.when(col("day") == 1, 1).otherwise(0))
    spark_df = spark_df.withColumn("is_month_end", 
                                  F.when(F.dayofmonth(F.last_day(col(date_col))) == col("day"), 1).otherwise(0))
    
    # Create cyclical features for day of week, month, and day of year
    spark_df = spark_df.withColumn("day_of_week_sin", F.sin(2 * np.pi * col("day_of_week") / 7))
    spark_df = spark_df.withColumn("day_of_week_cos", F.cos(2 * np.pi * col("day_of_week") / 7))
    spark_df = spark_df.withColumn("month_sin", F.sin(2 * np.pi * col("month") / 12))
    spark_df = spark_df.withColumn("month_cos", F.cos(2 * np.pi * col("month") / 12))
    
    logger.info("Created time-based features using Spark")
    return spark_df


def create_lag_features_spark(spark_df, target_col, group_cols=None, lag_periods=None):
    """
    Create lag features using Spark window functions.
    
    Args:
        spark_df (pyspark.sql.DataFrame): Input Spark DataFrame
        target_col (str): Target column for which to create lags
        group_cols (list): Columns to group by
        lag_periods (list): List of lag periods to create
        
    Returns:
        pyspark.sql.DataFrame: Spark DataFrame with lag features
    """
    if group_cols is None:
        group_cols = ["store_id", "product_id"]
    
    if lag_periods is None:
        lag_periods = [1, 7, 14, 28]
    
    # Create a window specification for each group, ordered by date
    window_spec = window.Window.partitionBy(*group_cols).orderBy("date")
    
    # Create lag features
    for lag in lag_periods:
        lag_col_name = f"{target_col}_lag_{lag}"
        spark_df = spark_df.withColumn(lag_col_name, F.lag(target_col, lag).over(window_spec))
    
    logger.info(f"Created {len(lag_periods)} lag features for {target_col}")
    return spark_df


def create_rolling_features_spark(spark_df, target_col, group_cols=None, windows=None, funcs=None):
    """
    Create rolling window features using Spark.
    
    Args:
        spark_df (pyspark.sql.DataFrame): Input Spark DataFrame
        target_col (str): Target column for which to create rolling features
        group_cols (list): Columns to group by
        windows (list): List of window sizes
        funcs (list): List of functions to apply ('mean', 'min', 'max', 'std')
        
    Returns:
        pyspark.sql.DataFrame: Spark DataFrame with rolling features
    """
    if group_cols is None:
        group_cols = ["store_id", "product_id"]
    
    if windows is None:
        windows = [7, 14, 28]
    
    if funcs is None:
        funcs = ["mean", "std", "min", "max"]
    
    # Use pandas UDF for rolling window calculations
    schema = StructType([
        StructField("group_key", StringType(), True),
        StructField("date", DateType(), True),
        *[StructField(f"{target_col}_rolling_{w}_{f}", DoubleType(), True) 
          for w in windows for f in funcs]
    ])
    
    @pandas_udf(schema, PandasUDFType.GROUPED_MAP)
    def create_rolling_features_udf(group_df):
        # Sort by date
        df = group_df.sort_values("date")
        
        # Create a group key for joining later
        group_vals = [str(group_df[col].iloc[0]) for col in group_cols]
        group_key = "_".join(group_vals)
        
        # Initialize result with date and group key
        result = pd.DataFrame({"group_key": group_key, "date": df["date"]})
        
        # Create rolling features
        for window_size in windows:
            for func_name in funcs:
                col_name = f"{target_col}_rolling_{window_size}_{func_name}"
                if func_name == "mean":
                    result[col_name] = df[target_col].rolling(window=window_size).mean()
                elif func_name == "std":
                    result[col_name] = df[target_col].rolling(window=window_size).std()
                elif func_name == "min":
                    result[col_name] = df[target_col].rolling(window=window_size).min()
                elif func_name == "max":
                    result[col_name] = df[target_col].rolling(window=window_size).max()
        
        return result
    
    # Apply the UDF
    rolling_features = spark_df.groupby(*group_cols).apply(create_rolling_features_udf)
    
    # Join with original DataFrame
    join_cond = [rolling_features["date"] == spark_df["date"]]
    for i, col in enumerate(group_cols):
        join_cond.append(rolling_features["group_key"].contains(col))
    
    result_df = spark_df.join(rolling_features, on=join_cond, how="left")
    
    logger.info(f"Created {len(windows) * len(funcs)} rolling features for {target_col}")
    return result_df


def train_prophet_distributed(spark, input_df, output_path, group_cols=None, target_col="sales", 
                           date_col="date", forecast_periods=30, freq="D"):
    """
    Train Prophet models in a distributed manner.
    
    Args:
        spark (SparkSession): Spark session
        input_df (pyspark.sql.DataFrame): Input data
        output_path (str): Path to save results
        group_cols (list): Columns to group by
        target_col (str): Column to forecast
        date_col (str): Date column
        forecast_periods (int): Periods to forecast
        freq (str): Frequency for forecasting
        
    Returns:
        pyspark.sql.DataFrame: Forecast results
    """
    from pyspark.sql.types import StructType, StructField, StringType, DoubleType, DateType
    from pyspark.sql.functions import pandas_udf, PandasUDFType
    
    if group_cols is None:
        group_cols = ["store_id", "product_id"]
    
    # Create a schema for the results
    result_schema = StructType([
        StructField("group_id", StringType(), True),
        StructField("ds", DateType(), True),
        StructField("y", DoubleType(), True),
        StructField("yhat", DoubleType(), True),
        StructField("yhat_lower", DoubleType(), True),
        StructField("yhat_upper", DoubleType(), True)
    ])
    
    # Define a Pandas UDF to train and predict with Prophet
    @pandas_udf(result_schema, PandasUDFType.GROUPED_MAP)
    def forecast_prophet(group_df):
        try:
            from prophet import Prophet
            import pandas as pd
            
            # Create a group ID
            group_id = "_".join([str(group_df[col].iloc[0]) for col in group_cols])
            
            # Prepare data for Prophet
            prophet_df = pd.DataFrame({
                'ds': pd.to_datetime(group_df[date_col]),
                'y': group_df[target_col]
            })
            
            # Create and fit model
            model = Prophet()
            model.fit(prophet_df)
            
            # Create future dataframe and predict
            future = model.make_future_dataframe(periods=forecast_periods, freq=freq)
            forecast = model.predict(future)
            
            # Add group ID and original values
            forecast['group_id'] = group_id
            
            # Merge with original data to get actual values
            result = pd.merge(
                forecast[['group_id', 'ds', 'yhat', 'yhat_lower', 'yhat_upper']],
                prophet_df[['ds', 'y']],
                on='ds',
                how='left'
            )
            
            # Save model if output path is provided
            if output_path:
                os.makedirs(output_path, exist_ok=True)
                with open(f"{output_path}/prophet_{group_id}.pkl", 'wb') as f:
                    pickle.dump(model, f)
            
            return result
        
        except Exception as e:
            # Log error and return empty DataFrame
            logger.error(f"Error in Prophet training: {str(e)}")
            return pd.DataFrame(columns=['group_id', 'ds', 'y', 'yhat', 'yhat_lower', 'yhat_upper'])
    
    # Apply the UDF to each group
    result_df = input_df.groupby(*group_cols).apply(forecast_prophet)
    
    # Cache the result
    result_df.cache()
    
    # Log completion
    logger.info(f"Completed Prophet training for {result_df.select('group_id').distinct().count()} groups")
    
    return result_df


def train_lgb_distributed(spark, input_df, output_path, group_cols=None, target_col="sales", 
                        date_col="date", feature_cols=None, forecast_periods=30):
    """
    Train LightGBM models in a distributed manner.
    
    Args:
        spark (SparkSession): Spark session
        input_df (pyspark.sql.DataFrame): Input data
        output_path (str): Path to save results
        group_cols (list): Columns to group by
        target_col (str): Column to forecast
        date_col (str): Date column
        feature_cols (list): Feature columns to use
        forecast_periods (int): Periods to forecast
        
    Returns:
        pyspark.sql.DataFrame: Forecast results
    """
    from pyspark.sql.types import StructType, StructField, StringType, DoubleType, DateType
    from pyspark.sql.functions import pandas_udf, PandasUDFType
    
    if group_cols is None:
        group_cols = ["store_id", "product_id"]
    
    # Get all column names if feature_cols is None
    if feature_cols is None:
        feature_cols = [col for col in input_df.columns 
                      if col not in [*group_cols, date_col, target_col]]
    
    # Create a schema for the results
    result_schema = StructType([
        StructField("group_id", StringType(), True),
        StructField("ds", DateType(), True),
        StructField("y", DoubleType(), True),
        StructField("yhat", DoubleType(), True)
    ])
    
    # Define a Pandas UDF to train and predict with LightGBM
    @pandas_udf(result_schema, PandasUDFType.GROUPED_MAP)
    def forecast_lgb(group_df):
        try:
            import lightgbm as lgb
            import pandas as pd
            
            # Create a group ID
            group_id = "_".join([str(group_df[col].iloc[0]) for col in group_cols])
            
            # Prepare data
            df = group_df.copy()
            df['ds'] = pd.to_datetime(df[date_col])
            df = df.sort_values('ds')
            
            # Create lag features
            for lag in [1, 7, 14, 28]:
                df[f'lag_{lag}'] = df[target_col].shift(lag)
            
            # Add date features
            df['day_of_week'] = df['ds'].dt.dayofweek
            df['day_of_month'] = df['ds'].dt.day
            df['month'] = df['ds'].dt.month
            df['year'] = df['ds'].dt.year
            
            # Additional features
            date_features = ['day_of_week', 'day_of_month', 'month', 'year']
            lag_features = [f'lag_{lag}' for lag in [1, 7, 14, 28]]
            all_features = feature_cols + date_features + lag_features
            
            # Drop NaNs
            df = df.dropna()
            
            # Train-test split
            train_size = int(len(df) * 0.8)
            train_df = df.iloc[:train_size]
            test_df = df.iloc[train_size:]
            
            # Prepare training data
            X_train = train_df[all_features]
            y_train = train_df[target_col]
            
            # Train model
            model = lgb.LGBMRegressor(
                objective='regression',
                n_estimators=100,
                learning_rate=0.05,
                num_leaves=31,
                verbosity=-1
            )
            model.fit(X_train, y_train)
            
            # Save model
            if output_path:
                os.makedirs(output_path, exist_ok=True)
                model.booster_.save_model(f"{output_path}/lgb_{group_id}.txt")
            
            # Generate forecast
            last_date = df['ds'].iloc[-1]
            future_dates = pd.date_range(start=last_date, periods=forecast_periods+1)[1:]
            
            # Prepare result DataFrame
            result = pd.DataFrame({
                'group_id': group_id,
                'ds': pd.concat([df['ds'], pd.Series(future_dates)]),
                'y': pd.concat([df[target_col], pd.Series([None] * forecast_periods)])
            })
            
            # Generate predictions for historical data
            historical_features = df[all_features]
            historical_preds = model.predict(historical_features)
            
            # Initialize future predictions
            future_preds = []
            
            # Recursively predict future values
            future_row = df.iloc[-1:].copy()
            
            for future_date in future_dates:
                # Update date features
                future_row['ds'] = future_date
                future_row['day_of_week'] = future_date.dayofweek
                future_row['day_of_month'] = future_date.day
                future_row['month'] = future_date.month
                future_row['year'] = future_date.year
                
                # Make prediction
                pred = model.predict(future_row[all_features])[0]
                future_preds.append(pred)
                
                # Update lags for next prediction
                future_row[f'lag_1'] = pred
                if len(future_preds) >= 7:
                    future_row[f'lag_7'] = future_preds[-7]
                if len(future_preds) >= 14:
                    future_row[f'lag_14'] = future_preds[-14]
                if len(future_preds) >= 28:
                    future_row[f'lag_28'] = future_preds[-28]
            
            # Combine predictions
            all_preds = np.concatenate([historical_preds, future_preds])
            result['yhat'] = all_preds
            
            # Return only the columns in the schema
            return result[['group_id', 'ds', 'y', 'yhat']]
        
        except Exception as e:
            # Log error and return empty DataFrame
            logger.error(f"Error in LightGBM training: {str(e)}")
            return pd.DataFrame(columns=['group_id', 'ds', 'y', 'yhat'])
    
    # Apply the UDF to each group
    result_df = input_df.groupby(*group_cols).apply(forecast_lgb)
    
    # Cache the result
    result_df.cache()
    
    # Log completion
    logger.info(f"Completed LightGBM training for {result_df.select('group_id').distinct().count()} groups")
    
    return result_df


def ensemble_forecasts_distributed(spark, forecast_dfs, output_path, weights=None):
    """
    Create ensemble forecasts from multiple models in a distributed manner.
    
    Args:
        spark (SparkSession): Spark session
        forecast_dfs (dict): Dictionary of forecast DataFrames by model name
        output_path (str): Path to save results
        weights (dict): Dictionary of model weights
        
    Returns:
        pyspark.sql.DataFrame: Ensemble forecast results
    """
    from pyspark.sql.types import StructType, StructField, StringType, DoubleType, DateType
    from pyspark.sql.functions import col, expr
    
    # Default to equal weights if not provided
    if weights is None:
        model_names = list(forecast_dfs.keys())
        weights = {model: 1.0 / len(model_names) for model in model_names}
    
    # Create a common key for joining
    for model_name, df in forecast_dfs.items():
        df = df.withColumn("model", expr(f"'{model_name}'"))
        forecast_dfs[model_name] = df
    
    # Union all forecasts
    all_forecasts = None
    for df in forecast_dfs.values():
        if all_forecasts is None:
            all_forecasts = df
        else:
            all_forecasts = all_forecasts.union(df)
    
    # Pivot to get model predictions side by side
    pivoted = all_forecasts.groupBy("group_id", "ds").pivot("model").agg({"yhat": "first"})
    
    # Create weighted ensemble
    ensemble_expr = " + ".join([f"{weights[model]} * {model}" for model in weights.keys()])
    pivoted = pivoted.withColumn("ensemble", expr(ensemble_expr))
    
    # Add actual values
    pivoted = pivoted.join(
        all_forecasts.select("group_id", "ds", "y").dropDuplicates(),
        on=["group_id", "ds"],
        how="left"
    )
    
    # Save results if output path is provided
    if output_path:
        pivoted.write.parquet(f"{output_path}/ensemble_forecasts")
    
    # Log completion
    logger.info(f"Created ensemble forecasts for {pivoted.select('group_id').distinct().count()} groups")
    
    return pivoted


def evaluate_forecasts_distributed(spark, forecast_df, metrics=None):
    """
    Evaluate forecasts in a distributed manner.
    
    Args:
        spark (SparkSession): Spark session
        forecast_df (pyspark.sql.DataFrame): Forecast DataFrame
        metrics (list): List of metrics to calculate
        
    Returns:
        pyspark.sql.DataFrame: Evaluation metrics
    """
    from pyspark.sql.types import StructType, StructField, StringType, DoubleType
    from pyspark.sql.functions import pandas_udf, PandasUDFType, col
    
    if metrics is None:
        metrics = ["rmse", "mae", "mape"]
    
    # Create a schema for the metrics
    metrics_schema = StructType([
        StructField("group_id", StringType(), True),
        StructField("model", StringType(), True),
        *[StructField(metric, DoubleType(), True) for metric in metrics]
    ])
    
    # Collect list of all model columns (including ensemble)
    model_cols = [col for col in forecast_df.columns 
                if col not in ["group_id", "ds", "y"]]
    
    # Define a Pandas UDF to calculate metrics
    @pandas_udf(metrics_schema, PandasUDFType.GROUPED_MAP)
    def calculate_metrics(group_df):
        results = []
        
        # Calculate metrics for each model
        for model in model_cols:
            # Filter out missing values
            df = group_df.dropna(subset=["y", model])
            
            if len(df) > 0:
                y_true = df["y"]
                y_pred = df[model]
                
                # Initialize metrics dictionary
                metrics_dict = {"group_id": group_df["group_id"].iloc[0], "model": model}
                
                # Calculate each metric
                if "rmse" in metrics:
                    metrics_dict["rmse"] = np.sqrt(np.mean((y_true - y_pred) ** 2))
                
                if "mae" in metrics:
                    metrics_dict["mae"] = np.mean(np.abs(y_true - y_pred))
                
                if "mape" in metrics:
                    # Avoid division by zero
                    mask = y_true != 0
                    if mask.any():
                        metrics_dict["mape"] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
                    else:
                        metrics_dict["mape"] = None
                
                results.append(metrics_dict)
        
        return pd.DataFrame(results)
    
    # Apply the UDF to each group
    metrics_df = forecast_df.groupBy("group_id").apply(calculate_metrics)
    
    # Log completion
    logger.info(f"Calculated evaluation metrics for {metrics_df.select('group_id').distinct().count()} groups")
    
    return metrics_df


def main():
    """Main function to run distributed training."""
    try:
        # Create Spark session
        spark = create_spark_session()
        
        # Set up paths
        data_path = "../retail_data"
        output_path = "../models/spark_models"
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Load data
        sales_df = load_data_to_spark(spark, f"{data_path}/sales.csv")
        stores_df = load_data_to_spark(spark, f"{data_path}/stores.csv")
        products_df = load_data_to_spark(spark, f"{data_path}/products.csv")
        
        # Join data
        from pyspark.sql.functions import col
        
        joined_df = sales_df.join(
            stores_df, on="store_id"
        ).join(
            products_df, on="product_id"
        )
        
        # Train Prophet models
        prophet_forecasts = train_prophet_distributed(
            spark, joined_df, f"{output_path}/prophet",
            target_col="sales_quantity"
        )
        
        # Train LightGBM models
        lgb_forecasts = train_lgb_distributed(
            spark, joined_df, f"{output_path}/lgb",
            target_col="sales_quantity"
        )
        
        # Create ensemble forecasts
        ensemble_forecasts = ensemble_forecasts_distributed(
            spark, 
            {"prophet": prophet_forecasts, "lgb": lgb_forecasts}, 
            f"{output_path}/ensemble"
        )
        
        # Evaluate forecasts
        metrics = evaluate_forecasts_distributed(spark, ensemble_forecasts)
        
        # Show average metrics by model
        metrics.groupBy("model").avg("rmse", "mae", "mape").show()
        
        # Stop Spark session
        spark.stop()
        
        logger.info("Distributed training completed successfully")
    
    except Exception as e:
        logger.error(f"Error in distributed training: {str(e)}")
        # Stop Spark session if exists
        try:
            if 'spark' in locals():
                spark.stop()
        except:
            pass


if __name__ == "__main__":
    main() 