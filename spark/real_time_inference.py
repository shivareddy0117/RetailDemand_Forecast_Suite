#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Real-time inference module using PySpark.
This module provides functions for batch and streaming predictions.
"""

import pandas as pd
import numpy as np
import os
import logging
import json
import pickle
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_spark_session(app_name="RetailForecastingInference", master="local[*]", config=None):
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
        # PySpark imports
        from pyspark.sql import SparkSession
        
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


def load_models(model_dir, model_type):
    """
    Load trained models from a directory.
    
    Args:
        model_dir (str): Directory containing model files
        model_type (str): Type of model ('prophet', 'lgbm', 'var')
        
    Returns:
        dict: Dictionary of loaded models with IDs as keys
    """
    models = {}
    
    # Check if directory exists
    if not os.path.exists(model_dir):
        logger.error(f"Model directory {model_dir} does not exist")
        return models
    
    # Load models based on type
    for filename in os.listdir(model_dir):
        if model_type == 'prophet' and filename.endswith('.pkl'):
            model_id = filename.replace('.pkl', '')
            try:
                with open(os.path.join(model_dir, filename), 'rb') as f:
                    models[model_id] = pickle.load(f)
                logger.info(f"Loaded Prophet model: {model_id}")
            except Exception as e:
                logger.error(f"Error loading Prophet model {filename}: {str(e)}")
        
        elif model_type == 'lgbm' and filename.endswith('.txt'):
            model_id = filename.replace('.txt', '')
            try:
                import lightgbm as lgb
                models[model_id] = lgb.Booster(model_file=os.path.join(model_dir, filename))
                logger.info(f"Loaded LightGBM model: {model_id}")
            except Exception as e:
                logger.error(f"Error loading LightGBM model {filename}: {str(e)}")
        
        elif model_type == 'var' and filename.endswith('.pkl'):
            model_id = filename.replace('.pkl', '')
            try:
                with open(os.path.join(model_dir, filename), 'rb') as f:
                    models[model_id] = pickle.load(f)
                logger.info(f"Loaded VAR model: {model_id}")
            except Exception as e:
                logger.error(f"Error loading VAR model {filename}: {str(e)}")
    
    logger.info(f"Loaded {len(models)} {model_type} models")
    return models


def batch_inference(spark, input_df, models, model_type, group_cols=None, forecast_horizon=30):
    """
    Perform batch inference on a DataFrame.
    
    Args:
        spark (SparkSession): Spark session
        input_df (pyspark.sql.DataFrame): Input DataFrame
        models (dict): Dictionary of loaded models
        model_type (str): Type of model ('prophet', 'lgbm', 'var')
        group_cols (list): Group column names for inference
        forecast_horizon (int): Number of periods to forecast
        
    Returns:
        pyspark.sql.DataFrame: DataFrame with predictions
    """
    from pyspark.sql.functions import pandas_udf, PandasUDFType, col, lit, struct
    from pyspark.sql.types import StructType, StructField, StringType, DoubleType, DateType, ArrayType
    
    if group_cols is None:
        group_cols = ["store_id", "product_id"]
    
    # Define result schema based on model type
    if model_type == 'prophet':
        result_schema = StructType([
            *[StructField(col, StringType(), True) for col in group_cols],
            StructField("ds", DateType(), True),
            StructField("yhat", DoubleType(), True),
            StructField("yhat_lower", DoubleType(), True),
            StructField("yhat_upper", DoubleType(), True)
        ])
    else:
        result_schema = StructType([
            *[StructField(col, StringType(), True) for col in group_cols],
            StructField("ds", DateType(), True),
            StructField("yhat", DoubleType(), True)
        ])
    
    # Create a broadcast variable for models
    model_broadcast = spark.sparkContext.broadcast(models)
    
    # Define UDF for Prophet predictions
    @pandas_udf(result_schema, PandasUDFType.GROUPED_MAP)
    def predict_prophet(df):
        # Create model ID from group columns
        model_id = "_".join([str(df[col].iloc[0]) for col in group_cols])
        
        # Check if model exists
        if model_id not in model_broadcast.value:
            # Return empty DataFrame with correct schema
            empty_df = pd.DataFrame(columns=result_schema.fieldNames())
            return empty_df
        
        # Get the model
        model = model_broadcast.value[model_id]
        
        # Make future dataframe
        future = model.make_future_dataframe(periods=forecast_horizon)
        
        # Make forecast
        forecast = model.predict(future)
        
        # Add group columns
        for col in group_cols:
            forecast[col] = df[col].iloc[0]
        
        # Select relevant columns
        forecast = forecast[group_cols + ['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        
        return forecast
    
    # Define UDF for LightGBM predictions
    @pandas_udf(result_schema, PandasUDFType.GROUPED_MAP)
    def predict_lgbm(df):
        import numpy as np
        from datetime import timedelta
        
        # Create model ID from group columns
        model_id = "_".join([str(df[col].iloc[0]) for col in group_cols])
        
        # Check if model exists
        if model_id not in model_broadcast.value:
            # Return empty DataFrame with correct schema
            empty_df = pd.DataFrame(columns=result_schema.fieldNames())
            return empty_df
        
        # Get the model
        model = model_broadcast.value[model_id]
        
        # Sort by date
        df = df.sort_values('date')
        
        # Get feature columns (exclude group cols and date)
        feature_cols = [col for col in df.columns if col not in group_cols + ['date']]
        
        # Create future dates
        last_date = df['date'].iloc[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_horizon)
        
        # Initialize predictions DataFrame
        predictions = []
        
        # For simplicity, we'll just use the last row's features for all future predictions
        # In a real application, you'd need a more sophisticated approach
        features = df[feature_cols].iloc[-1:].values
        
        # Get the prediction
        yhat = model.predict(features)[0]
        
        # Add to predictions with group information
        for future_date in future_dates:
            pred_row = {
                'ds': future_date,
                'yhat': yhat
            }
            
            # Add group columns
            for col in group_cols:
                pred_row[col] = df[col].iloc[0]
            
            predictions.append(pred_row)
        
        # Convert to DataFrame
        predictions_df = pd.DataFrame(predictions)
        
        return predictions_df
    
    # Define UDF for VAR predictions
    @pandas_udf(result_schema, PandasUDFType.GROUPED_MAP)
    def predict_var(df):
        # Create model ID from group columns
        model_id = "_".join([str(df[col].iloc[0]) for col in group_cols])
        
        # Check if model exists
        if model_id not in model_broadcast.value:
            # Return empty DataFrame with correct schema
            empty_df = pd.DataFrame(columns=result_schema.fieldNames())
            return empty_df
        
        # Get the model
        model = model_broadcast.value[model_id]
        
        # Make forecast (implementation depends on VAR model structure)
        # This is a simplified example
        forecast = model.forecast(steps=forecast_horizon)
        
        # Convert to DataFrame
        forecast_df = pd.DataFrame({
            'ds': pd.date_range(start=df['date'].iloc[-1] + timedelta(days=1), periods=forecast_horizon),
            'yhat': forecast
        })
        
        # Add group columns
        for col in group_cols:
            forecast_df[col] = df[col].iloc[0]
        
        return forecast_df
    
    # Apply the appropriate UDF
    if model_type == 'prophet':
        result_df = input_df.groupBy(*group_cols).apply(predict_prophet)
    elif model_type == 'lgbm':
        result_df = input_df.groupBy(*group_cols).apply(predict_lgbm)
    elif model_type == 'var':
        result_df = input_df.groupBy(*group_cols).apply(predict_var)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    logger.info(f"Performed batch inference with {model_type} models")
    return result_df


def setup_streaming_source(spark, kafka_bootstrap_servers, kafka_topic, schema=None):
    """
    Set up a streaming source from Kafka.
    
    Args:
        spark (SparkSession): Spark session
        kafka_bootstrap_servers (str): Kafka bootstrap servers
        kafka_topic (str): Kafka topic to subscribe to
        schema (StructType): Schema of the data
        
    Returns:
        pyspark.sql.DataFrame: Streaming DataFrame
    """
    from pyspark.sql.functions import from_json, col
    from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType
    
    # Define default schema if not provided
    if schema is None:
        schema = StructType([
            StructField("store_id", StringType(), True),
            StructField("product_id", StringType(), True),
            StructField("timestamp", TimestampType(), True),
            StructField("sales", DoubleType(), True)
        ])
    
    # Read from Kafka
    kafka_df = spark \
        .readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", kafka_bootstrap_servers) \
        .option("subscribe", kafka_topic) \
        .load()
    
    # Parse value as JSON
    parsed_df = kafka_df \
        .selectExpr("CAST(value AS STRING)") \
        .select(from_json(col("value"), schema).alias("data")) \
        .select("data.*")
    
    logger.info(f"Set up streaming source from Kafka topic: {kafka_topic}")
    return parsed_df


def streaming_inference(spark, stream_df, models, model_type, output_path, checkpoint_path,
                     group_cols=None, forecast_horizon=30):
    """
    Perform streaming inference.
    
    Args:
        spark (SparkSession): Spark session
        stream_df (pyspark.sql.DataFrame): Streaming DataFrame
        models (dict): Dictionary of loaded models
        model_type (str): Type of model ('prophet', 'lgbm', 'var')
        output_path (str): Path to write output
        checkpoint_path (str): Path for checkpointing
        group_cols (list): Group column names for inference
        forecast_horizon (int): Number of periods to forecast
        
    Returns:
        pyspark.sql.streaming.StreamingQuery: Streaming query
    """
    from pyspark.sql.functions import pandas_udf, PandasUDFType, col, to_json, struct, lit, current_timestamp
    
    if group_cols is None:
        group_cols = ["store_id", "product_id"]
    
    # Create a broadcast variable for models
    model_broadcast = spark.sparkContext.broadcast(models)
    
    # Process each batch
    def process_batch(batch_df, batch_id):
        # Skip empty batches
        if batch_df.count() == 0:
            return
        
        # Add batch metadata
        batch_df = batch_df.withColumn("batch_id", lit(batch_id))
        batch_df = batch_df.withColumn("processing_time", current_timestamp())
        
        # Perform inference based on model type
        if model_type == 'prophet':
            # Group the data and make predictions
            predictions = batch_inference(
                spark, batch_df, model_broadcast.value, 'prophet',
                group_cols=group_cols, forecast_horizon=forecast_horizon
            )
        elif model_type == 'lgbm':
            # Group the data and make predictions
            predictions = batch_inference(
                spark, batch_df, model_broadcast.value, 'lgbm',
                group_cols=group_cols, forecast_horizon=forecast_horizon
            )
        elif model_type == 'var':
            # Group the data and make predictions
            predictions = batch_inference(
                spark, batch_df, model_broadcast.value, 'var',
                group_cols=group_cols, forecast_horizon=forecast_horizon
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Write the predictions
        if predictions.count() > 0:
            predictions.write \
                .mode("append") \
                .parquet(f"{output_path}/batch_{batch_id}")
            
            logger.info(f"Wrote predictions for batch {batch_id}")
    
    # Start the streaming query
    query = stream_df.writeStream \
        .foreachBatch(process_batch) \
        .option("checkpointLocation", checkpoint_path) \
        .trigger(processingTime="10 seconds") \
        .start()
    
    logger.info(f"Started streaming inference with {model_type} models")
    return query


def real_time_api(spark, models, model_type, host="0.0.0.0", port=5000):
    """
    Start a REST API server for real-time predictions.
    
    Args:
        spark (SparkSession): Spark session
        models (dict): Dictionary of loaded models
        model_type (str): Type of model ('prophet', 'lgbm', 'var')
        host (str): Host to bind to
        port (int): Port to listen on
        
    Returns:
        None
    """
    try:
        from flask import Flask, request, jsonify
    except ImportError:
        logger.error("Flask is not installed. Install it with: pip install flask")
        return
    
    app = Flask(__name__)
    
    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            # Get data from request
            data = request.json
            
            # Check required fields
            if 'model_id' not in data:
                return jsonify({'error': 'Missing model_id field'}), 400
            
            model_id = data['model_id']
            
            # Check if model exists
            if model_id not in models:
                return jsonify({'error': f'Model {model_id} not found'}), 404
            
            # Get the model
            model = models[model_id]
            
            # Get prediction based on model type
            if model_type == 'prophet':
                # Create future dataframe
                horizon = data.get('horizon', 30)
                freq = data.get('freq', 'D')
                include_history = data.get('include_history', False)
                
                future = model.make_future_dataframe(periods=horizon, freq=freq, include_history=include_history)
                forecast = model.predict(future)
                
                # Format the response
                result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(horizon)
                result['ds'] = result['ds'].dt.strftime('%Y-%m-%d')
                
                return jsonify({
                    'forecast': result.to_dict(orient='records'),
                    'model_id': model_id,
                    'model_type': 'prophet'
                })
            
            elif model_type == 'lgbm':
                # Check for features
                if 'features' not in data:
                    return jsonify({'error': 'Missing features field'}), 400
                
                features = data['features']
                
                # Make prediction
                prediction = model.predict([features])[0]
                
                return jsonify({
                    'prediction': float(prediction),
                    'model_id': model_id,
                    'model_type': 'lgbm'
                })
            
            elif model_type == 'var':
                # Get horizon
                horizon = data.get('horizon', 30)
                
                # Make prediction
                forecast = model.forecast(steps=horizon)
                
                # Format the response
                result = {'forecast': forecast.tolist()}
                
                return jsonify({
                    'forecast': result,
                    'model_id': model_id,
                    'model_type': 'var'
                })
            
            else:
                return jsonify({'error': f'Unsupported model type: {model_type}'}), 400
        
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    # Start the server
    app.run(host=host, port=port)
    logger.info(f"Started API server at http://{host}:{port}")


def main():
    """Main function to demonstrate real-time inference."""
    # Create Spark session
    spark = create_spark_session()
    
    # Define paths
    model_dir = "../models"
    data_path = "../retail_data/sales.csv"
    output_path = "../predictions"
    checkpoint_path = "../checkpoints"
    
    # Create directories if they don't exist
    for path in [output_path, checkpoint_path]:
        os.makedirs(path, exist_ok=True)
    
    # Load models (adjust paths based on your project structure)
    prophet_models = load_models(f"{model_dir}/prophet", "prophet")
    lgbm_models = load_models(f"{model_dir}/lgbm", "lgbm")
    var_models = load_models(f"{model_dir}/var", "var")
    
    # Print loaded models
    logger.info(f"Loaded {len(prophet_models)} Prophet models")
    logger.info(f"Loaded {len(lgbm_models)} LightGBM models")
    logger.info(f"Loaded {len(var_models)} VAR models")
    
    # Example: Batch inference with Prophet models
    if prophet_models:
        # Load test data
        test_df = spark.read.csv(data_path, header=True, inferSchema=True)
        
        # Run batch inference
        predictions = batch_inference(
            spark, test_df, prophet_models, "prophet",
            group_cols=["store_id", "product_id"], forecast_horizon=30
        )
        
        # Write predictions
        predictions.write.mode("overwrite").parquet(f"{output_path}/prophet_predictions")
        logger.info(f"Wrote Prophet predictions to {output_path}/prophet_predictions")
    
    # Example: Start API server with LightGBM models
    # Uncomment to run API server (it will block until stopped)
    # if lgbm_models:
    #     real_time_api(spark, lgbm_models, "lgbm")
    
    # Example: Streaming inference (requires Kafka setup)
    # Uncomment and configure Kafka to use streaming inference
    # if prophet_models:
    #     # Set up streaming source
    #     stream_df = setup_streaming_source(spark, "localhost:9092", "retail_data")
    #     
    #     # Start streaming inference
    #     query = streaming_inference(
    #         spark, stream_df, prophet_models, "prophet",
    #         output_path=f"{output_path}/streaming",
    #         checkpoint_path=checkpoint_path
    #     )
    #     
    #     # Wait for termination (or timeout after 60 seconds)
    #     query.awaitTermination(60)
    
    # Stop Spark session
    spark.stop()


if __name__ == "__main__":
    main() 