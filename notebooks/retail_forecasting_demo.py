#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Retail Demand Forecasting & Inventory Optimization Demo Script
This script demonstrates the end-to-end pipeline for retail demand forecasting
and inventory optimization using multiple models and techniques.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from models.prophet_models import OptimizedProphet, prepare_prophet_data
from models.var_models import OptimizedVAR, prepare_var_data
from models.lightgbm_models import TimeSeriesLGBMModel
from models.ensemble import ForecastEnsemble

from preprocessing.data_cleaning import clean_retail_data, handle_outliers
from preprocessing.feature_engineering import create_time_features, create_lag_features
from preprocessing.external_data import merge_external_data

from optimization.inventory_models import calculate_optimal_inventory
from optimization.cost_functions import calculate_inventory_costs

from evaluation.metrics import calculate_all_metrics, compare_models, calculate_prediction_intervals
from evaluation.visualization import (plot_forecast_vs_actual, plot_multiple_forecasts,
                                   plot_forecast_errors, plot_error_distribution,
                                   plot_metrics_comparison, plot_residual_analysis,
                                   plot_feature_importance, plot_seasonal_decomposition)

# Set up directories
data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

# Create results directory if it doesn't exist
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

print("=" * 80)
print("RETAIL DEMAND FORECASTING & INVENTORY OPTIMIZATION DEMO")
print("=" * 80)
print()

# Step 1: Generate or load sample data
print("Step 1: Generating Sample Data")
print("-" * 50)

# Import the data generator
from data.generate_sample_data import (generate_store_data, generate_product_data,
                                      generate_external_data, generate_sales_data,
                                      plot_sample_data)

# Generate sample data
print("Generating store and product metadata...")
stores_df = generate_store_data(n_stores=10)
products_df = generate_product_data(n_products=50)

# Set date range for the sample data
start_date = datetime(2020, 1, 1)
end_date = datetime(2022, 12, 31)
print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

# Generate external data (weather, holidays, economic indicators)
print("Generating external data (weather, holidays, economic indicators)...")
weather_df, economics_df, holidays_df = generate_external_data(start_date, end_date)
external_data = {'weather': weather_df, 'economics': economics_df, 'holidays': holidays_df}

# Generate sales data
print("Generating sales data with seasonal patterns and trends...")
sales_df = generate_sales_data(stores_df, products_df, (weather_df, economics_df, holidays_df), 
                             start_date=start_date, end_date=end_date)

# Plot sample data
print("Plotting sample data...")
fig = plot_sample_data(sales_df, stores_df, products_df, external_data)
plt.savefig(os.path.join(results_dir, "sample_data_visualization.png"))
plt.close()

print(f"Generated {len(sales_df)} sales records across {stores_df['store_id'].nunique()} stores and {products_df['product_id'].nunique()} products.")
print()

# Step 2: Data Preprocessing
print("Step 2: Data Preprocessing")
print("-" * 50)

# Clean the data
print("Cleaning data...")
cleaned_sales_df = clean_retail_data(sales_df)
print(f"Data shape after cleaning: {cleaned_sales_df.shape}")

# Handle outliers
print("Handling outliers...")
cleaned_sales_df = handle_outliers(cleaned_sales_df, method='winsorize', column='sales', limits=(0.01, 0.01))

# Create features
print("Creating time-based features...")
featured_sales_df = create_time_features(cleaned_sales_df, date_column='date')
print(f"Generated features: {[col for col in featured_sales_df.columns if col not in cleaned_sales_df.columns]}")

# Count data points by store and product before generating lag features
store_product_counts = featured_sales_df.groupby(['store_id', 'product_id']).size().reset_index(name='count')
store_product_counts = store_product_counts.sort_values('count', ascending=False)

# Create lag features only if we have enough data
if store_product_counts.iloc[0]['count'] > 30:  # At least 30 data points for the top combination
    print("Creating lag features...")
    featured_sales_df = create_lag_features(featured_sales_df, target_column='sales', 
                                          group_columns=['store_id', 'product_id'],
                                          lag_periods=[1, 7, 14, 30])
    print(f"Generated lag features: {[col for col in featured_sales_df.columns if 'lag' in col]}")
else:
    print("Skipping lag features due to limited data...")

# Merge external data
print("Merging external data...")
merged_df = merge_external_data(featured_sales_df, external_data, on='date')
print(f"Data shape after merging external data: {merged_df.shape}")

# Use data without lag features to ensure we have enough rows
final_df = merged_df.copy()
if 'sales_lag_1' in final_df.columns:
    # Drop rows with NaN values (due to lag features)
    final_df = merged_df.dropna()
    print(f"Final data shape after preprocessing: {final_df.shape}")
else:
    # No need to drop rows if we didn't create lag features
    print(f"Final data shape after preprocessing: {final_df.shape}")
print()

# Step 3: Select a product and store for demonstration
print("Step 3: Selecting Sample Data for Forecasting Demo")
print("-" * 50)

# Find a store-product combination with sufficient data
print("Finding a valid store-product combination with sufficient data...")
if len(final_df) < 10:  # Not enough data after preprocessing
    print("Limited data after preprocessing. Using the original merged data...")
    use_df = merged_df
else:
    use_df = final_df

store_product_counts = use_df.groupby(['store_id', 'product_id']).size().reset_index(name='count')
store_product_counts = store_product_counts.sort_values('count', ascending=False)

min_required_points = 10  # Minimum data points needed for a valid forecast

if len(store_product_counts) > 0 and store_product_counts.iloc[0]['count'] >= min_required_points:
    # Take the combination with the most data points
    store_id = store_product_counts.iloc[0]['store_id']
    product_id = store_product_counts.iloc[0]['product_id']
    data_count = store_product_counts.iloc[0]['count']
    print(f"Selected Store ID: {store_id}, Product ID: {product_id} with {data_count} data points")
else:
    # Not enough data for any combination, generate synthetic data
    print("Not enough data for any store-product combination. Generating synthetic data for demo...")
    
    # Generate synthetic time series data
    np.random.seed(42)
    date_range = pd.date_range(start=start_date, periods=100, freq='D')
    trend = np.linspace(0, 10, 100)
    seasonality = 5 * np.sin(np.linspace(0, 2*np.pi*3, 100))
    noise = np.random.normal(0, 1, 100)
    
    # Combine components
    synthetic_sales = trend + seasonality + noise
    
    # Create synthetic DataFrame
    use_df = pd.DataFrame({
        'date': date_range,
        'store_id': 999,
        'product_id': 999,
        'sales': synthetic_sales
    })
    
    # Add some basic time features
    use_df = create_time_features(use_df, date_column='date')
    
    store_id = 999
    product_id = 999
    print(f"Created synthetic data for Store ID: {store_id}, Product ID: {product_id} with {len(use_df)} data points")

# Filter data for the selected store and product
sample_df = use_df[(use_df['store_id'] == store_id) & (use_df['product_id'] == product_id)].copy()
sample_df = sample_df.sort_values('date')
print(f"Sample data shape: {sample_df.shape}")

# Split data into train and test sets (use last 20% for testing)
train_size = int(len(sample_df) * 0.8)
train_df = sample_df.iloc[:train_size].copy()
test_df = sample_df.iloc[train_size:].copy()
print(f"Training data: {train_df.shape}, Testing data: {test_df.shape}")

# Visualize the sample data
plt.figure(figsize=(12, 6))
plt.plot(train_df['date'], train_df['sales'], label='Training Data')
plt.plot(test_df['date'], test_df['sales'], label='Testing Data')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title(f'Sales Data for Store {store_id}, Product {product_id}')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "train_test_split.png"))
plt.close()
print("Train-test visualization saved to 'results/train_test_split.png'")
print()

# Step 4: Train and Evaluate Different Models
print("Step 4: Model Training and Evaluation")
print("-" * 50)

# Prepare actual values for evaluation
actual_dates = test_df['date']
actual_values = test_df['sales'].values

# Dictionary to store forecasts from different models
model_forecasts = {}

# 1. Prophet Model
print("\n1. Facebook Prophet Model")
print("-" * 30)

print("Preparing data for Prophet...")
prophet_train = prepare_prophet_data(train_df, 'date', 'sales', 
                                   regressors=['temperature', 'is_holiday'])

print("Training Prophet model...")
prophet_model = OptimizedProphet(seasonality_mode='multiplicative')
prophet_model.fit(prophet_train)

print("Generating Prophet forecast...")
prophet_forecast_df = prophet_model.predict(periods=len(test_df))
prophet_forecast = prophet_forecast_df['yhat'].values
model_forecasts['Prophet'] = prophet_forecast

# Evaluate Prophet model
prophet_metrics = calculate_all_metrics(actual_values, prophet_forecast, 
                                      y_train=train_df['sales'].values, 
                                      seasonality=7)
print("\nProphet Model Performance:")
for metric, value in prophet_metrics.items():
    print(f"{metric}: {value:.4f}")

# Plot Prophet forecast
fig = plot_forecast_vs_actual(actual_dates, actual_values, prophet_forecast, 
                            model_name='Prophet')
plt.savefig(os.path.join(results_dir, "prophet_forecast.png"))
plt.close()
print("Prophet forecast visualization saved to 'results/prophet_forecast.png'")

# 2. Vector Auto Regression Model
print("\n2. Vector Auto Regression (VAR) Model")
print("-" * 30)

print("Preparing data for VAR...")
var_train = prepare_var_data(train_df, 'date', ['sales', 'temperature', 'price'])

print("Training VAR model...")
var_model = OptimizedVAR()
var_model.fit(var_train)

print("Generating VAR forecast...")
var_forecast_df = var_model.predict(steps=len(test_df))
var_forecast = var_forecast_df['sales'].values
model_forecasts['VAR'] = var_forecast

# Evaluate VAR model
var_metrics = calculate_all_metrics(actual_values, var_forecast, 
                                  y_train=train_df['sales'].values, 
                                  seasonality=7)
print("\nVAR Model Performance:")
for metric, value in var_metrics.items():
    print(f"{metric}: {value:.4f}")

# Plot VAR forecast
fig = plot_forecast_vs_actual(actual_dates, actual_values, var_forecast, 
                            model_name='VAR')
plt.savefig(os.path.join(results_dir, "var_forecast.png"))
plt.close()
print("VAR forecast visualization saved to 'results/var_forecast.png'")

# 3. LightGBM Model
print("\n3. LightGBM Model")
print("-" * 30)

print("Preparing data for LightGBM...")
# Use all features except for the date column
feature_cols = [col for col in train_df.columns if col not in ['date', 'sales']]

print("Training LightGBM model...")
lgb_model = TimeSeriesLGBMModel()
lgb_model.fit(train_df, target_column='sales', feature_columns=feature_cols)

print("Generating LightGBM forecast...")
lgb_forecast = lgb_model.predict(test_df)
model_forecasts['LightGBM'] = lgb_forecast

# Evaluate LightGBM model
lgb_metrics = calculate_all_metrics(actual_values, lgb_forecast, 
                                  y_train=train_df['sales'].values, 
                                  seasonality=7)
print("\nLightGBM Model Performance:")
for metric, value in lgb_metrics.items():
    print(f"{metric}: {value:.4f}")

# Plot LightGBM forecast
fig = plot_forecast_vs_actual(actual_dates, actual_values, lgb_forecast, 
                            model_name='LightGBM')
plt.savefig(os.path.join(results_dir, "lightgbm_forecast.png"))
plt.close()
print("LightGBM forecast visualization saved to 'results/lightgbm_forecast.png'")

# Get feature importance from LightGBM
feature_importance = lgb_model.get_feature_importance()
fig = plot_feature_importance(feature_importance['Feature'].values, 
                            feature_importance['Importance'].values, 
                            model_name='LightGBM')
plt.savefig(os.path.join(results_dir, "lightgbm_feature_importance.png"))
plt.close()
print("LightGBM feature importance visualization saved to 'results/lightgbm_feature_importance.png'")

# 4. Ensemble Model
print("\n4. Ensemble Model")
print("-" * 30)

print("Creating ensemble forecast...")
ensemble = ForecastEnsemble(method='weighted_average')
ensemble.fit(model_forecasts, actual_values)
ensemble_forecast = ensemble.predict(model_forecasts)
model_forecasts['Ensemble'] = ensemble_forecast

# Evaluate Ensemble model
ensemble_metrics = calculate_all_metrics(actual_values, ensemble_forecast, 
                                       y_train=train_df['sales'].values, 
                                       seasonality=7)
print("\nEnsemble Model Performance:")
for metric, value in ensemble_metrics.items():
    print(f"{metric}: {value:.4f}")

# Plot Ensemble forecast
fig = plot_forecast_vs_actual(actual_dates, actual_values, ensemble_forecast, 
                            model_name='Ensemble')
plt.savefig(os.path.join(results_dir, "ensemble_forecast.png"))
plt.close()
print("Ensemble forecast visualization saved to 'results/ensemble_forecast.png'")

# Calculate prediction intervals for ensemble
lower_bound, upper_bound = calculate_prediction_intervals(model_forecasts, confidence_level=0.95)
fig = plot_forecast_vs_actual(actual_dates, actual_values, ensemble_forecast, 
                            model_name='Ensemble', 
                            prediction_intervals=(lower_bound, upper_bound))
plt.savefig(os.path.join(results_dir, "ensemble_forecast_with_intervals.png"))
plt.close()
print("Ensemble forecast with prediction intervals saved to 'results/ensemble_forecast_with_intervals.png'")

# Step 5: Compare Models
print("\nStep 5: Model Comparison")
print("-" * 50)

# Compile all metrics
all_metrics = {
    'Prophet': prophet_metrics,
    'VAR': var_metrics,
    'LightGBM': lgb_metrics,
    'Ensemble': ensemble_metrics
}

# Convert to DataFrame
metrics_df = pd.DataFrame(all_metrics).T
print("\nModel Comparison:")
print(metrics_df)

# Save metrics to CSV
metrics_df.to_csv(os.path.join(results_dir, "model_comparison.csv"))
print("Model comparison metrics saved to 'results/model_comparison.csv'")

# Plot comparison of all forecasts
fig = plot_multiple_forecasts(actual_dates, actual_values, model_forecasts)
plt.savefig(os.path.join(results_dir, "all_models_comparison.png"))
plt.close()
print("All models comparison visualization saved to 'results/all_models_comparison.png'")

# Plot metrics comparison
fig = plot_metrics_comparison(metrics_df)
plt.savefig(os.path.join(results_dir, "metrics_comparison.png"))
plt.close()
print("Metrics comparison visualization saved to 'results/metrics_comparison.png'")

# Step 6: Residual Analysis for Best Model
print("\nStep 6: Residual Analysis")
print("-" * 50)

# Use the ensemble model for residual analysis
residuals = actual_values - ensemble_forecast
fig = plot_residual_analysis(actual_dates, residuals, model_name='Ensemble')
plt.savefig(os.path.join(results_dir, "residual_analysis.png"))
plt.close()
print("Residual analysis visualization saved to 'results/residual_analysis.png'")

# Plot error distribution
fig = plot_error_distribution(residuals, model_name='Ensemble')
plt.savefig(os.path.join(results_dir, "error_distribution.png"))
plt.close()
print("Error distribution visualization saved to 'results/error_distribution.png'")

# Step 7: Inventory Optimization
print("\nStep 7: Inventory Optimization")
print("-" * 50)

# Create a future date range for forecasting
future_dates = pd.date_range(start=test_df['date'].iloc[-1] + timedelta(days=1), periods=30, freq='D')

# Generate future forecasts using a simple approach
# For demonstration purposes, we'll use the average of the last 7 days of ensemble forecasts
# In a real application, you would use each model to generate future forecasts
print("Generating future forecasts...")
last_7_days_avg = np.mean(ensemble_forecast[-7:])
future_forecast = np.array([last_7_days_avg] * len(future_dates))

# Create a forecast DataFrame for inventory optimization
forecast_df = pd.DataFrame({
    'date': future_dates,
    'forecast': future_forecast
})

# Use the last known price for future calculations
if 'price' in test_df.columns:
    last_price = test_df['price'].iloc[-1]
else:
    # Default price if not available
    last_price = 10.0
forecast_df['price'] = last_price

# Get product details
if product_id in products_df['product_id'].values:
    product_details = products_df[products_df['product_id'] == product_id].iloc[0]
    product_name = product_details['product_name']
    product_price = product_details['price']
    product_cost = product_details['cost']
    lead_time_days = product_details['lead_time_days']
    min_order_quantity = product_details['min_order_quantity']
else:
    # Default values for synthetic product
    product_name = f"Product {product_id}"
    product_price = last_price
    product_cost = last_price * 0.6  # 60% of price
    lead_time_days = 7
    min_order_quantity = 10

print(f"Optimizing inventory for Product: {product_name}")
print(f"Price: ${product_price:.2f}, Cost: ${product_cost:.2f}")
print(f"Lead Time: {lead_time_days} days, Min Order Quantity: {min_order_quantity} units")

# Calculate optimal inventory parameters
print("\nCalculating optimal inventory parameters...")
inventory_params = calculate_optimal_inventory(
    demand_forecast=forecast_df['forecast'].values,
    service_level=0.95,
    lead_time=lead_time_days,
    holding_cost=0.2,  # 20% of product cost
    stockout_cost=0.5,  # 50% of product price
    order_cost=50,  # $50 per order
    min_order_qty=min_order_quantity,
    safety_stock_method='normal'
)

print("\nOptimal Inventory Parameters:")
for param, value in inventory_params.items():
    if isinstance(value, float):
        print(f"{param}: {value:.2f}")
    else:
        print(f"{param}: {value}")

# Calculate inventory costs
print("\nCalculating inventory costs...")
# Convert inventory_params to a dictionary for easier use
inventory_policy = {
    'reorder_point': inventory_params['reorder_point'].iloc[0],
    'economic_order_quantity': inventory_params['economic_order_quantity'].iloc[0],
    'safety_stock': inventory_params['safety_stock'].iloc[0],
    'max_inventory': inventory_params['max_inventory'].iloc[0]
}

costs = calculate_inventory_costs(
    demand_forecast=forecast_df['forecast'].values,
    inventory_policy=inventory_policy,
    unit_cost=product_cost,
    holding_cost_rate=0.2,  # 20% of product cost
    ordering_cost=50,  # $50 per order
    stockout_cost_per_unit=product_price * 0.5,  # 50% of product price
    unit_price=product_price
)

print("\nInventory Costs:")
for cost_type, amount in costs.items():
    # Convert numpy arrays to float if needed
    if isinstance(amount, np.ndarray):
        if amount.size == 1:
            amount = float(amount)
        else:
            # For arrays with multiple values, use the first one or the sum
            amount = float(amount[0]) if amount.size > 0 else 0.0
    print(f"{cost_type}: ${amount:.2f}")

# Use the correct key for total cost
if 'total_annual_cost' in costs:
    print(f"\nTotal Cost: ${costs['total_annual_cost']:.2f}")
elif 'total_cost' in costs:
    print(f"\nTotal Cost: ${costs['total_cost']:.2f}")
else:
    print("\nTotal Cost: Not available")

# Save inventory parameters to CSV
# Convert inventory_params to a dictionary for easier handling
inventory_dict = {}
for col in inventory_params.columns:
    # Extract the value from the DataFrame
    value = inventory_params[col].iloc[0]
    # Convert to float if it's a numpy array or pandas Series
    if hasattr(value, 'item'):
        inventory_dict[col] = value.item()
    else:
        inventory_dict[col] = value

# Create a new DataFrame from the dictionary
inventory_df = pd.DataFrame([inventory_dict])
inventory_df.to_csv(os.path.join(results_dir, "inventory_parameters.csv"), index=False)
print("Inventory parameters saved to 'results/inventory_parameters.csv'")

# Step 8: Visualize Inventory Policy
print("\nStep 8: Visualizing Inventory Policy")
print("-" * 50)

# Create a simulation of inventory levels over time
def simulate_inventory(forecast, reorder_point, order_quantity, lead_time, initial_inventory=None):
    """
    Simulate inventory levels over time based on forecast and inventory policy.
    
    Parameters:
    -----------
    forecast : array-like
        Forecasted demand
    reorder_point : float
        Inventory level at which to place an order
    order_quantity : float
        Quantity to order when placing an order
    lead_time : int
        Lead time in days for order delivery
    initial_inventory : float, optional
        Initial inventory level
        
    Returns:
    --------
    tuple
        (inventory_levels, orders, deliveries)
    """
    # Convert inputs to appropriate types
    if isinstance(forecast, pd.Series):
        forecast = forecast.values
    
    # Extract scalar values from Series if needed
    if hasattr(reorder_point, 'item'):
        reorder_point = reorder_point.item()
    
    if hasattr(order_quantity, 'item'):
        order_quantity = order_quantity.item()
    
    # Initialize inventory
    if initial_inventory is None:
        initial_inventory = order_quantity
    
    inventory = [initial_inventory]
    orders = []
    deliveries = []
    
    for i in range(len(forecast)):
        # Process any deliveries due today
        delivery_amount = 0
        for j, (order_day, amount) in enumerate(orders):
            if order_day + lead_time == i:
                delivery_amount += amount
                deliveries.append((i, amount))
                orders[j] = (order_day, 0)  # Mark as delivered
        
        # Add delivery to inventory
        current_inventory = inventory[-1] + delivery_amount
        
        # Subtract demand
        current_inventory -= forecast[i]
        
        # Check if we need to place an order
        if current_inventory <= reorder_point:
            order_amount = order_quantity
            orders.append((i, order_amount))
        
        # Append current inventory
        inventory.append(current_inventory)
        
        # Remove processed orders
        orders = [(day, amount) for day, amount in orders if amount > 0]
    
    return inventory[1:], orders, deliveries

# Simulate inventory
inventory_levels, orders, deliveries = simulate_inventory(
    forecast_df['forecast'].values,
    inventory_params['reorder_point'],
    inventory_params['economic_order_quantity'],
    lead_time_days
)

# Plot inventory simulation
plt.figure(figsize=(12, 8))
plt.plot(forecast_df['date'], inventory_levels, 'b-', label='Inventory Level')

# Extract values from Series if needed
reorder_point_value = inventory_dict['reorder_point'] if 'reorder_point' in inventory_dict else inventory_params['reorder_point'].iloc[0]
safety_stock_value = inventory_dict['safety_stock'] if 'safety_stock' in inventory_dict else inventory_params['safety_stock'].iloc[0]

plt.axhline(y=reorder_point_value, color='r', linestyle='--', 
           label=f'Reorder Point ({reorder_point_value:.0f})')
plt.axhline(y=safety_stock_value, color='g', linestyle='--', 
           label=f'Safety Stock ({safety_stock_value:.0f})')

# Plot orders
for order_day, amount in orders:
    if order_day < len(forecast_df):
        plt.scatter(forecast_df['date'].iloc[order_day], inventory_levels[order_day], 
                   color='red', s=100, marker='^', 
                   label='Order Placed' if order_day == orders[0][0] else "")

# Plot deliveries
for delivery_day, amount in deliveries:
    if delivery_day < len(forecast_df):
        plt.scatter(forecast_df['date'].iloc[delivery_day], inventory_levels[delivery_day], 
                   color='green', s=100, marker='v', 
                   label='Delivery Received' if delivery_day == deliveries[0][0] else "")

plt.fill_between(forecast_df['date'], 0, inventory_params['safety_stock'], 
                alpha=0.2, color='red', label='Stock-out Risk Zone')

plt.xlabel('Date')
plt.ylabel('Inventory Level')
plt.title('Inventory Simulation with (Q,R) Policy')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "inventory_simulation.png"))
plt.close()
print("Inventory simulation visualization saved to 'results/inventory_simulation.png'")

# Step 9: PySpark Integration (Pseudo-implementation)
print("\nStep 9: PySpark Integration (Example Only)")
print("-" * 50)
print("Note: This is an example of how PySpark would be used in a production environment.")
print("The actual PySpark implementation requires a Spark cluster to run.")

print("""
# Example PySpark code for distributed training:
from spark.distributed_training import SparkLGBMTrainer

# Initialize Spark trainer
spark_trainer = SparkLGBMTrainer(
    spark_session=spark,
    partition_column='store_id',
    target_column='sales'
)

# Train model in distributed fashion
distributed_model = spark_trainer.fit(
    spark_df=spark.createDataFrame(final_df),
    hyperparameters={
        'num_leaves': 31,
        'learning_rate': 0.05,
        'n_estimators': 100
    }
)

# Generate predictions
predictions = spark_trainer.transform(
    spark_df=spark.createDataFrame(test_df)
)
""")

print("\nExample of using PySpark for real-time inference:")
print("""
from spark.real_time_inference import SparkModelServer

# Deploy model to Spark model server
model_server = SparkModelServer(
    model=distributed_model,
    spark_session=spark,
    batch_size=1000
)

# Start streaming predictions
model_server.start_streaming(
    input_stream='kafka:input_topic',
    output_stream='kafka:output_topic',
    checkpoint_location='s3://checkpoints/'
)
""")

# Step 10: Summary and Next Steps
print("\nStep 10: Summary and Next Steps")
print("-" * 50)

print("Summary of Results:")
print("-" * 20)
print(f"Best performing model: {metrics_df['RMSE'].idxmin()} (RMSE: {metrics_df['RMSE'].min():.4f})")
print(f"Forecast reliability improvement: {(1 - metrics_df['RMSE'].min() / metrics_df['RMSE'].max()) * 100:.2f}%")
print(f"Optimal order quantity: {inventory_dict['economic_order_quantity']:.0f} units")
print(f"Reorder point: {inventory_dict['reorder_point']:.0f} units")
print(f"Safety stock: {inventory_dict['safety_stock']:.0f} units")
print(f"Total annual inventory cost: ${costs['total_annual_cost']:.2f}")

print("\nNext Steps:")
print("-" * 20)
print("1. Implement deep learning models (LSTM, Transformer-based models)")
print("2. Develop automated hyperparameter optimization")
print("3. Add advanced anomaly detection for outlier management")
print("4. Create interactive dashboard for business users")
print("5. Deploy models to production using cloud infrastructure")

print("\n" + "=" * 80)
print("DEMO COMPLETED SUCCESSFULLY")
print("=" * 80)
print(f"Results saved to: {results_dir}")

# Save this summary to a file
with open(os.path.join(results_dir, "summary.txt"), "w") as f:
    f.write("RETAIL DEMAND FORECASTING & INVENTORY OPTIMIZATION SUMMARY\n")
    f.write("=" * 60 + "\n\n")
    f.write("Summary of Results:\n")
    f.write("-" * 20 + "\n")
    f.write(f"Best performing model: {metrics_df['RMSE'].idxmin()} (RMSE: {metrics_df['RMSE'].min():.4f})\n")
    f.write(f"Forecast reliability improvement: {(1 - metrics_df['RMSE'].min() / metrics_df['RMSE'].max()) * 100:.2f}%\n")
    f.write(f"Optimal order quantity: {inventory_dict['economic_order_quantity']:.0f} units\n")
    f.write(f"Reorder point: {inventory_dict['reorder_point']:.0f} units\n")
    f.write(f"Safety stock: {inventory_dict['safety_stock']:.0f} units\n")
    f.write(f"Total annual inventory cost: ${costs['total_annual_cost']:.2f}\n")
    
    f.write("\nModel Comparison:\n")
    f.write(metrics_df.to_string() + "\n")
    
    f.write("\nInventory Parameters:\n")
    for param, value in inventory_params.items():
        if isinstance(value, float):
            f.write(f"{param}: {value:.2f}\n")
        else:
            f.write(f"{param}: {value}\n")
            
    f.write("\nInventory Costs:\n")
    for cost_type, amount in costs.items():
        # Convert numpy arrays to float if needed
        if isinstance(amount, np.ndarray):
            if amount.size == 1:
                amount = float(amount)
            else:
                # For arrays with multiple values, use the first one or the sum
                amount = float(amount[0]) if amount.size > 0 else 0.0
        f.write(f"{cost_type}: ${amount:.2f}\n")

print("\nSummary saved to 'results/summary.txt'")

# If this was a Jupyter notebook, we'd display the plots inline
# But since this is a script, we'll just save them to files 