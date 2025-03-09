#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Inventory optimization module for retail demand forecasting.
This module provides functions to optimize inventory levels based on forecasts.
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_safety_stock(demand_forecast, lead_time, service_level=0.95):
    """
    Calculate safety stock based on demand forecast and service level.
    
    Args:
        demand_forecast (pd.Series): Forecasted demand
        lead_time (int): Lead time in days
        service_level (float): Service level (0-1)
        
    Returns:
        float: Safety stock level
    """
    # Calculate standard deviation of demand
    demand_std = demand_forecast.std()
    
    # Calculate z-score for service level
    z_score = stats.norm.ppf(service_level)
    
    # Calculate safety stock
    safety_stock = z_score * demand_std * np.sqrt(lead_time)
    
    return safety_stock


def calculate_reorder_point(demand_forecast, lead_time, service_level=0.95):
    """
    Calculate reorder point based on demand forecast.
    
    Args:
        demand_forecast (pd.Series): Forecasted demand
        lead_time (int): Lead time in days
        service_level (float): Service level (0-1)
        
    Returns:
        float: Reorder point
    """
    # Calculate average daily demand
    daily_demand = demand_forecast.mean()
    
    # Calculate lead time demand
    lead_time_demand = daily_demand * lead_time
    
    # Calculate safety stock
    safety_stock = calculate_safety_stock(demand_forecast, lead_time, service_level)
    
    # Calculate reorder point
    reorder_point = lead_time_demand + safety_stock
    
    return reorder_point


def calculate_economic_order_quantity(annual_demand, ordering_cost, holding_cost):
    """
    Calculate economic order quantity (EOQ).
    
    Args:
        annual_demand (float): Annual demand
        ordering_cost (float): Cost per order
        holding_cost (float): Annual holding cost per unit
        
    Returns:
        float: Economic order quantity
    """
    # Calculate EOQ
    eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost)
    
    return eoq


def optimize_inventory(forecast_df, lead_time, ordering_cost, holding_cost_rate, 
                     unit_cost, service_level=0.95, current_stock=0):
    """
    Optimize inventory parameters based on forecast.
    
    Args:
        forecast_df (pd.DataFrame): Forecast DataFrame with 'ds' and 'yhat' columns
        lead_time (int): Lead time in days
        ordering_cost (float): Cost per order
        holding_cost_rate (float): Annual holding cost rate (percentage of unit cost)
        unit_cost (float): Cost per unit
        service_level (float): Service level (0-1)
        current_stock (int): Current stock level
        
    Returns:
        dict: Dictionary of inventory parameters
    """
    # Extract demand forecast
    demand_forecast = forecast_df['yhat']
    
    # Calculate daily and annual demand
    daily_demand = demand_forecast.mean()
    annual_demand = daily_demand * 365
    
    # Calculate holding cost per unit
    holding_cost = holding_cost_rate * unit_cost
    
    # Calculate safety stock
    safety_stock = calculate_safety_stock(demand_forecast, lead_time, service_level)
    
    # Calculate reorder point
    reorder_point = calculate_reorder_point(demand_forecast, lead_time, service_level)
    
    # Calculate economic order quantity
    eoq = calculate_economic_order_quantity(annual_demand, ordering_cost, holding_cost)
    
    # Check if reorder is needed
    reorder_needed = current_stock <= reorder_point
    
    # Calculate order quantity
    order_quantity = eoq if reorder_needed else 0
    
    # Calculate days of supply
    days_of_supply = current_stock / daily_demand if daily_demand > 0 else float('inf')
    
    # Calculate costs
    annual_ordering_cost = (annual_demand / eoq) * ordering_cost
    average_inventory = safety_stock + (eoq / 2)
    annual_holding_cost = average_inventory * holding_cost
    total_annual_cost = annual_ordering_cost + annual_holding_cost
    
    return {
        'safety_stock': safety_stock,
        'reorder_point': reorder_point,
        'economic_order_quantity': eoq,
        'reorder_needed': reorder_needed,
        'order_quantity': order_quantity,
        'days_of_supply': days_of_supply,
        'daily_demand': daily_demand,
        'annual_demand': annual_demand,
        'annual_ordering_cost': annual_ordering_cost,
        'annual_holding_cost': annual_holding_cost,
        'total_annual_cost': total_annual_cost
    }


def simulate_inventory(demand_forecast, initial_stock, reorder_point, order_quantity, lead_time):
    """
    Simulate inventory levels over time.
    
    Args:
        demand_forecast (pd.Series): Forecasted demand
        initial_stock (int): Initial stock level
        reorder_point (float): Reorder point
        order_quantity (int): Order quantity
        lead_time (int): Lead time in days
        
    Returns:
        pd.DataFrame: DataFrame with inventory simulation
    """
    # Initialize simulation
    n_days = len(demand_forecast)
    inventory = np.zeros(n_days + 1)
    inventory[0] = initial_stock
    
    orders = []  # List of (order_day, arrival_day, quantity)
    
    # Simulate inventory over time
    for day in range(n_days):
        # Process arriving orders
        arriving_orders = [order for order in orders if order[1] == day]
        for order in arriving_orders:
            inventory[day] += order[2]
            orders.remove(order)
        
        # Check if reorder is needed
        pending_orders = sum(order[2] for order in orders)
        if inventory[day] <= reorder_point and not orders:
            # Place order
            arrival_day = day + lead_time
            orders.append((day, arrival_day, order_quantity))
        
        # Consume inventory
        demand = demand_forecast.iloc[day]
        inventory[day+1] = max(0, inventory[day] - demand)
    
    # Create simulation DataFrame
    days = list(range(n_days + 1))
    simulation_df = pd.DataFrame({
        'day': days,
        'inventory': inventory,
        'reorder_point': reorder_point
    })
    
    # Add order information
    simulation_df['order_placed'] = 0
    simulation_df['order_received'] = 0
    
    for order_day, arrival_day, quantity in orders:
        if order_day < len(simulation_df):
            simulation_df.loc[order_day, 'order_placed'] = quantity
        if arrival_day < len(simulation_df):
            simulation_df.loc[arrival_day, 'order_received'] = quantity
    
    return simulation_df


def plot_inventory_simulation(simulation_df, safety_stock=None):
    """
    Plot inventory simulation.
    
    Args:
        simulation_df (pd.DataFrame): Simulation DataFrame
        safety_stock (float): Safety stock level
        
    Returns:
        matplotlib.figure.Figure: Plot figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot inventory level
    ax.plot(simulation_df['day'], simulation_df['inventory'], label='Inventory Level')
    
    # Plot reorder point
    ax.axhline(y=simulation_df['reorder_point'].iloc[0], color='r', linestyle='--', 
              label='Reorder Point')
    
    # Plot safety stock if provided
    if safety_stock is not None:
        ax.axhline(y=safety_stock, color='g', linestyle='--', label='Safety Stock')
    
    # Plot order placement points
    orders_placed = simulation_df[simulation_df['order_placed'] > 0]
    if not orders_placed.empty:
        ax.scatter(orders_placed['day'], orders_placed['inventory'], color='r', marker='v', 
                  label='Order Placed')
    
    # Plot order arrival points
    orders_received = simulation_df[simulation_df['order_received'] > 0]
    if not orders_received.empty:
        ax.scatter(orders_received['day'], orders_received['inventory'], color='g', marker='^', 
                  label='Order Received')
    
    # Add labels and legend
    ax.set_xlabel('Day')
    ax.set_ylabel('Inventory')
    ax.set_title('Inventory Simulation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def inventory_holding_cost(inventory_levels, unit_cost, holding_cost_rate):
    """
    Calculate inventory holding cost.
    
    Args:
        inventory_levels (pd.Series): Daily inventory levels
        unit_cost (float): Cost per unit
        holding_cost_rate (float): Annual holding cost rate
        
    Returns:
        float: Total holding cost
    """
    # Convert annual rate to daily rate
    daily_rate = holding_cost_rate / 365
    
    # Calculate daily holding cost
    daily_cost = inventory_levels * unit_cost * daily_rate
    
    # Calculate total holding cost
    total_cost = daily_cost.sum()
    
    return total_cost


def inventory_ordering_cost(n_orders, ordering_cost):
    """
    Calculate inventory ordering cost.
    
    Args:
        n_orders (int): Number of orders placed
        ordering_cost (float): Cost per order
        
    Returns:
        float: Total ordering cost
    """
    return n_orders * ordering_cost


def inventory_shortage_cost(stockouts, unit_shortage_cost):
    """
    Calculate inventory shortage cost.
    
    Args:
        stockouts (pd.Series): Daily stockout quantities
        unit_shortage_cost (float): Cost per unit short
        
    Returns:
        float: Total shortage cost
    """
    return stockouts.sum() * unit_shortage_cost


def optimize_multi_location_inventory(forecast_dfs, lead_times, ordering_costs, 
                                    holding_cost_rates, unit_costs, service_levels=None, 
                                    current_stocks=None):
    """
    Optimize inventory for multiple locations.
    
    Args:
        forecast_dfs (dict): Dictionary of forecast DataFrames by location
        lead_times (dict): Dictionary of lead times by location
        ordering_costs (dict): Dictionary of ordering costs by location
        holding_cost_rates (dict): Dictionary of holding cost rates by location
        unit_costs (dict): Dictionary of unit costs by location
        service_levels (dict): Dictionary of service levels by location
        current_stocks (dict): Dictionary of current stock levels by location
        
    Returns:
        dict: Dictionary of inventory parameters by location
    """
    if service_levels is None:
        service_levels = {loc: 0.95 for loc in forecast_dfs.keys()}
    
    if current_stocks is None:
        current_stocks = {loc: 0 for loc in forecast_dfs.keys()}
    
    # Optimize inventory for each location
    inventory_params = {}
    
    for location, forecast_df in forecast_dfs.items():
        inventory_params[location] = optimize_inventory(
            forecast_df=forecast_df,
            lead_time=lead_times[location],
            ordering_cost=ordering_costs[location],
            holding_cost_rate=holding_cost_rates[location],
            unit_cost=unit_costs[location],
            service_level=service_levels[location],
            current_stock=current_stocks[location]
        )
    
    return inventory_params


def calculate_optimal_inventory(demand_forecast, forecast_std=None, service_level=0.95, 
                               lead_time=1, holding_cost=0.2, stockout_cost=0.8,
                               order_cost=10, min_order_qty=0, safety_stock_method='normal'):
    """
    Calculate optimal inventory levels using economic order quantity and safety stock.
    
    Parameters:
    -----------
    demand_forecast : pandas.Series or numpy.ndarray
        Forecasted demand values
    forecast_std : pandas.Series or numpy.ndarray, optional
        Standard deviation of forecast (for safety stock calculation)
    service_level : float, optional
        Desired service level (probability of not stocking out)
    lead_time : int or float, optional
        Lead time for replenishment in same time units as demand
    holding_cost : float, optional
        Holding cost per unit per time period as a fraction of item cost
    stockout_cost : float, optional
        Stockout cost per unit as a fraction of item cost
    order_cost : float, optional
        Fixed cost of placing an order
    min_order_qty : int, optional
        Minimum order quantity
    safety_stock_method : str, optional
        Method to calculate safety stock: 'normal', 'poisson', 'empirical', or 'fixed'
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with optimal inventory levels and related metrics
    """
    # Convert to numpy arrays for easier handling
    if isinstance(demand_forecast, pd.Series):
        demand_forecast = demand_forecast.values
    
    if forecast_std is None:
        # If std not provided, estimate it
        if len(demand_forecast) > 1:
            forecast_std = np.std(demand_forecast)
        else:
            # Default to 20% of forecast if only one value
            forecast_std = 0.2 * demand_forecast
    elif isinstance(forecast_std, pd.Series):
        forecast_std = forecast_std.values
    
    # Calculate average demand
    avg_demand = np.mean(demand_forecast)
    
    # Calculate Economic Order Quantity (EOQ)
    eoq = np.sqrt((2 * avg_demand * order_cost) / holding_cost)
    eoq = max(eoq, min_order_qty)
    
    # Calculate safety stock based on service level
    if safety_stock_method == 'normal':
        # Normal distribution assumption
        z_score = stats.norm.ppf(service_level)
        safety_stock = z_score * forecast_std * np.sqrt(lead_time)
    
    elif safety_stock_method == 'poisson':
        # Poisson distribution (good for low-volume items)
        lambda_lead_time = avg_demand * lead_time
        safety_stock = stats.poisson.ppf(service_level, lambda_lead_time) - lambda_lead_time
    
    elif safety_stock_method == 'empirical':
        # Empirical quantile from historical data
        if len(demand_forecast) >= 10:
            safety_stock = np.percentile(demand_forecast, service_level * 100) - avg_demand
        else:
            # Fall back to normal if not enough data
            z_score = stats.norm.ppf(service_level)
            safety_stock = z_score * forecast_std * np.sqrt(lead_time)
    
    elif safety_stock_method == 'fixed':
        # Fixed percentage of average demand
        safety_stock = avg_demand * 0.3  # 30% of average demand
    
    else:
        logger.warning(f"Unknown safety stock method: {safety_stock_method}. Using normal distribution.")
        z_score = stats.norm.ppf(service_level)
        safety_stock = z_score * forecast_std * np.sqrt(lead_time)
    
    # Calculate reorder point
    reorder_point = avg_demand * lead_time + safety_stock
    
    # Calculate maximum inventory level
    max_inventory = eoq + safety_stock
    
    # Calculate expected stockout probability
    if safety_stock_method == 'normal':
        stockout_prob = 1 - stats.norm.cdf(safety_stock / (forecast_std * np.sqrt(lead_time)))
    elif safety_stock_method == 'poisson':
        stockout_prob = 1 - stats.poisson.cdf(reorder_point, avg_demand * lead_time)
    else:
        stockout_prob = 1 - service_level
    
    # Calculate expected holding cost
    avg_inventory = eoq / 2 + safety_stock
    expected_holding_cost = avg_inventory * holding_cost
    
    # Calculate expected stockout cost
    expected_stockout_units = stockout_prob * avg_demand
    expected_stockout_cost = expected_stockout_units * stockout_cost
    
    # Calculate expected order cost
    orders_per_period = avg_demand / eoq
    expected_order_cost = orders_per_period * order_cost
    
    # Calculate total expected cost
    total_cost = expected_holding_cost + expected_stockout_cost + expected_order_cost
    
    # Create results dictionary
    results = {
        'avg_demand': avg_demand,
        'forecast_std': np.mean(forecast_std) if isinstance(forecast_std, np.ndarray) else forecast_std,
        'economic_order_quantity': eoq,
        'safety_stock': safety_stock,
        'reorder_point': reorder_point,
        'max_inventory': max_inventory,
        'avg_inventory': avg_inventory,
        'stockout_probability': stockout_prob,
        'expected_holding_cost': expected_holding_cost,
        'expected_stockout_cost': expected_stockout_cost,
        'expected_order_cost': expected_order_cost,
        'total_expected_cost': total_cost
    }
    
    return pd.DataFrame([results])


def main():
    """Main function to demonstrate inventory optimization."""
    # Create sample forecast
    days = 90
    dates = pd.date_range(start='2023-01-01', periods=days)
    
    # Generate random demand with trend and seasonality
    np.random.seed(42)
    trend = np.linspace(0, 10, days)
    seasonality = 5 * np.sin(2 * np.pi * np.arange(days) / 7)  # Weekly seasonality
    noise = np.random.normal(0, 2, days)
    
    demand = 10 + trend + seasonality + noise
    demand = np.maximum(0, demand)  # Ensure non-negative demand
    
    # Create forecast DataFrame
    forecast_df = pd.DataFrame({
        'ds': dates,
        'yhat': demand
    })
    
    # Define product parameters
    lead_time = 7
    service_level = 0.95
    ordering_cost = 50
    holding_cost_rate = 0.2
    unit_cost = 10
    initial_stock = 50
    
    # Calculate optimal inventory parameters
    params = optimize_inventory(
        forecast_df,
        lead_time=lead_time,
        ordering_cost=ordering_cost,
        holding_cost_rate=holding_cost_rate,
        unit_cost=unit_cost,
        service_level=service_level,
        current_stock=initial_stock
    )
    
    # Print parameters
    print("Inventory Parameters:")
    for key, value in params.items():
        print(f"  {key}: {value:.2f}")
    
    # Simulate inventory over time
    simulation_df = simulate_inventory(
        forecast_df['yhat'],
        initial_stock=initial_stock,
        reorder_point=params['reorder_point'],
        order_quantity=params['economic_order_quantity'],
        lead_time=lead_time
    )
    
    # Plot simulation results
    fig = plot_inventory_simulation(simulation_df, safety_stock=params['safety_stock'])
    plt.show()


if __name__ == "__main__":
    main() 