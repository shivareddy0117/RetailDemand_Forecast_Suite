#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cost functions module for inventory optimization.
This module provides functions to calculate various inventory-related costs.
"""

import numpy as np
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_holding_cost(inventory_levels, unit_cost, holding_cost_rate):
    """
    Calculate inventory holding cost.
    
    Args:
        inventory_levels (np.ndarray or pd.Series): Inventory levels over time
        unit_cost (float): Cost per unit
        holding_cost_rate (float): Annual holding cost rate as percentage of unit cost
        
    Returns:
        float: Total holding cost
    """
    # Convert annual rate to daily rate
    daily_rate = holding_cost_rate / 365
    
    # Calculate daily holding cost
    daily_holding_cost = inventory_levels * unit_cost * daily_rate
    
    # Calculate total holding cost
    total_holding_cost = np.sum(daily_holding_cost)
    
    return total_holding_cost


def calculate_ordering_cost(num_orders, ordering_cost):
    """
    Calculate inventory ordering cost.
    
    Args:
        num_orders (int): Number of orders placed
        ordering_cost (float): Cost per order
        
    Returns:
        float: Total ordering cost
    """
    return num_orders * ordering_cost


def calculate_stockout_cost(stockout_quantities, stockout_cost_per_unit):
    """
    Calculate stockout cost.
    
    Args:
        stockout_quantities (np.ndarray or pd.Series): Quantities of stockouts over time
        stockout_cost_per_unit (float): Cost per unit of stockout
        
    Returns:
        float: Total stockout cost
    """
    return np.sum(stockout_quantities) * stockout_cost_per_unit


def calculate_lost_sales_cost(stockout_quantities, unit_price, unit_cost):
    """
    Calculate lost sales cost (profit margin lost due to stockouts).
    
    Args:
        stockout_quantities (np.ndarray or pd.Series): Quantities of stockouts over time
        unit_price (float): Selling price per unit
        unit_cost (float): Cost per unit
        
    Returns:
        float: Total lost sales cost
    """
    profit_margin = unit_price - unit_cost
    return np.sum(stockout_quantities) * profit_margin


def calculate_total_inventory_cost(holding_cost, ordering_cost, stockout_cost=0, lost_sales_cost=0):
    """
    Calculate total inventory cost.
    
    Args:
        holding_cost (float): Total holding cost
        ordering_cost (float): Total ordering cost
        stockout_cost (float): Total stockout cost
        lost_sales_cost (float): Total lost sales cost
        
    Returns:
        float: Total inventory cost
    """
    return holding_cost + ordering_cost + stockout_cost + lost_sales_cost


def calculate_annual_inventory_costs(average_inventory, annual_demand, order_quantity, 
                                  unit_cost, holding_cost_rate, ordering_cost, 
                                  stockout_rate=0, stockout_cost_per_unit=0, 
                                  unit_price=None):
    """
    Calculate annual inventory costs.
    
    Args:
        average_inventory (float): Average inventory level
        annual_demand (float): Annual demand in units
        order_quantity (float): Order quantity
        unit_cost (float): Cost per unit
        holding_cost_rate (float): Annual holding cost rate as percentage of unit cost
        ordering_cost (float): Cost per order
        stockout_rate (float): Stockout rate (0-1)
        stockout_cost_per_unit (float): Cost per unit of stockout
        unit_price (float): Selling price per unit (for lost sales calculation)
        
    Returns:
        dict: Dictionary of inventory costs
    """
    # Calculate number of orders per year
    num_orders = annual_demand / order_quantity if order_quantity > 0 else 0
    
    # Calculate annual holding cost
    annual_holding_cost = average_inventory * unit_cost * holding_cost_rate
    
    # Calculate annual ordering cost
    annual_ordering_cost = num_orders * ordering_cost
    
    # Calculate annual stockout cost
    annual_stockout_units = annual_demand * stockout_rate
    annual_stockout_cost = annual_stockout_units * stockout_cost_per_unit
    
    # Calculate annual lost sales cost
    annual_lost_sales_cost = 0
    if unit_price is not None:
        profit_margin = unit_price - unit_cost
        annual_lost_sales_cost = annual_stockout_units * profit_margin
    
    # Calculate total annual inventory cost
    total_annual_cost = annual_holding_cost + annual_ordering_cost + annual_stockout_cost + annual_lost_sales_cost
    
    return {
        'annual_holding_cost': annual_holding_cost,
        'annual_ordering_cost': annual_ordering_cost,
        'annual_stockout_cost': annual_stockout_cost,
        'annual_lost_sales_cost': annual_lost_sales_cost,
        'total_annual_cost': total_annual_cost,
        'num_orders': num_orders
    }


def calculate_inventory_turnover(annual_demand, average_inventory):
    """
    Calculate inventory turnover ratio.
    
    Args:
        annual_demand (float): Annual demand in units
        average_inventory (float): Average inventory level
        
    Returns:
        float: Inventory turnover ratio
    """
    return annual_demand / average_inventory if average_inventory > 0 else float('inf')


def calculate_days_of_supply(inventory_level, daily_demand):
    """
    Calculate days of supply.
    
    Args:
        inventory_level (float): Current inventory level
        daily_demand (float): Average daily demand
        
    Returns:
        float: Days of supply
    """
    return inventory_level / daily_demand if daily_demand > 0 else float('inf')


def calculate_service_level(total_demand, stockouts):
    """
    Calculate service level (fill rate).
    
    Args:
        total_demand (float): Total demand
        stockouts (float): Total stockouts
        
    Returns:
        float: Service level (0-1)
    """
    fulfilled_demand = total_demand - stockouts
    return fulfilled_demand / total_demand if total_demand > 0 else 0


def calculate_cost_breakdown(inventory_params):
    """
    Calculate cost breakdown for inventory parameters.
    
    Args:
        inventory_params (dict): Dictionary of inventory parameters
        
    Returns:
        dict: Dictionary of cost breakdown
    """
    # Extract relevant parameters
    safety_stock = inventory_params['safety_stock']
    economic_order_quantity = inventory_params['economic_order_quantity']
    annual_demand = inventory_params['annual_demand']
    daily_demand = inventory_params['daily_demand']
    unit_cost = inventory_params.get('unit_cost', 0)
    holding_cost_rate = inventory_params.get('holding_cost_rate', 0)
    ordering_cost = inventory_params.get('ordering_cost', 0)
    
    # Calculate average inventory
    average_inventory = safety_stock + economic_order_quantity / 2
    
    # Calculate number of orders per year
    num_orders = annual_demand / economic_order_quantity if economic_order_quantity > 0 else 0
    
    # Calculate annual costs
    annual_holding_cost = average_inventory * unit_cost * holding_cost_rate
    annual_ordering_cost = num_orders * ordering_cost
    total_annual_cost = annual_holding_cost + annual_ordering_cost
    
    # Calculate cost percentages
    holding_cost_percentage = annual_holding_cost / total_annual_cost * 100 if total_annual_cost > 0 else 0
    ordering_cost_percentage = annual_ordering_cost / total_annual_cost * 100 if total_annual_cost > 0 else 0
    
    # Calculate inventory turnover
    inventory_turnover = annual_demand / average_inventory if average_inventory > 0 else float('inf')
    
    # Calculate days of supply
    days_of_supply = average_inventory / daily_demand * 365 if daily_demand > 0 else float('inf')
    
    return {
        'average_inventory': average_inventory,
        'num_orders': num_orders,
        'annual_holding_cost': annual_holding_cost,
        'annual_ordering_cost': annual_ordering_cost,
        'total_annual_cost': total_annual_cost,
        'holding_cost_percentage': holding_cost_percentage,
        'ordering_cost_percentage': ordering_cost_percentage,
        'inventory_turnover': inventory_turnover,
        'days_of_supply': days_of_supply
    }


def calculate_multi_echelon_costs(inventory_levels, unit_costs, holding_cost_rates, ordering_costs, order_quantities):
    """
    Calculate costs for a multi-echelon inventory system.
    
    Args:
        inventory_levels (dict): Dictionary of inventory levels by echelon
        unit_costs (dict): Dictionary of unit costs by echelon
        holding_cost_rates (dict): Dictionary of holding cost rates by echelon
        ordering_costs (dict): Dictionary of ordering costs by echelon
        order_quantities (dict): Dictionary of order quantities by echelon
        
    Returns:
        dict: Dictionary of costs by echelon and total
    """
    # Initialize cost dictionaries
    holding_costs = {}
    ordering_costs_dict = {}
    total_costs = {}
    
    # Calculate costs for each echelon
    for echelon, inventory in inventory_levels.items():
        # Get parameters for this echelon
        unit_cost = unit_costs[echelon]
        holding_cost_rate = holding_cost_rates[echelon]
        ordering_cost = ordering_costs[echelon]
        order_quantity = order_quantities[echelon]
        
        # Calculate average inventory
        average_inventory = np.mean(inventory)
        
        # Calculate annual demand (assuming 365 days of data)
        daily_demand = np.diff(inventory).mean()  # Approximation of daily demand
        annual_demand = daily_demand * 365
        
        # Calculate number of orders per year
        num_orders = annual_demand / order_quantity if order_quantity > 0 else 0
        
        # Calculate costs
        holding_cost = average_inventory * unit_cost * holding_cost_rate
        ordering_cost_value = num_orders * ordering_cost
        total_cost = holding_cost + ordering_cost_value
        
        # Store costs
        holding_costs[echelon] = holding_cost
        ordering_costs_dict[echelon] = ordering_cost_value
        total_costs[echelon] = total_cost
    
    # Calculate total costs across all echelons
    total_holding_cost = sum(holding_costs.values())
    total_ordering_cost = sum(ordering_costs_dict.values())
    total_cost = sum(total_costs.values())
    
    return {
        'holding_costs': holding_costs,
        'ordering_costs': ordering_costs_dict,
        'total_costs': total_costs,
        'total_holding_cost': total_holding_cost,
        'total_ordering_cost': total_ordering_cost,
        'total_cost': total_cost
    }


def calculate_inventory_costs(demand_forecast, inventory_policy, unit_cost=10, 
                             holding_cost_rate=0.25, ordering_cost=50, 
                             stockout_cost_per_unit=20, unit_price=None):
    """
    Calculate inventory costs based on demand forecast and inventory policy.
    
    Parameters:
    -----------
    demand_forecast : pandas.Series or numpy.ndarray
        Forecasted demand over time
    inventory_policy : dict
        Dictionary with inventory policy parameters (reorder_point, order_quantity)
    unit_cost : float, optional
        Cost per unit
    holding_cost_rate : float, optional
        Annual holding cost rate as percentage of unit cost
    ordering_cost : float, optional
        Cost per order
    stockout_cost_per_unit : float, optional
        Cost per unit of stockout
    unit_price : float, optional
        Selling price per unit (for lost sales calculation)
    
    Returns:
    --------
    dict:
        Dictionary with inventory costs and metrics
    """
    # Convert to numpy array if it's a pandas Series
    if isinstance(demand_forecast, pd.Series):
        demand_forecast = demand_forecast.values
    
    # Extract policy parameters
    reorder_point = inventory_policy.get('reorder_point', 0)
    order_quantity = inventory_policy.get('order_quantity', 0)
    
    if 'economic_order_quantity' in inventory_policy:
        order_quantity = inventory_policy['economic_order_quantity']
    
    # Initialize variables
    inventory_level = reorder_point + order_quantity  # Start with full inventory
    inventory_levels = []
    order_placed = False
    orders = []
    stockouts = []
    
    # Simulate inventory over time
    for demand in demand_forecast:
        # Record current inventory level
        inventory_levels.append(inventory_level)
        
        # Check if we need to place an order
        if inventory_level <= reorder_point and not order_placed:
            orders.append(order_quantity)
            order_placed = True
        else:
            orders.append(0)
        
        # Fulfill demand (or record stockout)
        if demand <= inventory_level:
            inventory_level -= demand
            stockouts.append(0)
        else:
            stockout_qty = demand - inventory_level
            stockouts.append(stockout_qty)
            inventory_level = 0
        
        # Receive order (simplified: receive immediately after ordering)
        if order_placed:
            inventory_level += order_quantity
            order_placed = False
    
    # Convert lists to arrays
    inventory_levels = np.array(inventory_levels)
    orders = np.array(orders)
    stockouts = np.array(stockouts)
    
    # Calculate costs
    total_demand = np.sum(demand_forecast)
    total_stockouts = np.sum(stockouts)
    average_inventory = np.mean(inventory_levels)
    annual_factor = 365 / len(demand_forecast)  # Scale to annual if needed
    
    # Calculate annual costs
    annual_demand = total_demand * annual_factor
    num_orders = np.sum(orders > 0) * annual_factor
    
    # Calculate holding cost
    holding_cost = calculate_holding_cost(inventory_levels, unit_cost, holding_cost_rate/365) * annual_factor
    
    # Calculate ordering cost
    ordering_cost_total = calculate_ordering_cost(num_orders, ordering_cost)
    
    # Calculate stockout cost
    stockout_cost = calculate_stockout_cost(stockouts, stockout_cost_per_unit) * annual_factor
    
    # Calculate lost sales cost if unit price is provided
    lost_sales_cost = 0
    if unit_price is not None:
        lost_sales_cost = calculate_lost_sales_cost(stockouts, unit_price, unit_cost) * annual_factor
    
    # Calculate total cost
    total_cost = holding_cost + ordering_cost_total + stockout_cost + lost_sales_cost
    
    # Calculate service level
    service_level = calculate_service_level(total_demand, total_stockouts)
    
    # Calculate inventory turnover
    inventory_turnover = calculate_inventory_turnover(annual_demand, average_inventory)
    
    return {
        'average_inventory': average_inventory,
        'annual_demand': annual_demand,
        'num_orders': num_orders,
        'service_level': service_level,
        'inventory_turnover': inventory_turnover,
        'annual_holding_cost': holding_cost,
        'annual_ordering_cost': ordering_cost_total,
        'annual_stockout_cost': stockout_cost,
        'annual_lost_sales_cost': lost_sales_cost,
        'total_annual_cost': total_cost,
        'inventory_levels': inventory_levels,
        'stockouts': stockouts
    }


def main():
    """Main function to demonstrate cost calculations."""
    # Create sample inventory data
    inventory_levels = np.random.randint(100, 200, size=365)  # Daily inventory levels for a year
    stockout_quantities = np.zeros(365)
    stockout_quantities[inventory_levels < 10] = 10 - inventory_levels[inventory_levels < 10]
    inventory_levels[inventory_levels < 0] = 0
    
    # Define parameters
    unit_cost = 10
    unit_price = 25
    holding_cost_rate = 0.2
    ordering_cost = 50
    stockout_cost_per_unit = 5
    
    # Calculate costs
    holding_cost = calculate_holding_cost(inventory_levels, unit_cost, holding_cost_rate)
    ordering_cost_value = calculate_ordering_cost(12, ordering_cost)  # 12 orders per year
    stockout_cost = calculate_stockout_cost(stockout_quantities, stockout_cost_per_unit)
    lost_sales_cost = calculate_lost_sales_cost(stockout_quantities, unit_price, unit_cost)
    total_cost = calculate_total_inventory_cost(holding_cost, ordering_cost_value, stockout_cost, lost_sales_cost)
    
    # Print results
    print("Inventory Costs:")
    print(f"  Holding Cost: ${holding_cost:.2f}")
    print(f"  Ordering Cost: ${ordering_cost_value:.2f}")
    print(f"  Stockout Cost: ${stockout_cost:.2f}")
    print(f"  Lost Sales Cost: ${lost_sales_cost:.2f}")
    print(f"  Total Cost: ${total_cost:.2f}")
    
    # Calculate annual costs
    annual_demand = 1200
    order_quantity = 100
    average_inventory = np.mean(inventory_levels)
    
    annual_costs = calculate_annual_inventory_costs(
        average_inventory, annual_demand, order_quantity,
        unit_cost, holding_cost_rate, ordering_cost,
        stockout_rate=np.sum(stockout_quantities) / annual_demand,
        stockout_cost_per_unit=stockout_cost_per_unit,
        unit_price=unit_price
    )
    
    # Print annual costs
    print("\nAnnual Inventory Costs:")
    for key, value in annual_costs.items():
        print(f"  {key}: ${value:.2f}")
    
    # Calculate additional metrics
    inventory_turnover = calculate_inventory_turnover(annual_demand, average_inventory)
    days_of_supply = calculate_days_of_supply(average_inventory, annual_demand / 365)
    service_level = calculate_service_level(annual_demand, np.sum(stockout_quantities))
    
    # Print metrics
    print("\nInventory Metrics:")
    print(f"  Inventory Turnover: {inventory_turnover:.2f}")
    print(f"  Days of Supply: {days_of_supply:.2f}")
    print(f"  Service Level: {service_level:.2%}")


if __name__ == "__main__":
    main() 