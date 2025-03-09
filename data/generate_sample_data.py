#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate sample retail sales data for demand forecasting and inventory optimization.
This script creates realistic retail datasets with seasonal patterns, trends, and external factors.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_store_data(n_stores=10):
    """Generate store metadata."""
    store_types = ['Supermarket', 'Hypermarket', 'Convenience', 'Department Store', 'Specialty']
    regions = ['North', 'South', 'East', 'West', 'Central']
    countries = ['USA', 'Canada', 'UK', 'Germany', 'France', 'Australia', 'Japan', 'China', 'Brazil', 'India']
    
    stores = []
    for i in range(1, n_stores + 1):
        store = {
            'store_id': i,
            'store_name': f'Store_{i}',
            'store_type': random.choice(store_types),
            'region': random.choice(regions),
            'country': random.choice(countries),
            'size_sqft': random.randint(1000, 50000),
            'opening_date': datetime(2010, 1, 1) + timedelta(days=random.randint(0, 3650)),
            'is_active': random.random() > 0.1  # 90% of stores are active
        }
        stores.append(store)
    
    return pd.DataFrame(stores)

def generate_product_data(n_products=50):
    """Generate product metadata."""
    categories = ['Food', 'Beverages', 'Household', 'Electronics', 'Clothing', 'Health & Beauty', 'Toys']
    subcategories = {
        'Food': ['Dairy', 'Bakery', 'Meat', 'Produce', 'Frozen', 'Snacks'],
        'Beverages': ['Soft Drinks', 'Alcohol', 'Coffee', 'Tea', 'Juice'],
        'Household': ['Cleaning', 'Laundry', 'Kitchen', 'Bathroom'],
        'Electronics': ['Phones', 'Computers', 'TV', 'Audio', 'Accessories'],
        'Clothing': ['Men', 'Women', 'Children', 'Footwear', 'Accessories'],
        'Health & Beauty': ['Skincare', 'Haircare', 'Makeup', 'Pharmacy', 'Personal Care'],
        'Toys': ['Games', 'Educational', 'Outdoor', 'Dolls', 'Building']
    }
    
    products = []
    for i in range(1, n_products + 1):
        category = random.choice(categories)
        product = {
            'product_id': i,
            'product_name': f'Product_{i}',
            'category': category,
            'subcategory': random.choice(subcategories[category]),
            'price': round(random.uniform(1.99, 199.99), 2),
            'cost': 0,  # Will be calculated as a percentage of price
            'weight_kg': round(random.uniform(0.1, 20.0), 2),
            'is_perishable': random.random() < 0.3,  # 30% of products are perishable
            'shelf_life_days': random.choice([7, 14, 30, 60, 90, 180, 365, 999]) if random.random() < 0.3 else None,
            'lead_time_days': random.randint(1, 30),  # Days to restock
            'min_order_quantity': random.choice([1, 5, 10, 20, 50, 100])
        }
        # Cost is typically 40-80% of price
        product['cost'] = round(product['price'] * random.uniform(0.4, 0.8), 2)
        products.append(product)
    
    return pd.DataFrame(products)

def generate_external_data(start_date, end_date):
    """Generate external factors data (weather, holidays, economic indicators)."""
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate weather data
    weather = []
    base_temp = 15  # Base temperature in Celsius
    for date in date_range:
        # Seasonal temperature variation
        day_of_year = date.dayofyear
        seasonal_temp = base_temp + 15 * np.sin(2 * np.pi * day_of_year / 365)
        
        # Add random variation
        temp = seasonal_temp + np.random.normal(0, 3)
        
        # Precipitation (more likely in winter)
        winter_factor = 1 + np.cos(2 * np.pi * day_of_year / 365)
        precip_prob = 0.3 * winter_factor
        precipitation = np.random.exponential(5) if random.random() < precip_prob else 0
        
        weather.append({
            'date': date,
            'temperature': round(temp, 1),
            'precipitation': round(precipitation, 1),
            'is_sunny': precipitation == 0 and random.random() < 0.7
        })
    
    weather_df = pd.DataFrame(weather)
    
    # Generate economic indicators
    economics = []
    base_consumer_index = 100
    base_unemployment = 5.0
    base_inflation = 2.0
    
    for date in date_range:
        if date.day == 1:  # Monthly data
            # Add trends and cycles to economic data
            month_num = (date.year - start_date.year) * 12 + date.month - start_date.month
            
            # Consumer confidence with a slight upward trend and seasonal variation
            consumer_confidence = base_consumer_index + 0.1 * month_num + 5 * np.sin(2 * np.pi * month_num / 12) + np.random.normal(0, 2)
            
            # Unemployment with a cyclical pattern
            unemployment = base_unemployment + 1.5 * np.sin(2 * np.pi * month_num / 48) + np.random.normal(0, 0.2)
            
            # Inflation with a slight upward trend
            inflation = base_inflation + 0.02 * month_num + np.random.normal(0, 0.3)
            
            economics.append({
                'date': date,
                'consumer_confidence_index': round(consumer_confidence, 1),
                'unemployment_rate': round(max(0, unemployment), 1),
                'inflation_rate': round(max(0, inflation), 1)
            })
    
    economics_df = pd.DataFrame(economics)
    
    # Generate holiday data
    holidays = []
    major_holidays = {
        '01-01': 'New Year',
        '07-04': 'Independence Day',
        '11-25': 'Thanksgiving',
        '12-25': 'Christmas'
    }
    
    for date in date_range:
        date_key = date.strftime('%m-%d')
        if date_key in major_holidays:
            holidays.append({
                'date': date,
                'holiday_name': major_holidays[date_key],
                'is_major_holiday': True
            })
        elif random.random() < 0.01:  # Add some random minor holidays
            holidays.append({
                'date': date,
                'holiday_name': f'Local Holiday {date.strftime("%B %d")}',
                'is_major_holiday': False
            })
    
    holidays_df = pd.DataFrame(holidays) if holidays else pd.DataFrame(columns=['date', 'holiday_name', 'is_major_holiday'])
    
    return weather_df, economics_df, holidays_df

def generate_sales_data(stores_df, products_df, external_data=None, start_date=None, end_date=None):
    """Generate daily sales data with realistic patterns."""
    # Convert external_data to start_date and end_date if needed
    if external_data is not None and (start_date is None or end_date is None):
        if isinstance(external_data, tuple) and len(external_data) >= 1:
            # Handle case where external_data is a tuple of dataframes
            weather_df = external_data[0]
            if 'date' in weather_df.columns:
                if start_date is None:
                    start_date = weather_df['date'].min()
                if end_date is None:
                    end_date = weather_df['date'].max()
        elif isinstance(external_data, datetime):
            # Handle case where external_data is actually a start_date
            if start_date is None:
                start_date = external_data
            if end_date is None and isinstance(start_date, datetime):
                # Default to 3 years of data
                end_date = start_date + timedelta(days=3*365)
    
    # If start_date and end_date are not provided, use default values
    if start_date is None:
        start_date = datetime(2020, 1, 1)
    if end_date is None:
        end_date = datetime(2022, 12, 31)
    
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create base sales patterns for each product
    product_patterns = {}
    for product_id in products_df['product_id']:
        # Base sales level
        base_sales = np.random.lognormal(3, 1)
        
        # Seasonality parameters (yearly, monthly, weekly)
        yearly_amp = random.uniform(0.1, 0.5) * base_sales
        monthly_amp = random.uniform(0.05, 0.2) * base_sales
        weekly_amp = random.uniform(0.1, 0.3) * base_sales
        
        # Trend parameter
        trend_factor = random.uniform(-0.0005, 0.001)  # Some products grow, some decline
        
        product_patterns[product_id] = {
            'base_sales': base_sales,
            'yearly_amp': yearly_amp,
            'monthly_amp': monthly_amp,
            'weekly_amp': weekly_amp,
            'trend_factor': trend_factor
        }
    
    # Create store multipliers
    store_multipliers = {}
    for store_id in stores_df['store_id']:
        store_multipliers[store_id] = random.uniform(0.5, 2.0)
    
    # Generate sales data
    sales_data = []
    
    for date in date_range:
        day_of_year = date.dayofyear
        day_of_month = date.day
        day_of_week = date.dayofweek  # 0 = Monday, 6 = Sunday
        
        # Weekend effect
        weekend_multiplier = 1.3 if day_of_week >= 5 else 1.0
        
        # Month-end effect (higher sales at end of month)
        month_end_multiplier = 1.2 if day_of_month >= 28 else 1.0
        
        # Sample a subset of store-product combinations for each day
        # (not all products are sold in all stores every day)
        n_combinations = random.randint(50, 200)
        store_ids = np.random.choice(stores_df['store_id'], n_combinations)
        product_ids = np.random.choice(products_df['product_id'], n_combinations)
        
        for i in range(n_combinations):
            store_id = store_ids[i]
            product_id = product_ids[i]
            
            # Skip if store is not active
            if not stores_df.loc[stores_df['store_id'] == store_id, 'is_active'].values[0]:
                continue
                
            pattern = product_patterns[product_id]
            
            # Calculate sales with seasonality, trend, and noise
            days_since_start = (date - start_date).days
            
            # Yearly seasonality
            yearly_seasonal = pattern['yearly_amp'] * np.sin(2 * np.pi * day_of_year / 365)
            
            # Monthly seasonality
            monthly_seasonal = pattern['monthly_amp'] * np.sin(2 * np.pi * day_of_month / 30)
            
            # Weekly seasonality
            weekly_seasonal = pattern['weekly_amp'] * np.sin(2 * np.pi * day_of_week / 7)
            
            # Trend component
            trend = pattern['trend_factor'] * days_since_start * pattern['base_sales']
            
            # Combine components
            expected_sales = (pattern['base_sales'] + yearly_seasonal + monthly_seasonal + 
                             weekly_seasonal + trend) * store_multipliers[store_id] * weekend_multiplier * month_end_multiplier
            
            # Add noise
            noise_factor = np.random.normal(1, 0.2)
            sales_quantity = max(0, int(expected_sales * noise_factor))
            
            # Skip if no sales
            if sales_quantity == 0:
                continue
                
            # Product info
            product_info = products_df.loc[products_df['product_id'] == product_id].iloc[0]
            price = product_info['price']
            cost = product_info['cost'] if 'cost' in product_info else price * 0.6
            
            # Record sales data
            sales_data.append({
                'date': date,
                'store_id': store_id,
                'product_id': product_id,
                'quantity': sales_quantity,
                'price': price,
                'revenue': price * sales_quantity,
                'cost': cost,
                'profit': (price - cost) * sales_quantity
            })
    
    # Create DataFrame
    sales_df = pd.DataFrame(sales_data)
    
    # Rename 'quantity' to 'sales' to ensure consistency across the application
    if 'quantity' in sales_df.columns and 'sales' not in sales_df.columns:
        sales_df = sales_df.rename(columns={'quantity': 'sales'})
    
    return sales_df

def generate_inventory_data(sales_df, products_df):
    """Generate inventory data based on sales."""
    # Group sales by store and product
    store_product_sales = sales_df.groupby(['store_id', 'product_id'])['sales'].sum().reset_index()
    
    inventory_data = []
    current_date = datetime.now().date()
    
    for _, row in store_product_sales.iterrows():
        store_id = row['store_id']
        product_id = row['product_id']
        avg_daily_sales = row['sales'] / 365  # Approximate
        
        # Get product details
        product = products_df[products_df['product_id'] == product_id].iloc[0]
        lead_time = product['lead_time_days']
        
        # Calculate safety stock (cover for 2x lead time)
        safety_stock = int(avg_daily_sales * lead_time * 2)
        
        # Calculate reorder point
        reorder_point = int(avg_daily_sales * lead_time * 1.5)
        
        # Calculate economic order quantity (simplified)
        eoq = int(np.sqrt(2 * avg_daily_sales * 365 * 50 / (0.2 * product['cost'])))
        
        # Current stock level (random between safety stock and 2x safety stock)
        current_stock = random.randint(safety_stock, safety_stock * 2)
        
        inventory_data.append({
            'store_id': store_id,
            'product_id': product_id,
            'current_stock': current_stock,
            'safety_stock': safety_stock,
            'reorder_point': reorder_point,
            'economic_order_quantity': eoq,
            'last_restock_date': current_date - timedelta(days=random.randint(1, 30)),
            'next_delivery_date': current_date + timedelta(days=random.randint(1, 15)) if current_stock < reorder_point else None
        })
    
    return pd.DataFrame(inventory_data)

def generate_promotion_data(start_date, end_date, products_df):
    """Generate promotion data."""
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    promotion_types = ['Discount', 'BOGO', 'Bundle', 'Flash Sale', 'Clearance']
    promotion_data = []
    
    # Generate about 100 promotions
    for _ in range(100):
        # Random start date within the range
        promo_start = random.choice(date_range[:-14])  # Ensure at least 2 weeks before end
        
        # Duration between 3 and 14 days
        duration = random.randint(3, 14)
        promo_end = promo_start + timedelta(days=duration)
        
        # Select random products for the promotion (1-5 products)
        n_products = random.randint(1, 5)
        promo_products = np.random.choice(products_df['product_id'], n_products, replace=False)
        
        promo_type = random.choice(promotion_types)
        
        # Different discount rates based on promotion type
        if promo_type == 'Discount':
            discount_rate = random.choice([0.1, 0.15, 0.2, 0.25, 0.3, 0.5])
        elif promo_type == 'BOGO':
            discount_rate = 0.5  # Buy one get one 50% off
        elif promo_type == 'Bundle':
            discount_rate = 0.2  # 20% off when buying multiple
        elif promo_type == 'Flash Sale':
            discount_rate = random.choice([0.3, 0.4, 0.5])
        else:  # Clearance
            discount_rate = random.choice([0.5, 0.6, 0.7])
        
        for product_id in promo_products:
            promotion_data.append({
                'promotion_id': len(promotion_data) + 1,
                'product_id': product_id,
                'promotion_type': promo_type,
                'discount_rate': discount_rate,
                'start_date': promo_start,
                'end_date': promo_end,
                'description': f"{promo_type} - {int(discount_rate*100)}% off"
            })
    
    return pd.DataFrame(promotion_data)

def main():
    """Generate all datasets and save to CSV files."""
    print("Generating retail datasets for demand forecasting...")
    
    # Define date range (3 years of data)
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2022, 12, 31)
    
    # Create output directory if it doesn't exist
    os.makedirs('retail_data', exist_ok=True)
    
    # Generate store data
    print("Generating store data...")
    stores_df = generate_store_data(n_stores=20)
    stores_df.to_csv('retail_data/stores.csv', index=False)
    
    # Generate product data
    print("Generating product data...")
    products_df = generate_product_data(n_products=100)
    products_df.to_csv('retail_data/products.csv', index=False)
    
    # Generate external data
    print("Generating external factors data...")
    weather_df, economics_df, holidays_df = generate_external_data(start_date, end_date)
    weather_df.to_csv('retail_data/weather.csv', index=False)
    economics_df.to_csv('retail_data/economic_indicators.csv', index=False)
    holidays_df.to_csv('retail_data/holidays.csv', index=False)
    
    # Generate sales data
    print("Generating sales data (this may take a while)...")
    sales_df = generate_sales_data(stores_df, products_df, weather_df, start_date, end_date)
    sales_df.to_csv('retail_data/sales.csv', index=False)
    
    # Generate inventory data
    print("Generating inventory data...")
    inventory_df = generate_inventory_data(sales_df, products_df)
    inventory_df.to_csv('retail_data/inventory.csv', index=False)
    
    # Generate promotion data
    print("Generating promotion data...")
    promotions_df = generate_promotion_data(start_date, end_date, products_df)
    promotions_df.to_csv('retail_data/promotions.csv', index=False)
    
    # Generate some visualizations
    print("Creating sample visualizations...")
    os.makedirs('retail_data/visualizations', exist_ok=True)
    
    # Plot total sales over time
    plt.figure(figsize=(12, 6))
    daily_sales = sales_df.groupby('date')['revenue'].sum().reset_index()
    plt.plot(daily_sales['date'], daily_sales['revenue'])
    plt.title('Total Daily Sales Revenue')
    plt.xlabel('Date')
    plt.ylabel('Revenue ($)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('retail_data/visualizations/daily_sales.png')
    
    # Plot sales by category
    plt.figure(figsize=(12, 6))
    sales_with_category = sales_df.merge(products_df[['product_id', 'category']], on='product_id')
    category_sales = sales_with_category.groupby('category')['revenue'].sum().sort_values(ascending=False)
    sns.barplot(x=category_sales.index, y=category_sales.values)
    plt.title('Total Sales by Product Category')
    plt.xlabel('Category')
    plt.ylabel('Revenue ($)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('retail_data/visualizations/category_sales.png')
    
    print("Dataset generation complete! Files saved in the 'retail_data' directory.")

def plot_sample_data(sales_df, stores_df=None, products_df=None, external_data=None, 
                    n_products=5, n_stores=3, figsize=(15, 12)):
    """
    Create visualizations of sample retail data.
    
    Parameters:
    -----------
    sales_df : pandas.DataFrame
        Sales data with date, store_id, product_id, and quantity/sales
    stores_df : pandas.DataFrame, optional
        Store metadata
    products_df : pandas.DataFrame, optional
        Product metadata
    external_data : dict, optional
        Dictionary of external data DataFrames
    n_products : int, optional
        Number of products to plot
    n_stores : int, optional
        Number of stores to plot
    figsize : tuple, optional
        Figure size
    
    Returns:
    --------
    list of matplotlib.figure.Figure
        List of generated figures
    """
    figures = []
    
    # Ensure date column is datetime
    sales_df['date'] = pd.to_datetime(sales_df['date'])
    
    # Determine sales column name (can be 'quantity' or 'sales')
    sales_col = 'quantity' if 'quantity' in sales_df.columns else 'sales'
    
    # Aggregate sales by date
    daily_sales = sales_df.groupby('date')[sales_col].sum().reset_index()
    
    # Create figure 1: Overall sales trend
    fig1, ax1 = plt.subplots(figsize=figsize)
    ax1.plot(daily_sales['date'], daily_sales[sales_col], 'b-')
    ax1.set_title('Overall Daily Sales', fontsize=16)
    ax1.set_xlabel('Date', fontsize=14)
    ax1.set_ylabel(f'Total {sales_col.title()} Sold', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(np.arange(len(daily_sales)), daily_sales[sales_col], 1)
    p = np.poly1d(z)
    ax1.plot(daily_sales['date'], p(np.arange(len(daily_sales))), "r--", alpha=0.8)
    
    figures.append(fig1)
    
    # Create figure 2: Sales by week day
    fig2, ax2 = plt.subplots(figsize=figsize)
    
    # Add weekday column
    daily_sales['weekday'] = daily_sales['date'].dt.day_name()
    
    # Get weekday order
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Aggregate by weekday
    weekday_sales = daily_sales.groupby('weekday')[sales_col].mean().reindex(weekday_order)
    
    ax2.bar(weekday_sales.index, weekday_sales.values, color='skyblue')
    ax2.set_title('Average Sales by Day of Week', fontsize=16)
    ax2.set_xlabel('Day of Week', fontsize=14)
    ax2.set_ylabel(f'Average {sales_col.title()} Sold', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    
    figures.append(fig2)
    
    # Create figure 3: Sales by month
    fig3, ax3 = plt.subplots(figsize=figsize)
    
    # Add month column
    daily_sales['month'] = daily_sales['date'].dt.month_name()
    
    # Get month order
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                  'July', 'August', 'September', 'October', 'November', 'December']
    
    # Aggregate by month
    monthly_sales = daily_sales.groupby('month')[sales_col].mean().reindex(month_order)
    
    ax3.bar(monthly_sales.index, monthly_sales.values, color='lightgreen')
    ax3.set_title('Average Sales by Month', fontsize=16)
    ax3.set_xlabel('Month', fontsize=14)
    ax3.set_ylabel(f'Average {sales_col.title()} Sold', fontsize=14)
    ax3.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45)
    
    figures.append(fig3)
    
    # If products_df is provided, plot sales for top products
    if products_df is not None:
        # Aggregate sales by product
        product_sales = sales_df.groupby('product_id')[sales_col].sum().reset_index()
        
        # Get top n products
        top_products = product_sales.sort_values(sales_col, ascending=False).head(n_products)
        
        # Merge with product info
        top_products = top_products.merge(products_df, on='product_id')
        
        # Create figure 4: Top products sales
        fig4, ax4 = plt.subplots(figsize=figsize)
        
        ax4.bar(top_products['product_name'], top_products[sales_col], color='coral')
        ax4.set_title(f'Top {n_products} Products by Sales', fontsize=16)
        ax4.set_xlabel('Product', fontsize=14)
        ax4.set_ylabel(f'Total {sales_col.title()} Sold', fontsize=14)
        ax4.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        
        figures.append(fig4)
        
        # Create figure 5: Product category distribution
        if 'category' in products_df.columns:
            fig5, ax5 = plt.subplots(figsize=figsize)
            
            # Merge sales with product info
            sales_with_categories = sales_df.merge(products_df[['product_id', 'category']], on='product_id')
            
            # Aggregate by category
            category_sales = sales_with_categories.groupby('category')[sales_col].sum().sort_values(ascending=False)
            
            ax5.pie(category_sales, labels=category_sales.index, autopct='%1.1f%%', 
                   startangle=90, shadow=True)
            ax5.set_title('Sales Distribution by Product Category', fontsize=16)
            
            figures.append(fig5)
    
    # If stores_df is provided, plot sales for top stores
    if stores_df is not None:
        # Aggregate sales by store
        store_sales = sales_df.groupby('store_id')[sales_col].sum().reset_index()
        
        # Get top n stores
        top_stores = store_sales.sort_values(sales_col, ascending=False).head(n_stores)
        
        # Merge with store info
        top_stores = top_stores.merge(stores_df, on='store_id')
        
        # Create figure 6: Top stores sales
        fig6, ax6 = plt.subplots(figsize=figsize)
        
        ax6.bar(top_stores['store_name'], top_stores[sales_col], color='purple')
        ax6.set_title(f'Top {n_stores} Stores by Sales', fontsize=16)
        ax6.set_xlabel('Store', fontsize=14)
        ax6.set_ylabel(f'Total {sales_col.title()} Sold', fontsize=14)
        ax6.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        
        figures.append(fig6)
        
        # Create figure 7: Store region distribution
        if 'region' in stores_df.columns:
            fig7, ax7 = plt.subplots(figsize=figsize)
            
            # Merge sales with store info
            sales_with_regions = sales_df.merge(stores_df[['store_id', 'region']], on='store_id')
            
            # Aggregate by region
            region_sales = sales_with_regions.groupby('region')[sales_col].sum().sort_values(ascending=False)
            
            ax7.barh(region_sales.index, region_sales.values, color='teal')
            ax7.set_title('Sales by Store Region', fontsize=16)
            ax7.set_xlabel(f'Total {sales_col.title()} Sold', fontsize=14)
            ax7.set_ylabel('Region', fontsize=14)
            ax7.grid(True, alpha=0.3, axis='x')
            
            figures.append(fig7)
    
    # If external data is provided, plot correlations with sales
    if external_data is not None and 'weather' in external_data:
        weather_df = external_data['weather']
        
        # Ensure date column is datetime
        weather_df['date'] = pd.to_datetime(weather_df['date'])
        
        # Merge weather with daily sales
        sales_weather = daily_sales.merge(weather_df, on='date')
        
        # Create figure 8: Sales vs Temperature
        fig8, ax8 = plt.subplots(figsize=figsize)
        
        ax8.scatter(sales_weather['temperature'], sales_weather[sales_col], alpha=0.5, color='orange')
        ax8.set_title('Sales vs Temperature', fontsize=16)
        ax8.set_xlabel('Temperature (°C)', fontsize=14)
        ax8.set_ylabel(f'Daily {sales_col.title()}', fontsize=14)
        ax8.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(sales_weather['temperature'], sales_weather[sales_col], 1)
        p = np.poly1d(z)
        ax8.plot(sales_weather['temperature'], p(sales_weather['temperature']), "r--", alpha=0.8)
        
        figures.append(fig8)
    
    return figures

if __name__ == "__main__":
    main() 