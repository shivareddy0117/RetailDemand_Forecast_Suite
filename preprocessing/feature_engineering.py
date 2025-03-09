#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature engineering module for retail demand forecasting.
This module provides functions to create advanced features from retail data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import holidays
import warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_promotion_features(sales_df, promo_df, date_col='date', store_col='store_id', 
                            product_col='product_id', start_col='start_date', end_col='end_date',
                            discount_col='discount_rate', promo_type_col='promotion_type'):
    """
    Create promotion-related features for sales data.
    
    Args:
        sales_df (pd.DataFrame): Sales DataFrame
        promo_df (pd.DataFrame): Promotions DataFrame
        date_col (str): Name of the date column
        store_col (str): Name of the store ID column
        product_col (str): Name of the product ID column
        start_col (str): Name of the promotion start date column
        end_col (str): Name of the promotion end date column
        discount_col (str): Name of the discount rate column
        promo_type_col (str): Name of the promotion type column
        
    Returns:
        pd.DataFrame: Sales DataFrame with promotion features
    """
    # Make a copy of the sales DataFrame
    sales_copy = sales_df.copy()
    
    # Convert dates to datetime if needed
    sales_copy[date_col] = pd.to_datetime(sales_copy[date_col])
    promo_df[start_col] = pd.to_datetime(promo_df[start_col])
    promo_df[end_col] = pd.to_datetime(promo_df[end_col])
    
    # Initialize promotion features
    sales_copy['is_promotion'] = 0
    sales_copy['discount_rate'] = 0.0
    sales_copy['days_into_promotion'] = 0
    sales_copy['days_to_promotion_end'] = 0
    
    # Create dictionary for promotion types
    promo_types = promo_df[promo_type_col].unique()
    for promo_type in promo_types:
        sales_copy[f'is_{promo_type.lower().replace(" ", "_")}'] = 0
    
    # Iterate through promotions
    for _, promo in promo_df.iterrows():
        # Filter sales data for this promotion
        mask = ((sales_copy[date_col] >= promo[start_col]) & 
                (sales_copy[date_col] <= promo[end_col]) & 
                (sales_copy[product_col] == promo[product_col]))
        
        if mask.sum() > 0:
            # Update promotion flags
            sales_copy.loc[mask, 'is_promotion'] = 1
            sales_copy.loc[mask, 'discount_rate'] = promo[discount_col]
            
            # Update promotion type
            promo_type_col_name = f'is_{promo[promo_type_col].lower().replace(" ", "_")}'
            sales_copy.loc[mask, promo_type_col_name] = 1
            
            # Calculate days into promotion and days to end
            for idx in sales_copy.loc[mask].index:
                current_date = sales_copy.loc[idx, date_col]
                days_into = (current_date - promo[start_col]).days
                days_to_end = (promo[end_col] - current_date).days
                
                sales_copy.loc[idx, 'days_into_promotion'] = days_into
                sales_copy.loc[idx, 'days_to_promotion_end'] = days_to_end
    
    # Create promotion density features: number of active promotions per day
    # Group by date and count unique promotions
    promo_count = pd.DataFrame(columns=[date_col, 'promo_count'])
    
    # Create a date range covering all promotions
    all_dates = pd.date_range(start=promo_df[start_col].min(),
                              end=promo_df[end_col].max(),
                              freq='D')
    
    # Count promotions active on each date
    promo_counts = []
    for date in all_dates:
        count = ((promo_df[start_col] <= date) & (promo_df[end_col] >= date)).sum()
        promo_counts.append({'date': date, 'promo_count': count})
    
    promo_count = pd.DataFrame(promo_counts)
    
    # Merge promotion counts with sales data
    sales_copy = pd.merge(sales_copy, promo_count, on=date_col, how='left')
    sales_copy['promo_count'] = sales_copy['promo_count'].fillna(0).astype(int)
    
    logger.info(f"Created {2 + len(promo_types)} promotion-related features")
    
    return sales_copy


def create_holiday_features(df, date_col='date', country='US', state=None, years=None):
    """
    Create holiday features for sales data.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        date_col (str): Name of the date column
        country (str): Country code for holidays
        state (str): State code for regional holidays (optional)
        years (list): List of years to include
        
    Returns:
        pd.DataFrame: DataFrame with holiday features
    """
    # Make a copy of the DataFrame
    df_copy = df.copy()
    
    # Convert date to datetime if needed
    df_copy[date_col] = pd.to_datetime(df_copy[date_col])
    
    # Get the range of years from the data if not provided
    if years is None:
        years = list(range(df_copy[date_col].dt.year.min(), df_copy[date_col].dt.year.max() + 1))
    
    # Get holidays for the specified country and years
    country_holidays = holidays.country_holidays(country, subdiv=state, years=years)
    
    # Initialize holiday features
    df_copy['is_holiday'] = 0
    df_copy['is_major_holiday'] = 0
    df_copy['days_to_holiday'] = 99
    df_copy['days_after_holiday'] = 99
    
    # Map of major holidays (based on US holidays)
    major_holidays = [
        "New Year's Day", 
        "Independence Day", 
        "Thanksgiving", 
        "Christmas Day",
        "Memorial Day",
        "Labor Day"
    ]
    
    # Create dictionary of holidays
    holiday_dict = {}
    for date, name in country_holidays.items():
        holiday_dict[date] = name
    
    # Iterate through the DataFrame
    for idx, row in df_copy.iterrows():
        current_date = row[date_col].date()
        
        # Check if current date is a holiday
        if current_date in holiday_dict:
            df_copy.loc[idx, 'is_holiday'] = 1
            
            # Check if it's a major holiday
            if any(major in holiday_dict[current_date] for major in major_holidays):
                df_copy.loc[idx, 'is_major_holiday'] = 1
            
            df_copy.loc[idx, 'days_to_holiday'] = 0
            df_copy.loc[idx, 'days_after_holiday'] = 0
        else:
            # Find nearest upcoming holiday
            upcoming_holidays = [date for date in holiday_dict.keys() if date > current_date]
            if upcoming_holidays:
                nearest_upcoming = min(upcoming_holidays)
                df_copy.loc[idx, 'days_to_holiday'] = (nearest_upcoming - current_date).days
            
            # Find nearest past holiday
            past_holidays = [date for date in holiday_dict.keys() if date < current_date]
            if past_holidays:
                nearest_past = max(past_holidays)
                df_copy.loc[idx, 'days_after_holiday'] = (current_date - nearest_past).days
    
    # Create individual features for major holidays
    for major in major_holidays:
        df_copy[f'is_{major.lower().replace(" ", "_").replace("\'", "")}'] = 0
        
        # Find dates for this major holiday
        major_dates = [date for date, name in holiday_dict.items() if major in name]
        
        # Update the flag for these dates
        for date in major_dates:
            mask = df_copy[date_col].dt.date == date
            if mask.any():
                df_copy.loc[mask, f'is_{major.lower().replace(" ", "_").replace("\'", "")}'] = 1
    
    logger.info(f"Created {4 + len(major_holidays)} holiday-related features")
    
    return df_copy


def create_weather_features(sales_df, weather_df, date_col='date', store_col='store_id',
                           temp_col='temperature', precip_col='precipitation', 
                           sunny_col='is_sunny'):
    """
    Create weather-related features for sales data.
    
    Args:
        sales_df (pd.DataFrame): Sales DataFrame
        weather_df (pd.DataFrame): Weather DataFrame
        date_col (str): Name of the date column
        store_col (str): Name of the store ID column
        temp_col (str): Name of the temperature column
        precip_col (str): Name of the precipitation column
        sunny_col (str): Name of the sunny indicator column
        
    Returns:
        pd.DataFrame: Sales DataFrame with weather features
    """
    # Make a copy of the sales DataFrame
    sales_copy = sales_df.copy()
    
    # Convert dates to datetime if needed
    sales_copy[date_col] = pd.to_datetime(sales_copy[date_col])
    weather_df[date_col] = pd.to_datetime(weather_df[date_col])
    
    # Merge weather data with sales data
    sales_copy = pd.merge(sales_copy, weather_df, on=date_col, how='left')
    
    # Create temperature bins
    sales_copy['temp_bin'] = pd.cut(sales_copy[temp_col], 
                                   bins=[-100, 0, 10, 20, 30, 100],
                                   labels=['freezing', 'cold', 'mild', 'warm', 'hot'])
    
    # Create precipitation bins
    sales_copy['precip_bin'] = pd.cut(sales_copy[precip_col], 
                                     bins=[-1, 0, 5, 10, 20, 100],
                                     labels=['none', 'light', 'moderate', 'heavy', 'extreme'])
    
    # Create lagged weather features
    weather_cols = [temp_col, precip_col, sunny_col]
    for col in weather_cols:
        # Previous day
        sales_copy[f'{col}_lag1'] = sales_copy.groupby(store_col)[col].shift(1)
        
        # Previous week average
        sales_copy[f'{col}_lag_7d_avg'] = sales_copy.groupby(store_col)[col].transform(
            lambda x: x.shift(1).rolling(window=7).mean())
    
    # Create extreme weather flags
    sales_copy['is_extreme_hot'] = (sales_copy[temp_col] > 30).astype(int)
    sales_copy['is_extreme_cold'] = (sales_copy[temp_col] < 0).astype(int)
    sales_copy['is_heavy_rain'] = (sales_copy[precip_col] > 10).astype(int)
    
    # Create weather change features
    sales_copy['temp_change'] = sales_copy.groupby(store_col)[temp_col].diff()
    sales_copy['precip_change'] = sales_copy.groupby(store_col)[precip_col].diff()
    
    # Create temperature volatility (rolling standard deviation)
    sales_copy['temp_volatility'] = sales_copy.groupby(store_col)[temp_col].transform(
        lambda x: x.rolling(window=7).std())
    
    logger.info(f"Created 14 weather-related features")
    
    return sales_copy


def create_store_features(sales_df, store_df, date_col='date', store_col='store_id',
                         store_type_col='store_type', region_col='region', size_col='size_sqft',
                         opening_date_col='opening_date'):
    """
    Create store-related features for sales data.
    
    Args:
        sales_df (pd.DataFrame): Sales DataFrame
        store_df (pd.DataFrame): Store metadata DataFrame
        date_col (str): Name of the date column
        store_col (str): Name of the store ID column
        store_type_col (str): Name of the store type column
        region_col (str): Name of the region column
        size_col (str): Name of the store size column
        opening_date_col (str): Name of the store opening date column
        
    Returns:
        pd.DataFrame: Sales DataFrame with store features
    """
    # Make a copy of the sales DataFrame
    sales_copy = sales_df.copy()
    
    # Convert dates to datetime if needed
    sales_copy[date_col] = pd.to_datetime(sales_copy[date_col])
    
    if opening_date_col in store_df.columns:
        store_df[opening_date_col] = pd.to_datetime(store_df[opening_date_col])
    
    # Merge store data with sales data
    sales_copy = pd.merge(sales_copy, store_df, on=store_col, how='left')
    
    # Create store age feature
    if opening_date_col in store_df.columns:
        sales_copy['store_age_days'] = (sales_copy[date_col] - sales_copy[opening_date_col]).dt.days
        
        # Create store age bins
        sales_copy['store_age_bin'] = pd.cut(sales_copy['store_age_days'], 
                                           bins=[-1, 30, 90, 365, 730, 1825, 3650, 999999],
                                           labels=['new', '1-3m', '3-12m', '1-2y', '2-5y', '5-10y', '10y+'])
    
    # Create store size bins
    if size_col in store_df.columns:
        sales_copy['store_size_bin'] = pd.cut(sales_copy[size_col], 
                                            bins=[-1, 2000, 5000, 10000, 20000, 50000, 999999],
                                            labels=['tiny', 'small', 'medium', 'large', 'xlarge', 'mega'])
    
    # One-hot encode categorical store features
    categorical_cols = [col for col in [store_type_col, region_col] if col in store_df.columns]
    for col in categorical_cols:
        dummies = pd.get_dummies(sales_copy[col], prefix=col, dummy_na=False)
        sales_copy = pd.concat([sales_copy, dummies], axis=1)
    
    logger.info(f"Created store-related features from {len(store_df.columns) - 1} store attributes")
    
    return sales_copy


def create_product_features(sales_df, product_df, date_col='date', product_col='product_id',
                          category_col='category', subcategory_col='subcategory', 
                          price_col='price', cost_col='cost', perishable_col='is_perishable'):
    """
    Create product-related features for sales data.
    
    Args:
        sales_df (pd.DataFrame): Sales DataFrame
        product_df (pd.DataFrame): Product metadata DataFrame
        date_col (str): Name of the date column
        product_col (str): Name of the product ID column
        category_col (str): Name of the category column
        subcategory_col (str): Name of the subcategory column
        price_col (str): Name of the price column
        cost_col (str): Name of the cost column
        perishable_col (str): Name of the perishable indicator column
        
    Returns:
        pd.DataFrame: Sales DataFrame with product features
    """
    # Make a copy of the sales DataFrame
    sales_copy = sales_df.copy()
    
    # Convert dates to datetime if needed
    sales_copy[date_col] = pd.to_datetime(sales_copy[date_col])
    
    # Merge product data with sales data
    sales_copy = pd.merge(sales_copy, product_df, on=product_col, how='left')
    
    # Create price-related features
    if price_col in product_df.columns and cost_col in product_df.columns:
        # Margin
        sales_copy['margin'] = sales_copy[price_col] - sales_copy[cost_col]
        sales_copy['margin_ratio'] = sales_copy['margin'] / sales_copy[price_col]
        
        # Price bins
        sales_copy['price_bin'] = pd.cut(sales_copy[price_col], 
                                       bins=[-1, 5, 10, 25, 50, 100, 999999],
                                       labels=['budget', 'low', 'medium', 'high', 'premium', 'luxury'])
    
    # Create product type features
    if perishable_col in product_df.columns:
        # Already a flag, keep as is
        pass
    
    # One-hot encode categorical product features
    categorical_cols = [col for col in [category_col, subcategory_col] if col in product_df.columns]
    for col in categorical_cols:
        dummies = pd.get_dummies(sales_copy[col], prefix=col, dummy_na=False)
        sales_copy = pd.concat([sales_copy, dummies], axis=1)
    
    logger.info(f"Created product-related features from {len(product_df.columns) - 1} product attributes")
    
    return sales_copy


def create_economic_features(df, economic_df, date_col='date', 
                           consumer_index_col='consumer_confidence_index',
                           unemployment_col='unemployment_rate',
                           inflation_col='inflation_rate'):
    """
    Create economic indicator features for sales data.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        economic_df (pd.DataFrame): Economic indicators DataFrame
        date_col (str): Name of the date column
        consumer_index_col (str): Name of the consumer confidence index column
        unemployment_col (str): Name of the unemployment rate column
        inflation_col (str): Name of the inflation rate column
        
    Returns:
        pd.DataFrame: DataFrame with economic features
    """
    # Make a copy of the DataFrame
    df_copy = df.copy()
    
    # Convert dates to datetime if needed
    df_copy[date_col] = pd.to_datetime(df_copy[date_col])
    economic_df[date_col] = pd.to_datetime(economic_df[date_col])
    
    # Since economic data is typically monthly, resample if needed
    if economic_df[date_col].dt.day.nunique() == 1:
        # Data is likely monthly, so we'll fill in the missing dates
        date_range = pd.date_range(start=economic_df[date_col].min(),
                                  end=economic_df[date_col].max(),
                                  freq='D')
        
        # Create a template DataFrame
        template_df = pd.DataFrame({date_col: date_range})
        
        # Merge and forward fill
        economic_filled = pd.merge(template_df, economic_df, on=date_col, how='left')
        economic_filled = economic_filled.fillna(method='ffill')
        
        # Use the filled DataFrame
        economic_df = economic_filled
    
    # Merge economic data with input data
    df_copy = pd.merge(df_copy, economic_df, on=date_col, how='left')
    
    # Create lagged economic features
    economic_cols = [col for col in [consumer_index_col, unemployment_col, inflation_col] 
                    if col in economic_df.columns]
    
    for col in economic_cols:
        # Previous month
        df_copy[f'{col}_lag1m'] = df_copy[col].shift(30)
        
        # Change from previous month
        df_copy[f'{col}_mom_change'] = df_copy[col] - df_copy[f'{col}_lag1m']
        df_copy[f'{col}_mom_pct_change'] = df_copy[f'{col}_mom_change'] / df_copy[f'{col}_lag1m']
        
        # Year-over-year change
        df_copy[f'{col}_lag1y'] = df_copy[col].shift(365)
        df_copy[f'{col}_yoy_change'] = df_copy[col] - df_copy[f'{col}_lag1y']
        df_copy[f'{col}_yoy_pct_change'] = df_copy[f'{col}_yoy_change'] / df_copy[f'{col}_lag1y']
    
    # Create composite economic indicator
    if consumer_index_col in economic_df.columns and unemployment_col in economic_df.columns:
        # Simple economic health indicator (higher is better)
        # Standardize to avoid scale issues
        scaler = StandardScaler()
        consumer_scaled = scaler.fit_transform(df_copy[[consumer_index_col]])
        unemployment_scaled = scaler.fit_transform(df_copy[[unemployment_col]])
        
        # Higher consumer confidence is good, higher unemployment is bad
        df_copy['economic_health'] = consumer_scaled.flatten() - unemployment_scaled.flatten()
    
    logger.info(f"Created economic features from {len(economic_cols)} economic indicators")
    
    return df_copy


def create_sales_velocity_features(df, date_col='date', product_col='product_id', 
                                 store_col='store_id', sales_col='sales_quantity'):
    """
    Create sales velocity features for forecasting.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        date_col (str): Name of the date column
        product_col (str): Name of the product ID column
        store_col (str): Name of the store ID column
        sales_col (str): Name of the sales quantity column
        
    Returns:
        pd.DataFrame: DataFrame with sales velocity features
    """
    # Make a copy of the DataFrame
    df_copy = df.copy()
    
    # Convert dates to datetime if needed
    df_copy[date_col] = pd.to_datetime(df_copy[date_col])
    
    # Sort by date
    df_copy = df_copy.sort_values(by=[store_col, product_col, date_col])
    
    # Create sales velocity features (rate of change in sales)
    df_copy['sales_diff_1d'] = df_copy.groupby([store_col, product_col])[sales_col].diff(1)
    df_copy['sales_diff_7d'] = df_copy.groupby([store_col, product_col])[sales_col].diff(7)
    df_copy['sales_diff_30d'] = df_copy.groupby([store_col, product_col])[sales_col].diff(30)
    
    # Create rolling velocity (average rate of change)
    df_copy['sales_velocity_7d'] = df_copy.groupby([store_col, product_col])['sales_diff_1d'].transform(
        lambda x: x.rolling(window=7).mean())
    
    df_copy['sales_velocity_30d'] = df_copy.groupby([store_col, product_col])['sales_diff_1d'].transform(
        lambda x: x.rolling(window=30).mean())
    
    # Create acceleration features (change in velocity)
    df_copy['sales_accel_7d'] = df_copy.groupby([store_col, product_col])['sales_velocity_7d'].diff(1)
    
    # Create momentum features
    # Exponentially weighted moving average gives more weight to recent observations
    df_copy['sales_momentum_7d'] = df_copy.groupby([store_col, product_col])[sales_col].transform(
        lambda x: x.ewm(span=7).mean())
    
    df_copy['sales_momentum_30d'] = df_copy.groupby([store_col, product_col])[sales_col].transform(
        lambda x: x.ewm(span=30).mean())
    
    # Create sales trend features
    # Ratio of short-term to long-term momentum indicates trend direction
    df_copy['sales_trend'] = df_copy['sales_momentum_7d'] / df_copy['sales_momentum_30d']
    
    # Create normalized sales features
    # These help compare sales across different products/stores
    
    # Z-score normalization by product-store combination
    df_copy['sales_zscore'] = df_copy.groupby([store_col, product_col])[sales_col].transform(
        lambda x: (x - x.mean()) / x.std())
    
    # Min-max normalization by product-store combination
    df_copy['sales_normalized'] = df_copy.groupby([store_col, product_col])[sales_col].transform(
        lambda x: (x - x.min()) / (x.max() - x.min()))
    
    # Create percentile rank of sales
    df_copy['sales_percentile'] = df_copy.groupby([store_col, product_col])[sales_col].transform(
        lambda x: x.rank(pct=True))
    
    logger.info(f"Created 12 sales velocity and momentum features")
    
    return df_copy


def create_inventory_features(sales_df, inventory_df, date_col='date', store_col='store_id', 
                            product_col='product_id', sales_col='sales_quantity',
                            stock_col='current_stock', safety_col='safety_stock',
                            reorder_col='reorder_point'):
    """
    Create inventory-related features for sales data.
    
    Args:
        sales_df (pd.DataFrame): Sales DataFrame
        inventory_df (pd.DataFrame): Inventory DataFrame
        date_col (str): Name of the date column
        store_col (str): Name of the store ID column
        product_col (str): Name of the product ID column
        sales_col (str): Name of the sales quantity column
        stock_col (str): Name of the current stock column
        safety_col (str): Name of the safety stock column
        reorder_col (str): Name of the reorder point column
        
    Returns:
        pd.DataFrame: Sales DataFrame with inventory features
    """
    # Make a copy of the sales DataFrame
    sales_copy = sales_df.copy()
    
    # Convert dates to datetime if needed
    sales_copy[date_col] = pd.to_datetime(sales_copy[date_col])
    
    # Merge inventory data with sales data
    sales_copy = pd.merge(sales_copy, inventory_df, on=[store_col, product_col], how='left')
    
    # Create inventory level features
    if stock_col in inventory_df.columns:
        # Inventory coverage (days of supply)
        avg_daily_sales = sales_copy.groupby([store_col, product_col])[sales_col].transform('mean')
        sales_copy['days_of_supply'] = sales_copy[stock_col] / avg_daily_sales
        
        # Binned days of supply
        sales_copy['supply_bin'] = pd.cut(sales_copy['days_of_supply'], 
                                         bins=[-1, 3, 7, 14, 30, 60, 999999],
                                         labels=['critical', 'low', 'medium', 'adequate', 'high', 'excess'])
        
        # Stock to safety ratio
        if safety_col in inventory_df.columns:
            sales_copy['stock_to_safety_ratio'] = sales_copy[stock_col] / sales_copy[safety_col]
        
        # Stock to reorder ratio
        if reorder_col in inventory_df.columns:
            sales_copy['stock_to_reorder_ratio'] = sales_copy[stock_col] / sales_copy[reorder_col]
            
            # Flag if stock is below reorder point
            sales_copy['below_reorder_point'] = (sales_copy[stock_col] < sales_copy[reorder_col]).astype(int)
        
        # Stock to sales ratio
        weekly_sales = sales_copy.groupby([store_col, product_col])[sales_col].transform(
            lambda x: x.rolling(window=7).sum())
        sales_copy['stock_to_weekly_sales_ratio'] = sales_copy[stock_col] / weekly_sales
    
    logger.info(f"Created inventory-related features from inventory data")
    
    return sales_copy


def create_cross_product_features(df, date_col='date', store_col='store_id', 
                                product_col='product_id', category_col='category',
                                subcategory_col='subcategory', sales_col='sales_quantity'):
    """
    Create features that capture relationships between products.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        date_col (str): Name of the date column
        store_col (str): Name of the store ID column
        product_col (str): Name of the product ID column
        category_col (str): Name of the category column
        subcategory_col (str): Name of the subcategory column
        sales_col (str): Name of the sales quantity column
        
    Returns:
        pd.DataFrame: DataFrame with cross-product features
    """
    # Make a copy of the DataFrame
    df_copy = df.copy()
    
    # Convert dates to datetime if needed
    df_copy[date_col] = pd.to_datetime(df_copy[date_col])
    
    # Calculate daily sales by category and subcategory
    if category_col in df_copy.columns:
        # Daily category sales
        category_sales = df_copy.groupby([date_col, store_col, category_col])[sales_col].sum().reset_index()
        category_sales = category_sales.rename(columns={sales_col: 'category_sales'})
        
        # Merge back to main data
        df_copy = pd.merge(df_copy, category_sales, on=[date_col, store_col, category_col], how='left')
        
        # Calculate product's share of category sales
        df_copy['category_sales_share'] = df_copy[sales_col] / df_copy['category_sales']
        
        # Category sales excluding current product
        df_copy['category_sales_others'] = df_copy['category_sales'] - df_copy[sales_col]
    
    if subcategory_col in df_copy.columns:
        # Daily subcategory sales
        subcategory_sales = df_copy.groupby([date_col, store_col, subcategory_col])[sales_col].sum().reset_index()
        subcategory_sales = subcategory_sales.rename(columns={sales_col: 'subcategory_sales'})
        
        # Merge back to main data
        df_copy = pd.merge(df_copy, subcategory_sales, on=[date_col, store_col, subcategory_col], how='left')
        
        # Calculate product's share of subcategory sales
        df_copy['subcategory_sales_share'] = df_copy[sales_col] / df_copy['subcategory_sales']
        
        # Subcategory sales excluding current product
        df_copy['subcategory_sales_others'] = df_copy['subcategory_sales'] - df_copy[sales_col]
    
    # Total store sales for the day
    store_sales = df_copy.groupby([date_col, store_col])[sales_col].sum().reset_index()
    store_sales = store_sales.rename(columns={sales_col: 'store_sales'})
    
    # Merge back to main data
    df_copy = pd.merge(df_copy, store_sales, on=[date_col, store_col], how='left')
    
    # Calculate product's share of store sales
    df_copy['store_sales_share'] = df_copy[sales_col] / df_copy['store_sales']
    
    # Store sales excluding current product
    df_copy['store_sales_others'] = df_copy['store_sales'] - df_copy[sales_col]
    
    logger.info(f"Created cross-product relationship features")
    
    return df_copy


def standardize_features(df, numeric_cols=None, method='standard'):
    """
    Standardize numeric features.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        numeric_cols (list): List of numeric column names to standardize
        method (str): Standardization method ('standard' or 'minmax')
        
    Returns:
        pd.DataFrame: DataFrame with standardized features
    """
    # Make a copy of the DataFrame
    df_copy = df.copy()
    
    # If no columns are specified, use all numeric columns
    if numeric_cols is None:
        numeric_cols = df_copy.select_dtypes(include=['number']).columns.tolist()
    
    # Create a scaler based on the method
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Apply scaling to selected columns
    if numeric_cols:
        df_copy[numeric_cols] = scaler.fit_transform(df_copy[numeric_cols])
    
    logger.info(f"Standardized {len(numeric_cols)} numeric features using {method} scaling")
    
    return df_copy


def feature_selection(df, target_col, n_features=20, method='correlation'):
    """
    Select the most important features.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        target_col (str): Name of the target column
        n_features (int): Number of features to select
        method (str): Feature selection method ('correlation', 'mutual_info', or 'model_based')
        
    Returns:
        tuple: (Selected features DataFrame, list of selected feature names)
    """
    # Make a copy of the DataFrame
    df_copy = df.copy()
    
    # Remove non-numeric columns
    numeric_df = df_copy.select_dtypes(include=['number'])
    
    # Ensure target column is included
    if target_col not in numeric_df.columns:
        logger.warning(f"Target column {target_col} is not numeric or not in DataFrame")
        return df_copy, df_copy.columns.tolist()
    
    # Feature columns (excluding target)
    feature_cols = [col for col in numeric_df.columns if col != target_col]
    
    if method == 'correlation':
        # Calculate correlation with target
        correlations = numeric_df[feature_cols].corrwith(numeric_df[target_col]).abs().sort_values(ascending=False)
        
        # Select top features
        selected_features = correlations.head(n_features).index.tolist()
    
    elif method == 'mutual_info':
        from sklearn.feature_selection import mutual_info_regression
        
        # Calculate mutual information
        X = numeric_df[feature_cols]
        y = numeric_df[target_col]
        
        mutual_info = mutual_info_regression(X, y)
        mutual_info_df = pd.DataFrame({'feature': feature_cols, 'mutual_info': mutual_info})
        mutual_info_df = mutual_info_df.sort_values('mutual_info', ascending=False)
        
        # Select top features
        selected_features = mutual_info_df.head(n_features)['feature'].tolist()
    
    elif method == 'model_based':
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.feature_selection import SelectFromModel
        
        # Use random forest for feature importance
        X = numeric_df[feature_cols]
        y = numeric_df[target_col]
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        selector = SelectFromModel(model, threshold='median', max_features=n_features)
        selector.fit(X, y)
        
        # Get selected feature mask
        selected_mask = selector.get_support()
        selected_features = [feature for feature, selected in zip(feature_cols, selected_mask) if selected]
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Add target column to selected features
    selected_features.append(target_col)
    
    # Return DataFrame with only selected features
    selected_df = df_copy[selected_features]
    
    logger.info(f"Selected {len(selected_features)-1} features using {method} method")
    
    return selected_df, selected_features


def create_time_features(df, date_column='date'):
    """
    Create time-based features from a date column.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    date_column : str, optional
        Name of the date column
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added time features
    """
    # Create a copy to avoid modifying the original
    df_features = df.copy()
    
    # Ensure date column is datetime type
    if date_column in df_features.columns:
        df_features[date_column] = pd.to_datetime(df_features[date_column])
    elif isinstance(df_features.index, pd.DatetimeIndex):
        # If date is already the index, use it directly
        df_features = df_features.reset_index()
        df_features = df_features.rename(columns={'index': date_column})
        date_is_index = True
    else:
        raise ValueError(f"Date column '{date_column}' not found in DataFrame")
    
    # Extract basic date components
    df_features['day_of_week'] = df_features[date_column].dt.dayofweek
    df_features['day_of_month'] = df_features[date_column].dt.day
    df_features['day_of_year'] = df_features[date_column].dt.dayofyear
    df_features['month'] = df_features[date_column].dt.month
    df_features['quarter'] = df_features[date_column].dt.quarter
    df_features['year'] = df_features[date_column].dt.year
    df_features['week_of_year'] = df_features[date_column].dt.isocalendar().week
    
    # Create weekend indicator
    df_features['is_weekend'] = df_features['day_of_week'].isin([5, 6]).astype(int)
    
    # Create month-end and month-start indicators
    df_features['is_month_end'] = df_features[date_column].dt.is_month_end.astype(int)
    df_features['is_month_start'] = df_features[date_column].dt.is_month_start.astype(int)
    
    # Create quarter-end and quarter-start indicators
    df_features['is_quarter_end'] = df_features[date_column].dt.is_quarter_end.astype(int)
    df_features['is_quarter_start'] = df_features[date_column].dt.is_quarter_start.astype(int)
    
    # Create year-end and year-start indicators
    df_features['is_year_end'] = df_features[date_column].dt.is_year_end.astype(int)
    df_features['is_year_start'] = df_features[date_column].dt.is_year_start.astype(int)
    
    # Create cyclical features for day of week, month, and day of year
    df_features['day_of_week_sin'] = np.sin(2 * np.pi * df_features['day_of_week'] / 7)
    df_features['day_of_week_cos'] = np.cos(2 * np.pi * df_features['day_of_week'] / 7)
    
    df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
    df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
    
    df_features['day_of_year_sin'] = np.sin(2 * np.pi * df_features['day_of_year'] / 365)
    df_features['day_of_year_cos'] = np.cos(2 * np.pi * df_features['day_of_year'] / 365)
    
    # If date was originally the index, set it back as index
    if 'date_is_index' in locals() and date_is_index:
        df_features = df_features.set_index(date_column)
    
    return df_features


def create_lag_features(df, target_column, lag_periods=[1, 7, 14, 30], group_columns=None):
    """
    Create lag features for a target column.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    target_column : str
        Name of the target column
    lag_periods : list, optional
        List of lag periods to create
    group_columns : list, optional
        List of columns to group by before creating lags
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added lag features
    """
    # Create a copy to avoid modifying the original
    df_lags = df.copy()
    
    # Sort by date if date column exists
    if 'date' in df_lags.columns:
        df_lags = df_lags.sort_values('date')
    
    # Create lag features
    if group_columns is not None:
        # Create lags within groups
        for lag in lag_periods:
            lag_name = f'{target_column}_lag_{lag}'
            df_lags[lag_name] = df_lags.groupby(group_columns)[target_column].shift(lag)
    else:
        # Create lags globally
        for lag in lag_periods:
            lag_name = f'{target_column}_lag_{lag}'
            df_lags[lag_name] = df_lags[target_column].shift(lag)
    
    return df_lags


def create_rolling_features(df, target_column, window_sizes=[7, 14, 30], functions=['mean', 'std', 'min', 'max'], group_columns=None):
    """
    Create rolling window features for a target column.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    target_column : str
        Name of the target column
    window_sizes : list, optional
        List of window sizes to create
    functions : list, optional
        List of aggregation functions to apply
    group_columns : list, optional
        List of columns to group by before creating rolling features
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added rolling features
    """
    # Create a copy to avoid modifying the original
    df_rolling = df.copy()
    
    # Sort by date if date column exists
    if 'date' in df_rolling.columns:
        df_rolling = df_rolling.sort_values('date')
    
    # Create rolling features
    if group_columns is not None:
        # Create rolling features within groups
        for window in window_sizes:
            for func in functions:
                feature_name = f'{target_column}_rolling_{window}_{func}'
                
                if func == 'mean':
                    df_rolling[feature_name] = df_rolling.groupby(group_columns)[target_column].transform(
                        lambda x: x.rolling(window=window, min_periods=1).mean()
                    )
                elif func == 'std':
                    df_rolling[feature_name] = df_rolling.groupby(group_columns)[target_column].transform(
                        lambda x: x.rolling(window=window, min_periods=1).std()
                    )
                elif func == 'min':
                    df_rolling[feature_name] = df_rolling.groupby(group_columns)[target_column].transform(
                        lambda x: x.rolling(window=window, min_periods=1).min()
                    )
                elif func == 'max':
                    df_rolling[feature_name] = df_rolling.groupby(group_columns)[target_column].transform(
                        lambda x: x.rolling(window=window, min_periods=1).max()
                    )
                elif func == 'median':
                    df_rolling[feature_name] = df_rolling.groupby(group_columns)[target_column].transform(
                        lambda x: x.rolling(window=window, min_periods=1).median()
                    )
                elif func == 'sum':
                    df_rolling[feature_name] = df_rolling.groupby(group_columns)[target_column].transform(
                        lambda x: x.rolling(window=window, min_periods=1).sum()
                    )
    else:
        # Create rolling features globally
        for window in window_sizes:
            for func in functions:
                feature_name = f'{target_column}_rolling_{window}_{func}'
                
                if func == 'mean':
                    df_rolling[feature_name] = df_rolling[target_column].rolling(window=window, min_periods=1).mean()
                elif func == 'std':
                    df_rolling[feature_name] = df_rolling[target_column].rolling(window=window, min_periods=1).std()
                elif func == 'min':
                    df_rolling[feature_name] = df_rolling[target_column].rolling(window=window, min_periods=1).min()
                elif func == 'max':
                    df_rolling[feature_name] = df_rolling[target_column].rolling(window=window, min_periods=1).max()
                elif func == 'median':
                    df_rolling[feature_name] = df_rolling[target_column].rolling(window=window, min_periods=1).median()
                elif func == 'sum':
                    df_rolling[feature_name] = df_rolling[target_column].rolling(window=window, min_periods=1).sum()
    
    return df_rolling


def create_expanding_features(df, target_column, functions=['mean', 'std', 'min', 'max'], group_columns=None):
    """
    Create expanding window features for a target column.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    target_column : str
        Name of the target column
    functions : list, optional
        List of aggregation functions to apply
    group_columns : list, optional
        List of columns to group by before creating expanding features
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added expanding features
    """
    # Create a copy to avoid modifying the original
    df_expanding = df.copy()
    
    # Sort by date if date column exists
    if 'date' in df_expanding.columns:
        df_expanding = df_expanding.sort_values('date')
    
    # Create expanding features
    if group_columns is not None:
        # Create expanding features within groups
        for func in functions:
            feature_name = f'{target_column}_expanding_{func}'
            
            if func == 'mean':
                df_expanding[feature_name] = df_expanding.groupby(group_columns)[target_column].transform(
                    lambda x: x.expanding(min_periods=1).mean()
                )
            elif func == 'std':
                df_expanding[feature_name] = df_expanding.groupby(group_columns)[target_column].transform(
                    lambda x: x.expanding(min_periods=1).std()
                )
            elif func == 'min':
                df_expanding[feature_name] = df_expanding.groupby(group_columns)[target_column].transform(
                    lambda x: x.expanding(min_periods=1).min()
                )
            elif func == 'max':
                df_expanding[feature_name] = df_expanding.groupby(group_columns)[target_column].transform(
                    lambda x: x.expanding(min_periods=1).max()
                )
            elif func == 'median':
                df_expanding[feature_name] = df_expanding.groupby(group_columns)[target_column].transform(
                    lambda x: x.expanding(min_periods=1).median()
                )
            elif func == 'sum':
                df_expanding[feature_name] = df_expanding.groupby(group_columns)[target_column].transform(
                    lambda x: x.expanding(min_periods=1).sum()
                )
    else:
        # Create expanding features globally
        for func in functions:
            feature_name = f'{target_column}_expanding_{func}'
            
            if func == 'mean':
                df_expanding[feature_name] = df_expanding[target_column].expanding(min_periods=1).mean()
            elif func == 'std':
                df_expanding[feature_name] = df_expanding[target_column].expanding(min_periods=1).std()
            elif func == 'min':
                df_expanding[feature_name] = df_expanding[target_column].expanding(min_periods=1).min()
            elif func == 'max':
                df_expanding[feature_name] = df_expanding[target_column].expanding(min_periods=1).max()
            elif func == 'median':
                df_expanding[feature_name] = df_expanding[target_column].expanding(min_periods=1).median()
            elif func == 'sum':
                df_expanding[feature_name] = df_expanding[target_column].expanding(min_periods=1).sum()
    
    return df_expanding 