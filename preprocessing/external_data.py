#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
External data integration module for retail demand forecasting.
This module provides functions to import and integrate external data sources.
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import logging
import os
import json
import csv
import holidays
from io import StringIO
import warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_holiday_data(start_date, end_date, country='US', state=None):
    """
    Get holiday data for a specified date range and country.
    
    Args:
        start_date (str or datetime): Start date
        end_date (str or datetime): End date
        country (str): Country code (e.g., 'US', 'UK', 'CA')
        state (str): State or region code for regional holidays (optional)
        
    Returns:
        pd.DataFrame: DataFrame with holiday information
    """
    # Convert dates to datetime if they are strings
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    # Get the years covered by the date range
    years = range(start_date.year, end_date.year + 1)
    
    # Get holidays for the specified country and years
    country_holidays = holidays.country_holidays(country, subdiv=state, years=years)
    
    # Create a list of holiday data
    holiday_data = []
    
    for date, name in country_holidays.items():
        # Check if the holiday is within the specified date range
        if start_date <= pd.Timestamp(date) <= end_date:
            # Define major holidays (based on US holidays, customize as needed)
            major_holidays = [
                "New Year's Day", 
                "Independence Day", 
                "Thanksgiving", 
                "Christmas Day",
                "Memorial Day",
                "Labor Day"
            ]
            
            is_major = any(major in name for major in major_holidays)
            
            holiday_data.append({
                'date': date,
                'holiday_name': name,
                'is_major_holiday': is_major,
                'country': country,
                'state': state
            })
    
    # Create DataFrame from holiday data
    holidays_df = pd.DataFrame(holiday_data)
    
    if len(holidays_df) > 0:
        # Ensure date column is datetime type
        holidays_df['date'] = pd.to_datetime(holidays_df['date'])
        
        logger.info(f"Retrieved {len(holidays_df)} holidays for {country} from {start_date} to {end_date}")
    else:
        logger.warning(f"No holidays found for {country} from {start_date} to {end_date}")
        
        # Create empty DataFrame with correct columns
        holidays_df = pd.DataFrame(columns=['date', 'holiday_name', 'is_major_holiday', 'country', 'state'])
    
    return holidays_df


def get_weather_data(api_key, locations, start_date, end_date, frequency='daily'):
    """
    Get historical weather data from OpenWeatherMap or similar API.
    NOTE: This is a placeholder function. You'll need a valid API key and
    to implement the actual API call based on the provider's documentation.
    
    Args:
        api_key (str): API key for the weather service
        locations (list): List of location dictionaries with lat, lon, and name keys
        start_date (str or datetime): Start date
        end_date (str or datetime): End date
        frequency (str): Data frequency ('daily' or 'hourly')
        
    Returns:
        pd.DataFrame: DataFrame with weather information
    """
    # Convert dates to datetime if they are strings
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    # Create a list to store weather data
    weather_data = []
    
    # In a real implementation, you would loop through locations and dates
    # and make API calls to get the actual weather data
    
    # For this example, we'll generate synthetic weather data
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    for location in locations:
        location_name = location.get('name', 'Unknown')
        base_temp = 15 + np.random.normal(0, 3)  # Base temperature varies by location
        
        for date in date_range:
            # Seasonal temperature variation
            day_of_year = date.dayofyear
            seasonal_temp = base_temp + 15 * np.sin(2 * np.pi * day_of_year / 365)
            
            # Add random variation
            temp = seasonal_temp + np.random.normal(0, 3)
            
            # Precipitation (more likely in winter)
            winter_factor = 1 + np.cos(2 * np.pi * day_of_year / 365)
            precip_prob = 0.3 * winter_factor
            precipitation = np.random.exponential(5) if np.random.random() < precip_prob else 0
            
            # Is it sunny?
            is_sunny = precipitation == 0 and np.random.random() < 0.7
            
            weather_data.append({
                'date': date,
                'location': location_name,
                'temperature': round(temp, 1),
                'precipitation': round(precipitation, 1),
                'humidity': round(np.random.uniform(40, 90), 1),
                'wind_speed': round(np.random.gamma(2, 2), 1),
                'is_sunny': is_sunny
            })
    
    # Create DataFrame from weather data
    weather_df = pd.DataFrame(weather_data)
    
    logger.info(f"Retrieved weather data for {len(locations)} locations from {start_date} to {end_date}")
    
    return weather_df


def get_economic_data(indicators, start_date, end_date, frequency='monthly'):
    """
    Get economic indicator data (e.g., from FRED API or similar).
    NOTE: This is a placeholder function. You'll need to implement the actual API
    call based on the provider's documentation or use a library like pandas-datareader.
    
    Args:
        indicators (list): List of economic indicator codes
        start_date (str or datetime): Start date
        end_date (str or datetime): End date
        frequency (str): Data frequency ('monthly', 'quarterly', or 'annual')
        
    Returns:
        pd.DataFrame: DataFrame with economic indicators
    """
    # Convert dates to datetime if they are strings
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    # In a real implementation, you would use pandas-datareader or a similar
    # library to fetch actual economic data from FRED or another source
    
    # For this example, we'll generate synthetic economic data
    if frequency == 'monthly':
        date_range = pd.date_range(start=start_date.replace(day=1), 
                                  end=end_date.replace(day=1),
                                  freq='MS')  # Month start frequency
    elif frequency == 'quarterly':
        date_range = pd.date_range(start=start_date.replace(day=1), 
                                  end=end_date.replace(day=1),
                                  freq='QS')  # Quarter start frequency
    elif frequency == 'annual':
        date_range = pd.date_range(start=datetime(start_date.year, 1, 1), 
                                  end=datetime(end_date.year, 1, 1),
                                  freq='AS')  # Year start frequency
    else:
        raise ValueError(f"Unknown frequency: {frequency}")
    
    # Create a list to store economic data
    economic_data = []
    
    # Base values for economic indicators
    base_values = {
        'consumer_confidence_index': 100,
        'unemployment_rate': 5.0,
        'inflation_rate': 2.0,
        'gdp_growth': 2.5,
        'retail_sales_growth': 3.0,
        'interest_rate': 3.0
    }
    
    for date in date_range:
        # Calculate month number from start date
        month_num = (date.year - start_date.year) * 12 + date.month - start_date.month
        
        data_point = {'date': date}
        
        for indicator in indicators:
            if indicator in base_values:
                base = base_values[indicator]
                
                # Add trend, cyclical component, and random noise
                trend = 0.02 * month_num  # Slight upward trend
                cycle = 1.5 * np.sin(2 * np.pi * month_num / 48)  # 4-year cycle
                noise = np.random.normal(0, 0.3)
                
                value = base + trend + cycle + noise
                
                # Ensure non-negative values for rates
                if 'rate' in indicator:
                    value = max(0, value)
                
                data_point[indicator] = round(value, 1)
        
        economic_data.append(data_point)
    
    # Create DataFrame from economic data
    economic_df = pd.DataFrame(economic_data)
    
    logger.info(f"Retrieved {len(indicators)} economic indicators from {start_date} to {end_date}")
    
    return economic_df


def get_social_media_trends(keywords, start_date, end_date, frequency='daily'):
    """
    Get social media trend data for specified keywords.
    NOTE: This is a placeholder function. You'll need API access to platforms
    like Twitter, Reddit, or use a service like Google Trends.
    
    Args:
        keywords (list): List of keywords or hashtags to track
        start_date (str or datetime): Start date
        end_date (str or datetime): End date
        frequency (str): Data frequency ('daily', 'weekly', or 'monthly')
        
    Returns:
        pd.DataFrame: DataFrame with social media trend data
    """
    # Convert dates to datetime if they are strings
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    # In a real implementation, you would use an API to fetch actual social media data
    
    # For this example, we'll generate synthetic social media trend data
    if frequency == 'daily':
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    elif frequency == 'weekly':
        date_range = pd.date_range(start=start_date, end=end_date, freq='W')
    elif frequency == 'monthly':
        date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
    else:
        raise ValueError(f"Unknown frequency: {frequency}")
    
    # Create a list to store social media trend data
    trend_data = []
    
    for date in date_range:
        # Calculate day number from start date
        day_num = (date - start_date).days
        
        data_point = {'date': date}
        
        for keyword in keywords:
            # Base trend level
            base = 100
            
            # Add trend, cyclical component, and random noise
            trend = 0.1 * day_num  # Slight upward trend
            cycle = 20 * np.sin(2 * np.pi * day_num / 30)  # Monthly cycle
            noise = np.random.normal(0, 10)
            
            # Special events can cause spikes
            spike = 50 if np.random.random() < 0.01 else 0  # 1% chance of a spike
            
            value = base + trend + cycle + noise + spike
            
            # Ensure non-negative values
            value = max(0, value)
            
            data_point[f"{keyword}_trend"] = round(value, 1)
        
        trend_data.append(data_point)
    
    # Create DataFrame from trend data
    trend_df = pd.DataFrame(trend_data)
    
    logger.info(f"Retrieved social media trends for {len(keywords)} keywords from {start_date} to {end_date}")
    
    return trend_df


def get_competitor_data(competitors, start_date, end_date, metrics=['price', 'promotion']):
    """
    Get competitor data for analysis.
    NOTE: This is a placeholder function. In practice, this data might come from
    web scraping, market research firms, or internal competitive intelligence.
    
    Args:
        competitors (list): List of competitor names
        start_date (str or datetime): Start date
        end_date (str or datetime): End date
        metrics (list): List of metrics to track
        
    Returns:
        pd.DataFrame: DataFrame with competitor data
    """
    # Convert dates to datetime if they are strings
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    # Create a weekly date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='W')
    
    # Create a list to store competitor data
    competitor_data = []
    
    for date in date_range:
        for competitor in competitors:
            data_point = {
                'date': date,
                'competitor': competitor
            }
            
            # Add metrics
            if 'price' in metrics:
                # Base price with some randomness
                data_point['price_index'] = round(100 + np.random.normal(0, 5), 1)
            
            if 'promotion' in metrics:
                # Random promotion intensity (0-100)
                data_point['promotion_intensity'] = round(np.random.gamma(2, 10), 1)
            
            if 'product_count' in metrics:
                # Number of products
                data_point['product_count'] = int(1000 + np.random.normal(0, 50))
            
            if 'customer_satisfaction' in metrics:
                # Customer satisfaction score (0-100)
                data_point['customer_satisfaction'] = round(75 + np.random.normal(0, 5), 1)
            
            competitor_data.append(data_point)
    
    # Create DataFrame from competitor data
    competitor_df = pd.DataFrame(competitor_data)
    
    logger.info(f"Retrieved data for {len(competitors)} competitors from {start_date} to {end_date}")
    
    return competitor_df


def get_google_trends_data(keywords, start_date, end_date, geo='', gprop=''):
    """
    Get Google Trends data for specified keywords.
    NOTE: This function requires the pytrends library.
    
    Args:
        keywords (list): List of keywords to track (max 5)
        start_date (str): Start date in format 'yyyy-mm-dd'
        end_date (str): End date in format 'yyyy-mm-dd'
        geo (str): Geographic location (e.g., 'US', 'GB')
        gprop (str): Google property to filter results ('', 'images', 'news', 'youtube', or 'froogle')
        
    Returns:
        pd.DataFrame: DataFrame with Google Trends data
    """
    try:
        from pytrends.request import TrendReq
    except ImportError:
        logger.error("pytrends library not installed. Run: pip install pytrends")
        return pd.DataFrame()
    
    # Check that we have max 5 keywords (Google Trends limit)
    if len(keywords) > 5:
        logger.warning("Google Trends allows maximum 5 keywords. Using first 5.")
        keywords = keywords[:5]
    
    # Initialize pytrends
    pytrends = TrendReq(hl='en-US', tz=360)
    
    # Build payload
    timeframe = f"{start_date} {end_date}"
    pytrends.build_payload(keywords, cat=0, timeframe=timeframe, geo=geo, gprop=gprop)
    
    # Get interest over time
    trends_df = pytrends.interest_over_time()
    
    # Reset index to make date a column
    if not trends_df.empty:
        trends_df = trends_df.reset_index()
        trends_df = trends_df.rename(columns={'date': 'date'})
        
        # Drop isPartial column if it exists
        if 'isPartial' in trends_df.columns:
            trends_df = trends_df.drop('isPartial', axis=1)
        
        logger.info(f"Retrieved Google Trends data for {len(keywords)} keywords from {start_date} to {end_date}")
    else:
        logger.warning(f"No Google Trends data found for the specified parameters")
    
    return trends_df


def merge_external_data(df, external_df, on='date', how='left'):
    """
    Merge external data with the main dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Main DataFrame
    external_df : pandas.DataFrame or dict
        External data DataFrame or dictionary of DataFrames
    on : str, optional
        Column to merge on
    how : str, optional
        Type of merge to perform
        
    Returns:
    --------
    pandas.DataFrame
        Merged DataFrame
    """
    # Create a copy to avoid modifying the original
    df_merged = df.copy()
    
    # Handle dictionary of external data
    if isinstance(external_df, dict):
        for name, ext_df in external_df.items():
            if ext_df is None or len(ext_df) == 0:
                continue
                
            # Ensure join column is in the right format
            if on == 'date':
                if on in df_merged.columns:
                    df_merged[on] = pd.to_datetime(df_merged[on])
                if on in ext_df.columns:
                    ext_df = ext_df.copy()
                    ext_df[on] = pd.to_datetime(ext_df[on])
            
            # Merge dataset
            suffix = f'_{name}'
            df_merged = df_merged.merge(ext_df, on=on, how=how, suffixes=('', suffix))
    
    # Handle single DataFrame (original behavior)
    elif isinstance(external_df, pd.DataFrame):
        # Ensure join column is in the right format
        if on == 'date':
            if on in df_merged.columns:
                df_merged[on] = pd.to_datetime(df_merged[on])
            if on in external_df.columns:
                external_df = external_df.copy()
                external_df[on] = pd.to_datetime(external_df[on])
        
        # Merge datasets
        df_merged = df_merged.merge(external_df, on=on, how=how, suffixes=('', '_ext'))
    
    # Handle tuple of DataFrames
    elif isinstance(external_df, tuple):
        for i, ext_df in enumerate(external_df):
            if ext_df is None or len(ext_df) == 0:
                continue
                
            # Ensure join column is in the right format
            if on == 'date':
                if on in df_merged.columns:
                    df_merged[on] = pd.to_datetime(df_merged[on])
                if on in ext_df.columns:
                    ext_df = ext_df.copy()
                    ext_df[on] = pd.to_datetime(ext_df[on])
            
            # Merge dataset
            suffix = f'_ext_{i}'
            df_merged = df_merged.merge(ext_df, on=on, how=how, suffixes=('', suffix))
    
    else:
        raise TypeError("external_df must be a DataFrame, a dictionary of DataFrames, or a tuple of DataFrames")
    
    return df_merged


def create_holiday_features(df, date_column='date', country='US'):
    """
    Create holiday indicators for a DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    date_column : str, optional
        Name of the date column
    country : str, optional
        Country code for holidays
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with holiday features
    """
    # Create a copy to avoid modifying the original
    df_holidays = df.copy()
    
    # Ensure date column is datetime type
    if date_column in df_holidays.columns:
        df_holidays[date_column] = pd.to_datetime(df_holidays[date_column])
    else:
        raise ValueError(f"Date column '{date_column}' not found in DataFrame")
    
    # Get holiday information
    start_year = df_holidays[date_column].min().year
    end_year = df_holidays[date_column].max().year
    years = range(start_year, end_year + 1)
    
    # Get holidays for the specified country
    country_holidays = holidays.country_holidays(country, years=years)
    
    # Add holiday indicator
    df_holidays['is_holiday'] = df_holidays[date_column].apply(lambda x: x in country_holidays).astype(int)
    
    # Add holiday name
    df_holidays['holiday_name'] = df_holidays[date_column].apply(
        lambda x: country_holidays.get(x) if x in country_holidays else None
    )
    
    # Create indicators for major holidays
    major_holidays = [
        'New Year', 'Christmas', 'Thanksgiving', 'Independence', 'Labor Day',
        'Memorial Day', 'Easter', 'Halloween', 'Valentine'
    ]
    
    df_holidays['is_major_holiday'] = df_holidays['holiday_name'].apply(
        lambda x: 1 if x and any(holiday in str(x) for holiday in major_holidays) else 0
    )
    
    # Create days before/after holiday features
    df_holidays['days_before_holiday'] = 0
    df_holidays['days_after_holiday'] = 0
    
    # Get all holiday dates
    holiday_dates = df_holidays[df_holidays['is_holiday'] == 1][date_column].dt.date.tolist()
    
    # Calculate days before/after nearest holiday
    for i, row in df_holidays.iterrows():
        date = row[date_column].date()
        
        if row['is_holiday'] == 1:
            # Skip if it's a holiday
            continue
        
        # Calculate days to the nearest holiday
        days_before = [date - holiday for holiday in holiday_dates if date < holiday]
        days_after = [holiday - date for holiday in holiday_dates if date > holiday]
        
        if days_before:
            df_holidays.loc[i, 'days_before_holiday'] = min(days_before).days
        
        if days_after:
            df_holidays.loc[i, 'days_after_holiday'] = min(days_after).days
    
    return df_holidays


def add_weather_data(df, weather_df, date_column='date', location_column=None):
    """
    Add weather data to the main dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Main DataFrame
    weather_df : pandas.DataFrame
        Weather data DataFrame
    date_column : str, optional
        Name of the date column
    location_column : str, optional
        Name of the location column for matching weather data
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added weather data
    """
    # Create a copy to avoid modifying the original
    df_weather = df.copy()
    
    # Ensure date column is datetime type
    if date_column in df_weather.columns:
        df_weather[date_column] = pd.to_datetime(df_weather[date_column])
    
    if date_column in weather_df.columns:
        weather_df = weather_df.copy()
        weather_df[date_column] = pd.to_datetime(weather_df[date_column])
    
    # Merge datasets
    if location_column is not None and location_column in df_weather.columns and location_column in weather_df.columns:
        # Merge by date and location
        df_weather = df_weather.merge(
            weather_df, 
            on=[date_column, location_column], 
            how='left',
            suffixes=('', '_weather')
        )
    else:
        # Merge by date only
        df_weather = df_weather.merge(
            weather_df, 
            on=date_column, 
            how='left',
            suffixes=('', '_weather')
        )
    
    # Fill missing weather data
    weather_columns = [col for col in df_weather.columns if col.endswith('_weather')]
    for col in weather_columns:
        # Fill with forward fill then backward fill
        df_weather[col] = df_weather[col].fillna(method='ffill').fillna(method='bfill')
    
    return df_weather


def add_economic_indicators(df, economic_df, date_column='date'):
    """
    Add economic indicators to the main dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Main DataFrame
    economic_df : pandas.DataFrame
        Economic indicators DataFrame
    date_column : str, optional
        Name of the date column
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added economic indicators
    """
    # Create a copy to avoid modifying the original
    df_economic = df.copy()
    
    # Ensure date column is datetime type
    if date_column in df_economic.columns:
        df_economic[date_column] = pd.to_datetime(df_economic[date_column])
    
    if date_column in economic_df.columns:
        economic_df = economic_df.copy()
        economic_df[date_column] = pd.to_datetime(economic_df[date_column])
    
    # Economic data is often monthly or quarterly
    # Resample to daily if needed
    min_date = df_economic[date_column].min()
    max_date = df_economic[date_column].max()
    
    # Check the frequency of economic data
    if economic_df[date_column].nunique() < (max_date - min_date).days / 28:
        # Looks like monthly or less frequent data
        # Resample to daily with forward fill
        economic_df = economic_df.set_index(date_column)
        
        # Create a daily date range
        daily_idx = pd.date_range(min_date, max_date, freq='D')
        
        # Reindex and forward fill
        economic_df = economic_df.reindex(daily_idx, method='ffill')
        economic_df = economic_df.reset_index().rename(columns={'index': date_column})
    
    # Merge datasets
    df_economic = df_economic.merge(
        economic_df, 
        on=date_column, 
        how='left',
        suffixes=('', '_economic')
    )
    
    return df_economic


def add_events_data(df, events_df, date_column='date', event_date_column='event_date'):
    """
    Add events data to the main dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Main DataFrame
    events_df : pandas.DataFrame
        Events data DataFrame
    date_column : str, optional
        Name of the date column in main DataFrame
    event_date_column : str, optional
        Name of the date column in events DataFrame
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added events data
    """
    # Create a copy to avoid modifying the original
    df_events = df.copy()
    
    # Ensure date column is datetime type
    if date_column in df_events.columns:
        df_events[date_column] = pd.to_datetime(df_events[date_column])
    
    if event_date_column in events_df.columns:
        events_df = events_df.copy()
        events_df[event_date_column] = pd.to_datetime(events_df[event_date_column])
    
    # Create event indicators
    event_types = events_df['event_type'].unique() if 'event_type' in events_df.columns else ['event']
    
    for event_type in event_types:
        if 'event_type' in events_df.columns:
            event_dates = events_df[events_df['event_type'] == event_type][event_date_column].dt.date.tolist()
        else:
            event_dates = events_df[event_date_column].dt.date.tolist()
        
        col_name = f'is_{event_type.lower().replace(" ", "_")}'
        df_events[col_name] = df_events[date_column].dt.date.apply(lambda x: 1 if x in event_dates else 0)
    
    return df_events


def merge_all_external_data(sales_df, date_col='date', store_col='store_id', product_col='product_id'):
    """
    Merge all available external data sources with sales data.
    This function demonstrates how to integrate multiple external data sources.
    
    Args:
        sales_df (pd.DataFrame): Sales DataFrame
        date_col (str): Name of the date column
        store_col (str): Name of the store ID column
        product_col (str): Name of the product ID column
        
    Returns:
        pd.DataFrame: Sales DataFrame with external data
    """
    # Make a copy of the sales DataFrame
    sales_copy = sales_df.copy()
    
    # Ensure date column is datetime
    sales_copy[date_col] = pd.to_datetime(sales_copy[date_col])
    
    # Get date range from sales data
    start_date = sales_copy[date_col].min()
    end_date = sales_copy[date_col].max()
    
    # Get holiday data
    holidays_df = get_holiday_data(start_date, end_date, country='US')
    
    # Get weather data for a few locations
    locations = [
        {'name': 'New York', 'lat': 40.7128, 'lon': -74.0060},
        {'name': 'Los Angeles', 'lat': 34.0522, 'lon': -118.2437},
        {'name': 'Chicago', 'lat': 41.8781, 'lon': -87.6298}
    ]
    weather_df = get_weather_data('dummy_api_key', locations, start_date, end_date)
    
    # Get economic indicators
    indicators = ['consumer_confidence_index', 'unemployment_rate', 'inflation_rate', 'retail_sales_growth']
    economic_df = get_economic_data(indicators, start_date, end_date)
    
    # Get social media trends for relevant keywords
    keywords = ['retail', 'shopping', 'sale', 'discount']
    social_df = get_social_media_trends(keywords, start_date, end_date)
    
    # Get competitor data
    competitors = ['Competitor A', 'Competitor B', 'Competitor C']
    competitor_df = get_competitor_data(competitors, start_date, end_date)
    
    # Merge holiday data with sales data
    # We only need to know if a given date is a holiday
    holiday_features = holidays_df[['date', 'is_major_holiday']].copy()
    holiday_features['is_holiday'] = 1
    
    # Pivot to get one row per date
    holiday_features = holiday_features.drop_duplicates('date')
    
    sales_copy = pd.merge(sales_copy, holiday_features, on=date_col, how='left')
    sales_copy['is_holiday'] = sales_copy['is_holiday'].fillna(0).astype(int)
    sales_copy['is_major_holiday'] = sales_copy['is_major_holiday'].fillna(0).astype(int)
    
    # Merge weather data with sales data
    # For simplicity, we'll use the first location's weather for all stores
    # In a real application, you would map stores to the nearest weather station
    weather_features = weather_df[weather_df['location'] == locations[0]['name']].copy()
    weather_features = weather_features.drop('location', axis=1)
    
    sales_copy = pd.merge(sales_copy, weather_features, on=date_col, how='left')
    
    # Forward fill any missing weather data
    weather_cols = ['temperature', 'precipitation', 'humidity', 'wind_speed', 'is_sunny']
    sales_copy[weather_cols] = sales_copy[weather_cols].fillna(method='ffill')
    
    # Merge economic data with sales data
    # Since economic data is usually monthly, we'll forward fill to daily
    economic_features = economic_df.copy()
    
    # Create a template with all dates
    template_dates = pd.DataFrame({date_col: pd.date_range(start=start_date, end=end_date, freq='D')})
    
    # Merge economic data to the template and forward fill
    economic_daily = pd.merge(template_dates, economic_features, on=date_col, how='left')
    economic_daily = economic_daily.fillna(method='ffill')
    
    sales_copy = pd.merge(sales_copy, economic_daily, on=date_col, how='left')
    
    # Merge social media trends with sales data
    social_features = social_df.copy()
    
    # If data is not daily, resample to daily
    if len(social_features) < (end_date - start_date).days + 1:
        social_features = social_features.set_index('date')
        social_features = social_features.resample('D').interpolate()
        social_features = social_features.reset_index()
    
    sales_copy = pd.merge(sales_copy, social_features, on=date_col, how='left')
    
    # Merge competitor data with sales data
    # Average across competitors for each date
    competitor_features = competitor_df.groupby('date').mean().reset_index()
    
    sales_copy = pd.merge(sales_copy, competitor_features, on=date_col, how='left')
    
    # Forward fill any missing data
    sales_copy = sales_copy.fillna(method='ffill')
    
    logger.info(f"Merged all external data sources with sales data")
    
    return sales_copy


def clean_external_data(df, date_col='date'):
    """
    Clean and validate external data.
    
    Args:
        df (pd.DataFrame): DataFrame with external data
        date_col (str): Name of the date column
        
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    # Make a copy of the DataFrame
    df_copy = df.copy()
    
    # Ensure date column is datetime
    df_copy[date_col] = pd.to_datetime(df_copy[date_col])
    
    # Sort by date
    df_copy = df_copy.sort_values(by=date_col)
    
    # Check for duplicate dates
    duplicates = df_copy[date_col].duplicated()
    if duplicates.any():
        logger.warning(f"Found {duplicates.sum()} duplicate dates. Keeping first occurrence.")
        df_copy = df_copy.drop_duplicates(subset=[date_col], keep='first')
    
    # Check for missing dates
    date_range = pd.date_range(start=df_copy[date_col].min(), end=df_copy[date_col].max(), freq='D')
    missing_dates = set(date_range) - set(df_copy[date_col])
    
    if missing_dates:
        logger.warning(f"Found {len(missing_dates)} missing dates. Filling with NaN.")
        missing_df = pd.DataFrame({date_col: list(missing_dates)})
        df_copy = pd.concat([df_copy, missing_df], ignore_index=True)
        df_copy = df_copy.sort_values(by=date_col)
    
    # Handle missing values in numeric columns
    numeric_cols = df_copy.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        # Check for missing values
        missing = df_copy[col].isnull().sum()
        if missing > 0:
            logger.warning(f"Column {col} has {missing} missing values. Interpolating.")
            df_copy[col] = df_copy[col].interpolate(method='linear')
            # Fill any remaining NaNs at the edges
            df_copy[col] = df_copy[col].fillna(method='ffill').fillna(method='bfill')
    
    # Check for outliers in numeric columns
    for col in numeric_cols:
        # Calculate z-scores
        z_scores = np.abs((df_copy[col] - df_copy[col].mean()) / df_copy[col].std())
        outliers = z_scores > 3
        
        if outliers.any():
            logger.warning(f"Column {col} has {outliers.sum()} outliers. Consider handling them.")
    
    logger.info(f"Cleaned and validated external data")
    
    return df_copy 