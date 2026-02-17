"""
Feature Engineering Module for Traffic Flow Prediction
Creates temporal, lagged, and statistical features
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class TrafficFeatureEngineer:
    """
    Create features for traffic flow prediction
    """
    
    def __init__(self, date_column='date_time'):
        self.date_column = date_column
        
    def extract_temporal_features(self, df):
        """
        Extract time-based features from datetime column
        
        Args:
            df: Input DataFrame with datetime column
            
        Returns:
            DataFrame with temporal features
        """
        df = df.copy()
        
        # Basic temporal features
        df['year'] = df[self.date_column].dt.year
        df['month'] = df[self.date_column].dt.month
        df['day'] = df[self.date_column].dt.day
        df['hour'] = df[self.date_column].dt.hour
        df['day_of_week'] = df[self.date_column].dt.dayofweek  # 0=Monday, 6=Sunday
        df['day_of_year'] = df[self.date_column].dt.dayofyear
        df['week_of_year'] = df[self.date_column].dt.isocalendar().week
        
        # Binary features
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_rush_hour_morning'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
        df['is_rush_hour_evening'] = ((df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        
        # Season (1=Winter, 2=Spring, 3=Summer, 4=Fall)
        df['season'] = df['month'].apply(lambda x: (x%12 + 3)//3)
        
        print(f"Temporal features created: {['year', 'month', 'day', 'hour', 'day_of_week', 'is_weekend', 'season', etc.]}")
        
        return df
    
    def create_cyclical_features(self, df):
        """
        Create cyclical encoding for temporal features
        Useful for capturing circular nature of time
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with cyclical features
        """
        df = df.copy()
        
        # Hour (24-hour cycle)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Day of week (7-day cycle)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Month (12-month cycle)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Day of year (365-day cycle)
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        print("Cyclical features created: hour_sin/cos, day_of_week_sin/cos, month_sin/cos, day_of_year_sin/cos")
        
        return df
    
    def create_lag_features(self, df, target_column='traffic_volume', lags=[1, 2, 3, 24, 48, 168]):
        """
        Create lagged features from target variable
        
        Args:
            df: Input DataFrame
            target_column: Column to create lags from
            lags: List of lag periods (in hours if hourly data)
            
        Returns:
            DataFrame with lag features
        """
        df = df.copy()
        
        for lag in lags:
            df[f'{target_column}_lag_{lag}'] = df[target_column].shift(lag)
        
        print(f"Lag features created: {len(lags)} lags {lags}")
        
        return df
    
    def create_rolling_features(self, df, target_column='traffic_volume', 
                                windows=[3, 6, 12, 24, 168]):
        """
        Create rolling window statistics
        
        Args:
            df: Input DataFrame
            target_column: Column to calculate rolling stats
            windows: List of window sizes
            
        Returns:
            DataFrame with rolling features
        """
        df = df.copy()
        
        for window in windows:
            # Rolling mean
            df[f'{target_column}_rolling_mean_{window}'] = df[target_column].rolling(
                window=window, min_periods=1).mean()
            
            # Rolling std
            df[f'{target_column}_rolling_std_{window}'] = df[target_column].rolling(
                window=window, min_periods=1).std()
            
            # Rolling min
            df[f'{target_column}_rolling_min_{window}'] = df[target_column].rolling(
                window=window, min_periods=1).min()
            
            # Rolling max
            df[f'{target_column}_rolling_max_{window}'] = df[target_column].rolling(
                window=window, min_periods=1).max()
        
        print(f"Rolling features created: {len(windows)} windows {windows}")
        
        return df
    
    def create_difference_features(self, df, target_column='traffic_volume', periods=[1, 24, 168]):
        """
        Create difference features (current - previous)
        
        Args:
            df: Input DataFrame
            target_column: Column to difference
            periods: List of periods to difference
            
        Returns:
            DataFrame with difference features
        """
        df = df.copy()
        
        for period in periods:
            df[f'{target_column}_diff_{period}'] = df[target_column].diff(period)
        
        print(f"Difference features created: {len(periods)} periods {periods}")
        
        return df
    
    def create_holiday_features(self, df):
        """
        Create holiday indicators
        This is a simplified version - expand based on your region
        
        Args:
            df: Input DataFrame with date_time column
            
        Returns:
            DataFrame with holiday features
        """
        df = df.copy()
        
        # Define major holidays (US format - adjust for your region)
        holidays = [
            (1, 1),   # New Year
            (7, 4),   # Independence Day
            (12, 25), # Christmas
            (11, 25), # Thanksgiving (approximate)
        ]
        
        df['is_holiday'] = df.apply(
            lambda row: 1 if (row[self.date_column].month, row[self.date_column].day) in holidays else 0,
            axis=1
        )
        
        # Day before/after holiday
        df['day_before_holiday'] = df['is_holiday'].shift(-1, fill_value=0)
        df['day_after_holiday'] = df['is_holiday'].shift(1, fill_value=0)
        
        print("Holiday features created: is_holiday, day_before_holiday, day_after_holiday")
        
        return df
    
    def create_weather_features(self, df):
        """
        Encode weather-related features if available
        
        Args:
            df: Input DataFrame with weather columns
            
        Returns:
            DataFrame with encoded weather features
        """
        df = df.copy()
        
        # Check if weather columns exist
        if 'weather_condition' in df.columns:
            # One-hot encode weather conditions
            weather_dummies = pd.get_dummies(df['weather_condition'], prefix='weather')
            df = pd.concat([df, weather_dummies], axis=1)
            print(f"Weather features created: {list(weather_dummies.columns)}")
        
        # Temperature bins
        if 'temperature' in df.columns:
            df['temp_cold'] = (df['temperature'] < 10).astype(int)
            df['temp_mild'] = ((df['temperature'] >= 10) & (df['temperature'] < 25)).astype(int)
            df['temp_hot'] = (df['temperature'] >= 25).astype(int)
            print("Temperature bins created: temp_cold, temp_mild, temp_hot")
        
        return df
    
    def create_interaction_features(self, df):
        """
        Create interaction features between important variables
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with interaction features
        """
        df = df.copy()
        
        # Hour × Weekend interaction
        if 'hour' in df.columns and 'is_weekend' in df.columns:
            df['hour_weekend_interaction'] = df['hour'] * df['is_weekend']
        
        # Rush hour × Weekday interaction
        if 'is_rush_hour_morning' in df.columns and 'is_weekend' in df.columns:
            df['rush_weekday'] = df['is_rush_hour_morning'] * (1 - df['is_weekend'])
        
        print("Interaction features created")
        
        return df
    
    def create_all_features(self, df, target_column='traffic_volume'):
        """
        Create all features in one go
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
            
        Returns:
            DataFrame with all features
        """
        print("\n=== Starting Feature Engineering ===\n")
        
        # Temporal features
        df = self.extract_temporal_features(df)
        df = self.create_cyclical_features(df)
        
        # Lag features
        df = self.create_lag_features(df, target_column)
        
        # Rolling features
        df = self.create_rolling_features(df, target_column)
        
        # Difference features
        df = self.create_difference_features(df, target_column)
        
        # Holiday features
        df = self.create_holiday_features(df)
        
        # Weather features
        df = self.create_weather_features(df)
        
        # Interaction features
        df = self.create_interaction_features(df)
        
        # Drop rows with NaN created by lag/rolling features
        initial_rows = len(df)
        df = df.dropna()
        print(f"\nDropped {initial_rows - len(df)} rows with NaN values from feature creation")
        
        print(f"\n=== Feature Engineering Complete ===")
        print(f"Total features: {df.shape[1]}")
        print(f"Total samples: {df.shape[0]}")
        
        return df


# Example usage
if __name__ == "__main__":
    # Load synthetic data
    df = pd.read_csv('/home/claude/traffic_data_synthetic.csv')
    df['date_time'] = pd.to_datetime(df['date_time'])
    
    # Create feature engineer
    feature_engineer = TrafficFeatureEngineer(date_column='date_time')
    
    # Create all features
    df_featured = feature_engineer.create_all_features(df, target_column='traffic_volume')
    
    # Save featured data
    df_featured.to_csv('/home/claude/traffic_data_featured.csv', index=False)
    print("\nFeatured data saved to traffic_data_featured.csv")
    
    # Display feature names
    print("\nAll features:")
    print(df_featured.columns.tolist())
