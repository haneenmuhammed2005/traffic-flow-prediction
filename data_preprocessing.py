"""
Data Preprocessing Module for Traffic Flow Prediction
Handles data loading, cleaning, and initial transformations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')


class TrafficDataPreprocessor:
    """
    Comprehensive data preprocessing for traffic flow prediction
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_scaler = MinMaxScaler()
        
    def load_data(self, filepath, date_column='date_time', target_column='traffic_volume'):
        """
        Load traffic data from CSV file
        
        Args:
            filepath: Path to CSV file
            date_column: Name of datetime column
            target_column: Name of target variable column
            
        Returns:
            DataFrame with parsed datetime
        """
        df = pd.read_csv(filepath)
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.sort_values(date_column).reset_index(drop=True)
        
        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"Date range: {df[date_column].min()} to {df[date_column].max()}")
        
        return df
    
    def handle_missing_values(self, df, method='interpolate'):
        """
        Handle missing values in the dataset
        
        Args:
            df: Input DataFrame
            method: 'interpolate', 'forward_fill', 'backward_fill', or 'drop'
            
        Returns:
            DataFrame with missing values handled
        """
        print(f"\nMissing values before: {df.isnull().sum().sum()}")
        
        if method == 'interpolate':
            df = df.interpolate(method='time')
        elif method == 'forward_fill':
            df = df.fillna(method='ffill')
        elif method == 'backward_fill':
            df = df.fillna(method='bfill')
        elif method == 'drop':
            df = df.dropna()
            
        print(f"Missing values after: {df.isnull().sum().sum()}")
        return df
    
    def detect_outliers(self, df, column, method='iqr', threshold=1.5):
        """
        Detect outliers using IQR or Z-score method
        
        Args:
            df: Input DataFrame
            column: Column to check for outliers
            method: 'iqr' or 'zscore'
            threshold: Threshold for outlier detection
            
        Returns:
            Boolean mask of outliers
        """
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
        
        elif method == 'zscore':
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            outliers = z_scores > threshold
        
        print(f"Outliers detected: {outliers.sum()} ({outliers.sum()/len(df)*100:.2f}%)")
        return outliers
    
    def treat_outliers(self, df, column, method='cap', threshold=1.5):
        """
        Treat outliers in the data
        
        Args:
            df: Input DataFrame
            column: Column to treat
            method: 'cap', 'remove', or 'median'
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outliers treated
        """
        outliers = self.detect_outliers(df, column, threshold=threshold)
        
        if method == 'cap':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
        
        elif method == 'remove':
            df = df[~outliers]
        
        elif method == 'median':
            median = df[column].median()
            df.loc[outliers, column] = median
        
        return df
    
    def normalize_features(self, df, columns, method='standard'):
        """
        Normalize/standardize features
        
        Args:
            df: Input DataFrame
            columns: List of columns to normalize
            method: 'standard' or 'minmax'
            
        Returns:
            DataFrame with normalized features
        """
        if method == 'standard':
            df[columns] = self.scaler.fit_transform(df[columns])
        elif method == 'minmax':
            df[columns] = self.feature_scaler.fit_transform(df[columns])
        
        return df
    
    def create_time_based_split(self, df, train_ratio=0.7, val_ratio=0.15):
        """
        Create time-based train/val/test split (no shuffling)
        
        Args:
            df: Input DataFrame
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            
        Returns:
            train_df, val_df, test_df
        """
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_df = df[:train_end]
        val_df = df[train_end:val_end]
        test_df = df[val_end:]
        
        print(f"\nDataset split:")
        print(f"Train: {len(train_df)} samples ({len(train_df)/n*100:.1f}%)")
        print(f"Val: {len(val_df)} samples ({len(val_df)/n*100:.1f}%)")
        print(f"Test: {len(test_df)} samples ({len(test_df)/n*100:.1f}%)")
        
        return train_df, val_df, test_df
    
    def generate_synthetic_data(self, start_date='2023-01-01', periods=8760, freq='H'):
        """
        Generate synthetic traffic data for testing
        
        Args:
            start_date: Start date for data
            periods: Number of time periods
            freq: Frequency ('H' for hourly, 'D' for daily)
            
        Returns:
            DataFrame with synthetic traffic data
        """
        dates = pd.date_range(start=start_date, periods=periods, freq=freq)
        
        # Base traffic pattern
        base_traffic = 1000
        
        # Hour of day pattern (higher during rush hours)
        hour_pattern = np.array([0.3, 0.2, 0.15, 0.15, 0.2, 0.4, 0.8, 1.0, 
                                 0.9, 0.7, 0.6, 0.6, 0.7, 0.8, 0.9, 1.0,
                                 1.0, 0.9, 0.7, 0.6, 0.5, 0.4, 0.35, 0.3])
        
        # Day of week pattern (lower on weekends)
        day_pattern = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.7, 0.6])
        
        traffic_volume = []
        for date in dates:
            hour = date.hour
            day = date.dayofweek
            
            # Combine patterns
            traffic = base_traffic * hour_pattern[hour] * day_pattern[day]
            
            # Add random noise
            noise = np.random.normal(0, traffic * 0.15)
            traffic_volume.append(max(0, traffic + noise))
        
        df = pd.DataFrame({
            'date_time': dates,
            'traffic_volume': traffic_volume,
            'temperature': np.random.normal(20, 10, periods),
            'humidity': np.random.uniform(30, 90, periods),
            'weather_condition': np.random.choice(['Clear', 'Rain', 'Clouds', 'Snow'], periods)
        })
        
        print(f"Synthetic data generated: {len(df)} records")
        return df


# Example usage
if __name__ == "__main__":
    preprocessor = TrafficDataPreprocessor()
    
    # Generate synthetic data
    df = preprocessor.generate_synthetic_data(periods=8760)
    
    # Save to CSV
    df.to_csv('/home/claude/traffic_data_synthetic.csv', index=False)
    print("\nSynthetic data saved to traffic_data_synthetic.csv")
    
    # Basic preprocessing
    df = preprocessor.handle_missing_values(df)
    df = preprocessor.treat_outliers(df, 'traffic_volume', method='cap')
    
    # Split data
    train_df, val_df, test_df = preprocessor.create_time_based_split(df)
    
    print("\nPreprocessing complete!")
