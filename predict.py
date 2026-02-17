"""
Prediction Script for Traffic Flow Prediction
Use trained models to make predictions on new data
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from data_preprocessing import TrafficDataPreprocessor
from feature_engineering import TrafficFeatureEngineer
import warnings
warnings.filterwarnings('ignore')


class TrafficPredictor:
    """
    Make predictions using trained models
    """
    
    def __init__(self, model_path='models/traffic_model_xgboost.pkl'):
        """
        Initialize predictor with trained model
        
        Args:
            model_path: Path to saved model file
        """
        self.model = joblib.load(model_path)
        self.feature_engineer = TrafficFeatureEngineer()
        print(f"Model loaded from: {model_path}")
    
    def prepare_input_data(self, df):
        """
        Prepare input data with all necessary features
        
        Args:
            df: Input DataFrame with date_time and other columns
            
        Returns:
            DataFrame with engineered features
        """
        # Create all features
        df_featured = self.feature_engineer.create_all_features(df, target_column='traffic_volume')
        
        # Get feature columns (exclude target and metadata)
        exclude_cols = ['traffic_volume', 'date_time', 'weather_condition']
        feature_cols = [col for col in df_featured.columns if col not in exclude_cols]
        
        return df_featured[feature_cols], df_featured
    
    def predict_single_timestamp(self, timestamp, temperature=20, humidity=50, weather='Clear'):
        """
        Predict traffic for a single timestamp
        
        Args:
            timestamp: datetime object or string
            temperature: Temperature in Celsius
            humidity: Humidity percentage
            weather: Weather condition
            
        Returns:
            Predicted traffic volume
        """
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)
        
        # Create single row DataFrame
        df = pd.DataFrame({
            'date_time': [timestamp],
            'traffic_volume': [0],  # Placeholder
            'temperature': [temperature],
            'humidity': [humidity],
            'weather_condition': [weather]
        })
        
        # Prepare features
        X, _ = self.prepare_input_data(df)
        
        # Predict
        prediction = self.model.predict(X)[0]
        
        return max(0, prediction)  # Ensure non-negative
    
    def predict_date_range(self, start_date, end_date, freq='H'):
        """
        Predict traffic for a date range
        
        Args:
            start_date: Start datetime
            end_date: End datetime
            freq: Frequency ('H' for hourly, 'D' for daily)
            
        Returns:
            DataFrame with predictions
        """
        # Generate date range
        dates = pd.date_range(start=start_date, end=end_date, freq=freq)
        
        # Create DataFrame
        df = pd.DataFrame({
            'date_time': dates,
            'traffic_volume': [0] * len(dates),  # Placeholder
            'temperature': np.random.normal(20, 5, len(dates)),
            'humidity': np.random.uniform(40, 80, len(dates)),
            'weather_condition': np.random.choice(['Clear', 'Clouds', 'Rain'], len(dates))
        })
        
        # Prepare features
        X, df_full = self.prepare_input_data(df)
        
        # Predict
        predictions = self.model.predict(X)
        predictions = np.maximum(predictions, 0)  # Ensure non-negative
        
        # Add predictions to DataFrame
        result_df = pd.DataFrame({
            'date_time': df_full['date_time'],
            'predicted_traffic': predictions,
            'hour': df_full['hour'],
            'day_of_week': df_full['day_of_week'],
            'is_weekend': df_full['is_weekend']
        })
        
        return result_df
    
    def predict_next_n_hours(self, start_time, n_hours=24):
        """
        Predict traffic for next N hours
        
        Args:
            start_time: Starting datetime
            n_hours: Number of hours to predict
            
        Returns:
            DataFrame with predictions
        """
        end_time = start_time + timedelta(hours=n_hours)
        return self.predict_date_range(start_time, end_time, freq='H')
    
    def predict_rush_hours(self, date):
        """
        Predict traffic specifically for rush hours on a given date
        
        Args:
            date: Date to predict (datetime or string)
            
        Returns:
            DataFrame with rush hour predictions
        """
        if isinstance(date, str):
            date = pd.to_datetime(date)
        
        # Morning rush hour: 7-9 AM
        morning_start = date.replace(hour=7, minute=0, second=0)
        morning_end = date.replace(hour=9, minute=0, second=0)
        
        # Evening rush hour: 5-7 PM
        evening_start = date.replace(hour=17, minute=0, second=0)
        evening_end = date.replace(hour=19, minute=0, second=0)
        
        # Predict both periods
        morning_pred = self.predict_date_range(morning_start, morning_end, freq='H')
        evening_pred = self.predict_date_range(evening_start, evening_end, freq='H')
        
        # Combine
        rush_hour_pred = pd.concat([morning_pred, evening_pred]).reset_index(drop=True)
        rush_hour_pred['rush_period'] = ['Morning'] * len(morning_pred) + ['Evening'] * len(evening_pred)
        
        return rush_hour_pred


def demo_predictions():
    """
    Demonstrate various prediction capabilities
    """
    print("="*80)
    print("TRAFFIC FLOW PREDICTION - DEMO")
    print("="*80)
    
    # Load predictor
    predictor = TrafficPredictor('models/traffic_model_xgboost.pkl')
    
    # ========================================================================
    # Demo 1: Single timestamp prediction
    # ========================================================================
    print("\n" + "="*60)
    print("DEMO 1: Single Timestamp Prediction")
    print("="*60)
    
    timestamp = datetime(2024, 3, 15, 8, 0)  # March 15, 2024, 8:00 AM
    prediction = predictor.predict_single_timestamp(
        timestamp=timestamp,
        temperature=18,
        humidity=60,
        weather='Clear'
    )
    
    print(f"\nTimestamp: {timestamp.strftime('%Y-%m-%d %H:%M')}")
    print(f"Predicted Traffic Volume: {prediction:.0f} vehicles")
    
    # ========================================================================
    # Demo 2: Next 24 hours prediction
    # ========================================================================
    print("\n" + "="*60)
    print("DEMO 2: Next 24 Hours Prediction")
    print("="*60)
    
    start_time = datetime.now()
    next_24h = predictor.predict_next_n_hours(start_time, n_hours=24)
    
    print(f"\nPredictions from {start_time.strftime('%Y-%m-%d %H:%M')}:")
    print(next_24h.head(10))
    
    # Statistics
    print(f"\n24-Hour Statistics:")
    print(f"Average Traffic: {next_24h['predicted_traffic'].mean():.0f}")
    print(f"Peak Traffic: {next_24h['predicted_traffic'].max():.0f} at {next_24h.loc[next_24h['predicted_traffic'].idxmax(), 'date_time']}")
    print(f"Lowest Traffic: {next_24h['predicted_traffic'].min():.0f} at {next_24h.loc[next_24h['predicted_traffic'].idxmin(), 'date_time']}")
    
    # Save predictions
    next_24h.to_csv('results/predictions_next_24h.csv', index=False)
    print("\nPredictions saved to: results/predictions_next_24h.csv")
    
    # ========================================================================
    # Demo 3: Rush hour predictions
    # ========================================================================
    print("\n" + "="*60)
    print("DEMO 3: Rush Hour Predictions")
    print("="*60)
    
    target_date = datetime(2024, 3, 18)  # Monday
    rush_hours = predictor.predict_rush_hours(target_date)
    
    print(f"\nRush Hour Predictions for {target_date.strftime('%A, %B %d, %Y')}:")
    print(rush_hours)
    
    # Compare morning vs evening
    morning_avg = rush_hours[rush_hours['rush_period'] == 'Morning']['predicted_traffic'].mean()
    evening_avg = rush_hours[rush_hours['rush_period'] == 'Evening']['predicted_traffic'].mean()
    
    print(f"\nMorning Rush Average: {morning_avg:.0f}")
    print(f"Evening Rush Average: {evening_avg:.0f}")
    print(f"Busier Period: {'Evening' if evening_avg > morning_avg else 'Morning'}")
    
    # ========================================================================
    # Demo 4: Weekly prediction
    # ========================================================================
    print("\n" + "="*60)
    print("DEMO 4: Full Week Prediction")
    print("="*60)
    
    start_date = datetime(2024, 3, 18)  # Monday
    end_date = start_date + timedelta(days=7)
    
    weekly_pred = predictor.predict_date_range(start_date, end_date, freq='H')
    
    # Daily averages
    weekly_pred['date'] = pd.to_datetime(weekly_pred['date_time']).dt.date
    daily_avg = weekly_pred.groupby('date')['predicted_traffic'].mean().reset_index()
    daily_avg.columns = ['Date', 'Average Traffic']
    
    print("\nDaily Average Traffic:")
    print(daily_avg.to_string(index=False))
    
    # Save weekly predictions
    weekly_pred.to_csv('results/predictions_weekly.csv', index=False)
    print("\nWeekly predictions saved to: results/predictions_weekly.csv")
    
    print("\n" + "="*80)
    print("DEMO COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    demo_predictions()
