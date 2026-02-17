"""
Main Training Script for Traffic Flow Prediction
Complete end-to-end pipeline
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from data_preprocessing import TrafficDataPreprocessor
from feature_engineering import TrafficFeatureEngineer
from models import TrafficFlowModels, EnsembleModel, evaluate_model
from visualization import TrafficVisualizer


def main():
    """
    Main training pipeline
    """
    print("="*80)
    print("TRAFFIC FLOW PREDICTION - COMPLETE PIPELINE")
    print("="*80)
    
    # Create necessary directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # ========================================================================
    # STEP 1: DATA GENERATION/LOADING
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 1: DATA LOADING")
    print("="*80)
    
    preprocessor = TrafficDataPreprocessor()
    
    # Generate synthetic data (replace with real data loading)
    print("\nGenerating synthetic traffic data...")
    df = preprocessor.generate_synthetic_data(
        start_date='2023-01-01',
        periods=8760,  # 1 year of hourly data
        freq='H'
    )
    
    # Save raw data
    df.to_csv('data/raw/traffic_data_raw.csv', index=False)
    print("Raw data saved to: data/raw/traffic_data_raw.csv")
    
    # ========================================================================
    # STEP 2: DATA PREPROCESSING
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 2: DATA PREPROCESSING")
    print("="*80)
    
    # Handle missing values
    df = preprocessor.handle_missing_values(df, method='interpolate')
    
    # Treat outliers
    df = preprocessor.treat_outliers(df, 'traffic_volume', method='cap', threshold=1.5)
    
    print("\nPreprocessing complete!")
    
    # ========================================================================
    # STEP 3: EXPLORATORY DATA ANALYSIS
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 3: EXPLORATORY DATA ANALYSIS")
    print("="*80)
    
    viz = TrafficVisualizer(save_dir='plots/')
    
    print("\nCreating visualizations...")
    viz.plot_time_series(df, save_name='01_time_series.png')
    viz.plot_hourly_pattern(df, save_name='02_hourly_pattern.png')
    viz.plot_weekly_pattern(df, save_name='03_weekly_pattern.png')
    viz.plot_heatmap(df, save_name='04_traffic_heatmap.png')
    
    print("\nEDA visualizations saved to: plots/")
    
    # ========================================================================
    # STEP 4: FEATURE ENGINEERING
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 4: FEATURE ENGINEERING")
    print("="*80)
    
    feature_engineer = TrafficFeatureEngineer(date_column='date_time')
    df_featured = feature_engineer.create_all_features(df, target_column='traffic_volume')
    
    # Save featured data
    df_featured.to_csv('data/processed/traffic_data_featured.csv', index=False)
    print("\nFeatured data saved to: data/processed/traffic_data_featured.csv")
    
    # ========================================================================
    # STEP 5: PREPARE DATA FOR MODELING
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 5: PREPARING DATA FOR MODELING")
    print("="*80)
    
    # Define target and features
    target_col = 'traffic_volume'
    exclude_cols = [target_col, 'date_time', 'weather_condition']
    feature_cols = [col for col in df_featured.columns if col not in exclude_cols]
    
    print(f"\nTotal features: {len(feature_cols)}")
    print(f"Target variable: {target_col}")
    
    X = df_featured[feature_cols]
    y = df_featured[target_col]
    dates = df_featured['date_time']
    
    # Time-based split (70% train, 15% val, 15% test)
    train_size = int(0.7 * len(df_featured))
    val_size = int(0.15 * len(df_featured))
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    dates_train = dates[:train_size]
    
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    dates_val = dates[train_size:train_size+val_size]
    
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]
    dates_test = dates[train_size+val_size:]
    
    print(f"\nData split:")
    print(f"Train: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Val:   {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"Test:  {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    # ========================================================================
    # STEP 6: MODEL TRAINING
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 6: MODEL TRAINING")
    print("="*80)
    
    # Dictionary to store results
    all_results = {}
    
    # Train multiple models
    model_types = ['random_forest', 'xgboost', 'lightgbm']
    
    for model_type in model_types:
        print(f"\n{'='*60}")
        print(f"Training {model_type.upper()}")
        print(f"{'='*60}")
        
        model = TrafficFlowModels(model_type)
        
        # Model-specific hyperparameters
        if model_type == 'random_forest':
            model.create_model(n_estimators=100, max_depth=15, min_samples_split=10)
        elif model_type == 'xgboost':
            model.create_model(n_estimators=200, max_depth=6, learning_rate=0.1)
        elif model_type == 'lightgbm':
            model.create_model(n_estimators=200, max_depth=6, learning_rate=0.1)
        
        # Train
        model.train(X_train, y_train, X_val, y_val)
        
        # Predict on test set
        y_pred = model.predict(X_test)
        
        # Evaluate
        metrics = evaluate_model(y_test, y_pred, model_name=model_type.upper())
        all_results[model_type] = metrics
        
        # Save model
        model.save_model(f'models/traffic_model_{model_type}.pkl')
        
        # Plot predictions
        viz.plot_predictions(y_test.values, y_pred, dates_test.values,
                           title=f'{model_type.upper()} - Actual vs Predicted',
                           save_name=f'05_predictions_{model_type}.png')
        
        # Plot residuals
        viz.plot_residuals(y_test.values, y_pred,
                         save_name=f'06_residuals_{model_type}.png')
        
        # Plot feature importance
        if model.feature_importance is not None:
            viz.plot_feature_importance(model.feature_importance,
                                      save_name=f'07_feature_importance_{model_type}.png')
    
    # ========================================================================
    # STEP 7: ENSEMBLE MODEL
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 7: ENSEMBLE MODEL")
    print("="*80)
    
    ensemble = EnsembleModel(model_types=['xgboost', 'lightgbm', 'random_forest'])
    ensemble.train_all(X_train, y_train, X_val, y_val)
    
    # Predict with ensemble
    y_pred_ensemble = ensemble.predict(X_test, method='weighted_average')
    
    # Evaluate ensemble
    metrics_ensemble = evaluate_model(y_test, y_pred_ensemble, model_name="ENSEMBLE")
    all_results['ensemble'] = metrics_ensemble
    
    # Plot ensemble predictions
    viz.plot_predictions(y_test.values, y_pred_ensemble, dates_test.values,
                       title='ENSEMBLE - Actual vs Predicted',
                       save_name='08_predictions_ensemble.png')
    
    # ========================================================================
    # STEP 8: MODEL COMPARISON
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 8: MODEL COMPARISON")
    print("="*80)
    
    # Create comparison DataFrame
    results_df = pd.DataFrame(all_results).T
    results_df.insert(0, 'Model', results_df.index)
    results_df = results_df.reset_index(drop=True)
    
    print("\n" + "="*60)
    print("FINAL MODEL COMPARISON")
    print("="*60)
    print(results_df.to_string(index=False))
    
    # Save results
    results_df.to_csv('results/model_comparison.csv', index=False)
    print("\nResults saved to: results/model_comparison.csv")
    
    # Plot comparison
    viz.plot_model_comparison(results_df, save_name='09_model_comparison.png')
    
    # ========================================================================
    # STEP 9: FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    
    # Find best model
    best_model = results_df.loc[results_df['RMSE'].idxmin(), 'Model']
    best_rmse = results_df.loc[results_df['RMSE'].idxmin(), 'RMSE']
    best_r2 = results_df.loc[results_df['RMSE'].idxmin(), 'R2']
    
    print(f"\n🏆 BEST MODEL: {best_model.upper()}")
    print(f"   RMSE: {best_rmse:.2f}")
    print(f"   R²:   {best_r2:.4f}")
    
    print("\n📁 Output Locations:")
    print("   - Raw data:        data/raw/traffic_data_raw.csv")
    print("   - Featured data:   data/processed/traffic_data_featured.csv")
    print("   - Models:          models/")
    print("   - Plots:           plots/")
    print("   - Results:         results/model_comparison.csv")
    
    print("\n" + "="*80)
    print("Next Steps:")
    print("1. Review visualizations in plots/ directory")
    print("2. Check model comparison in results/model_comparison.csv")
    print("3. Use best model for predictions with predict.py")
    print("4. Fine-tune hyperparameters for better performance")
    print("="*80)


if __name__ == "__main__":
    start_time = datetime.now()
    print(f"\nStarted at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    main()
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"\nCompleted at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time: {duration/60:.2f} minutes")
