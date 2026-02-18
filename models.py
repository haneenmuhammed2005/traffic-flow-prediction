"""
Machine Learning Models for Traffic Flow Prediction
Includes classical ML and deep learning approaches
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')


class TrafficFlowModels:
    """
    Collection of ML models for traffic prediction
    """
    
    def __init__(self, model_type='xgboost'):
        """
        Initialize model
        
        Args:
            model_type: 'linear', 'random_forest', 'xgboost', 'lightgbm', 'ensemble'
        """
        self.model_type = model_type
        self.model = None
        self.feature_importance = None
        
    def create_model(self, **kwargs):
        """
        Create model based on type
        
        Args:
            **kwargs: Model-specific parameters
            
        Returns:
            Initialized model
        """
        if self.model_type == 'linear':
            self.model = LinearRegression(**kwargs)
            
        elif self.model_type == 'ridge':
            self.model = Ridge(alpha=kwargs.get('alpha', 1.0))
            
        elif self.model_type == 'lasso':
            self.model = Lasso(alpha=kwargs.get('alpha', 1.0))
            
        elif self.model_type == 'decision_tree':
            self.model = DecisionTreeRegressor(
                max_depth=kwargs.get('max_depth', 10),
                min_samples_split=kwargs.get('min_samples_split', 20),
                random_state=42
            )
            
        elif self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 15),
                min_samples_split=kwargs.get('min_samples_split', 10),
                n_jobs=-1,
                random_state=42
            )
            
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                learning_rate=kwargs.get('learning_rate', 0.1),
                max_depth=kwargs.get('max_depth', 5),
                random_state=42
            )
            
        elif self.model_type == 'xgboost':
            self.model = xgb.XGBRegressor(
                n_estimators=kwargs.get('n_estimators', 200),
                learning_rate=kwargs.get('learning_rate', 0.1),
                max_depth=kwargs.get('max_depth', 6),
                subsample=kwargs.get('subsample', 0.8),
                colsample_bytree=kwargs.get('colsample_bytree', 0.8),
                random_state=42,
                n_jobs=-1
            )
            
        elif self.model_type == 'lightgbm':
            self.model = lgb.LGBMRegressor(
                n_estimators=kwargs.get('n_estimators', 200),
                learning_rate=kwargs.get('learning_rate', 0.1),
                max_depth=kwargs.get('max_depth', 6),
                num_leaves=kwargs.get('num_leaves', 31),
                subsample=kwargs.get('subsample', 0.8),
                colsample_bytree=kwargs.get('colsample_bytree', 0.8),
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        print(f"{self.model_type.upper()} model created")
        return self.model
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            
        Returns:
            Trained model
        """
        if self.model is None:
            self.create_model()
        
        print(f"\nTraining {self.model_type} model...")
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Number of features: {X_train.shape[1]}")
        
        # Train model with validation if using tree-based models
        if self.model_type in ['xgboost', 'lightgbm'] and X_val is not None:
            if self.model_type == 'xgboost':
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
            else:  # lightgbm
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.log_evaluation(0)]
                )
        else:
            self.model.fit(X_train, y_train)
        
        print("Training complete!")
        
        # Extract feature importance
        self.extract_feature_importance(X_train)
        
        return self.model
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        predictions = self.model.predict(X)
        return predictions
    
    def extract_feature_importance(self, X_train):
        """
        Extract feature importance from trained model
        
        Args:
            X_train: Training features (to get feature names)
        """
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X_train.columns if hasattr(X_train, 'columns') else [f'feature_{i}' for i in range(X_train.shape[1])],
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 Most Important Features:")
            print(self.feature_importance.head(10))
    
    def save_model(self, filepath):
        """
        Save trained model to disk
        
        Args:
            filepath: Path to save model
        """
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load model from disk
        
        Args:
            filepath: Path to load model from
        """
        self.model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return self.model


class EnsembleModel:
    """
    Ensemble of multiple models for improved predictions
    """
    
    def __init__(self, model_types=['xgboost', 'lightgbm', 'random_forest']):
        """
        Initialize ensemble
        
        Args:
            model_types: List of model types to include
        """
        self.models = {}
        self.weights = None
        
        for model_type in model_types:
            self.models[model_type] = TrafficFlowModels(model_type)
    
    def train_all(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train all models in ensemble
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
        """
        print("\n=== Training Ensemble Models ===\n")
        
        val_scores = {}
        
        for name, model in self.models.items():
            print(f"\n--- Training {name} ---")
            model.create_model()
            model.train(X_train, y_train, X_val, y_val)
            
            # Evaluate on validation set
            if X_val is not None:
                val_pred = model.predict(X_val)
                val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
                val_scores[name] = val_rmse
                print(f"Validation RMSE: {val_rmse:.2f}")
        
        # Calculate weights based on validation performance (inverse of error)
        if val_scores:
            total_inv_error = sum(1/score for score in val_scores.values())
            self.weights = {name: (1/score)/total_inv_error for name, score in val_scores.items()}
            print("\n=== Ensemble Weights ===")
            for name, weight in self.weights.items():
                print(f"{name}: {weight:.3f}")
    
    def predict(self, X, method='weighted_average'):
        """
        Make ensemble predictions
        
        Args:
            X: Feature matrix
            method: 'average', 'weighted_average', or 'median'
            
        Returns:
            Ensemble predictions
        """
        predictions = []
        
        for name, model in self.models.items():
            pred = model.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        if method == 'average':
            return np.mean(predictions, axis=0)
        
        elif method == 'weighted_average' and self.weights is not None:
            weights = np.array([self.weights[name] for name in self.models.keys()])
            return np.average(predictions, axis=0, weights=weights)
        
        elif method == 'median':
            return np.median(predictions, axis=0)
        
        else:
            return np.mean(predictions, axis=0)


def evaluate_model(y_true, y_pred, model_name="Model"):
    """
    Evaluate model performance with multiple metrics
    
    Args:
        y_true: True values
        y_pred: Predicted values
        model_name: Name of model for display
        
    Returns:
        Dictionary of metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n=== {model_name} Performance ===")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"R²:   {r2:.4f}")
    
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2
    }
    
    return metrics


# Example usage
if __name__ == "__main__":
    # Load featured data
    df = pd.read_csv('data/processed/traffic_data_featured.csv')
    
    # Prepare features and target
    target_col = 'traffic_volume'
    feature_cols = [col for col in df.columns if col not in [target_col, 'date_time', 'weather_condition']]
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Split data (70-15-15)
    train_size = int(0.7 * len(df))
    val_size = int(0.15 * len(df))
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Train single model
    model = TrafficFlowModels('xgboost')
    model.create_model(n_estimators=100, max_depth=6)
    model.train(X_train, y_train, X_val, y_val)
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred, "XGBoost")
    
    # Save model
    model.save_model('models/traffic_model_xgboost.pkl')
