"""
Visualization Module for Traffic Flow Prediction
Creates comprehensive plots and visualizations
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 8)


class TrafficVisualizer:
    """
    Visualization tools for traffic flow analysis
    """
    
    def __init__(self, save_dir='/home/claude/plots/'):
        """
        Initialize visualizer
        
        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = save_dir
        
    def plot_time_series(self, df, date_column='date_time', value_column='traffic_volume', 
                        title='Traffic Volume Over Time', save_name='time_series.png'):
        """
        Plot time series of traffic volume
        
        Args:
            df: DataFrame with time series data
            date_column: Name of datetime column
            value_column: Name of value column
            title: Plot title
            save_name: Filename to save plot
        """
        fig, ax = plt.subplots(figsize=(15, 6))
        
        ax.plot(df[date_column], df[value_column], linewidth=0.5, alpha=0.7)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Traffic Volume', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir + save_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved: {save_name}")
    
    def plot_hourly_pattern(self, df, date_column='date_time', value_column='traffic_volume',
                           save_name='hourly_pattern.png'):
        """
        Plot average traffic by hour of day
        
        Args:
            df: DataFrame with time series data
            date_column: Name of datetime column
            value_column: Name of value column
            save_name: Filename to save plot
        """
        df = df.copy()
        df['hour'] = df[date_column].dt.hour
        
        hourly_avg = df.groupby('hour')[value_column].agg(['mean', 'std']).reset_index()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(hourly_avg['hour'], hourly_avg['mean'], marker='o', linewidth=2, 
                markersize=8, label='Average')
        ax.fill_between(hourly_avg['hour'], 
                        hourly_avg['mean'] - hourly_avg['std'],
                        hourly_avg['mean'] + hourly_avg['std'],
                        alpha=0.3, label='±1 Std Dev')
        
        ax.set_xlabel('Hour of Day', fontsize=12)
        ax.set_ylabel('Traffic Volume', fontsize=12)
        ax.set_title('Average Traffic Volume by Hour', fontsize=14, fontweight='bold')
        ax.set_xticks(range(0, 24))
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir + save_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved: {save_name}")
    
    def plot_weekly_pattern(self, df, date_column='date_time', value_column='traffic_volume',
                           save_name='weekly_pattern.png'):
        """
        Plot average traffic by day of week
        
        Args:
            df: DataFrame with time series data
            date_column: Name of datetime column
            value_column: Name of value column
            save_name: Filename to save plot
        """
        df = df.copy()
        df['day_of_week'] = df[date_column].dt.day_name()
        
        # Order days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        df['day_of_week'] = pd.Categorical(df['day_of_week'], categories=day_order, ordered=True)
        
        daily_avg = df.groupby('day_of_week')[value_column].agg(['mean', 'std']).reset_index()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.bar(daily_avg['day_of_week'], daily_avg['mean'], 
               yerr=daily_avg['std'], capsize=5, alpha=0.7, color='steelblue')
        
        ax.set_xlabel('Day of Week', fontsize=12)
        ax.set_ylabel('Traffic Volume', fontsize=12)
        ax.set_title('Average Traffic Volume by Day of Week', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.save_dir + save_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved: {save_name}")
    
    def plot_heatmap(self, df, date_column='date_time', value_column='traffic_volume',
                    save_name='traffic_heatmap.png'):
        """
        Create heatmap of traffic by hour and day of week
        
        Args:
            df: DataFrame with time series data
            date_column: Name of datetime column
            value_column: Name of value column
            save_name: Filename to save plot
        """
        df = df.copy()
        df['hour'] = df[date_column].dt.hour
        df['day_of_week'] = df[date_column].dt.day_name()
        
        # Pivot for heatmap
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        pivot_data = df.pivot_table(values=value_column, 
                                     index='day_of_week', 
                                     columns='hour', 
                                     aggfunc='mean')
        pivot_data = pivot_data.reindex(day_order)
        
        fig, ax = plt.subplots(figsize=(15, 6))
        
        sns.heatmap(pivot_data, cmap='YlOrRd', annot=False, fmt='.0f', 
                   cbar_kws={'label': 'Traffic Volume'}, ax=ax)
        
        ax.set_xlabel('Hour of Day', fontsize=12)
        ax.set_ylabel('Day of Week', fontsize=12)
        ax.set_title('Traffic Volume Heatmap (Day vs Hour)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.save_dir + save_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved: {save_name}")
    
    def plot_predictions(self, y_true, y_pred, dates=None, 
                        title='Actual vs Predicted Traffic Volume',
                        save_name='predictions.png'):
        """
        Plot actual vs predicted values
        
        Args:
            y_true: True values
            y_pred: Predicted values
            dates: Optional datetime index
            title: Plot title
            save_name: Filename to save plot
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Time series comparison
        if dates is not None:
            ax1.plot(dates, y_true, label='Actual', alpha=0.7, linewidth=1)
            ax1.plot(dates, y_pred, label='Predicted', alpha=0.7, linewidth=1)
        else:
            ax1.plot(y_true, label='Actual', alpha=0.7, linewidth=1)
            ax1.plot(y_pred, label='Predicted', alpha=0.7, linewidth=1)
        
        ax1.set_xlabel('Time' if dates is not None else 'Sample Index', fontsize=12)
        ax1.set_ylabel('Traffic Volume', fontsize=12)
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Scatter plot
        ax2.scatter(y_true, y_pred, alpha=0.5, s=10)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        ax2.set_xlabel('Actual Traffic Volume', fontsize=12)
        ax2.set_ylabel('Predicted Traffic Volume', fontsize=12)
        ax2.set_title('Actual vs Predicted Scatter Plot', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir + save_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved: {save_name}")
    
    def plot_residuals(self, y_true, y_pred, save_name='residuals.png'):
        """
        Plot residual analysis
        
        Args:
            y_true: True values
            y_pred: Predicted values
            save_name: Filename to save plot
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Residuals over time
        axes[0, 0].plot(residuals, linewidth=0.5, alpha=0.7)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0, 0].set_xlabel('Sample Index', fontsize=10)
        axes[0, 0].set_ylabel('Residuals', fontsize=10)
        axes[0, 0].set_title('Residuals Over Time', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Residual histogram
        axes[0, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('Residuals', fontsize=10)
        axes[0, 1].set_ylabel('Frequency', fontsize=10)
        axes[0, 1].set_title('Residual Distribution', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Residuals vs predicted
        axes[1, 0].scatter(y_pred, residuals, alpha=0.5, s=10)
        axes[1, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[1, 0].set_xlabel('Predicted Values', fontsize=10)
        axes[1, 0].set_ylabel('Residuals', fontsize=10)
        axes[1, 0].set_title('Residuals vs Predicted', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir + save_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved: {save_name}")
    
    def plot_feature_importance(self, feature_importance_df, top_n=20, 
                               save_name='feature_importance.png'):
        """
        Plot feature importance
        
        Args:
            feature_importance_df: DataFrame with 'feature' and 'importance' columns
            top_n: Number of top features to show
            save_name: Filename to save plot
        """
        top_features = feature_importance_df.head(top_n)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        ax.barh(range(len(top_features)), top_features['importance'], alpha=0.7)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.set_title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(self.save_dir + save_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved: {save_name}")
    
    def plot_model_comparison(self, results_df, save_name='model_comparison.png'):
        """
        Compare multiple models
        
        Args:
            results_df: DataFrame with model names and metrics
            save_name: Filename to save plot
        """
        metrics = [col for col in results_df.columns if col != 'Model']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for idx, metric in enumerate(metrics[:4]):
            ax = axes[idx]
            ax.bar(results_df['Model'], results_df[metric], alpha=0.7)
            ax.set_xlabel('Model', fontsize=10)
            ax.set_ylabel(metric, fontsize=10)
            ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(self.save_dir + save_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved: {save_name}")


# Example usage
if __name__ == "__main__":
    import os
    
    # Create plots directory
    os.makedirs('/home/claude/plots', exist_ok=True)
    
    # Load data
    df = pd.read_csv('/home/claude/traffic_data_synthetic.csv')
    df['date_time'] = pd.to_datetime(df['date_time'])
    
    # Create visualizer
    viz = TrafficVisualizer(save_dir='/home/claude/plots/')
    
    # Create plots
    viz.plot_time_series(df)
    viz.plot_hourly_pattern(df)
    viz.plot_weekly_pattern(df)
    viz.plot_heatmap(df)
    
    print("\nAll visualizations created!")
