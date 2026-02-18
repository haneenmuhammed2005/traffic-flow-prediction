"""
Traffic Flow Prediction - Simple UI Version with Streamlit
Run with: streamlit run app_ui.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os

# Set page config
st.set_page_config(
    page_title="Traffic Flow Prediction",
    page_icon="🚗",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main {background-color: #f5f5f5;}
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        padding: 10px 24px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🚗 Traffic Flow Prediction System")
st.markdown("### Complete ML Pipeline with Real-time Visualization")

# Sidebar
st.sidebar.title("⚙️ Settings")
action = st.sidebar.radio(
    "Choose Action",
    ["📊 Generate Data", "🔧 Train Models", "🔮 Make Predictions", "📈 View Results"]
)

# Create directories
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# ============================================================================
# ACTION 1: GENERATE DATA
# ============================================================================
if action == "📊 Generate Data":
    st.header("📊 Generate Synthetic Traffic Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input("Start Date", value=datetime(2023, 1, 1))
        periods = st.slider("Number of Hours", 100, 8760, 2000)
    
    with col2:
        st.info(f"**Total Duration:** {periods/24:.0f} days")
        st.info(f"**Data Points:** {periods:,}")
    
    if st.button("🎲 Generate Data", key="gen"):
        with st.spinner("Generating synthetic traffic data..."):
            # Generate dates
            dates = pd.date_range(start=start_date, periods=periods, freq='h')
            
            # Base traffic pattern
            base_traffic = 1000
            hour_pattern = np.array([0.3, 0.2, 0.15, 0.15, 0.2, 0.4, 0.8, 1.0, 
                                     0.9, 0.7, 0.6, 0.6, 0.7, 0.8, 0.9, 1.0,
                                     1.0, 0.9, 0.7, 0.6, 0.5, 0.4, 0.35, 0.3])
            day_pattern = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.7, 0.6])
            
            traffic_volume = []
            for date in dates:
                hour = date.hour
                day = date.dayofweek
                traffic = base_traffic * hour_pattern[hour] * day_pattern[day]
                noise = np.random.normal(0, traffic * 0.15)
                traffic_volume.append(max(0, traffic + noise))
            
            # Create dataframe
            df = pd.DataFrame({
                'date_time': dates,
                'traffic_volume': traffic_volume,
                'temperature': np.random.normal(20, 10, periods),
                'humidity': np.random.uniform(30, 90, periods),
                'weather_condition': np.random.choice(['Clear', 'Rain', 'Clouds'], periods)
            })
            
            # Save
            df.to_csv('data/raw/traffic_data_raw.csv', index=False)
            
            st.success(f"✅ Generated {len(df):,} records!")
            
            # Show preview
            st.subheader("📋 Data Preview")
            st.dataframe(df.head(20))
            
            # Show statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", f"{len(df):,}")
            with col2:
                st.metric("Avg Traffic", f"{df['traffic_volume'].mean():.0f}")
            with col3:
                st.metric("Peak Traffic", f"{df['traffic_volume'].max():.0f}")
            
            # Plot time series
            st.subheader("📈 Traffic Over Time")
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(df['date_time'], df['traffic_volume'], linewidth=0.5, alpha=0.7)
            ax.set_xlabel('Date')
            ax.set_ylabel('Traffic Volume')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Plot hourly pattern
            st.subheader("⏰ Average Traffic by Hour")
            df['hour'] = pd.to_datetime(df['date_time']).dt.hour
            hourly_avg = df.groupby('hour')['traffic_volume'].mean()
            
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.bar(hourly_avg.index, hourly_avg.values, color='steelblue', alpha=0.7)
            ax.set_xlabel('Hour of Day')
            ax.set_ylabel('Average Traffic')
            ax.grid(True, alpha=0.3, axis='y')
            st.pyplot(fig)

# ============================================================================
# ACTION 2: TRAIN MODELS
# ============================================================================
elif action == "🔧 Train Models":
    st.header("🔧 Train Machine Learning Models")
    
    # Check if data exists
    if not os.path.exists('data/raw/traffic_data_raw.csv'):
        st.error("❌ No data found! Please generate data first.")
        st.stop()
    
    # Load data
    df = pd.read_csv('data/raw/traffic_data_raw.csv')
    df['date_time'] = pd.to_datetime(df['date_time'])
    
    st.info(f"📁 Loaded {len(df):,} records")
    
    # Model selection
    model_type = st.selectbox(
        "Choose Model",
        ["Random Forest", "XGBoost", "LightGBM"]
    )
    
    col1, col2 = st.columns(2)
    with col1:
        n_estimators = st.slider("Number of Trees", 10, 200, 100)
    with col2:
        max_depth = st.slider("Max Depth", 3, 20, 10)
    
    if st.button("🚀 Train Model", key="train"):
        with st.spinner(f"Training {model_type}..."):
            
            # Feature engineering
            st.write("**Step 1:** Creating features...")
            progress_bar = st.progress(0)
            
            df['hour'] = df['date_time'].dt.hour
            df['day_of_week'] = df['date_time'].dt.dayofweek
            df['month'] = df['date_time'].dt.month
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            progress_bar.progress(25)
            
            # Lag features
            df['traffic_lag_1'] = df['traffic_volume'].shift(1)
            df['traffic_lag_24'] = df['traffic_volume'].shift(24)
            progress_bar.progress(50)
            
            # Rolling features
            df['traffic_rolling_mean_3'] = df['traffic_volume'].rolling(3, min_periods=1).mean()
            df['traffic_rolling_std_24'] = df['traffic_volume'].rolling(24, min_periods=1).std()
            progress_bar.progress(75)
            
            df = df.dropna()
            progress_bar.progress(100)
            
            st.success(f"✅ Created {df.shape[1]} features")
            
            # Prepare data
            st.write("**Step 2:** Preparing train/test split...")
            feature_cols = ['hour', 'day_of_week', 'month', 'is_weekend', 
                           'traffic_lag_1', 'traffic_lag_24', 
                           'traffic_rolling_mean_3', 'traffic_rolling_std_24',
                           'temperature', 'humidity']
            
            X = df[feature_cols]
            y = df['traffic_volume']
            
            train_size = int(0.8 * len(df))
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            st.info(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
            
            # Train model
            st.write("**Step 3:** Training model...")
            
            if model_type == "Random Forest":
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            elif model_type == "XGBoost":
                import xgboost as xgb
                model = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            else:
                import lightgbm as lgb
                model = lgb.LGBMRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, verbose=-1)
            
            model.fit(X_train, y_train)
            
            # Evaluate
            st.write("**Step 4:** Evaluating...")
            y_pred = model.predict(X_test)
            
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            # Save model
            import joblib
            joblib.dump(model, f'models/traffic_model_{model_type.lower().replace(" ", "_")}.pkl')
            
            # Show results
            st.success("✅ Training Complete!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("MAE", f"{mae:.2f}")
            with col2:
                st.metric("RMSE", f"{rmse:.2f}")
            with col3:
                st.metric("R² Score", f"{r2:.4f}")
            
            # Plot predictions
            st.subheader("📊 Predictions vs Actual")
            fig, ax = plt.subplots(figsize=(12, 4))
            test_dates = df['date_time'].iloc[train_size:]
            ax.plot(test_dates, y_test.values, label='Actual', alpha=0.7, linewidth=1)
            ax.plot(test_dates, y_pred, label='Predicted', alpha=0.7, linewidth=1)
            ax.legend()
            ax.set_xlabel('Date')
            ax.set_ylabel('Traffic Volume')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                st.subheader("🎯 Feature Importance")
                importance = pd.DataFrame({
                    'feature': feature_cols,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(importance['feature'], importance['importance'])
                ax.set_xlabel('Importance')
                ax.invert_yaxis()
                st.pyplot(fig)

# ============================================================================
# ACTION 3: MAKE PREDICTIONS
# ============================================================================
elif action == "🔮 Make Predictions":
    st.header("🔮 Make Traffic Predictions")
    
    # Check if model exists
    model_files = [f for f in os.listdir('models') if f.endswith('.pkl')]
    
    if not model_files:
        st.error("❌ No trained models found! Please train a model first.")
        st.stop()
    
    selected_model = st.selectbox("Choose Model", model_files)
    
    col1, col2 = st.columns(2)
    
    with col1:
        pred_date = st.date_input("Prediction Date", value=datetime.now() + timedelta(days=1))
        pred_hour = st.slider("Hour", 0, 23, 8)
    
    with col2:
        temperature = st.slider("Temperature (°C)", -10, 40, 20)
        humidity = st.slider("Humidity (%)", 0, 100, 50)
    
    if st.button("🔮 Predict", key="predict"):
        with st.spinner("Making prediction..."):
            try:
                import joblib
                from feature_engineering import TrafficFeatureEngineer
                
                # Load model
                model = joblib.load(f'models/{selected_model}')
                
                # Load historical data to get proper features
                if os.path.exists('data/processed/traffic_data_featured.csv'):
                    df_hist = pd.read_csv('data/processed/traffic_data_featured.csv')
                    df_hist['date_time'] = pd.to_datetime(df_hist['date_time'])
                    
                    # Get the most recent record as reference
                    last_record = df_hist.iloc[-1]
                    
                    # Create prediction row with all required features
                    pred_data = pd.DataFrame({
                        'date_time': [pd.Timestamp(pred_date) + pd.Timedelta(hours=pred_hour)],
                        'traffic_volume': [last_record['traffic_volume']],  # Placeholder
                        'temperature': [temperature],
                        'humidity': [humidity],
                        'weather_condition': ['Clear']
                    })
                    
                    # Apply feature engineering
                    feature_engineer = TrafficFeatureEngineer()
                    df_combined = pd.concat([df_hist.tail(200), pred_data], ignore_index=True)
                    df_featured = feature_engineer.create_all_features(df_combined, target_column='traffic_volume')
                    
                    # Get the last row (our prediction row) with all features
                    pred_row = df_featured.iloc[-1:].copy()
                    
                    # Remove non-feature columns
                    exclude_cols = ['traffic_volume', 'date_time', 'weather_condition']
                    feature_cols = [col for col in pred_row.columns if col not in exclude_cols]
                    
                    X_pred = pred_row[feature_cols]
                    prediction = model.predict(X_pred)[0]
                    
                else:
                    st.error("❌ No historical data found! Please train the model first.")
                    st.stop()
                    
            except Exception as e:
                st.error(f"❌ Prediction Error: {str(e)}")
                st.info("💡 Tip: Make sure you've trained the model using the same data!")
                st.stop()
            
            st.success("✅ Prediction Complete!")
            
            # Display prediction
            st.markdown(f"""
            ## Predicted Traffic: **{prediction:.0f} vehicles**
            
            **Details:**
            - Date: {pred_date.strftime('%A, %B %d, %Y')}
            - Time: {pred_hour:02d}:00
            - Temperature: {temperature}°C
            - Humidity: {humidity}%
            - Weekend: {'Yes' if is_weekend else 'No'}
            """)
            
            # Gauge chart
            fig, ax = plt.subplots(figsize=(8, 2))
            ax.barh([0], [prediction], color='green' if prediction < 700 else 'orange' if prediction < 900 else 'red')
            ax.set_xlim(0, 1200)
            ax.set_yticks([])
            ax.set_xlabel('Traffic Volume')
            ax.axvline(700, color='yellow', linestyle='--', alpha=0.5, label='Moderate')
            ax.axvline(900, color='red', linestyle='--', alpha=0.5, label='Heavy')
            ax.legend()
            st.pyplot(fig)

# ============================================================================
# ACTION 4: VIEW RESULTS
# ============================================================================
else:
    st.header("📈 View Results & Analysis")
    
    # Check if data exists
    if not os.path.exists('data/raw/traffic_data_raw.csv'):
        st.error("❌ No data found! Please generate data first.")
        st.stop()
    
    df = pd.read_csv('data/raw/traffic_data_raw.csv')
    df['date_time'] = pd.to_datetime(df['date_time'])
    
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "⏰ Patterns", "📁 Data"])
    
    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Avg Traffic", f"{df['traffic_volume'].mean():.0f}")
        with col3:
            st.metric("Peak Traffic", f"{df['traffic_volume'].max():.0f}")
        with col4:
            st.metric("Min Traffic", f"{df['traffic_volume'].min():.0f}")
        
        st.subheader("Time Series")
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df['date_time'], df['traffic_volume'], linewidth=0.5)
        ax.set_xlabel('Date')
        ax.set_ylabel('Traffic Volume')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with tab2:
        df['hour'] = df['date_time'].dt.hour
        df['day_name'] = df['date_time'].dt.day_name()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("By Hour")
            hourly = df.groupby('hour')['traffic_volume'].mean()
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(hourly.index, hourly.values, color='steelblue', alpha=0.7)
            ax.set_xlabel('Hour')
            ax.set_ylabel('Avg Traffic')
            ax.grid(True, alpha=0.3, axis='y')
            st.pyplot(fig)
        
        with col2:
            st.subheader("By Day of Week")
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            daily = df.groupby('day_name')['traffic_volume'].mean().reindex(day_order)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(range(7), daily.values, color='coral', alpha=0.7)
            ax.set_xticks(range(7))
            ax.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
            ax.set_ylabel('Avg Traffic')
            ax.grid(True, alpha=0.3, axis='y')
            st.pyplot(fig)
    
    with tab3:
        st.subheader("Raw Data")
        st.dataframe(df)
        
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name='traffic_data.csv',
            mime='text/csv'
        )

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Project Info")
st.sidebar.info("""
**Traffic Flow Prediction**  
Complete ML Pipeline  
Built with Streamlit
""")
