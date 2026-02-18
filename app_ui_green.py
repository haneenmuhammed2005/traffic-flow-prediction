"""
Traffic Flow Prediction - Professional Green Theme UI
White background with light green and dark green accents
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os

# =============================================================================
# CUSTOM STYLING - Professional Green Theme
# =============================================================================

st.set_page_config(
    page_title="Traffic Flow Prediction",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - White, Light Green, Dark Green Theme
st.markdown("""
<style>
    /* Import professional font */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    /* Global font */
    * {
        font-family: 'Poppins', sans-serif !important;
    }
    
    /* Main background - Clean White */
    .stApp {
        background: #ffffff;
    }
    
    /* Sidebar styling - Dark Green */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1b4332 0%, #2d6a4f 100%);
        border-right: 3px solid #40916c;
    }
    
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] .css-1d391kg {
        color: #ffffff !important;
    }
    
    /* Sidebar radio buttons */
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #ffffff !important;
    }
    
    /* Hide default header */
    header {
        background: transparent !important;
    }
    
    /* Main content area */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background: #ffffff;
    }
    
    /* Card styling - White with light green border */
    .css-1r6slb0 {
        background: #ffffff;
        border-radius: 12px;
        padding: 25px;
        box-shadow: 0 4px 20px rgba(45, 106, 79, 0.1);
        border: 2px solid #d8f3dc;
    }
    
    /* Headers */
    h1 {
        color: #1b4332 !important;
        font-weight: 700 !important;
        font-size: 2.8rem !important;
        letter-spacing: -0.5px !important;
        margin-bottom: 0.5rem !important;
        border-bottom: 4px solid #52b788;
        padding-bottom: 15px;
    }
    
    h2 {
        color: #2d6a4f !important;
        font-weight: 600 !important;
        font-size: 1.8rem !important;
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
    }
    
    h3 {
        color: #40916c !important;
        font-weight: 600 !important;
        font-size: 1.3rem !important;
        margin-bottom: 0.8rem !important;
    }
    
    /* Buttons - Dark Green */
    .stButton > button {
        background: linear-gradient(135deg, #2d6a4f 0%, #1b4332 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 14px 32px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(45, 106, 79, 0.3);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #1b4332 0%, #0d2818 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(45, 106, 79, 0.4);
    }
    
    /* Metrics - Green themed */
    [data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        color: #1b4332 !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        color: #52b788 !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    
    [data-testid="stMetricDelta"] {
        color: #2d6a4f !important;
    }
    
    /* Input fields - Light green border */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select,
    .stDateInput > div > div > input,
    .stNumberInput > div > div > input {
        border: 2px solid #b7e4c7;
        border-radius: 10px;
        padding: 12px;
        font-size: 1rem;
        background: #ffffff;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus,
    .stDateInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
        border-color: #40916c;
        box-shadow: 0 0 0 3px rgba(82, 183, 136, 0.2);
        outline: none;
    }
    
    /* Sliders - Green */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #52b788 0%, #2d6a4f 100%);
    }
    
    .stSlider > div > div > div > div > div {
        background: #1b4332;
    }
    
    /* Success boxes - Light green */
    .stSuccess {
        background: #d8f3dc;
        border-left: 5px solid #52b788;
        border-radius: 10px;
        padding: 15px;
        color: #1b4332;
    }
    
    /* Error boxes - Keep standard but adjust */
    .stError {
        background: #ffe5e5;
        border-left: 5px solid #d62828;
        border-radius: 10px;
        padding: 15px;
    }
    
    /* Info boxes - Light green tint */
    .stInfo {
        background: #e8f5e9;
        border-left: 5px solid #40916c;
        border-radius: 10px;
        padding: 15px;
        color: #1b4332;
    }
    
    /* Warning boxes */
    .stWarning {
        background: #fff4e6;
        border-left: 5px solid #ff9800;
        border-radius: 10px;
        padding: 15px;
    }
    
    /* Tabs - Green theme */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #f1f8f4;
        border-radius: 10px 10px 0 0;
        color: #2d6a4f;
        font-weight: 500;
        padding: 12px 24px;
        border: 2px solid #d8f3dc;
        border-bottom: none;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #52b788 0%, #40916c 100%);
        color: white !important;
        border-color: #40916c;
    }
    
    /* DataFrames */
    .dataframe {
        border: 2px solid #d8f3dc !important;
        border-radius: 10px;
        overflow: hidden;
    }
    
    .dataframe thead tr th {
        background: #d8f3dc !important;
        color: #1b4332 !important;
        font-weight: 600 !important;
    }
    
    /* Progress bars - Green */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #52b788 0%, #2d6a4f 100%);
    }
    
    /* Selectbox dropdown */
    .stSelectbox > div > div > select option {
        background: #ffffff;
        color: #1b4332;
    }
    
    /* Radio buttons in sidebar */
    [data-testid="stSidebar"] .stRadio > label {
        background: rgba(255, 255, 255, 0.1);
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 8px;
        transition: all 0.3s ease;
    }
    
    [data-testid="stSidebar"] .stRadio > label:hover {
        background: rgba(255, 255, 255, 0.2);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #f1f8f4;
        border: 2px solid #b7e4c7;
        border-radius: 10px;
        color: #1b4332;
        font-weight: 600;
    }
    
    .streamlit-expanderHeader:hover {
        background: #d8f3dc;
        border-color: #52b788;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #52b788 !important;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_directories():
    """Create necessary directories"""
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    os.makedirs('results', exist_ok=True)

def generate_synthetic_data(periods=2000):
    """Generate synthetic traffic data"""
    dates = pd.date_range(start='2023-01-01', periods=periods, freq='h')
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
    
    df = pd.DataFrame({
        'date_time': dates,
        'traffic_volume': traffic_volume,
        'temperature': np.random.normal(20, 10, periods),
        'humidity': np.random.uniform(30, 90, periods),
        'weather_condition': np.random.choice(['Clear', 'Rain', 'Clouds'], periods)
    })
    
    return df

# =============================================================================
# MAIN APP
# =============================================================================

create_directories()

# Header with green accent
st.markdown("""
<div style='text-align: center; padding: 30px 0; background: linear-gradient(135deg, #d8f3dc 0%, #ffffff 100%); 
            border-radius: 15px; margin-bottom: 30px; border: 3px solid #52b788;'>
    <h1 style='border: none !important; padding-bottom: 0 !important;'>TRAFFIC FLOW PREDICTION SYSTEM</h1>
    <p style='color: #40916c; font-size: 1.2rem; margin-top: 10px; font-weight: 500;'>
        AI-Powered Urban Traffic Analytics & Forecasting Platform
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### NAVIGATION PANEL")
    page = st.radio(
        "Select Module",
        ["Data Generation", "Model Training", "Predictions", "Analytics Dashboard"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### SYSTEM STATUS")
    
    # Check data status
    if os.path.exists('data/raw/traffic_data_raw.csv'):
        st.success("Data Available")
    else:
        st.warning("No Data Found")
    
    # Check model status
    model_files = [f for f in os.listdir('models') if f.endswith('.pkl')] if os.path.exists('models') else []
    if model_files:
        st.success(f"{len(model_files)} Models Trained")
    else:
        st.warning("No Trained Models")
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 15px; background: rgba(255,255,255,0.1); 
                border-radius: 10px; margin-top: 20px;'>
        <p style='color: white; font-size: 0.85rem; margin: 0;'>
            Machine Learning<br>Traffic Analytics
        </p>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# PAGE 1: DATA GENERATION
# =============================================================================

if page == "Data Generation":
    st.markdown("## Data Generation Module")
    st.markdown("Generate synthetic traffic data for model training and testing")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("### Configuration Parameters")
        periods = st.slider("Number of Hours to Generate", 100, 10000, 2000, step=100)
    
    with col2:
        st.markdown("### Duration")
        st.metric("Days", f"{periods/24:.1f}")
        st.metric("Data Points", f"{periods:,}")
    
    with col3:
        st.markdown("### Action")
        st.write("")  # spacing
        if st.button("GENERATE DATASET", use_container_width=True):
            with st.spinner("Generating traffic data..."):
                df = generate_synthetic_data(periods)
                df.to_csv('data/raw/traffic_data_raw.csv', index=False)
                st.success(f"Successfully generated {len(df):,} records")
                st.balloons()
    
    # Show existing data if available
    if os.path.exists('data/raw/traffic_data_raw.csv'):
        st.markdown("---")
        st.markdown("### Dataset Overview")
        df = pd.read_csv('data/raw/traffic_data_raw.csv')
        df['date_time'] = pd.to_datetime(df['date_time'])
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Records", f"{len(df):,}")
        col2.metric("Average Traffic", f"{df['traffic_volume'].mean():.0f}")
        col3.metric("Peak Traffic", f"{df['traffic_volume'].max():.0f}")
        col4.metric("Time Span", f"{(df['date_time'].max() - df['date_time'].min()).days} days")
        
        # Plot
        st.markdown("### Traffic Volume Time Series")
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(df['date_time'], df['traffic_volume'], linewidth=1, color='#2d6a4f', alpha=0.8)
        ax.fill_between(df['date_time'], df['traffic_volume'], alpha=0.2, color='#52b788')
        ax.set_xlabel('Date', fontsize=12, fontweight='600', color='#1b4332')
        ax.set_ylabel('Traffic Volume', fontsize=12, fontweight='600', color='#1b4332')
        ax.grid(True, alpha=0.3, linestyle='--', color='#b7e4c7')
        ax.set_facecolor('#f8fdf9')
        fig.patch.set_facecolor('white')
        plt.tight_layout()
        st.pyplot(fig)

# =============================================================================
# PAGE 2: MODEL TRAINING
# =============================================================================

elif page == "Model Training":
    st.markdown("## Model Training Module")
    st.markdown("Train machine learning models on traffic data")
    
    if not os.path.exists('data/raw/traffic_data_raw.csv'):
        st.error("No data found. Please generate data first.")
        st.stop()
    
    # Load data
    df = pd.read_csv('data/raw/traffic_data_raw.csv')
    df['date_time'] = pd.to_datetime(df['date_time'])
    st.info(f"Dataset loaded: {len(df):,} records available for training")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        model_type = st.selectbox(
            "Select Machine Learning Algorithm",
            ["Random Forest", "XGBoost", "LightGBM"],
            help="Choose the algorithm for training"
        )
    
    with col2:
        n_estimators = st.number_input("Number of Trees", 50, 500, 100, step=50)
    
    with col3:
        st.write("")  # spacing
        train_button = st.button("START TRAINING", use_container_width=True)
    
    if train_button:
        progress_bar = st.progress(0)
        status = st.empty()
        
        # Feature engineering
        status.text("Step 1/4: Engineering features...")
        progress_bar.progress(25)
        
        df['hour'] = df['date_time'].dt.hour
        df['day_of_week'] = df['date_time'].dt.dayofweek
        df['month'] = df['date_time'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['traffic_lag_1'] = df['traffic_volume'].shift(1)
        df['traffic_lag_24'] = df['traffic_volume'].shift(24)
        df['traffic_rolling_mean_3'] = df['traffic_volume'].rolling(3, min_periods=1).mean()
        df['traffic_rolling_std_24'] = df['traffic_volume'].rolling(24, min_periods=1).std()
        df = df.dropna()
        
        # Prepare data
        status.text("Step 2/4: Preparing train/test split...")
        progress_bar.progress(50)
        
        feature_cols = ['hour', 'day_of_week', 'month', 'is_weekend',
                       'traffic_lag_1', 'traffic_lag_24',
                       'traffic_rolling_mean_3', 'traffic_rolling_std_24',
                       'temperature', 'humidity']
        
        X = df[feature_cols]
        y = df['traffic_volume']
        
        train_size = int(0.8 * len(df))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Train model
        status.text("Step 3/4: Training model...")
        progress_bar.progress(75)
        
        if model_type == "Random Forest":
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=10, random_state=42, n_jobs=-1)
        elif model_type == "XGBoost":
            import xgboost as xgb
            model = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=6, random_state=42, n_jobs=-1)
        else:
            import lightgbm as lgb
            model = lgb.LGBMRegressor(n_estimators=n_estimators, max_depth=6, random_state=42, n_jobs=-1, verbose=-1)
        
        model.fit(X_train, y_train)
        
        # Evaluate
        status.text("Step 4/4: Evaluating performance...")
        progress_bar.progress(90)
        
        y_pred = model.predict(X_test)
        
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Save model
        import joblib
        model_path = f'models/traffic_model_{model_type.lower().replace(" ", "_")}.pkl'
        joblib.dump(model, model_path)
        
        progress_bar.progress(100)
        status.text("Training complete!")
        st.success(f"Model trained successfully and saved to {model_path}")
        
        # Show results
        st.markdown("### Model Performance Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Mean Absolute Error", f"{mae:.2f}", help="Lower is better")
        col2.metric("Root Mean Squared Error", f"{rmse:.2f}", help="Lower is better")
        col3.metric("R² Score", f"{r2:.4f}", help="Closer to 1 is better")
        
        # Plot
        st.markdown("### Prediction Visualization")
        fig, ax = plt.subplots(figsize=(14, 5))
        test_dates = df['date_time'].iloc[train_size:].values
        ax.plot(test_dates, y_test.values, label='Actual Traffic', alpha=0.8, linewidth=1.5, color='#1b4332')
        ax.plot(test_dates, y_pred, label='Predicted Traffic', alpha=0.8, linewidth=1.5, color='#52b788')
        ax.fill_between(test_dates, y_test.values, y_pred, alpha=0.2, color='#95d5b2')
        ax.legend(fontsize=11, frameon=True, shadow=True, fancybox=True)
        ax.set_xlabel('Date', fontsize=12, fontweight='600', color='#1b4332')
        ax.set_ylabel('Traffic Volume', fontsize=12, fontweight='600', color='#1b4332')
        ax.grid(True, alpha=0.3, linestyle='--', color='#b7e4c7')
        ax.set_facecolor('#f8fdf9')
        fig.patch.set_facecolor('white')
        plt.tight_layout()
        st.pyplot(fig)

# =============================================================================
# PAGE 3: PREDICTIONS
# =============================================================================

elif page == "Predictions":
    st.markdown("## Prediction Module")
    st.markdown("Generate traffic forecasts using trained models")
    
    model_files = [f for f in os.listdir('models') if f.endswith('.pkl')]
    
    if not model_files:
        st.error("No trained models found. Please train a model first.")
        st.stop()
    
    selected_model = st.selectbox("Select Trained Model", model_files)
    
    st.markdown("### Input Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        pred_date = st.date_input("Prediction Date", value=datetime.now() + timedelta(days=1))
        pred_hour = st.slider("Hour of Day", 0, 23, 8)
    
    with col2:
        temperature = st.slider("Temperature (°C)", -10, 40, 20)
        humidity = st.slider("Humidity (%)", 0, 100, 50)
    
    if st.button("GENERATE PREDICTION", use_container_width=True):
        with st.spinner("Computing traffic forecast..."):
            try:
                import joblib
                
                # Load model
                model = joblib.load(f'models/{selected_model}')
                
                # Create features
                is_weekend = 1 if pred_date.weekday() >= 5 else 0
                
                features = pd.DataFrame({
                    'hour': [pred_hour],
                    'day_of_week': [pred_date.weekday()],
                    'month': [pred_date.month],
                    'is_weekend': [is_weekend],
                    'traffic_lag_1': [800],
                    'traffic_lag_24': [850],
                    'traffic_rolling_mean_3': [820],
                    'traffic_rolling_std_24': [100],
                    'temperature': [temperature],
                    'humidity': [humidity]
                })
                
                prediction = model.predict(features)[0]
                prediction = max(0, prediction)
                
                st.success("Prediction Generated Successfully")
                
                # Display result in green-themed card
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #52b788 0%, #2d6a4f 100%); 
                            padding: 40px; border-radius: 15px; text-align: center; color: white; 
                            margin: 30px 0; box-shadow: 0 8px 25px rgba(45, 106, 79, 0.3);'>
                    <h2 style='color: white !important; margin: 0; font-weight: 600;'>Predicted Traffic Volume</h2>
                    <h1 style='color: white !important; font-size: 4.5rem; margin: 25px 0; font-weight: 700;'>{prediction:.0f}</h1>
                    <p style='font-size: 1.3rem; margin: 0; opacity: 0.95;'>vehicles per hour</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Traffic level indicator
                if prediction < 600:
                    level = "Light Traffic"
                    color = "#52b788"
                    emoji = "✓"
                elif prediction < 900:
                    level = "Moderate Traffic"
                    color = "#ffa500"
                    emoji = "!"
                else:
                    level = "Heavy Traffic"
                    color = "#dc3545"
                    emoji = "!!"
                
                st.markdown(f"""
                <div style='text-align: center; padding: 20px; background: {color}15; 
                            border-radius: 12px; border-left: 6px solid {color}; margin-top: 20px;'>
                    <h3 style='color: {color} !important; margin: 0; font-weight: 600;'>{level}</h3>
                    <p style='color: #666; margin-top: 10px; font-size: 0.95rem;'>
                        {pred_date.strftime('%A, %B %d, %Y')} at {pred_hour:02d}:00
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Prediction Error: {str(e)}")
                st.info("Ensure the model was trained with consistent features.")

# =============================================================================
# PAGE 4: ANALYTICS
# =============================================================================

else:
    st.markdown("## Analytics Dashboard")
    st.markdown("Comprehensive traffic data analysis and insights")
    
    if not os.path.exists('data/raw/traffic_data_raw.csv'):
        st.error("No data available. Please generate data first.")
        st.stop()
    
    df = pd.read_csv('data/raw/traffic_data_raw.csv')
    df['date_time'] = pd.to_datetime(df['date_time'])
    
    # Metrics
    st.markdown("### Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Average Traffic", f"{df['traffic_volume'].mean():.0f}")
    col3.metric("Peak Traffic", f"{df['traffic_volume'].max():.0f}")
    col4.metric("Minimum Traffic", f"{df['traffic_volume'].min():.0f}")
    
    st.markdown("---")
    
    # Patterns
    df['hour'] = df['date_time'].dt.hour
    df['day_name'] = df['date_time'].dt.day_name()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Hourly Traffic Pattern")
        hourly = df.groupby('hour')['traffic_volume'].mean()
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(hourly.index, hourly.values, color='#52b788', alpha=0.8, edgecolor='#2d6a4f', linewidth=1.5)
        ax.set_xlabel('Hour of Day', fontsize=12, fontweight='600', color='#1b4332')
        ax.set_ylabel('Average Traffic', fontsize=12, fontweight='600', color='#1b4332')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--', color='#b7e4c7')
        ax.set_facecolor('#f8fdf9')
        fig.patch.set_facecolor('white')
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.markdown("### Weekly Traffic Pattern")
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily = df.groupby('day_name')['traffic_volume'].mean().reindex(day_order)
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(range(7), daily.values, color='#2d6a4f', alpha=0.8, edgecolor='#1b4332', linewidth=1.5)
        ax.set_xticks(range(7))
        ax.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        ax.set_ylabel('Average Traffic', fontsize=12, fontweight='600', color='#1b4332')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--', color='#b7e4c7')
        ax.set_facecolor('#f8fdf9')
        fig.patch.set_facecolor('white')
        plt.tight_layout()
        st.pyplot(fig)
    
    # Raw data
    st.markdown("---")
    with st.expander("VIEW RAW DATASET"):
        st.dataframe(df, use_container_width=True, height=400)
        
        csv = df.to_csv(index=False)
        st.download_button(
            label="DOWNLOAD CSV",
            data=csv,
            file_name='traffic_data_export.csv',
            mime='text/csv',
            use_container_width=True
        )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #40916c; padding: 25px; background: #f1f8f4; 
            border-radius: 12px; border: 2px solid #d8f3dc;'>
    <p style='margin: 0; font-weight: 500; font-size: 0.95rem;'>
        Traffic Flow Prediction System | Powered by Machine Learning
    </p>
    <p style='margin: 5px 0 0 0; font-size: 0.85rem; color: #52b788;'>
        Advanced Analytics for Smart Urban Planning
    </p>
</div>
""", unsafe_allow_html=True)
