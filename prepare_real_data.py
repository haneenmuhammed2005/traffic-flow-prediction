import pandas as pd

# Load your downloaded data
df = pd.read_csv('data/raw/Metro_Interstate_Traffic_Volume.csv')

# Check what columns it has
print("Current columns:", df.columns.tolist())

# Rename columns to match our code
# ADJUST THESE based on your actual column names!
df = df.rename(columns={
    'date_time': 'date_time',  # or 'timestamp', 'datetime', etc.
    'volume': 'traffic_volume',  # or 'count', 'vehicles', etc.
    'temp': 'temperature',
    'clouds_all': 'humidity'  # adjust as needed
})

# Make sure date_time is datetime format
df['date_time'] = pd.to_datetime(df['date_time'])

# Save as traffic_data_raw.csv
df.to_csv('data/raw/traffic_data_raw.csv', index=False)

print(f"✅ Processed {len(df)} records")
print("Columns:", df.columns.tolist())
print("\nFirst few rows:")
print(df.head())