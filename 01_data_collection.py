import os
import requests
import numpy as np
import pandas as pd

def generate_colombo_dataset(output_path="colombo_air_quality.csv"):
    print("Fetching Colombo meteorological data from Open-Meteo API...")
    url = (
        "https://archive-api.open-meteo.com/v1/archive?"
        "latitude=6.9271&longitude=79.8612&"
        "start_date=2024-01-01&end_date=2024-06-30&"
        "hourly=temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m,precipitation&"
        "timezone=Asia%2FColombo"
    )
    
    response = requests.get(url, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(f"API request failed with status code {response.status_code}")
        
    data = response.json()["hourly"]
    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"])
    
    # Feature Engineering
    df["hour"] = df["time"].dt.hour
    df["day_of_week"] = df["time"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)
    
    # Environmental dynamics for Colombo PM2.5 ground truth
    np.random.seed(42)
    n = len(df)
    
    traffic_factor = np.where(df["hour"].isin([7, 8, 9, 17, 18, 19]), 1.8, 
                     np.where(df["hour"].isin([1, 2, 3, 4]), 0.4, 1.0))
    weekend_factor = np.where(df["is_weekend"] == 1, 0.75, 1.0)
    
    wind_dispersion = np.exp(-df["wind_speed_10m"] / 4.5)
    humidity_accumulation = 1.0 + (df["relative_humidity_2m"] / 100.0) * 0.4
    rain_washout = np.exp(-df["precipitation"] * 0.8)
    
    base_pm25 = 28.0
    synthetic_pm25 = (base_pm25 * traffic_factor * weekend_factor * 
                      wind_dispersion * humidity_accumulation * rain_washout)
    
    pm25_series = np.zeros(n)
    pm25_series[0] = synthetic_pm25.iloc[0]
    for t in range(1, n):
        pm25_series[t] = 0.72 * pm25_series[t-1] + 0.28 * synthetic_pm25.iloc[t] + np.random.normal(0, 1.8)
    
    df["pm25"] = np.clip(pm25_series, 4.0, 150.0).round(2)
    df["pm25_lag_1h"] = df["pm25"].shift(1)
    
    df = df.dropna().reset_index(drop=True)
    
    cols = [
        "time", "temperature_2m", "relative_humidity_2m", "surface_pressure",
        "wind_speed_10m", "wind_direction_10m", "precipitation",
        "hour", "day_of_week", "is_weekend", "pm25_lag_1h", "pm25"
    ]
    df = df[cols]
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully compiled and saved to '{output_path}'. Total rows: {len(df)}")
    return df

if __name__ == "__main__":
    generate_colombo_dataset()