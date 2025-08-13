import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
import numpy as np

# --- 1. Setup the Open-Meteo API client ---
# Using a cache to avoid re-downloading data and retrying on network errors.
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# --- 2. Define Saurashtra Locations ---
# A dictionary of key cities/towns in the Saurashtra region with their coordinates.
saurashtra_locations = {
    "Rajkot": {"latitude": 22.30, "longitude": 70.80},
    "Jamnagar": {"latitude": 22.47, "longitude": 70.06},
    "Junagadh": {"latitude": 21.52, "longitude": 70.47},
    "Bhavnagar": {"latitude": 21.76, "longitude": 72.15},
    "Porbandar": {"latitude": 21.64, "longitude": 69.63},
    "Amreli": {"latitude": 21.60, "longitude": 71.22}
}

# --- 3. Loop Through Each Location to Fetch Data and Provide Advisories ---
# The core of the regional analysis.
print("======================================================")
print(" TerraMoist Regional Soil Advisory for Saurashtra ")
print("======================================================")

for city, coords in saurashtra_locations.items():
    # --- API Parameters for the current city ---
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": coords["latitude"],
        "longitude": coords["longitude"],
        "hourly": ["soil_temperature_0cm", "soil_moisture_0_to_7cm"],
        "forecast_days": 7
    }
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    # --- Process Hourly Data ---
    hourly = response.Hourly()
    hourly_soil_temperature_0cm = hourly.Variables(0).ValuesAsNumpy()
    hourly_soil_moisture_0_to_7cm = hourly.Variables(1).ValuesAsNumpy()

    hourly_data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        ),
        "soil_moisture_0_to_7cm_m3m3": hourly_soil_moisture_0_to_7cm,
    }
    hourly_dataframe = pd.DataFrame(data=hourly_data)

    # --- Generate Daily Advisory from Hourly Data ---
    hourly_dataframe['date'] = hourly_dataframe['date'].dt.tz_convert('Asia/Kolkata')
    daily_summary = hourly_dataframe.resample('D', on='date').mean()

    print(f"\n Location: {city.upper()}")
    print("-" * 25)

    for date, row in daily_summary.iterrows():
        day = date.strftime('%A, %b %d')
        avg_moisture = row['soil_moisture_0_to_7cm_m3m3']

        # Simplified analysis based on average daily moisture
        advice = ""
        if avg_moisture < 0.15:
            advice = " High Stress. Irrigation needed."
        elif 0.15 <= avg_moisture < 0.25:
            advice = " Moderate Stress. Monitor soil."
        else:
            advice = " Low Stress. Moisture is adequate."

        print(f"  {day}: {advice} (Avg Moisture: {avg_moisture:.2f} m³/m³)")

print("\n======================================================")
print("Analysis Complete.")
print("======================================================")