import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

# --- 1. Setup the Open-Meteo API client ---
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# --- 2. Define Punjab Locations ---
punjab_locations = {
    "Amritsar": {"latitude": 31.63, "longitude": 74.87},
    "Ludhiana": {"latitude": 30.90, "longitude": 75.86},
    "Jalandhar": {"latitude": 31.33, "longitude": 75.58},
    "Patiala": {"latitude": 30.34, "longitude": 76.38},
    "Bathinda": {"latitude": 30.21, "longitude": 74.94},
    "Firozpur": {"latitude": 30.92, "longitude": 74.60}
}

# --- 3. Loop through each location ---
for city, coords in punjab_locations.items():
    print(f"\n#### **{city.upper()}**")
    
    # API parameters are updated for each city in the loop
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": coords["latitude"],
        "longitude": coords["longitude"],
        "hourly": ["temperature_2m", "soil_moisture_0_to_1cm", "soil_moisture_1_to_3cm", "soil_temperature_0cm", "soil_temperature_6cm"],
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process first location
    response = responses[0]
    print(f"Coordinates: {response.Latitude():.2f}°N {response.Longitude():.2f}°E")

    # Process hourly data
    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_soil_moisture_0_to_1cm = hourly.Variables(1).ValuesAsNumpy()
    hourly_soil_moisture_1_to_3cm = hourly.Variables(2).ValuesAsNumpy()
    hourly_soil_temperature_0cm = hourly.Variables(3).ValuesAsNumpy()
    hourly_soil_temperature_6cm = hourly.Variables(4).ValuesAsNumpy()

    hourly_data = {"date": pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    )}
    hourly_data["temperature_2m"] = hourly_temperature_2m
    hourly_data["soil_moisture_0_to_1cm"] = hourly_soil_moisture_0_to_1cm
    hourly_data["soil_moisture_1_to_3cm"] = hourly_soil_moisture_1_to_3cm
    hourly_data["soil_temperature_0cm"] = hourly_soil_temperature_0cm
    hourly_data["soil_temperature_6cm"] = hourly_soil_temperature_6cm

    hourly_dataframe = pd.DataFrame(data=hourly_data)
    
    # Print the head of the dataframe for a summarized view
    print(hourly_dataframe.head())