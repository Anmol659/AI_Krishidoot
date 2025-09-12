import requests

# API configuration
API_KEY = "b1e6b35205dc4102e12a861e549c56f4"  # Your OpenWeatherMap API key
BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

def get_hyperlocal_weather(location: str) -> str:
    """
    Fetches hyper-local weather data from OpenWeatherMap API.
    Args:
        location: City or District name (e.g., 'Amritsar', 'Ludhiana')
    Returns:
        A string describing current weather conditions or an error message.
    """
    params = {
        'q': f"{location},IN",  # Location + country code for accuracy
        'appid': API_KEY,
        'units': 'metric'      # Temperature in Celsius
    }

    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()  # Raises error for 4xx/5xx status
        data = response.json()

        # Extract weather details
        weather_description = data['weather'][0]['description']
        temperature = data['main']['temp']
        humidity = data['main']['humidity']
        wind_speed = data['wind']['speed']

        return (f"Weather in {location.title()}: "
                f"{weather_description.title()}, {temperature}°C, "
                f"{humidity}% humidity, Wind {wind_speed} m/s")

    except requests.exceptions.HTTPError as http_err:
        if response.status_code == 404:
            return f"Error: Could not find weather for '{location}'."
        elif response.status_code == 401:
            return "Error: Invalid or inactive API key."
        else:
            return f"HTTP error occurred: {http_err}"
    except Exception as e:
        return f"Unexpected error: {e}"

# --- Main execution block ---
if __name__ == "__main__":
    print("--- Fetching Weather for All Districts in Punjab ---")
    
    # Comprehensive list of all districts in Punjab
    punjab_districts = [
        "Amritsar", "Barnala", "Bathinda", "Faridkot", "Fatehgarh Sahib",
        "Fazilka", "Ferozepur", "Gurdaspur", "Hoshiarpur", "Jalandhar",
        "Kapurthala", "Ludhiana", "Malerkotla", "Mansa", "Moga", "Muktsar",
        "Pathankot", "Patiala", "Rupnagar", "Mohali", "Sangrur", "Nawanshahr",
        "Tarn Taran"
    ]
    
    for district in punjab_districts:
        print(get_hyperlocal_weather(district))