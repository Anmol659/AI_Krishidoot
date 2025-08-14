import requests

# API configuration
API_KEY = "b1e6b35205dc4102e12a861e549c56f4"  # Your OpenWeatherMap API key
BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

def get_hyperlocal_weather(location: str) -> str:
    """
    Fetches hyper-local weather data from OpenWeatherMap API.
    Args:
        location: City name (e.g., 'Rajkot', 'Jamnagar')
    Returns:
        A string describing current weather conditions or an error message.
    """
    params = {
        'q': f"{location},IN",  # City + country code for accuracy
        'appid': API_KEY,
        'units': 'metric'       # Temperature in Celsius
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

# Test mode (only runs when executed directly)
if __name__ == "__main__":
    print("--- Fetching Weather for Key Cities in Saurashtra ---")
    saurashtra_cities = [
        "Rajkot",
        "Jamnagar",
        "Bhavnagar",
        "Junagadh",
        "Porbandar",
        "Amreli",
        "Surendranagar"
    ]
    for city in saurashtra_cities:
        print(get_hyperlocal_weather(city))
