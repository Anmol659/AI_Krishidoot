import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import datetime

class PricePredictor:
    """
    A dedicated module to train a model and predict crop prices based on historical
    market and weather data. This will be used by the EcoAdvisor's other agents.
    """
    # The __init__ method should take variable names, not the file paths themselves.
    def __init__(self, price_data_path, weather_data_path, location_data_path):
        """
        Initializes the predictor by loading data and training the ML model.
        """
        print("Initializing PricePredictor: Loading data and training model...")
        self.model = None
        self.feature_names = []
        self.is_ready = False
        try:
            # Load, clean, and merge data
            self.data = self._load_and_prepare_data(r"C:/Users/anmol/Downloads/final_project_dataset.csv", r"C:/Users/anmol/Downloads/Weather data.xlsx", r"C:/Users/anmol/Downloads/Location information.xlsx")
            if self.data is not None:
                # Train the model upon initialization
                self._train_model()
                self.is_ready = True
                print("PricePredictor is trained and ready.")
        except Exception as e:
            print(f"Error during PricePredictor initialization: {e}")

    def _load_and_prepare_data(self, price_path, weather_path, location_path):
        """
        Private method to load, merge, and engineer features from the datasets.
        """
        try:
            # Added encoding='ISO-8859-1' to handle potential file format issues.
            price_df = pd.read_csv(price_path, encoding='ISO-8859-1')
            # Use read_excel for .xlsx files
            weather_df = pd.read_excel(weather_path)
            location_df = pd.read_excel(location_path)
        except FileNotFoundError as e:
            print(f"Data file not found: {e}. Please ensure all files are present.")
            return None
        except Exception as e:
            # Catch other potential errors, like needing to install 'openpyxl'
            print(f"An error occurred loading data. You might need to run 'pip install openpyxl'. Details: {e}")
            return None

        # --- Focus on Cotton in Saurashtra (e.g., Rajkot market) ---
        df = price_df[price_df['Commodity'] == 'Cotton (Unginned)'].copy()
        df['Price Date'] = pd.to_datetime(df['Price Date'], errors='coerce')

        # --- Process and Merge Weather Data ---
        df['avg_temp'] = weather_df['temperature_celsius'].mean()
        df['precipitation'] = weather_df['precip_mm'].mean()
        df['humidity'] = weather_df['humidity'].mean()

        # --- Feature Engineering ---
        df['day_of_year'] = df['Price Date'].dt.dayofyear
        df['month'] = df['Price Date'].dt.month
        df['year'] = df['Price Date'].dt.year
        df['day_of_week'] = df['Price Date'].dt.dayofweek
        
        # Drop rows with missing dates or target values
        df.dropna(subset=['Price Date', 'Modal_Price'], inplace=True)
        return df

    def _train_model(self):
        """
        Trains the RandomForestRegressor model.
        """
        print("   -> Training price prediction model...")
        self.feature_names = ['day_of_year', 'month', 'year', 'day_of_week', 'avg_temp', 'precipitation', 'humidity']
        target = 'Modal_Price'

        X = self.data[self.feature_names]
        y = self.data[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.model.fit(X_train, y_train)
        
        accuracy = self.model.score(X_test, y_test)
        print(f"   -> Model training complete. Accuracy (R^2 Score): {accuracy:.2f}")

    def predict_price(self, future_sale_date, weather_scenario="Normal"):
        """
        Predicts the price for a future date and given weather scenario.
        """
        if not self.is_ready:
            return {"error": "Model is not ready for prediction."}

        print(f"   -> Generating prediction for {future_sale_date} with '{weather_scenario}' weather...")
        
        future_date = pd.to_datetime(future_sale_date)
        future_features_df = pd.DataFrame({
            'day_of_year': [future_date.timetuple().tm_yday],
            'month': [future_date.month],
            'year': [future_date.year],
            'day_of_week': [future_date.weekday()],
            'avg_temp': [self.data['avg_temp'].mean()],
            'precipitation': [self.data['precipitation'].mean()],
            'humidity': [self.data['humidity'].mean()]
        })

        if weather_scenario == "Dry Spell (Low Rain)":
            future_features_df['precipitation'] *= 0.5
            future_features_df['avg_temp'] *= 1.05
        elif weather_scenario == "Heavy Monsoon (High Rain)":
            future_features_df['precipitation'] *= 1.5
            future_features_df['avg_temp'] *= 0.98

        future_features_df = future_features_df[self.feature_names]

        predicted_price = self.model.predict(future_features_df)[0]
        
        return {"predicted_price_per_quintal": round(predicted_price, 2)}


# --- Example of how EcoAdvisor would use this module ---
if __name__ == '__main__':
    # When you run the script, it expects the data files to be in the same directory,
    # or you can provide the full path.
    price_predictor = PricePredictor(
        price_data_path=r"C:\Users\anmol\Downloads\final_project_dataset.csv",
        weather_data_path=r"C:\Users\anmol\Downloads\Weather data.xlsx",
        location_data_path=r"C:\Users\anmol\Downloads\Location information.xlsx"
    )
    
    print("\n" + "="*50)

    # Simulate a farmer's query
    farmer_sale_date = datetime.date.today() + datetime.timedelta(days=60)
    farmer_weather_outlook = "Dry Spell (Low Rain)"

    print(f"EcoAdvisor: Farmer wants to know the price forecast for a sale on {farmer_sale_date}.")
    
    if price_predictor.is_ready:
        forecast = price_predictor.predict_price(
            future_sale_date=farmer_sale_date,
            weather_scenario=farmer_weather_outlook
        )
        
        print("\n--- Price Forecast Result ---")
        if "error" in forecast:
            print(f"Could not generate forecast: {forecast['error']}")
        else:
            price = forecast['predicted_price_per_quintal']
            print(f"The predicted price for cotton is: {price:,.2f} INR per Quintal")
            print("This information will now be passed to the ProfitPilot agent for financial analysis.")
    else:
        print("Could not generate forecast because the PricePredictor failed to initialize.")
        
    print("="*50)