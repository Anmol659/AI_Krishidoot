import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib # Used for saving and loading the trained model

def train_price_prediction_model(file_path: str):
    """
    Loads historical price data, engineers features, trains a RandomForestRegressor
    model, and saves it to a file. This creates the core of the ProfitPilot agent.
    """
    try:
        print(f"Loading historical data from '{file_path}'...")
        df = pd.read_csv(r"C:/Users/anmol/Downloads/gujarat_environmental_data.csv")

        # --- 1. Data Preparation and Feature Engineering ---
        print("Preparing data and creating time-based features...")

        # Convert 'price_date' to datetime objects
        df['price_date'] = pd.to_datetime(df['price_date'])

        # The target variable is 'modal_price'. We need to ensure it's numeric.
        # 'coerce' will turn any non-numeric prices into 'NaN' (Not a Number)
        df['modal_price'] = pd.to_numeric(df['modal_price'], errors='coerce')

        # Remove any rows where the price could not be converted
        df.dropna(subset=['modal_price'], inplace=True)

        # Create time-based features that the model can learn from
        df['year'] = df['price_date'].dt.year
        df['month'] = df['price_date'].dt.month
        df['day'] = df['price_date'].dt.day
        df['day_of_year'] = df['price_date'].dt.dayofyear # Helps capture seasonality

        # --- 2. Model Training ---
        print("Training the RandomForestRegressor model...")

        # Define our features (X) and the target we want to predict (y)
        features = ['year', 'month', 'day', 'day_of_year']
        X = df[features]
        y = df['modal_price']

        # Split data into a training set (to learn from) and a testing set (to evaluate on)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize and train the model.
        # n_estimators=100 means the model is an ensemble of 100 decision trees.
        # n_jobs=-1 uses all available CPU cores to speed up training.
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        # --- 3. Evaluation and Saving ---
        # Test the model's performance on the data it has never seen before
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        print(f"\nModel training complete. Mean Absolute Error: {mae:.2f}")
        print("This means the model's price predictions are, on average, off by this amount.")

        # Save the trained model to a file for later use
        model_filename = 'profitpilot_price_model.joblib'
        joblib.dump(model, model_filename)
        print(f"\nTrained ProfitPilot model saved as '{model_filename}'")
        
        return model

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

# --- Main execution block ---
if __name__ == "__main__":
    # Use the historical data file we created
    trained_model = train_price_prediction_model('historical_gujarat_cotton_prices.csv')
    
    if trained_model:
        # --- Example of how EcoAdvisor would use the trained model ---
        print("\n--- Example Prediction ---")
        # Let's predict the price for a future date during the next trading season
        future_date = pd.to_datetime('2025-11-15')
        
        # Create a DataFrame with the same features our model was trained on
        prediction_input = pd.DataFrame([{
            'year': future_date.year,
            'month': future_date.month,
            'day': future_date.day,
            'day_of_year': future_date.dayofyear
        }])
        
        predicted_price = trained_model.predict(prediction_input)
        print(f"Predicted cotton price for {future_date.date()}: Rs. {predicted_price[0]:.2f} per Quintal")

