import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import joblib

# ===============================
# Load dataset
# ===============================
df = pd.read_csv(r"C:\Users\theha\Downloads\agmarknet_merged_monthwise.csv")

# Parse date
df["Arrival Date"] = pd.to_datetime(df["Arrival Date"], format="%d/%m/%Y", errors="coerce")

# Drop rows where date or target price is missing
df = df.dropna(subset=["Arrival Date", "Modal Price(Rs./Quintal)"])

# Convert numeric columns from string to float
for col in ["Arrivals (Tonnes)", "Minimum Price(Rs./Quintal)", "Maximum Price(Rs./Quintal)", "Modal Price(Rs./Quintal)"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Extract date features
df["day"] = df["Arrival Date"].dt.day
df["month"] = df["Arrival Date"].dt.month
df["year"] = df["Arrival Date"].dt.year

# Fill missing numeric columns
df["Arrivals (Tonnes)"] = df["Arrivals (Tonnes)"].fillna(0)
df["Minimum Price(Rs./Quintal)"] = df["Minimum Price(Rs./Quintal)"].fillna(df["Minimum Price(Rs./Quintal)"].median())
df["Maximum Price(Rs./Quintal)"] = df["Maximum Price(Rs./Quintal)"].fillna(df["Maximum Price(Rs./Quintal)"].median())

# Encode categorical columns
label_encoders = {}
for col in ["Market", "Variety"]:
    df[col] = df[col].fillna("Unknown")
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features and target
feature_cols = [
    "Market",
    "Arrivals (Tonnes)",
    "Variety",
    "Minimum Price(Rs./Quintal)",
    "Maximum Price(Rs./Quintal)",
    "day",
    "month",
    "year"
]
X = df[feature_cols]
y = df["Modal Price(Rs./Quintal)"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
model.fit(X_train, y_train)

# Predictions for evaluation
y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# ===============================
# Save the trained model & encoders
# ===============================
joblib.dump(model, "cotton_price_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
print("Model and encoders saved successfully!")

# ===============================
# Historical averages for missing data
# ===============================
# Group by market & month for auto-filling
market_month_avg = df.groupby(["Market", "month"]).agg({
    "Arrivals (Tonnes)": "mean",
    "Minimum Price(Rs./Quintal)": "mean",
    "Maximum Price(Rs./Quintal)": "mean"
}).to_dict()

# ===============================
# Predict function for LLM pipeline
# ===============================
def predict_price(date_str, market_name, variety_name="Other", arrivals=None, min_price=None, max_price=None):
    date_obj = datetime.strptime(date_str, "%d/%m/%Y")
    month = date_obj.month
    
    # Encode market & variety
    market_encoded = label_encoders["Market"].transform([market_name])[0]
    variety_encoded = label_encoders["Variety"].transform([variety_name])[0]
    
    # Auto-fill from historical averages if missing
    if arrivals is None or min_price is None or max_price is None:
        avg_arrivals = market_month_avg[("Arrivals (Tonnes)")].get((market_encoded, month), 0)
        avg_min = market_month_avg[("Minimum Price(Rs./Quintal)")].get((market_encoded, month), df["Minimum Price(Rs./Quintal)"].median())
        avg_max = market_month_avg[("Maximum Price(Rs./Quintal)")].get((market_encoded, month), df["Maximum Price(Rs./Quintal)"].median())
        arrivals = arrivals if arrivals is not None else avg_arrivals
        min_price = min_price if min_price is not None else avg_min
        max_price = max_price if max_price is not None else avg_max
    
    # Prepare input features
    features_df = pd.DataFrame([[
        market_encoded,
        arrivals,
        variety_encoded,
        min_price,
        max_price,
        date_obj.day,
        date_obj.month,
        date_obj.year
    ]], columns=feature_cols)
    
    return model.predict(features_df)[0]

# Example usage
pred_price = predict_price(date_str="15/10/2025", market_name="Amreli")
print(f"Predicted Modal Price: Rs. {pred_price:.2f}")