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
df = pd.read_csv(r"C:/Users/anmol/OneDrive/Desktop/AI_Krishidoot/backend/models/ProfitPilot/crop_data.csv")

# Parse date
df["Last Updated On"] = pd.to_datetime(df["Last Updated On"], format="%d %b %Y", errors="coerce")

# Drop rows where date or target price is missing
df = df.dropna(subset=["Last Updated On", "Average Price (INR)"])

# Convert numeric columns
for col in ["Maximum Price (INR)", "Average Price (INR)", "Minimum Price (INR)"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Extract date features
df["day"] = df["Last Updated On"].dt.day
df["month"] = df["Last Updated On"].dt.month
df["year"] = df["Last Updated On"].dt.year

# Encode categorical columns
label_encoders = {}
for col in ["District", "Market", "Commodity", "Variety"]:
    df[col] = df[col].fillna("Unknown")
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# ===============================
# Features and target
# ===============================
feature_cols = [
    "District",
    "Market",
    "Commodity",
    "Variety",
    "Minimum Price (INR)",
    "Maximum Price (INR)",
    "day",
    "month",
    "year"
]
X = df[feature_cols]
y = df["Average Price (INR)"]

# ===============================
# Train-test split
# ===============================
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
joblib.dump(model, "punjab_crop_price_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
print("Model and encoders saved successfully!")

# ===============================
# Historical averages for missing data
# ===============================
# Group by market & month for auto-filling
market_month_avg = df.groupby(["Market", "month"]).agg({
    "Minimum Price (INR)": "mean",
    "Maximum Price (INR)": "mean"
}).to_dict()

# ===============================
# Predict function
# ===============================
def predict_price(date_str, district_name, market_name, commodity_name, variety_name="Other", min_price=None, max_price=None):
    date_obj = datetime.strptime(date_str, "%d/%m/%Y")
    month = date_obj.month

    # Encode categorical inputs
    district_encoded = label_encoders["District"].transform([district_name])[0]
    market_encoded = label_encoders["Market"].transform([market_name])[0]
    commodity_encoded = label_encoders["Commodity"].transform([commodity_name])[0]
    variety_encoded = label_encoders["Variety"].transform([variety_name])[0]

    # Auto-fill missing values from historical averages
    if min_price is None or max_price is None:
        avg_min = market_month_avg[("Minimum Price (INR)")].get((market_encoded, month), df["Minimum Price (INR)"].median())
        avg_max = market_month_avg[("Maximum Price (INR)")].get((market_encoded, month), df["Maximum Price (INR)"].median())
        min_price = min_price if min_price is not None else avg_min
        max_price = max_price if max_price is not None else avg_max

    # Prepare input features
    features_df = pd.DataFrame([[
        district_encoded,
        market_encoded,
        commodity_encoded,
        variety_encoded,
        min_price,
        max_price,
        date_obj.day,
        date_obj.month,
        date_obj.year
    ]], columns=feature_cols)

    return model.predict(features_df)[0]

# ===============================
# Example usage
# ===============================
pred_price = predict_price(
    date_str="15/10/2025",
    district_name="Amritsar",
    market_name="Amritsar",
    commodity_name="Wheat",
    variety_name="Other"
)
print(f"Predicted Average Price: Rs. {pred_price:.2f}")
