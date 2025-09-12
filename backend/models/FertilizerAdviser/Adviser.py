import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import pickle  # --- NEW: Import the pickle library ---

# --- 1. Load the Dataset ---
# Reads your CSV file into a pandas DataFrame.
df = pd.read_csv(r"C:/Users/anmol/OneDrive/Desktop/AI_Krishidoot/backend/models/FertilizerAdviser/Fertilizer_recommendation.csv")
print("Successfully loaded data. Here are the first 5 rows:")
print(df.head())
print("\n" + "="*50 + "\n")

# --- 2. Data Preprocessing ---
# Clean up column names to remove any extra spaces.
df.columns = df.columns.str.strip()

# Convert categorical text data (Soil Type, Crop Type) into numerical labels
# that the model can understand.
label_encoders = {}
for col in ['Soil Type', 'Crop Type']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define the features (X) and the target (y).
# The model will use all columns except 'Fertilizer Name' to make predictions.
X = df.drop(columns=['Fertilizer Name'])
y = df['Fertilizer Name']

# --- 3. Splitting the Data ---
# Divide the dataset into two parts: one for training the model (80%)
# and one for testing its performance (20%).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4. Training the Model ---
# Initialize and train the RandomForestClassifier.
# This model is an ensemble of decision trees, making it robust and accurate.
print("Training the RandomForest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Model training complete!")
print("\n" + "="*50 + "\n")

# --- 5. Evaluating the Model ---
# Use the trained model to make predictions on the unseen test data.
print("Evaluating the model on the test data...")
y_pred = model.predict(X_test)

# Calculate the model's accuracy.
accuracy = accuracy_score(y_test, y_pred)
print(f" Model Accuracy: {accuracy:.2%}\n")

# Display a detailed report of the model's performance for each fertilizer type.
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("\n" + "="*50 + "\n")

# --- 6. Saving the Model --- (NEW SECTION)
# Use pickle to save the trained model to a file.
model_filename = 'fertilizer_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)

print(f"Model saved successfully to '{model_filename}'")

# Also save the label encoders, which are needed for new predictions
encoder_filename = 'label_encoders.pkl'
with open(encoder_filename, 'wb') as file:
    pickle.dump(label_encoders, file)

print(f"Label encoders saved successfully to '{encoder_filename}'")
