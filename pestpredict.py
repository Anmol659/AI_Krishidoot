import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib

def train_pest_prediction_ensemble(file_path: str):
    """
    Loads data, trains multiple classification models (Random Forest, Gradient
    Boosting, Logistic Regression, SVC), and combines them into a robust
    ensemble model using a VotingClassifier.
    """
    try:
        print(f"Loading environmental data from '{file_path}'...")
        df = pd.read_csv(r"C:/Users/anmol/Downloads/gujarat_environmental_data.csv")

        # --- 1. Feature Engineering & Target Creation ---
        print("Preparing data and creating pest risk variable...")
        df['pest_risk'] = ((df['temperature_celsius'] > 25) & (df['humidity'] > 60)).astype(int)

        features = ['temperature_celsius', 'humidity', 'precip_mm', 'wind_kph', 'cloud']
        X = df[features]
        y = df['pest_risk']

        # --- 2. Model Training ---
        print("Splitting data and training individual models...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Define the individual models
        clf1 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf2 = GradientBoostingClassifier(n_estimators=100, random_state=42)
        # For SVC and Logistic Regression, it's good practice to scale the data first
        clf3 = make_pipeline(StandardScaler(), LogisticRegression(random_state=42))
        clf4 = make_pipeline(StandardScaler(), SVC(probability=True, random_state=42))

        # --- 3. Ensemble Model Creation ---
        print("Creating and training the ensemble model (VotingClassifier)...")
        
        # Create the VotingClassifier
        # 'soft' voting uses the average of predicted probabilities, which is often better
        ensemble_model = VotingClassifier(
            estimators=[('rf', clf1), ('gb', clf2), ('lr', clf3), ('svc', clf4)],
            voting='soft'
        )

        # Train the ensemble model
        ensemble_model.fit(X_train, y_train)

        # --- 4. Evaluation and Saving ---
        print("\nEvaluating ensemble model performance...")
        predictions = ensemble_model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        print(f"Ensemble Model Accuracy: {accuracy * 100:.2f}%")
        print("\nEnsemble Classification Report:")
        print(classification_report(y_test, predictions))

        # Save the trained ensemble model
        model_filename = 'pestpredict_ensemble_model.joblib'
        joblib.dump(ensemble_model, model_filename)
        print(f"\nTrained PestPredict ensemble model saved as '{model_filename}'")
        
        return ensemble_model

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

# --- Main execution block ---
if __name__ == "__main__":
    trained_ensemble = train_pest_prediction_ensemble('gujarat_environmental_data.csv')
    
    if trained_ensemble:
        print("\n--- Example Prediction with Ensemble Model ---")
        # Predict the pest risk for the same sample weather conditions
        sample_conditions = [[32, 75, 0.0, 5, 50]]
        prediction_input = pd.DataFrame(sample_conditions, columns=['temperature_celsius', 'humidity', 'precip_mm', 'wind_kph', 'cloud'])
        
        risk_prediction = trained_ensemble.predict(prediction_input)
        risk_probability = trained_ensemble.predict_proba(prediction_input)

        risk_level = "High" if risk_prediction[0] == 1 else "Low"
        # The confidence is the average probability from all models in the ensemble
        confidence = risk_probability[0][risk_prediction[0]] * 100
        
        print(f"Predicted Pest Risk for sample conditions: {risk_level} (Confidence: {confidence:.2f}%)")
